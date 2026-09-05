import { randomUUID } from "node:crypto"

import { digitizeRun } from "@/lib/digitizer"
import {
  compactStoredRun,
  getRun,
  releaseStoredRunProcessingClaim,
  tryClaimStoredRunProcessing,
  updateStoredRun,
  type DigitizerJobLifecycle,
  type RunRecord,
} from "@/lib/runs"

const DEFAULT_JOB_TIMEOUT_MS = 30 * 60 * 1_000
const MINIMUM_JOB_TIMEOUT_MS = 60_000
const MAXIMUM_JOB_TIMEOUT_MS = 2 * 60 * 60 * 1_000
const DEFAULT_HEARTBEAT_INTERVAL_MS = 5_000
const CLAIM_STALE_GRACE_MS = 60_000

type Digitize = (
  run: RunRecord,
  options: { signal: AbortSignal }
) => Promise<Partial<RunRecord>>

type JobDependencies = {
  getRun: typeof getRun
  updateRun: typeof updateStoredRun
  tryClaim: typeof tryClaimStoredRunProcessing
  releaseClaim: typeof releaseStoredRunProcessingClaim
  compact: typeof compactStoredRun
  digitize: Digitize
  now: () => number
  heartbeatIntervalMs: number
  timeoutOverrideMs?: number
  reportUnexpectedError?: (runId: string) => void
}

type QueueState = {
  waiting: string[]
  scheduled: Set<string>
  active: Set<string>
  draining: boolean
  idleWaiters: Set<() => void>
}

const queueGlobal = globalThis as typeof globalThis & {
  __ecgDigitizerLocalQueue?: QueueState
}

function queueState() {
  queueGlobal.__ecgDigitizerLocalQueue ??= {
    waiting: [],
    scheduled: new Set(),
    active: new Set(),
    draining: false,
    idleWaiters: new Set(),
  }
  return queueGlobal.__ecgDigitizerLocalQueue
}

const productionDependencies: JobDependencies = {
  getRun,
  updateRun: updateStoredRun,
  tryClaim: tryClaimStoredRunProcessing,
  releaseClaim: releaseStoredRunProcessingClaim,
  compact: compactStoredRun,
  digitize: (run, options) => digitizeRun(run, options),
  now: Date.now,
  heartbeatIntervalMs: DEFAULT_HEARTBEAT_INTERVAL_MS,
  reportUnexpectedError: (runId) => {
    console.error(
      `[local-digitizer] Job orchestration failed for ${runId}; the persisted active state remains available for recovery.`
    )
  },
}

export function configuredLocalJobTimeoutMs(
  value = process.env.ECG_DIGITIZER_JOB_TIMEOUT_MS
) {
  if (value === undefined || value.trim() === "") return DEFAULT_JOB_TIMEOUT_MS
  const parsed = Number(value)
  if (
    !Number.isSafeInteger(parsed) ||
    parsed < MINIMUM_JOB_TIMEOUT_MS ||
    parsed > MAXIMUM_JOB_TIMEOUT_MS
  ) {
    throw new Error(
      `ECG_DIGITIZER_JOB_TIMEOUT_MS must be an integer between ${MINIMUM_JOB_TIMEOUT_MS} and ${MAXIMUM_JOB_TIMEOUT_MS}.`
    )
  }
  return parsed
}

export async function enqueueLocalDigitization(runId: string) {
  const now = Date.now()
  const queuedAt = new Date(now).toISOString()
  const timeoutMs = configuredLocalJobTimeoutMs()
  const run = await updateStoredRun(runId, (current) => {
    if (current.source !== "upload") {
      throw new Error("Only local upload jobs can be queued for digitization.")
    }
    if (current.status === "queued" || current.status === "running") {
      return current
    }
    if (current.status !== "pending_digitizer") {
      throw new Error("Only a stored pending ECG can enter the digitizer queue.")
    }
    const previous = current.processing
    return {
      ...current,
      status: "queued",
      message: "Queued for local ECG digitization.",
      updatedAt: queuedAt,
      processing: {
        version: 1,
        state: "queued",
        attempt: (previous?.attempt ?? 0) + 1,
        recoveryCount: previous?.recoveryCount ?? 0,
        queuedAt,
        timeoutMs,
      },
    }
  })
  if (!run) throw new Error("Run not found.")
  scheduleLocalDigitization(run.id)
  return run
}

export function recoverLocalDigitizationJobs(runs: RunRecord[]) {
  for (const run of runs) {
    if (
      run.source === "upload" &&
      (run.status === "queued" || run.status === "running")
    ) {
      scheduleLocalDigitization(run.id)
    }
  }
}

function scheduleLocalDigitization(runId: string) {
  const state = queueState()
  if (state.scheduled.has(runId) || state.active.has(runId)) return
  state.scheduled.add(runId)
  state.waiting.push(runId)
  queueMicrotask(() => void drainLocalDigitizationQueue())
}

async function drainLocalDigitizationQueue() {
  const state = queueState()
  if (state.draining) return
  state.draining = true
  try {
    while (state.waiting.length > 0) {
      const runId = state.waiting.shift()
      if (!runId) continue
      state.scheduled.delete(runId)
      state.active.add(runId)
      try {
        await executeLocalDigitizationJobSafely(runId, productionDependencies)
      } finally {
        state.active.delete(runId)
      }
    }
  } finally {
    state.draining = false
    if (state.waiting.length === 0 && state.active.size === 0) {
      for (const resolve of state.idleWaiters) resolve()
      state.idleWaiters.clear()
    }
  }
}

async function executeLocalDigitizationJobSafely(
  runId: string,
  dependencies: JobDependencies
) {
  try {
    return await executeLocalDigitizationJob(runId, dependencies)
  } catch {
    dependencies.reportUnexpectedError?.(runId)
    return false
  }
}

async function executeLocalDigitizationJob(
  runId: string,
  dependencies: JobDependencies
) {
  const initial = await dependencies.getRun(runId)
  if (
    !initial ||
    initial.source !== "upload" ||
    (initial.status !== "queued" && initial.status !== "running")
  ) {
    return false
  }

  const timeoutMs =
    dependencies.timeoutOverrideMs ?? validTimeout(initial.processing?.timeoutMs)
  const ownerId = `local-${process.pid}-${randomUUID()}`
  const claimed = await dependencies.tryClaim({
    id: runId,
    ownerId,
    staleAfterMs: timeoutMs + CLAIM_STALE_GRACE_MS,
    now: dependencies.now(),
  })
  if (!claimed) return false

  let heartbeat: ReturnType<typeof setInterval> | undefined
  let heartbeatWrites = Promise.resolve()
  let timeout: ReturnType<typeof setTimeout> | undefined
  const controller = new AbortController()
  let timedOut = false

  try {
    const startedAtMs = dependencies.now()
    const startedAt = new Date(startedAtMs).toISOString()
    const deadlineAt = new Date(startedAtMs + timeoutMs).toISOString()
    const claimedRun = await dependencies.updateRun(runId, (current) => {
      if (current.status !== "queued" && current.status !== "running") {
        return current
      }
      const previous = lifecycleForRun(current, timeoutMs, startedAt)
      const recovered = current.status === "running"
      return {
        ...current,
        status: "running",
        message: recovered
          ? "Recovered an interrupted local digitization attempt."
          : "Digitizing this ECG locally.",
        updatedAt: startedAt,
        processing: {
          ...previous,
          state: "running",
          attempt: previous.attempt + (recovered ? 1 : 0),
          recoveryCount:
            previous.recoveryCount + (recovered ? 1 : 0),
          ownerId,
          startedAt,
          heartbeatAt: startedAt,
          deadlineAt,
          ...(recovered ? { recoveredAt: startedAt } : {}),
        },
      }
    })
    if (!claimedRun || claimedRun.processing?.ownerId !== ownerId) return false

    heartbeat = setInterval(() => {
      heartbeatWrites = heartbeatWrites
        .catch(() => undefined)
        .then(async () => {
          const heartbeatAt = new Date(dependencies.now()).toISOString()
          await dependencies.updateRun(runId, (current) => {
            if (
              current.status !== "running" ||
              current.processing?.ownerId !== ownerId
            ) {
              return current
            }
            return {
              ...current,
              updatedAt: heartbeatAt,
              processing: {
                ...current.processing,
                heartbeatAt,
              },
            }
          })
        })
    }, dependencies.heartbeatIntervalMs)

    timeout = setTimeout(() => {
      timedOut = true
      controller.abort(new Error("local-digitizer-job-timeout"))
    }, timeoutMs)

    let patch: Partial<RunRecord>
    try {
      patch = await dependencies.digitize(claimedRun, {
        signal: controller.signal,
      })
    } catch {
      patch = {}
    }

    if (heartbeat) clearInterval(heartbeat)
    heartbeat = undefined
    await heartbeatWrites.catch(() => undefined)
    const finishedAt = new Date(dependencies.now()).toISOString()

    if (timedOut) {
      await dependencies.updateRun(runId, (current) =>
        terminalLifecycleRun(current, ownerId, {
          status: "timed_out",
          state: "timed_out",
          failureCode: "job_timeout",
          message:
            "Local digitization exceeded its whole-job time limit and was stopped. The untouched source is preserved and can be run again.",
          finishedAt,
        })
      )
      await dependencies.compact(runId)
      return true
    }

    const terminalStatus = patch.status
    if (
      terminalStatus === undefined ||
      terminalStatus === "queued" ||
      terminalStatus === "running" ||
      terminalStatus === "timed_out"
    ) {
      await dependencies.updateRun(runId, (current) =>
        terminalLifecycleRun(current, ownerId, {
          status: "failed",
          state: "failed",
          failureCode: "worker_failure",
          message:
            "The local digitizer worker ended without a valid terminal result. The untouched source is preserved and can be run again.",
          finishedAt,
        })
      )
      await dependencies.compact(runId)
      return true
    }

    await dependencies.updateRun(runId, (current) => {
      if (current.processing?.ownerId !== ownerId) return current
      const blocked = terminalStatus === "pending_digitizer"
      const failed = terminalStatus === "failed"
      return {
        ...current,
        ...patch,
        assets: patch.assets
          ? { ...current.assets, ...patch.assets }
          : current.assets,
        updatedAt: finishedAt,
        processing: {
          ...current.processing,
          state: blocked ? "blocked" : failed ? "failed" : "completed",
          finishedAt,
          ...(blocked
            ? { failureCode: "digitizer_unavailable" as const }
            : {}),
        },
      }
    })
    await dependencies.compact(runId)
    return true
  } finally {
    if (heartbeat) clearInterval(heartbeat)
    if (timeout) clearTimeout(timeout)
    await heartbeatWrites.catch(() => undefined)
    await dependencies.releaseClaim(runId, ownerId).catch(() => false)
  }
}

function terminalLifecycleRun(
  current: RunRecord,
  ownerId: string,
  terminal: {
    status: "failed" | "timed_out"
    state: "failed" | "timed_out"
    failureCode: "worker_failure" | "job_timeout"
    message: string
    finishedAt: string
  }
) {
  if (current.processing?.ownerId !== ownerId) return current
  return {
    ...current,
    status: terminal.status,
    message: terminal.message,
    updatedAt: terminal.finishedAt,
    processing: {
      ...current.processing,
      state: terminal.state,
      failureCode: terminal.failureCode,
      finishedAt: terminal.finishedAt,
    },
  }
}

function lifecycleForRun(
  run: RunRecord,
  timeoutMs: number,
  queuedAt: string
): DigitizerJobLifecycle {
  return (
    run.processing ?? {
      version: 1,
      state: "queued",
      attempt: 1,
      recoveryCount: 0,
      queuedAt,
      timeoutMs,
    }
  )
}

function validTimeout(value: number | undefined) {
  return Number.isSafeInteger(value) && value! >= MINIMUM_JOB_TIMEOUT_MS
    ? Math.min(value!, MAXIMUM_JOB_TIMEOUT_MS)
    : DEFAULT_JOB_TIMEOUT_MS
}

async function waitForQueueIdleForTests() {
  const state = queueState()
  if (state.waiting.length === 0 && state.active.size === 0 && !state.draining) {
    return
  }
  await new Promise<void>((resolve) => state.idleWaiters.add(resolve))
}

export const localDigitizerJobTestUtils = {
  executeLocalDigitizationJob,
  executeLocalDigitizationJobSafely,
  waitForQueueIdleForTests,
}
