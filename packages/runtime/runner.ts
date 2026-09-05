import { promises as fs } from "node:fs"
import { execFile } from "node:child_process"
import path from "node:path"
import { promisify } from "node:util"
import { randomUUID, createHash } from "node:crypto"
import { digitizeRun } from "../../src/lib/digitizer"
import { preflightCaptureBytes } from "../../src/lib/digitizer/capture-preflight"
import { RESOURCE_ROOT, WORKSPACE_ROOT } from "../../src/lib/runtime-paths"
import {
  createStoredRunFromBytes, getRun, saveRun, compactStoredRun,
  appendRunReview, readVerifiedRunSource, readVerifiedRunAsset,
  tryClaimStoredRunProcessing, releaseStoredRunProcessingClaim,
  type RunRecord, type RunAssetKey,
} from "../../src/lib/runs"
import release from "../../config/waveparse-release.json"
import { runtimeDirectory } from "./runtime-location"

const execute = promisify(execFile)
const controller = new AbortController()
let activeId: string | undefined
let runtimeHash = ""
let device = "cpu"
const output = (value: unknown) => process.stdout.write(JSON.stringify(value) + "\n")
// Stdout is reserved exclusively for the versioned line protocol.
console.log = (...args) => console.error(...args)
process.on("SIGTERM", () => controller.abort(new Error("cancelled")))
process.on("SIGINT", () => controller.abort(new Error("cancelled")))

class RequestError extends Error {
  constructor(public code: string, message: string) { super(message) }
}

function textField(request: Record<string, unknown>, key: string): string {
  const value = request[key]
  if (typeof value !== "string" || !value.trim()) throw new RequestError("invalid_request", `${key} must be a non-empty string.`)
  return value
}

async function verifiedResult(run: RunRecord | null) {
  if (!run) return null
  await readVerifiedRunSource(run)
  const assets: Record<string, unknown> = {}
  for (const [key, asset] of Object.entries(run.assets)) {
    if (!asset) continue
    if (key !== "input") await readVerifiedRunAsset(run, key as Exclude<RunAssetKey, "input">)
    assets[key] = { ...asset, absolutePath: path.resolve(WORKSPACE_ROOT, asset.path) }
  }
  let waveparse = { version: release.version, runtimeManifestSha256: runtimeHash, device }
  try { waveparse = JSON.parse(await fs.readFile(path.join(WORKSPACE_ROOT, run.localPath, "waveparse.json"), "utf8")) } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error
  }
  return { schemaVersion: release.schemaVersion, ...run, waveparse, assets }
}

async function checkRuntime(runtimeDir: string) {
  const payload = JSON.parse(await fs.readFile(path.join(RESOURCE_ROOT, "payload-manifest.json"), "utf8")) as { files: Record<string, string> }
  for (const [relative, expected] of Object.entries(payload.files)) {
    const file = path.resolve(RESOURCE_ROOT, relative)
    if (!file.startsWith(RESOURCE_ROOT + path.sep) || createHash("sha256").update(await fs.readFile(file)).digest("hex") !== expected) throw new RequestError("runtime_unavailable", `Packaged resource integrity mismatch: ${relative}`)
  }
  const manifest = await fs.readFile(path.join(RESOURCE_ROOT, "config", "waveparse-runtime.json"))
  runtimeHash = createHash("sha256").update(manifest).digest("hex")
  const python = path.join(runtimeDir, "engine", ".venv", "bin", "python")
  try {
    await execute(python, [path.join(RESOURCE_ROOT, "runtime_setup.py"), "doctor", "--runtime-dir", runtimeDir, "--resource-root", RESOURCE_ROOT, "--device", device], {
      env: { ...process.env, WAVEPARSE_NODE: process.execPath },
      timeout: 360_000, maxBuffer: 1024 * 1024, signal: controller.signal,
    })
  } catch (error) {
    throw new RequestError("runtime_unavailable", `Run ecg-waveparse setup, then doctor. ${String((error as { stderr?: string }).stderr || (error as Error).message).slice(-2500)}`)
  }
}

async function recoverInterruptedRun(current: RunRecord | null) {
  if (current?.processing?.state !== "running" || !current.processing.ownerId?.startsWith("waveparse-")) return current
  const owner = `recovery-${process.pid}-${randomUUID()}`
  // Only reclaim a dead local owner; elapsed time alone cannot steal a live run.
  if (!await tryClaimStoredRunProcessing({ id: current.id, ownerId: owner, staleAfterMs: Number.MAX_SAFE_INTEGER })) return current
  try {
    const latest = await getRun(current.id)
    if (!latest || latest.processing?.state !== "running") return latest
    await readVerifiedRunSource(latest)
    latest.status = "failed"
    latest.message = "The local runner exited before completion; source preserved."
    latest.updatedAt = new Date().toISOString()
    latest.assets = { input: latest.assets.input }
    latest.processing = { ...latest.processing, state: "failed", failureCode: "worker_failure", finishedAt: latest.updatedAt }
    await saveRun(latest)
    return await compactStoredRun(latest.id)
  } finally { await releaseStoredRunProcessingClaim(current.id, owner) }
}

async function main() {
  let input = ""
  for await (const chunk of process.stdin) {
    input += chunk
    if (input.length > 64 * 1024) throw new RequestError("invalid_request", "Request is too large.")
  }
  const request = JSON.parse(input) as Record<string, unknown>
  if (!request || Array.isArray(request) || request.protocolVersion !== release.protocolVersion) throw new RequestError("invalid_request", "Unsupported protocol version.")
  if (path.resolve(textField(request, "workspaceDir")) !== WORKSPACE_ROOT) throw new RequestError("invalid_request", "Workspace mismatch.")
  if (WORKSPACE_ROOT === RESOURCE_ROOT || WORKSPACE_ROOT.startsWith(RESOURCE_ROOT + path.sep)) throw new RequestError("invalid_request", "Run storage must be outside the installed package.")
  const operation = textField(request, "operation")
  if (operation === "get" || operation === "review") {
    const id = textField(request, "runId")
    if (!/^run_[a-zA-Z0-9_-]+$/.test(id)) throw new RequestError("invalid_request", "Invalid WaveParse run ID.")
    const current = await recoverInterruptedRun(await getRun(id))
    if (operation === "get") return verifiedResult(current)
    if (!current) throw new RequestError("not_found", "Run not found.")
    if (current.processing?.state === "running") throw new RequestError("run_busy", "Cannot review a running job.")
    if (request.decision !== "accepted" && request.decision !== "rejected") throw new RequestError("invalid_request", "decision must be accepted or rejected.")
    const owner = `review-${process.pid}-${randomUUID()}`
    if (!await tryClaimStoredRunProcessing({ id, ownerId: owner, staleAfterMs: 120_000, now: Date.now() })) throw new RequestError("run_busy", "Another process is using this run.")
    try {
      await appendRunReview({ id, decision: request.decision, reviewer: textField(request, "reviewer"), notes: textField(request, "notes"), confirmations: request.confirmations as Parameters<typeof appendRunReview>[0]["confirmations"] })
      return verifiedResult(await compactStoredRun(id))
    } finally { await releaseStoredRunProcessingClaim(id, owner) }
  }
  if (operation !== "digitize") throw new RequestError("invalid_request", "Unknown operation.")
  const timeoutMs = request.timeoutMs ?? 1_800_000
  if (!Number.isSafeInteger(timeoutMs) || Number(timeoutMs) < 60_000 || Number(timeoutMs) > 7_200_000) throw new RequestError("invalid_request", "timeoutMs must be between 60000 and 7200000.")
  device = String(request.device ?? "cpu")
  if (device !== "cpu" && device !== "mps") throw new RequestError("invalid_request", "device must be cpu or mps.")
  const runtimeDir = request.runtimeDir ? path.resolve(textField(request, "runtimeDir")) : runtimeDirectory(RESOURCE_ROOT)
  process.env.OPEN_ECG_DIGITIZER_DIR = path.join(runtimeDir, "engine")
  process.env.ECG_DIGITIZER_DEVICE = device
  await checkRuntime(runtimeDir)
  const source = path.resolve(textField(request, "inputPath"))
  if (![".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"].includes(path.extname(source).toLowerCase())) throw new RequestError("invalid_input", "Unsupported image format.")
  const sourceStat = await fs.stat(source)
  if (!sourceStat.isFile() || sourceStat.size === 0 || sourceStat.size > 50 * 1024 * 1024) throw new RequestError("invalid_input", "Input must be a non-empty image file no larger than 50 MB.")
  const bytes = await fs.readFile(source)
  const captureQuality = await preflightCaptureBytes({ fileName: path.basename(source), bytes, captureMethod: "upload" })
  controller.signal.throwIfAborted()
  let run = await createStoredRunFromBytes({ id: `run_${randomUUID().replaceAll("-", "")}`, fileName: path.basename(source), bytes, captureQuality })
  activeId = run.id
  const owner = `waveparse-${process.pid}-${randomUUID()}`
  if (!await tryClaimStoredRunProcessing({ id: run.id, ownerId: owner, staleAfterMs: Number(timeoutMs) + 60_000, now: Date.now() })) throw new RequestError("run_busy", "Another process is using this run.")
  const startedAt = new Date().toISOString()
  run.processing = { version: 1, state: "running", attempt: 1, recoveryCount: 0, queuedAt: startedAt, startedAt, heartbeatAt: startedAt, deadlineAt: new Date(Date.now() + Number(timeoutMs)).toISOString(), timeoutMs: Number(timeoutMs), ownerId: owner }
  run.status = "running"
  await saveRun(run)
  await fs.writeFile(path.join(WORKSPACE_ROOT, run.localPath, "waveparse.json"), JSON.stringify({ version: release.version, runtimeManifestSha256: runtimeHash, device }), { mode: 0o600 })
  output({ type: "started", runId: run.id })
  let timedOut = false
  const deadline = setTimeout(() => { timedOut = true; controller.abort(new Error("timeout")) }, Number(timeoutMs))
  // Heartbeats update only mtime; metadata is written by a single run owner.
  const heartbeat = setInterval(() => { const now = new Date(); void fs.utimes(path.join(WORKSPACE_ROOT, run.localPath), now, now).catch(() => undefined) }, 5000)
  try {
    const patch = await digitizeRun(run, { signal: controller.signal })
    if (!controller.signal.aborted) run = { ...run, ...patch, assets: { ...run.assets, ...patch.assets } }
  } catch (error) {
    run.status = "failed"
    run.message = (error as Error).message
  } finally {
    clearTimeout(deadline); clearInterval(heartbeat)
    if (controller.signal.aborted) {
      run.status = timedOut ? "timed_out" : "failed"
      run.message = timedOut ? "Digitization timed out; source preserved." : "Digitization cancelled; source preserved."
      run.assets = { input: run.assets.input }
    }
    if (run.status === "running" || run.status === "pending_digitizer") { run.status = "failed"; run.message = "Engine did not produce a terminal result." }
    run.updatedAt = new Date().toISOString()
    const policyAbstention = run.status === "failed" && run.publicationDecision?.outcome === "failed"
    run.processing = { ...run.processing!, state: run.status === "timed_out" ? "timed_out" : run.status === "failed" ? "failed" : "completed", finishedAt: run.updatedAt, ...(run.status === "failed" && !policyAbstention ? { failureCode: "worker_failure" as const } : {}), ...(timedOut ? { failureCode: "job_timeout" as const } : {}) }
    try { await saveRun(run) } finally { await releaseStoredRunProcessingClaim(run.id, owner) }
    run = await compactStoredRun(run.id)
  }
  if (controller.signal.aborted && !timedOut) throw new RequestError("cancelled", "Digitization cancelled; admitted source preserved.")
  return verifiedResult(run)
}

main().then((result) => output({ type: "result", result })).catch((error) => {
  output({ type: "error", error: { code: controller.signal.aborted ? "cancelled" : error.code ?? "execution_error", message: error.message ?? String(error), ...(activeId ? { runId: activeId } : {}) } })
  process.exitCode = 1
})
