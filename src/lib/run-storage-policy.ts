import releaseRuntime from "../../config/release-runtime.json"
import { promises as fs } from "node:fs"

const CONTRACT_MINIMUM_FREE_BYTES =
  releaseRuntime.storage.minimumFreeBytes
const CONTRACT_MINIMUM_RUN_RESERVATION_BYTES =
  releaseRuntime.storage.minimumRunReservationBytes

export type RunStorageAdmission = {
  version: 1
  admittedAt: string
  minimumFreeBytes: number
  reservedBytes: number
}

export class RunStorageAdmissionError extends Error {
  readonly code: "insufficient_storage" | "storage_admission_busy"
  readonly status: 507 | 503

  constructor({
    code,
    message,
  }: {
    code: RunStorageAdmissionError["code"]
    message: string
  }) {
    super(message)
    this.name = "RunStorageAdmissionError"
    this.code = code
    this.status = code === "insufficient_storage" ? 507 : 503
  }
}

export function configuredStorageMinimumFreeBytes(
  value = process.env.ECG_DIGITIZER_MINIMUM_FREE_BYTES
) {
  if (value === undefined || value.trim() === "") {
    return CONTRACT_MINIMUM_FREE_BYTES
  }
  const parsed = Number(value)
  if (
    !Number.isSafeInteger(parsed) ||
    parsed < CONTRACT_MINIMUM_FREE_BYTES
  ) {
    throw new Error(
      `ECG_DIGITIZER_MINIMUM_FREE_BYTES must be an integer of at least ${CONTRACT_MINIMUM_FREE_BYTES}; the release safety floor cannot be lowered.`
    )
  }
  return parsed
}

export function requiredRunStorageReservationBytes(inputBytes: number) {
  if (!Number.isSafeInteger(inputBytes) || inputBytes < 0) {
    throw new Error("Run input size must be a non-negative safe integer.")
  }
  const scaledInputBytes = inputBytes * 16
  if (!Number.isSafeInteger(scaledInputBytes)) {
    throw new Error("Run storage reservation exceeds the safe integer range.")
  }
  return Math.max(CONTRACT_MINIMUM_RUN_RESERVATION_BYTES, scaledInputBytes)
}

export function assertRunStorageAdmission({
  availableBytes,
  activeReservedBytes,
  inputBytes,
  minimumFreeBytes = configuredStorageMinimumFreeBytes(),
}: {
  availableBytes: bigint
  activeReservedBytes: bigint
  inputBytes: number
  minimumFreeBytes?: number
}) {
  if (availableBytes < BigInt(0) || activeReservedBytes < BigInt(0)) {
    throw new Error("Storage byte counts cannot be negative.")
  }
  // Keep legacy run metadata/claim compatibility without reserving hypothetical
  // future outputs. Capacity depends on the actual bytes about to be written.
  const reservedBytes = requiredRunStorageReservationBytes(inputBytes)
  const { requiredAvailableBytes } = assertRunStorageWrite({
    availableBytes,
    writeBytes: inputBytes,
    minimumFreeBytes,
  })
  return {
    minimumFreeBytes,
    reservedBytes,
    requiredAvailableBytes,
  }
}

export function assertRunStorageWrite({
  availableBytes,
  writeBytes,
  minimumFreeBytes = configuredStorageMinimumFreeBytes(),
}: {
  availableBytes: bigint
  writeBytes: number | bigint
  minimumFreeBytes?: number
}) {
  if (availableBytes < BigInt(0)) {
    throw new Error("Storage byte counts cannot be negative.")
  }
  if (
    (typeof writeBytes === "number" && !Number.isSafeInteger(writeBytes)) ||
    writeBytes < 0
  ) {
    throw new Error("Imminent write size must be a non-negative safe byte count.")
  }
  if (!Number.isSafeInteger(minimumFreeBytes) || minimumFreeBytes < CONTRACT_MINIMUM_FREE_BYTES) {
    throw new Error(`Storage writes must preserve at least ${CONTRACT_MINIMUM_FREE_BYTES} free bytes.`)
  }
  const requiredAvailableBytes = BigInt(minimumFreeBytes) + BigInt(writeBytes)
  if (availableBytes < requiredAvailableBytes) {
    throw new RunStorageAdmissionError({
      code: "insufficient_storage",
      message: `Writing ${writeBytes} bytes needs ${requiredAvailableBytes} available bytes to preserve the ${minimumFreeBytes}-byte free-space floor; ${availableBytes} are available. The write was stopped. Preserve the checkpoint and free space before resuming.`,
    })
  }
  return { minimumFreeBytes, requiredAvailableBytes }
}

export async function assertRunStorageWriteAt(directory: string, writeBytes: number, minimumFreeBytes = configuredStorageMinimumFreeBytes()) {
  if (!Number.isSafeInteger(writeBytes) || writeBytes < 0) {
    throw new Error("Imminent write size must be a non-negative safe byte count.")
  }
  const stat = await fs.statfs(directory, { bigint: true })
  // A short file still consumes a complete filesystem allocation block.
  const blockBytes = stat.bsize
  if (blockBytes <= BigInt(0)) throw new Error("Invalid filesystem allocation block size.")
  const allocatedBytes = ((BigInt(writeBytes) + blockBytes - BigInt(1)) / blockBytes) * blockBytes
  const availableBytes = stat.bavail * blockBytes
  return { ...assertRunStorageWrite({
    availableBytes,
    writeBytes: allocatedBytes,
    minimumFreeBytes,
  }), availableBytes }
}

export const runStoragePolicy = {
  minimumFreeBytes: CONTRACT_MINIMUM_FREE_BYTES,
  minimumRunReservationBytes: CONTRACT_MINIMUM_RUN_RESERVATION_BYTES,
  // Interrupted intakes recorded under the prior policy remain recoverable.
  minimumRecordedFreeBytes: 5 * 1024 ** 3,
}
