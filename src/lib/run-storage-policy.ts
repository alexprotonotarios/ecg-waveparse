import releaseRuntime from "../../config/release-runtime.json"

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
  if (
    !Number.isSafeInteger(minimumFreeBytes) ||
    minimumFreeBytes < CONTRACT_MINIMUM_FREE_BYTES
  ) {
    throw new Error(
      `Storage admission must preserve at least ${CONTRACT_MINIMUM_FREE_BYTES} free bytes.`
    )
  }
  const reservedBytes = requiredRunStorageReservationBytes(inputBytes)
  const requiredAvailableBytes =
    BigInt(minimumFreeBytes) + activeReservedBytes + BigInt(reservedBytes)
  if (availableBytes < requiredAvailableBytes) {
    throw new RunStorageAdmissionError({
      code: "insufficient_storage",
      message:
        `This ECG was not stored because the run filesystem has ${availableBytes} available bytes but ${requiredAvailableBytes} are required to preserve the clinical storage floor and all active run reservations. Free space and retry; no run was admitted.`,
    })
  }
  return {
    minimumFreeBytes,
    reservedBytes,
    requiredAvailableBytes,
  }
}

export const runStoragePolicy = {
  minimumFreeBytes: CONTRACT_MINIMUM_FREE_BYTES,
  minimumRunReservationBytes: CONTRACT_MINIMUM_RUN_RESERVATION_BYTES,
}
