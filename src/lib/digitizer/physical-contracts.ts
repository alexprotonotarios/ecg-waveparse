export type EvidenceState = "verified" | "inferred" | "unresolved" | "not_applicable"

export class EvidenceContractError extends Error {
  readonly code = "invalid_evidence_contract"
  constructor(readonly issue: string, message: string) {
    super(message)
    this.name = "EvidenceContractError"
  }
}

export function positivePhysicalValue(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}

export function unitFraction(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
}

export function nonnegativePhysicalValue(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
}

export function invertibleTransform(matrix: number[][]): boolean {
  if (matrix.length !== 3 || matrix.some(row => row.length !== 3 || !row.every(Number.isFinite))) return false
  // Normalize rows independently: a valid large translation must not look singular.
  const normalized = matrix.map(row => {
    const scale = Math.max(...row.map(Math.abs))
    return row.map(value => value / scale)
  })
  const [a, b, c] = normalized
  const determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
  return Number.isFinite(determinant) && Math.abs(determinant) > Number.EPSILON
}

/** Absence of this route-specific inference is not a calibration confirmation. */
export function sourceTimingEvidenceState(inference: { quantitativeCalibrationConfirmed?: boolean } | undefined): EvidenceState {
  if (!inference) return "not_applicable"
  return inference.quantitativeCalibrationConfirmed === true ? "verified" : "unresolved"
}
