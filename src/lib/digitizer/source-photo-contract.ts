import { createHash } from "node:crypto"
import { promises as fs } from "node:fs"
import path from "node:path"
import { EvidenceContractError } from "./physical-contracts"

export const SOURCE_PHOTO_FILES = [
  "timeseries-canonical-uv.csv", "qualitative-review-required.csv", "primary-II-uv.csv",
  "segments-and-review.csv", "source-path-evidence.json", "source-context.json",
  "source-overlay.png", "ecg-paper.png",
] as const

export type SourcePhotoAttempt = {
  version: 2
  method: "source-photo-waveform-v1"
  state: "diagnostic_available" | "preferred_timing_available" | "unresolved" | "failed"
  originalSourceSha256: string
  workingSourceSha256: string
  runtimeMs: number
  originalPublicationReason: string
  manifestSha256?: string
  message?: string
  productionEligible: false
  semanticIdentityConfirmed: false
  quantitativeUseApproved: false
}

export type SourcePhotoCandidate = {
  schemaVersion: 2
  method: "source-photo-waveform-v1"
  candidateId: "source-photo-diagnostic-v1"
  state: Exclude<SourcePhotoAttempt["state"], "failed">
  complete: true
  publicationOutcome: "diagnostic_only" | "unresolved"
  requiresReview: true
  productionEligible: false
  semanticIdentityConfirmed: false
  quantitativeUseApproved: false
  source: {
    sha256: string
    originalFileSha256: string
    decodedRasterSha256: string
    imageSize: [number, number]
    identityKind: "encoded-source"
    unchanged: true
    workingToOriginalTransformVerified: false
  }
  files: Record<string, { sha256: string; bytes: number }>
  branch: string
}

export type SourcePhotoExpectedSource = {
  originalSha256: string
  workingSha256: string
  imageSize: [number, number]
}

const isHash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v)
const fail = (): never => { throw new EvidenceContractError("invalid_source_photo_candidate", "The diagnostic photo candidate is incomplete or does not match its source and artifacts.") }
const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== "object" || Array.isArray(v)) return fail()
  return v as Record<string, unknown>
}

export function parseSourcePhotoCandidate(value: unknown, expected: SourcePhotoExpectedSource): SourcePhotoCandidate {
  const v = object(value), source = object(v.source), files = object(v.files)
  const available = v.state === "diagnostic_available"
  if (v.schemaVersion !== 2 || v.method !== "source-photo-waveform-v1" || v.candidateId !== "source-photo-diagnostic-v1" ||
      !["diagnostic_available", "preferred_timing_available", "unresolved"].includes(String(v.state)) ||
      v.complete !== true || v.requiresReview !== true || v.productionEligible !== false ||
      v.semanticIdentityConfirmed !== false || v.quantitativeUseApproved !== false ||
      v.clinicalMeaning !== "unclassified" || v.reviewFlagsAreQualitative !== true ||
      v.sourceIdentityState !== (available ? "diagnostic_source_supported" : "unresolved") ||
      v.quantitativeUncertainty !== "unavailable" || v.zeroReviewFlagDoesNotMeanSafe !== true ||
      v.sourceGapsMayNotBeFilled !== true || v.gapFillingApplied !== false || v.absoluteStLevelsEstablished !== false ||
      v.publicationOutcome !== (available ? "diagnostic_only" : "unresolved") ||
      v.selectedArm !== (available ? "guarded-rays" : null) || v.selectedDiagnosticArm !== (available ? "guarded-rays" : null) ||
      typeof v.branch !== "string" || !v.branch) return fail()
  if (!isHash(source.sha256) || !isHash(source.originalFileSha256) || !isHash(source.decodedRasterSha256) ||
      source.sha256 !== expected.workingSha256 || source.originalFileSha256 !== expected.originalSha256 ||
      source.identityKind !== "encoded-source" || source.unchanged !== true || source.workingToOriginalTransformVerified !== false ||
      !Array.isArray(source.imageSize) || source.imageSize.length !== 2 ||
      !source.imageSize.every((n, i) => Number.isSafeInteger(n) && n >= 200 && n === expected.imageSize[i])) return fail()
  const names = available ? [...SOURCE_PHOTO_FILES] : ["unresolved-evidence.json"]
  if (JSON.stringify(Object.keys(files).sort()) !== JSON.stringify(names.sort())) return fail()
  let bytes = 0
  for (const item of Object.values(files)) {
    const file = object(item)
    if (!isHash(file.sha256) || typeof file.bytes !== "number" || !Number.isSafeInteger(file.bytes) || file.bytes <= 0 || file.bytes > 128 * 1024 ** 2) return fail()
    bytes += file.bytes
  }
  if (bytes > 128 * 1024 ** 2) return fail()
  if (available && (v.units !== "uV" || v.sampleRateHz !== 500 || v.layout !== "standard_3x4_with_r1" ||
      v.canonicalII !== "rhythm_II" || v.primaryIIPreservedSeparately !== true || v.paperPixelsPerMm !== 10 ||
      JSON.stringify(v.overlaySourceOffsetPixels) !== "[0,44]" ||
      JSON.stringify(v.canonicalLeadOrder) !== JSON.stringify(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]))) return fail()
  if (!available && (v.units !== null || v.sampleRateHz !== null || v.layout !== null ||
      v.canonicalII !== null || v.primaryIIPreservedSeparately !== false ||
      v.paperPixelsPerMm !== null || v.overlaySourceOffsetPixels !== null ||
      JSON.stringify(v.canonicalLeadOrder) !== "[]")) return fail()
  return v as SourcePhotoCandidate
}

export async function verifySourcePhotoBundle(directory: string, expected: SourcePhotoExpectedSource) {
  if (!(await fs.lstat(directory)).isDirectory()) return fail()
  const manifestPath = path.join(directory, "candidate.json")
  const stat = await fs.lstat(manifestPath)
  if (!stat.isFile() || stat.size > 64 * 1024) return fail()
  const manifest = await fs.readFile(manifestPath)
  const candidate = parseSourcePhotoCandidate(JSON.parse(manifest.toString("utf8")), expected)
  const names = [...Object.keys(candidate.files), "candidate.json"].sort()
  if (JSON.stringify((await fs.readdir(directory)).sort()) !== JSON.stringify(names)) return fail()
  for (const [name, identity] of Object.entries(candidate.files)) {
    const file = path.join(directory, name), stat = await fs.lstat(file)
    if (!stat.isFile() || stat.size !== identity.bytes) return fail()
    if (createHash("sha256").update(await fs.readFile(file)).digest("hex") !== identity.sha256) return fail()
  }
  return { candidate, manifestSha256: createHash("sha256").update(manifest).digest("hex"), names }
}

export function validateSourcePhotoAttempt(v: SourcePhotoAttempt, originalSha256?: string): void {
  if (v.version !== 2 || v.method !== "source-photo-waveform-v1" ||
      !["diagnostic_available", "preferred_timing_available", "unresolved", "failed"].includes(v.state) ||
      !isHash(v.originalSourceSha256) || !isHash(v.workingSourceSha256) ||
      (originalSha256 && originalSha256 !== v.originalSourceSha256) ||
      !Number.isFinite(v.runtimeMs) || v.runtimeMs < 0 ||
      !["no_publishable_candidate", "semantic_identity_unconfirmed"].includes(v.originalPublicationReason) ||
      v.productionEligible !== false || v.semanticIdentityConfirmed !== false || v.quantitativeUseApproved !== false ||
      (v.state !== "failed" && !isHash(v.manifestSha256))) fail()
}
