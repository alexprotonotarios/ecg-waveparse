import { execFile } from "node:child_process"
import { createHash, randomBytes } from "node:crypto"
import { promises as fs } from "node:fs"
import os from "node:os"
import path from "node:path"

import type {
  DigitizerCandidateCapabilities,
  DigitizerCandidateKind,
} from "@/lib/digitizer/domain"
import {
  isCaptureQualitySummary,
  type CaptureQualitySummary,
} from "@/lib/digitizer/capture-quality"
import type { PublicationReasonCode } from "@/lib/digitizer/publication-policy"
import { validateDurableRun } from "@/lib/digitizer/record-contracts"
import { describeOutcome } from "@/lib/digitizer/outcome-dimensions"
import {
  PERMANENT_EVIDENCE_ASSETS,
  quantitativeEvidenceAvailability,
  type QuantitativeEvidenceAvailability,
} from "@/lib/digitizer/evidence-retention"
import {
  canAcceptQuantitativeReview,
  hasAllReviewConfirmations,
  statusAfterReviewDecision,
  type DigitizerReviewConfirmations,
} from "@/lib/digitizer/review-policy"
import {
  assertRunStorageAdmission,
  requiredRunStorageReservationBytes,
  RunStorageAdmissionError,
  runStoragePolicy,
  type RunStorageAdmission,
} from "@/lib/run-storage-policy"

export const ASSET_KEYS = [
  "input",
  "preparedInput",
  "geometryCorrectedInput",
  "artifactPreprocessedInput",
  "annotationMask",
  "diagnostic",
  "paperRender",
  "probability",
  "canonicalCsv",
  "segmentsCsv",
  "uncertaintyCsv",
  "segmentMapJson",
  "metadataCsv",
  "provenanceJson",
  "reviewJson",
] as const

export type RunAssetKey = (typeof ASSET_KEYS)[number]

export type RunStatus =
  | "completed"
  | "needs_review"
  | "partial"
  | "queued"
  | "running"
  | "pending_digitizer"
  | "timed_out"
  | "failed"

export type DigitizerJobState =
  | "queued"
  | "running"
  | "completed"
  | "blocked"
  | "timed_out"
  | "failed"

export type DigitizerJobLifecycle = {
  version: 1
  state: DigitizerJobState
  attempt: number
  recoveryCount: number
  queuedAt: string
  timeoutMs: number
  ownerId?: string
  startedAt?: string
  heartbeatAt?: string
  deadlineAt?: string
  recoveredAt?: string
  finishedAt?: string
  failureCode?:
    | "digitizer_unavailable"
    | "job_timeout"
    | "worker_failure"
    | "intake_interrupted"
    | "intake_corrupt"
}

export type RunAsset = {
  label: string
  path: string
  mimeType?: string
  sizeBytes?: number
  identity?: RunAssetIdentity
}

export type RunAssetIdentity = {
  version: 1
  algorithm: "sha256"
  sha256: string
  sizeBytes: number
  sourceSha256: string
  recordedAt: string
}

export type RunSourceIdentity = {
  version: 1
  algorithm: "sha256"
  sha256: string
  sizeBytes: number
  recordedAt: string
}

export class RunSourceIntegrityError extends Error {
  readonly code = "source_integrity_mismatch"
  readonly status = 409

  constructor(message: string) {
    super(message)
    this.name = "RunSourceIntegrityError"
  }
}

export class RunArtifactIntegrityError extends Error {
  readonly code = "artifact_integrity_mismatch"
  readonly status = 409

  constructor(message: string) {
    super(message)
    this.name = "RunArtifactIntegrityError"
  }
}

export type RunRecord = {
  executionProfile?: { version: 1; stageDurationsMs: Record<string, number>; runnerMaxRssBytes: number; scope: string }
  id: string
  createdAt: string
  updatedAt: string
  status: RunStatus
  source: "reference" | "upload"
  fileName: string
  fileSizeBytes?: number
  sourceIdentity?: RunSourceIdentity
  captureQuality?: CaptureQualitySummary
  localPath: string
  message?: string
  layout?: string
  layoutCost?: number
  sampleRateHz?: number
  effectiveSampleRateHz?: number
  paperSpeedMmPerSecond?: number
  gainMmPerMv?: number
  calibrationConfidence?: number
  leadCount?: number
  selectedCandidateId?: string
  publicationDecision?: DigitizerPublicationDecision
  digitizer?: DigitizerRunSummary
  reliability?: DigitizerReliabilitySummary
  review?: DigitizerReviewSummary
  processing?: DigitizerJobLifecycle
  storage?: RunStorageAdmission
  retention?: RunRetentionSummary
  quantitativeEvidence?: QuantitativeEvidenceAvailability
  outcomeDimensions?: ReturnType<typeof describeOutcome>
  assets: Partial<Record<RunAssetKey, RunAsset>>
}

export type RunRetentionSummary = {
  version: 1 | 2
  policy: "lean-final-evidence-v1" | "lean-final-evidence-v2"
  compactedAt: string
  retainedAssets: RunAssetKey[]
  removedAssets: RunAssetKey[]
  reviewArtifactsRetained: boolean
  reclaimedBytes: number
  cumulativeReclaimedBytes: number
}

export type DigitizerQaSeverity = "warning" | "error"

export type DigitizerQaWarning = {
  lead?: string
  severity: DigitizerQaSeverity
  message: string
}

export type DigitizerLeadQa = {
  expectedSamples?: number
  finiteSamples: number
  missingSamples: number
  maxGapSamples: number
  outsideSegmentSamples: number
  candidateSpreadUvP95?: number
  candidateSpreadUvMax?: number
  uncertainSamples?: number
}

export type DigitizerQa = {
  passed: boolean
  summary: string
  warnings: DigitizerQaWarning[]
  leads: Record<string, DigitizerLeadQa>
  score?: number
  scoreBreakdown?: DigitizerCandidateScoreBreakdown
}

export type DigitizerInputVariant =
  | "original"
  | "annotation-masked"
  | "preprocessed"
  | "geometry-corrected"
  | "artifact-preprocessed"
export type DigitizerVectorizer =
  | "probability-centroid"
  | "dynamic-path"
  | "native-grid-path"

export type DigitizerSourceFidelityLead = {
  evidenceMedian: number
  evidenceP10: number
  maxNativeJumpPixels: number
  p95NativeJumpPixels: number
  coverage: number
  sourceStartPixel: number
  sourceEndPixel: number
  homeRowFraction?: number
  largeJumpCount?: number
  unsupportedLargeJumpCount?: number
  minimumLargeJumpSupport?: number
  qrsAnchorShiftPixels?: number
  qrsAnchorMatchCount?: number
  largeExcursionCount?: number
  recoveredExcursionCount?: number
  recoveredSampleCount?: number
  rejectedExcursionCount?: number
  rejectedSampleCount?: number
  disconnectedExcursionCount?: number
  verticalArtifactExcursionCount?: number
  unsafeExcursionCount?: number
}

export type DigitizerSourceFidelitySummary = {
  passed: boolean
  method: string
  pixelsPerMm: number
  layoutConfidence: number
  evidenceMedian: number
  evidenceP10: number
  maxNativeJumpPixels: number
  p95NativeJumpPixels: number
  minimumCoverage: number
  minimumRequiredCoverage?: number
  sourcePanelTimingDetected?: boolean
  rowLocalSourceTimingDetected?: boolean
  rowTimeOriginConsensus?: {
    method: string
    accepted: boolean
    applied: boolean
    correctedRowCount: number
    maximumCorrectedRows: number
    originTolerancePixels: number
    maximumRawResidualPixels: number
    rawStarts: number[]
    predictedStarts: number[]
    correctedStarts: number[] | null
  }
  sourceTimingInference?: {
    method: string
    inferredPixelsPerMm: number
    rawGridPeriodPixels: number
    inferredGridScaleMm: number
    gridScaleErrorFraction: number
    quantitativeCalibrationConfirmed: boolean
    quantitativeCalibrationMethod?: string
    calibrationConfidence?: number
    calibrationPixelsPerMmX?: number
    calibrationPixelsPerMmY?: number
    calibrationAgreementLimitFraction?: number
    sourceStartPixel?: number
    sourceEndPixel?: number
    sourceSpanFraction?: number
    correctedRowCount?: number
    projectedRowCount?: number
    maximumCorrectedRows?: number
    boundaryTolerancePixels?: number
    rawStarts?: number[]
    rawEnds?: number[]
    correctedStarts?: number[]
    correctedEnds?: number[]
    globalTimingRange?: [number, number]
    globalTimingConsistent?: boolean
    globalTimingTolerancePixels?: number
  }
  inkConnectedTransitionRecovery?: {
    method: string
    transitionScale: number
    displacementRewardPerPixel: number
    maximumMissingInkFraction: number
    maximumMissingInkPixelsFloor: number
  }
  minimumHomeRowFraction?: number
  largeJumpCount?: number
  unsupportedLargeJumpCount?: number
  maximumUnsupportedLargeJumps?: number
  rhythmAnchorCount?: number
  rhythmTracePassed?: boolean
  recoveredExcursionCount?: number
  recoveredSampleCount?: number
  rejectedExcursionCount?: number
  rejectedSampleCount?: number
  unsafeExcursionCount?: number
  leadOrderValidation?: {
    passed?: boolean
    selectedOrder?: string | null
    method?: string
  }
  leadLabelValidation?: {
    passed?: boolean
    order?: string | null
    method?: string
    semanticIdentityConfirmed?: boolean
  }
  precordialLabelValidation?: {
    passed?: boolean
    method?: string
    semanticIdentityConfirmed?: boolean
  }
  limbLabelValidation?: {
    passed?: boolean
    order?: string | null
    method?: string
    semanticIdentityConfirmed?: boolean
  }
  deterministicUpscale?: {
    method: string
    originalMaxDimension: number
    targetMaxDimension: number
    effectiveSampleRateCappedToSource: boolean
  }
  boundedSourceRaster?: {
    method: string
    originalWidth: number
    originalHeight: number
    workingWidth: number
    workingHeight: number
    scaleX: number
    scaleY: number
    morphologyReconstructed: false
  }
  leadMetrics: Record<string, DigitizerSourceFidelityLead>
}

export type DigitizerCandidateScoreBreakdown = {
  errorPenalty: number
  warningPenalty: number
  missingPenalty: number
  gapPenalty: number
  layoutPenalty: number
  annotationRiskPenalty: number
  permissiveThresholdPenalty: number
  candidateDisagreementPenalty: number
  layoutCalibrationPenalty: number
}

export type DigitizerPanelTimingCorrection = {
  column: number
  sourceStart: number
  sourceEnd: number
  scale: number
  referenceLeads: string[]
}

export type DigitizerComputeDevice = "cpu" | "mps"

export type DigitizerCandidateSummary = {
  id: string
  label: string
  localPath?: string
  workingFilesRetained?: boolean
  selected?: boolean
  leadSources?: Record<string, string>
  intervalLineage?: Record<string, import("@/lib/digitizer/lineage").IntervalLineage[]>
  leadCorroborators?: Record<string, string[]>
  status: "completed" | "failed"
  parameters: {
    kind?: DigitizerCandidateKind
    capabilities?: DigitizerCandidateCapabilities
    resampleSize: number
    upscaleToMaxDimension?: number
    darkInkEnhancement?: boolean
    darkInkSupportRadius?: number
    gainMmPerMv?: number
    labelThresh?: number
    layoutConstraint?: string
    geometryConfirmedLayout?: boolean
    geometryLayoutConfidence?: number
    semanticLeadIdentityConfirmed?: boolean
    semanticLeadIdentityMethod?: string
    semanticLeadIdentityOrder?: "standard" | "cabrera"
    semanticLeadIdentityLayout?: string
    adaptivePreprocessingEligible?: boolean
    artifactPreprocessingEligible?: boolean
    maximumLayoutCost?: number
    cropBox?: {
      left: number
      top: number
      right: number
      bottom: number
    }
    selectionEligible?: boolean
    inputVariant?: DigitizerInputVariant
    vectorizer?: DigitizerVectorizer
    device?: DigitizerComputeDevice
    planningPhase?: "core" | "recovery" | "exhaustive"
    scheduleReason?: string
  }
  layout?: string
  layoutCost?: number
  effectiveSampleRateHz?: number
  panelTimingCorrections?: DigitizerPanelTimingCorrection[]
  sourceFidelity?: DigitizerSourceFidelitySummary
  runtimeMs?: number
  score?: number
  scoreBreakdown?: DigitizerCandidateScoreBreakdown
  featureCacheHit?: boolean
  qa?: DigitizerQa
  message?: string
}

export type DigitizerRunSummary = {
  engine: "Open-ECG-Digitizer"
  selectedCandidateId?: string
  candidates: DigitizerCandidateSummary[]
  evidence?: DigitizerPipelineEvidence
}

export type DigitizerStabilityEvidence = {
  required: boolean
  reasons: string[]
  outcome:
    | "not-required"
    | "confirmed-repeat"
    | "confirmed-cpu"
    | "unstable"
  repeatCandidateIds: string[]
  cpuCandidateIds: string[]
  repeatAgreementLeadCount: number
  confirmedLeadCount: number
}

export type DigitizerPipelineEvidence = {
  version: 1
  selectorCalibrationProfileId: string
  neuralEscalated: boolean
  escalationReasons: string[]
  coreAgreementReached: boolean
  recoveryExpanded: boolean
  candidatePlan: Array<{
    candidateId: string
    phase: "core" | "recovery" | "exhaustive"
    reason: string
    executed: boolean
  }>
  selection?: {
    selectedCandidateId: string
    strongestAlternativeId?: string
    selectedSource: "native" | "neural" | "fused"
    reasonCode: PublicationReasonCode
    scoreGap?: number
    calibration: {
      source: string
      artifactContext?: string
      confidence:
        | "validation-activated-context"
        | "layout-only"
        | "uncalibrated"
      trainingCaseCount: number
      trainingPipelineCommit: string
      calibrationContractVersion: number
      requiredCalibrationContractVersion: number
      priorsApplied: boolean
      heldoutUsed: boolean
      exactCandidateCalibrated: boolean
      outOfDomainReasons: string[]
    }
    semanticLeadIdentity: {
      passed: boolean
      explicitlyAnchored: boolean
      inferredFromLayout?: boolean
      requiresManualVerification?: boolean
      expectedLeads: string[]
      sourceCandidateIds: string[]
    }
    quantitativeOutputEligible: boolean
  }
  stability: DigitizerStabilityEvidence
  layoutDetection: {
    outcome: "detected" | "not-detected" | "failed"
    inputVariant?: "original" | "annotation-masked" | "geometry-corrected"
    coordinateSpace?: "working" | "geometry-corrected"
    layoutHint?: string
    confidence?: number
    failure?: { method: string; message: string }
    semanticRecognition?: {
      passed: boolean
      method: string
      confidence: number
      recognizedLabels: string[]
      failureReasons: string[]
      engine?: {
        name?: string | null
        revision?: number | null
        platform?: string | null
        platformVersion?: string | null
      }
    }
  }
  neuralFeatureCache: { hits: number; misses: number }
}

export type DigitizerPublicationDecision = {
  policyId: string
  policyVersion: number
  outcome: "needs_review" | "partial" | "failed"
  reasonCode: PublicationReasonCode
  candidateId?: string
}

export type AnnotationComponent = {
  x0: number
  y0: number
  x1: number
  y1: number
  pixels: number
  fillFraction: number
  dominantColor: "red" | "blue"
}

export type DigitizerReliabilitySummary = {
  version: number
  originalPreserved: true
  morphologyReconstructed: false
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
  annotationMaskedPixels: number
  annotationMaskedFraction: number
  annotationComponents: AnnotationComponent[]
  nominalSampleRateHz: number
  effectiveSampleRateHz: number
  uncertainSampleCount: number
  missingSampleCount: number
  confidence?: "standard" | "lower"
  confidenceReasons?: string[]
  inputQualityOutcome?: "acceptable" | "review" | "insufficient"
  inputQualityReasons?: string[]
  reviewRequired: true
}

export type DigitizerReviewDecision = "accepted" | "rejected"

export type DigitizerReviewEvent = {
  id: string
  createdAt: string
  decision: DigitizerReviewDecision
  reviewer: string
  notes: string
  selectedCandidateId?: string
  sourceSha256?: string
  canonicalSha256?: string
  evidenceSha256?: Partial<Record<Exclude<RunAssetKey, "input" | "reviewJson">, string>>
  confirmations?: DigitizerReviewConfirmations
}

export type DigitizerReviewSummary = {
  decision: DigitizerReviewDecision
  reviewer: string
  notes: string
  updatedAt: string
  eventCount: number
  confirmations?: DigitizerReviewConfirmations
}

export type KnownDigitization = {
  sourcePath: string
  status: RunStatus
  message: string
  layout?: string
  layoutCost?: number
  sampleRateHz?: number
  leadCount?: number
  assets: Partial<Record<Exclude<RunAssetKey, "input">, string>>
}

import { WORKSPACE_ROOT as ROOT_DIR } from "@/lib/runtime-paths"
import { LOCAL_REFERENCE_DIGITIZATIONS } from "@/lib/local-reference"
const STORAGE_DIR = path.join(
  /* turbopackIgnore: true */ ROOT_DIR,
  "storage",
  "runs"
)
const METADATA_FILE = "metadata.json"
const JOB_CLAIM_FILE = ".digitizer-job-claim.json"
const STORAGE_ADMISSION_CLAIM_FILE = ".storage-admission-claim.json"
const RUN_INTAKE_PREFIX = ".run-intake-"
const RUN_INTAKE_JOURNAL_FILE = ".run-intake.json"
const RUN_INTAKE_RECOVERY_FILE = ".run-intake-recovery.json"
const STORAGE_ADMISSION_STALE_MS = 10 * 60 * 1_000
const STORAGE_ADMISSION_WAIT_MS = 5_000
const STORAGE_ADMISSION_RETRY_MS = 25
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024
const MAX_DERIVED_ASSET_BYTES = 256 * 1024 * 1024
const PINNED_FILE_READER = String.raw`
const fs = require("node:fs")
const path = require("node:path")

async function main() {
  const [fileName, expectedDirectory] = process.argv.slice(1)
  if (
    !fileName ||
    path.basename(fileName) !== fileName ||
    fileName === "." ||
    fileName === ".." ||
    !path.isAbsolute(expectedDirectory)
  ) {
    throw new Error("invalid pinned file request")
  }

  const pinnedDirectory = await fs.promises.realpath(".")
  if (pinnedDirectory !== expectedDirectory) {
    throw new Error("file directory is redirected")
  }

  const handle = await fs.promises.open(
    fileName,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW
  )
  try {
    const stat = await handle.stat()
    if (!stat.isFile()) throw new Error("target is not a regular file")
    const bytes = await handle.readFile()
    if (bytes.byteLength !== stat.size) {
      throw new Error("target changed while being read")
    }
    await new Promise((resolve, reject) => {
      process.stdout.write(bytes, (error) =>
        error ? reject(error) : resolve()
      )
    })
  } finally {
    await handle.close()
  }
}

main().catch((error) => {
  process.stderr.write(error instanceof Error ? error.message : "pinned file read failed")
  process.exitCode = 74
})
`
const runMutationGates = new Map<string, Promise<void>>()
let storageAdmissionGate = Promise.resolve()

type RunIntakeFaultPoint = "journal" | "source" | "metadata" | "commit"

type StoredRunIntakeJournal = {
  version: 1
  runId: string
  fileName: string
  safeFileName: string
  fileSizeBytes: number
  sourceSha256: string
  createdAt: string
  storage: RunStorageAdmission
  captureQuality?: CaptureQualitySummary
  owner: {
    ownerId: string
    pid: number
    hostname: string
  }
}

const KNOWN_DIGITIZATIONS = LOCAL_REFERENCE_DIGITIZATIONS

export function uploadLimitBytes() {
  return MAX_UPLOAD_BYTES
}

export async function createStoredRunFromBytes({
  id,
  fileName,
  bytes,
  captureQuality,
}: {
  id: string
  fileName: string
  bytes: Buffer
  captureQuality?: CaptureQualitySummary
}) {
  return createStoredRunFromBytesInternal({
    id,
    fileName,
    bytes,
    captureQuality,
  })
}

async function createStoredRunFromBytesInternal(
  {
    id,
    fileName,
    bytes,
    captureQuality,
  }: {
    id: string
    fileName: string
    bytes: Buffer
    captureQuality?: CaptureQualitySummary
  },
  fault?: (point: RunIntakeFaultPoint) => void | Promise<void>
) {
  if (bytes.byteLength > MAX_UPLOAD_BYTES) {
    throw new Error("The ECG image is larger than the 50 MB local upload limit.")
  }
  return withStorageAdmission(async (ownerId) => {
    await fs.mkdir(STORAGE_DIR, { recursive: true, mode: 0o700 })
    await fs.chmod(STORAGE_DIR, 0o700)
    await reconcileStoredRunIntakesUnlocked()
    const storedRuns = await listStoredRuns()
    const activeReservedBytes = storedRuns.reduce(
      (total, run) => total + activeRunReservationBytes(run),
      BigInt(0)
    )
    const stat = await fs.statfs(STORAGE_DIR, { bigint: true })
    const admission = assertRunStorageAdmission({
      availableBytes: stat.bavail * stat.bsize,
      activeReservedBytes,
      inputBytes: bytes.byteLength,
    })
    const runDir = runDirectory(id)
    const stagingDir = path.join(
      STORAGE_DIR,
      `${RUN_INTAKE_PREFIX}${id}-${cryptoRandom()}`
    )
    try {
      await fs.access(runDir)
      throw new Error(`Run ${id} already exists; choose a new run id.`)
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error
    }
    const createdAt = new Date().toISOString()
    const safeFileName = sanitizeFileName(fileName)
    const storage: RunStorageAdmission = {
      version: 1,
      admittedAt: createdAt,
      minimumFreeBytes: admission.minimumFreeBytes,
      reservedBytes: admission.reservedBytes,
    }
    const sourceSha256 = createHash("sha256").update(bytes).digest("hex")
    const sourceIdentity: RunSourceIdentity = {
      version: 1,
      algorithm: "sha256",
      sha256: sourceSha256,
      sizeBytes: bytes.byteLength,
      recordedAt: createdAt,
    }
    if (
      captureQuality &&
      (captureQuality.outcome !== "ready" ||
        captureQuality.sourceSha256 !== sourceSha256)
    ) {
      throw new Error(
        "The acquisition assessment is not a ready assessment for these exact source bytes."
      )
    }
    const journal: StoredRunIntakeJournal = {
      version: 1,
      runId: id,
      fileName,
      safeFileName,
      fileSizeBytes: bytes.byteLength,
      sourceSha256,
      createdAt,
      storage,
      ...(captureQuality ? { captureQuality } : {}),
      owner: {
        ownerId,
        pid: process.pid,
        hostname: os.hostname(),
      },
    }
    let stagingCreated = false
    let committed = false
    try {
      await fs.mkdir(stagingDir, { mode: 0o700 })
      stagingCreated = true
      await syncDirectory(STORAGE_DIR)
      await atomicWritePrivateFile(
        path.join(stagingDir, RUN_INTAKE_JOURNAL_FILE),
        `${JSON.stringify(journal, null, 2)}\n`
      )
      await fault?.("journal")

      const stagedInputDir = path.join(stagingDir, "input")
      const stagedInputPath = path.join(stagedInputDir, safeFileName)
      await fs.mkdir(stagedInputDir, { mode: 0o700 })
      await atomicWritePrivateFile(stagedInputPath, bytes, true)
      await syncDirectory(stagingDir)
      await fault?.("source")

      const inputPath = relativePath(
        joinRuntimePath(runDir, "input", safeFileName)
      )
      const initialRun = createInitialStoredRun({
        id,
        fileName,
        fileSizeBytes: bytes.byteLength,
        inputPath,
        createdAt,
        sourceIdentity,
        storage,
        captureQuality,
        inputAsset: {
          label: "Uploaded ECG",
          path: normalizeRelativePath(inputPath),
          mimeType: contentTypeForPath(inputPath),
          sizeBytes: bytes.byteLength,
        },
      })
      await writeRunToDirectory(initialRun, stagingDir)
      await syncDirectory(stagingDir)
      await fault?.("metadata")

      await fs.rename(stagingDir, runDir)
      committed = true
      await syncDirectory(STORAGE_DIR)
      await fault?.("commit")
      const run = await createRunFromStoredInput({
        id,
        fileName,
        fileSizeBytes: bytes.byteLength,
        inputPath,
        sourceSha256,
        storage,
        captureQuality,
      })
      await fs.rm(path.join(runDir, RUN_INTAKE_JOURNAL_FILE), {
        force: true,
      })
      await syncDirectory(runDir)
      return run
    } catch (error) {
      if (stagingCreated && !committed) {
        await fs.rm(stagingDir, { recursive: true, force: true })
        await syncDirectory(STORAGE_DIR).catch(() => undefined)
      }
      throw error
    }
  })
}

export function allowedUploadExtension(fileName: string) {
  const ext = path.extname(fileName).toLowerCase()
  return [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"].includes(ext)
}

export function sanitizeFileName(fileName: string) {
  const cleaned = fileName
    .replace(/[/\\?%*:|"<>]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 160)

  return cleaned || "ecg-upload.png"
}

export function contentTypeForPath(filePath: string) {
  const ext = path.extname(filePath).toLowerCase()

  if (ext === ".png") return "image/png"
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg"
  if (ext === ".webp") return "image/webp"
  if (ext === ".tif" || ext === ".tiff") return "image/tiff"
  if (ext === ".csv") return "text/csv; charset=utf-8"
  if (ext === ".json") return "application/json; charset=utf-8"

  return "application/octet-stream"
}

export async function listRuns(): Promise<RunRecord[]> {
  await reconcileStoredRunIntakes()
  const [referenceRun, storedRuns] = await Promise.all([
    getReferenceRun(),
    listStoredRuns(),
  ])

  return [referenceRun, ...storedRuns]
    .filter((run): run is RunRecord => Boolean(run))
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
}

export async function reconcileStoredRunIntakes() {
  return withStorageAdmission(() => reconcileStoredRunIntakesUnlocked())
}

export async function getRun(id: string): Promise<RunRecord | null> {
  if (id === referenceRunId()) {
    return getReferenceRun()
  }
  const stored = await readStoredRun(id)
  if (stored && !(await committedIntakeJournalExists(id))) return stored
  await reconcileStoredRunIntakes()
  return readStoredRun(id)
}

async function committedIntakeJournalExists(id: string) {
  try {
    await fs.access(path.join(runDirectory(id), RUN_INTAKE_JOURNAL_FILE))
    return true
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false
    throw error
  }
}

export async function createRunFromStoredInput({
  id,
  fileName,
  fileSizeBytes,
  inputPath,
  sourceSha256,
  storage,
  captureQuality,
}: {
  id: string
  fileName: string
  fileSizeBytes: number
  inputPath: string
  sourceSha256?: string
  storage?: RunStorageAdmission
  captureQuality?: CaptureQualitySummary
}): Promise<RunRecord> {
  const now = new Date().toISOString()
  const verifiedSource = await readVerifiedInputFile({
    id,
    inputPath,
    expectedSizeBytes: fileSizeBytes,
    expectedSha256: sourceSha256,
  })
  const inputAsset: RunAsset = {
    label: "Uploaded ECG",
    path: normalizeRelativePath(inputPath),
    mimeType: contentTypeForPath(inputPath),
    sizeBytes: verifiedSource.bytes.byteLength,
  }
  const matchedDigitization = await findKnownDigitization(
    verifiedSource.sha256
  )
  const run = createInitialStoredRun({
    id,
    fileName,
    fileSizeBytes,
    inputPath,
    createdAt: now,
    sourceIdentity: {
      version: 1,
      algorithm: "sha256",
      sha256: verifiedSource.sha256,
      sizeBytes: verifiedSource.bytes.byteLength,
      recordedAt: now,
    },
    storage,
    captureQuality,
    inputAsset,
  })

  if (matchedDigitization) {
    const copiedAssets = await copyKnownAssets(
      id,
      matchedDigitization,
      verifiedSource.sha256
    )
    run.status = matchedDigitization.status
    run.message = matchedDigitization.message
    run.layout = matchedDigitization.layout
    run.layoutCost = matchedDigitization.layoutCost
    run.sampleRateHz = matchedDigitization.sampleRateHz
    run.leadCount = matchedDigitization.leadCount
    run.assets = {
      ...run.assets,
      ...copiedAssets,
    }
  }

  await writeRun(run)
  return run
}

function createInitialStoredRun({
  id,
  fileName,
  fileSizeBytes,
  inputPath,
  createdAt,
  sourceIdentity,
  storage,
  captureQuality,
  inputAsset,
}: {
  id: string
  fileName: string
  fileSizeBytes: number
  inputPath: string
  createdAt: string
  sourceIdentity: RunSourceIdentity
  storage?: RunStorageAdmission
  captureQuality?: CaptureQualitySummary
  inputAsset: RunAsset
}): RunRecord {
  return {
    id,
    createdAt,
    updatedAt: createdAt,
    status: "pending_digitizer",
    source: "upload",
    fileName,
    fileSizeBytes,
    sourceIdentity,
    localPath: relativePath(runDirectory(id)),
    message:
      "Input saved locally. Configure or run Open-ECG-Digitizer to attach exports for this ECG.",
    ...(storage ? { storage } : {}),
    ...(captureQuality ? { captureQuality } : {}),
    assets: {
      input: {
        ...inputAsset,
        path: normalizeRelativePath(inputPath),
      },
    },
  }
}

export async function readVerifiedRunSource(run: RunRecord) {
  if (run.source !== "upload") {
    throw new RunSourceIntegrityError(
      "Only locally admitted uploads have a verifiable source identity."
    )
  }

  const inputAsset = run.assets.input
  if (!inputAsset) {
    throw new RunSourceIntegrityError(
      "The untouched source image is missing from this job."
    )
  }

  const identity = run.sourceIdentity
  if (identity) {
    if (
      identity.version !== 1 ||
      identity.algorithm !== "sha256" ||
      !isSha256(identity.sha256) ||
      !Number.isSafeInteger(identity.sizeBytes) ||
      identity.sizeBytes < 0 ||
      identity.sizeBytes > MAX_UPLOAD_BYTES ||
      !Number.isFinite(Date.parse(identity.recordedAt))
    ) {
      throw new RunSourceIntegrityError(
        "The recorded source identity is invalid."
      )
    }
    if (
      run.fileSizeBytes !== undefined &&
      run.fileSizeBytes !== identity.sizeBytes
    ) {
      throw new RunSourceIntegrityError(
        "The recorded source size metadata is inconsistent."
      )
    }
  }

  const legacyReliabilitySha256 = run.reliability?.sourceSha256
  if (!identity) {
    if (legacyReliabilitySha256 === undefined) {
      throw new RunSourceIntegrityError(
        "This legacy job has no trustworthy admitted-source hash. Upload the original ECG as a new job before viewing, reviewing, rerunning, or digitizing it."
      )
    }
    if (!isSha256(legacyReliabilitySha256)) {
      throw new RunSourceIntegrityError(
        "The legacy source identity is invalid."
      )
    }
  }

  return readVerifiedInputFile({
    id: run.id,
    inputPath: inputAsset.path,
    expectedSizeBytes: identity?.sizeBytes ?? run.fileSizeBytes,
    expectedSha256: identity?.sha256 ?? legacyReliabilitySha256,
  })
}

const DERIVED_ASSET_DIRECTORY: Record<
  Exclude<RunAssetKey, "input">,
  "exports" | "preprocessing" | "review"
> = {
  preparedInput: "preprocessing",
  geometryCorrectedInput: "preprocessing",
  artifactPreprocessedInput: "preprocessing",
  annotationMask: "preprocessing",
  diagnostic: "exports",
  paperRender: "exports",
  probability: "exports",
  canonicalCsv: "exports",
  segmentsCsv: "exports",
  uncertaintyCsv: "exports",
  segmentMapJson: "exports",
  metadataCsv: "exports",
  provenanceJson: "preprocessing",
  reviewJson: "review",
}

export async function readVerifiedRunAsset(
  run: RunRecord,
  key: Exclude<RunAssetKey, "input">
) {
  if (run.source !== "upload") {
    throw new RunArtifactIntegrityError(
      "Only locally admitted uploads have persistent derived-output identities."
    )
  }

  const asset = run.assets[key]
  if (!asset) return null
  const identity = asset.identity
  if (!identity) {
    throw new RunArtifactIntegrityError(
      "This derived output predates persistent artifact identity. Re-digitize the trusted source before viewing, downloading, rendering, or reviewing it."
    )
  }
  if (
    identity.version !== 1 ||
    identity.algorithm !== "sha256" ||
    !isSha256(identity.sha256) ||
    !isSha256(identity.sourceSha256) ||
    !Number.isSafeInteger(identity.sizeBytes) ||
    identity.sizeBytes < 0 ||
    identity.sizeBytes > MAX_DERIVED_ASSET_BYTES ||
    !Number.isFinite(Date.parse(identity.recordedAt))
  ) {
    throw new RunArtifactIntegrityError(
      "The recorded derived-output identity is invalid."
    )
  }
  if (asset.sizeBytes !== undefined && asset.sizeBytes !== identity.sizeBytes) {
    throw new RunArtifactIntegrityError(
      "The recorded derived-output size metadata is inconsistent."
    )
  }

  const verifiedSource = await readVerifiedRunSource(run)
  if (identity.sourceSha256 !== verifiedSource.sha256) {
    throw new RunArtifactIntegrityError(
      "The derived output is not bound to this job's verified source image."
    )
  }

  const expectedRunDirectory = runDirectory(run.id)
  const expectedAssetRoot = joinRuntimePath(
    expectedRunDirectory,
    DERIVED_ASSET_DIRECTORY[key]
  )
  const absolutePath = resolveWorkspacePath(asset.path)
  const relativeToAssetRoot = path.relative(expectedAssetRoot, absolutePath)
  if (
    !relativeToAssetRoot ||
    relativeToAssetRoot.startsWith(`..${path.sep}`) ||
    relativeToAssetRoot === ".." ||
    path.isAbsolute(relativeToAssetRoot)
  ) {
    throw new RunArtifactIntegrityError(
      "The derived output is not bound to this job's private artifact directory."
    )
  }

  try {
    const parentDirectory = path.dirname(absolutePath)
    const rootRealPath = await fs.realpath(ROOT_DIR)
    const expectedRealParentDirectory = path.join(
      rootRealPath,
      path.relative(ROOT_DIR, parentDirectory)
    )
    const bytes = await readFileFromPinnedDirectory({
      directory: parentDirectory,
      fileName: path.basename(absolutePath),
      expectedRealDirectory: expectedRealParentDirectory,
      maximumBytes: identity.sizeBytes,
    })
    if (bytes.byteLength !== identity.sizeBytes) {
      throw new RunArtifactIntegrityError(
        "The derived output size no longer matches its recorded identity."
      )
    }
    const digest = createHash("sha256").update(bytes).digest("hex")
    if (digest !== identity.sha256) {
      throw new RunArtifactIntegrityError(
        "The derived output hash no longer matches its recorded identity."
      )
    }
    return { run, asset, identity, absolutePath, bytes }
  } catch (error) {
    if (error instanceof RunArtifactIntegrityError) throw error
    throw new RunArtifactIntegrityError(
      "The derived output could not be safely opened and verified."
    )
  }
}

async function readVerifiedInputFile({
  id,
  inputPath,
  expectedSizeBytes,
  expectedSha256,
}: {
  id: string
  inputPath: string
  expectedSizeBytes?: number
  expectedSha256?: string
}) {
  if (
    expectedSizeBytes !== undefined &&
    (!Number.isSafeInteger(expectedSizeBytes) ||
      expectedSizeBytes < 0 ||
      expectedSizeBytes > MAX_UPLOAD_BYTES)
  ) {
    throw new RunSourceIntegrityError("The recorded source size is invalid.")
  }
  if (expectedSha256 !== undefined && !isSha256(expectedSha256)) {
    throw new RunSourceIntegrityError("The recorded source hash is invalid.")
  }

  const expectedRunDirectory = runDirectory(id)
  const expectedInputDirectory = path.join(expectedRunDirectory, "input")
  const absolutePath = resolveWorkspacePath(inputPath)
  if (path.dirname(absolutePath) !== expectedInputDirectory) {
    throw new RunSourceIntegrityError(
      "The source asset is not bound to this job's private input directory."
    )
  }

  try {
    const rootRealPath = await fs.realpath(ROOT_DIR)
    const expectedRealInputDirectory = path.join(
      rootRealPath,
      path.relative(ROOT_DIR, expectedInputDirectory)
    )
    const bytes = await readFileFromPinnedDirectory({
      directory: expectedInputDirectory,
      fileName: path.basename(absolutePath),
      expectedRealDirectory: expectedRealInputDirectory,
      maximumBytes: MAX_UPLOAD_BYTES,
    })
    if (
      expectedSizeBytes !== undefined &&
      bytes.byteLength !== expectedSizeBytes
    ) {
      throw new RunSourceIntegrityError(
        "The source image size no longer matches its admitted identity."
      )
    }
    const sourceSha256 = createHash("sha256").update(bytes).digest("hex")
    if (expectedSha256 !== undefined && sourceSha256 !== expectedSha256) {
      throw new RunSourceIntegrityError(
        "The source image hash no longer matches its admitted identity."
      )
    }

    return { absolutePath, bytes, sha256: sourceSha256 }
  } catch (error) {
    if (error instanceof RunSourceIntegrityError) throw error
    throw new RunSourceIntegrityError(
      "The untouched source image could not be safely opened and verified."
    )
  }
}

function readFileFromPinnedDirectory({
  directory,
  fileName,
  expectedRealDirectory,
  maximumBytes,
}: {
  directory: string
  fileName: string
  expectedRealDirectory: string
  maximumBytes: number
}) {
  return new Promise<Buffer>((resolve, reject) => {
    execFile(
      process.execPath,
      ["-e", PINNED_FILE_READER, fileName, expectedRealDirectory],
      {
        cwd: /* turbopackIgnore: true */ directory,
        encoding: "buffer",
        env: { NODE_ENV: process.env.NODE_ENV ?? "production" },
        maxBuffer: maximumBytes + 64 * 1024,
        timeout: 15_000,
        windowsHide: true,
      },
      (error, stdout) => {
        if (error || !Buffer.isBuffer(stdout)) {
          reject(error ?? new Error("Pinned source reader returned text."))
          return
        }
        resolve(stdout)
      }
    )
  })
}

export async function createRerunFromRun(
  sourceRun: RunRecord,
  captureQuality = sourceRun.captureQuality
) {
  if (sourceRun.source !== "upload") {
    throw new Error("Only local upload jobs can be run again.")
  }

  const { bytes } = await readVerifiedRunSource(sourceRun)
  const runId = createRunId()
  const rerun = await createStoredRunFromBytes({
    id: runId,
    fileName: sourceRun.fileName,
    bytes,
    captureQuality,
  })
  return rerun
}

export async function saveRun(run: RunRecord) {
  await withRunMutation(run.id, () => writeRun(run))
}

export async function updateStoredRun(
  id: string,
  update: (run: RunRecord) => RunRecord | Promise<RunRecord>
) {
  return withRunMutation(id, async () => {
    const run = await readStoredRun(id)
    if (!run) return null
    const updated = await update(run)
    if (updated.id !== id) {
      throw new Error("A stored run update cannot change the run id.")
    }
    await writeRun(updated)
    return updated
  })
}

export async function appendRunReview({
  id,
  decision,
  reviewer,
  notes,
  confirmations,
}: {
  id: string
  decision: DigitizerReviewDecision
  reviewer: string
  notes: string
  confirmations?: Partial<DigitizerReviewConfirmations>
}) {
  return withRunMutation(id, () =>
    appendRunReviewUnlocked({
      id,
      decision,
      reviewer,
      notes,
      confirmations,
    })
  )
}

async function appendRunReviewUnlocked({
  id,
  decision,
  reviewer,
  notes,
  confirmations,
}: {
  id: string
  decision: DigitizerReviewDecision
  reviewer: string
  notes: string
  confirmations?: Partial<DigitizerReviewConfirmations>
}) {
  const run = await readStoredRun(id)
  if (!run) throw new Error("Run not found.")
  if (run.source !== "upload") {
    throw new Error("Reference runs cannot be reviewed from the local workspace.")
  }

  const verifiedSource = await readVerifiedRunSource(run)
  const verifiedDerivedAssets = new Map<
    Exclude<RunAssetKey, "input">,
    NonNullable<Awaited<ReturnType<typeof readVerifiedRunAsset>>>
  >()
  for (const key of ASSET_KEYS) {
    if (key === "input" || key === "reviewJson" || !run.assets[key]) continue
    const verified = await readVerifiedRunAsset(run, key)
    if (verified) verifiedDerivedAssets.set(key, verified)
  }

  const cleanReviewer = reviewer.trim().slice(0, 120)
  const cleanNotes = notes.trim().slice(0, 4_000)
  if (!cleanReviewer) throw new Error("Reviewer name or identifier is required.")
  if (!cleanNotes) throw new Error("Review notes are required.")
  if (decision === "accepted" && !canAcceptQuantitativeReview(run)) {
    throw new Error(
      "Only a quantitative needs-review result with canonical and compact CSV assets can be accepted."
    )
  }
  if (
    decision === "accepted" &&
    run.reliability?.sourceSha256 !== verifiedSource.sha256
  ) {
    throw new Error(
      "The quantitative result is not cryptographically bound to the verified source image and cannot be accepted."
    )
  }
  if (decision === "accepted" && !hasAllReviewConfirmations(confirmations)) {
    throw new Error(
      "All source, lead-identity, and scale/gap review confirmations are required for acceptance."
    )
  }
  const acceptedConfirmations =
    decision === "accepted" && hasAllReviewConfirmations(confirmations)
      ? confirmations
      : undefined

  const canonicalSha256 = verifiedDerivedAssets.get("canonicalCsv")?.identity.sha256
  const createdAt = new Date().toISOString()
  const event: DigitizerReviewEvent = {
    id: `review_${createHash("sha256")
      .update(`${id}:${createdAt}:${cleanReviewer}:${decision}`)
      .digest("hex")
      .slice(0, 10)}`,
    createdAt,
    decision,
    reviewer: cleanReviewer,
    notes: cleanNotes,
    selectedCandidateId: run.selectedCandidateId,
    sourceSha256: verifiedSource.sha256,
    canonicalSha256,
    evidenceSha256: Object.fromEntries(
      [...verifiedDerivedAssets].map(([key, asset]) => [key, asset.identity.sha256])
    ),
    ...(acceptedConfirmations ? { confirmations: acceptedConfirmations } : {}),
  }
  const reviewDir = path.join(runDirectory(id), "review")
  const reviewPath = path.join(reviewDir, "review_audit.json")
  let events: DigitizerReviewEvent[] = []

  try {
    const verifiedReview = await readVerifiedRunAsset(run, "reviewJson")
    if (!verifiedReview) {
      await fs.access(reviewPath)
      throw new RunArtifactIntegrityError(
        "An existing review audit trail has no persistent artifact identity. Preserve it for forensics and re-digitize the trusted source before recording another review."
      )
    }
    const existing = JSON.parse(verifiedReview.bytes.toString("utf8")) as {
      events?: DigitizerReviewEvent[]
    }
    if (Array.isArray(existing.events)) events = existing.events
  } catch (error) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error
  }

  events.push(event)
  await fs.mkdir(reviewDir, { recursive: true, mode: 0o700 })
  await fs.chmod(reviewDir, 0o700)
  await atomicWritePrivateFile(
    reviewPath,
    `${JSON.stringify({ version: 1, runId: id, events }, null, 2)}\n`
  )
  run.review = {
    decision,
    reviewer: cleanReviewer,
    notes: cleanNotes,
    updatedAt: createdAt,
    eventCount: events.length,
    ...(acceptedConfirmations ? { confirmations: acceptedConfirmations } : {}),
  }
  run.assets.reviewJson = await assetFromRelativePath(
    "Review audit trail",
    relativePath(reviewPath),
    undefined,
    verifiedSource.sha256
  )
  run.status = statusAfterReviewDecision(
    decision,
    run.publicationDecision?.outcome
  )
  run.message =
    decision === "accepted"
      ? `Accepted after visual review by ${cleanReviewer}.`
      : `Rejected during visual review by ${cleanReviewer}; correction or re-digitization is required.`
  run.updatedAt = createdAt
  await writeRun(run)
  return run
}

const RUN_WORKING_DIRECTORIES = [
  "candidates",
  "candidate-inputs",
  "neural-cache",
] as const

export async function compactStoredRun(id: string) {
  return withRunMutation(id, async () => {
    const run = await readStoredRun(id)
    if (!run) throw new Error("Run not found.")
    if (run.source !== "upload" || !isTerminalStoredRun(run)) return run

    await readVerifiedRunSource(run)
    const retainedKeys = ASSET_KEYS.filter((key) => {
      if (!run.assets[key]) return false
      return PERMANENT_EVIDENCE_ASSETS.has(key)
    })
    const reviewArtifactsRetained = retainedKeys.some(
      (key) => key === "segmentsCsv" || key === "uncertaintyCsv"
    )
    const retainedKeySet = new Set(retainedKeys)
    const removedAssets = ASSET_KEYS.filter(
      (key) => run.assets[key] && !retainedKeySet.has(key)
    )
    const retainedPaths = new Set<string>()
    const retainedAssets: RunRecord["assets"] = {}

    for (const key of retainedKeys) {
      const asset = run.assets[key]
      if (!asset) continue
      if (key === "input") {
        retainedAssets.input = asset
        continue
      }
      const verified = await readVerifiedRunAsset(run, key)
      if (!verified) continue
      retainedAssets[key] = asset
      retainedPaths.add(verified.absolutePath)
    }

    const compactedAt = new Date().toISOString()
    const previousRemovedAssets = run.retention?.removedAssets ?? []
    const cumulativeRemovedAssets = ASSET_KEYS.filter(
      (key) => previousRemovedAssets.includes(key) || removedAssets.includes(key)
    )
    const compacted: RunRecord = {
      ...run,
      digitizer: run.digitizer
        ? {
            ...run.digitizer,
            candidates: run.digitizer.candidates.map((candidate) => {
              const next = { ...candidate }
              delete next.localPath
              next.workingFilesRetained = false
              return next
            }),
          }
        : undefined,
      assets: retainedAssets,
      retention: {
        version: 2,
        policy: "lean-final-evidence-v2",
        compactedAt,
        retainedAssets: retainedKeys,
        removedAssets: cumulativeRemovedAssets,
        reviewArtifactsRetained,
        reclaimedBytes: 0,
        cumulativeReclaimedBytes:
          run.retention?.cumulativeReclaimedBytes ?? 0,
      },
    }
    compacted.quantitativeEvidence = quantitativeEvidenceAvailability(compacted)
    compacted.outcomeDimensions = describeOutcome(compacted)

    const runDir = runDirectory(id)
    await atomicWritePrivateFile(
      path.join(runDir, METADATA_FILE),
      `${JSON.stringify(compacted, null, 2)}\n`
    )

    let reclaimedBytes = 0
    for (const directoryName of RUN_WORKING_DIRECTORIES) {
      reclaimedBytes += await removeTreeAndCount(
        path.join(runDir, directoryName)
      )
    }
    for (const directoryName of ["exports", "preprocessing", "review"] as const) {
      reclaimedBytes += await pruneDirectoryToRetainedFiles(
        path.join(runDir, directoryName),
        retainedPaths
      )
    }

    compacted.retention = {
      ...compacted.retention!,
      reclaimedBytes,
      cumulativeReclaimedBytes:
        (run.retention?.cumulativeReclaimedBytes ?? 0) + reclaimedBytes,
    }
    await writeRun(compacted)
    return compacted
  })
}

async function removeTreeAndCount(target: string) {
  const bytes = await logicalTreeBytes(target)
  await fs.rm(target, { recursive: true, force: true })
  return bytes
}

async function logicalTreeBytes(target: string): Promise<number> {
  let stat
  try {
    stat = await fs.lstat(target)
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return 0
    throw error
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) return stat.size
  const entries = await fs.readdir(target)
  let total = 0
  for (const entry of entries) {
    total += await logicalTreeBytes(path.join(target, entry))
  }
  return total
}

async function pruneDirectoryToRetainedFiles(
  directory: string,
  retainedPaths: ReadonlySet<string>
): Promise<number> {
  let entries
  try {
    entries = await fs.readdir(directory, { withFileTypes: true })
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return 0
    throw error
  }

  let reclaimedBytes = 0
  for (const entry of entries) {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      reclaimedBytes += await pruneDirectoryToRetainedFiles(
        target,
        retainedPaths
      )
      try {
        await fs.rmdir(target)
      } catch (error) {
        if (
          !isNodeError(error) ||
          (error.code !== "ENOENT" && error.code !== "ENOTEMPTY")
        ) {
          throw error
        }
      }
      continue
    }
    if (entry.isFile() && retainedPaths.has(target)) continue
    const stat = await fs.lstat(target)
    reclaimedBytes += stat.size
    await fs.rm(target, { recursive: true, force: true })
  }
  return reclaimedBytes
}

export async function removeStoredRun(id: string) {
  await withRunMutation(id, () =>
    fs.rm(runDirectory(id), { recursive: true, force: true })
  )
}

type StorageAdmissionClaim = {
  version: 1
  ownerId: string
  pid: number
  hostname: string
  claimedAt: string
}

async function withStorageAdmission<T>(
  operation: (ownerId: string) => Promise<T>
) {
  const previous = storageAdmissionGate
  let releaseGate: () => void = () => undefined
  const gate = new Promise<void>((resolve) => {
    releaseGate = resolve
  })
  const chain = previous.catch(() => undefined).then(() => gate)
  storageAdmissionGate = chain
  await previous.catch(() => undefined)

  const ownerId = `storage-${process.pid}-${cryptoRandom()}`
  let claimed = false
  try {
    claimed = await waitForStorageAdmissionClaim(ownerId)
    if (!claimed) {
      throw new RunStorageAdmissionError({
        code: "storage_admission_busy",
        message:
          "Another process is admitting an ECG run. Retry after that storage transaction completes; no run was admitted.",
      })
    }
    return await operation(ownerId)
  } finally {
    try {
      if (claimed) await releaseStorageAdmissionClaim(ownerId)
    } finally {
      releaseGate()
      if (storageAdmissionGate === chain) {
        storageAdmissionGate = Promise.resolve()
      }
    }
  }
}

async function waitForStorageAdmissionClaim(ownerId: string) {
  const deadline = Date.now() + STORAGE_ADMISSION_WAIT_MS
  do {
    if (await tryClaimStorageAdmission(ownerId)) return true
    if (Date.now() >= deadline) return false
    await new Promise<void>((resolve) =>
      setTimeout(resolve, STORAGE_ADMISSION_RETRY_MS)
    )
  } while (true)
}

async function tryClaimStorageAdmission(
  ownerId: string,
  now = Date.now(),
  storageDir = STORAGE_DIR
) {
  await fs.mkdir(storageDir, { recursive: true, mode: 0o700 })
  const claimPath = path.join(storageDir, STORAGE_ADMISSION_CLAIM_FILE)
  const claim: StorageAdmissionClaim = {
    version: 1,
    ownerId,
    pid: process.pid,
    hostname: os.hostname(),
    claimedAt: new Date(now).toISOString(),
  }
  const createClaim = async () => {
    try {
      await fs.writeFile(claimPath, `${JSON.stringify(claim, null, 2)}\n`, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600,
      })
      return true
    } catch (error) {
      if (isNodeError(error) && error.code === "EEXIST") return false
      throw error
    }
  }
  if (await createClaim()) return true

  let existingSource: string | null = null
  let existing: StorageAdmissionClaim | null = null
  let modifiedAt = now
  try {
    const [source, stat] = await Promise.all([
      fs.readFile(claimPath, "utf8"),
      fs.stat(claimPath),
    ])
    existingSource = source
    existing = parseStorageAdmissionClaim(source)
    modifiedAt = stat.mtimeMs
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return createClaim()
    throw error
  }
  const sameHost = existing?.hostname === os.hostname()
  const ownerAlive =
    sameHost && typeof existing?.pid === "number"
      ? processIsAlive(existing.pid)
      : true
  if (ownerAlive && now - modifiedAt < STORAGE_ADMISSION_STALE_MS) {
    return false
  }
  try {
    const source = await fs.readFile(claimPath, "utf8")
    if (source !== existingSource) return false
    await fs.unlink(claimPath)
  } catch (error) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error
  }
  return createClaim()
}

async function releaseStorageAdmissionClaim(
  ownerId: string,
  storageDir = STORAGE_DIR
) {
  const claimPath = path.join(storageDir, STORAGE_ADMISSION_CLAIM_FILE)
  try {
    const source = await fs.readFile(claimPath, "utf8")
    const claim = parseStorageAdmissionClaim(source)
    if (!claim || claim.ownerId !== ownerId) {
      throw new Error("The storage admission claim changed ownership.")
    }
    await fs.unlink(claimPath)
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return
    throw error
  }
}

function activeRunReservationBytes(run: RunRecord) {
  const mayProduceMoreArtifacts =
    run.status === "queued" ||
    run.status === "running" ||
    (run.status === "pending_digitizer" &&
      run.storage !== undefined &&
      run.processing?.state !== "blocked")
  if (!mayProduceMoreArtifacts) return BigInt(0)
  const minimum = requiredRunStorageReservationBytes(run.fileSizeBytes ?? 0)
  const recorded = run.storage?.reservedBytes
  return BigInt(
    Number.isSafeInteger(recorded) && recorded! > 0
      ? Math.max(minimum, recorded!)
      : minimum
  )
}

type IntakeRecoveryResult = "recovered" | "failed" | "deferred"

async function reconcileStoredRunIntakesUnlocked(
  storageDir = STORAGE_DIR,
  now = Date.now()
) {
  await fs.mkdir(storageDir, { recursive: true, mode: 0o700 })
  const entries = await fs.readdir(storageDir, { withFileTypes: true })
  const report = { recovered: 0, failed: 0, deferred: 0 }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const staged = parseRunIntakeDirectoryName(entry.name)
    const result = staged
      ? await recoverStagedRunIntake({
          storageDir,
          stagingDirectory: joinRuntimePath(storageDir, entry.name),
          staged,
          now,
        })
      : isSafeRunId(entry.name)
        ? await recoverCommittedRunIntake({
            directory: joinRuntimePath(storageDir, entry.name),
            runId: entry.name,
            now,
          })
        : null
    if (!result) continue
    report[result] += 1
  }
  return report
}

async function recoverStagedRunIntake({
  storageDir,
  stagingDirectory,
  staged,
  now,
}: {
  storageDir: string
  stagingDirectory: string
  staged: { runId: string; token: string }
  now: number
}): Promise<IntakeRecoveryResult> {
  const stat = await fs.stat(stagingDirectory)
  const recordedJournal = await readRunIntakeJournal(stagingDirectory)
  if (intakeRecoveryDeferred(recordedJournal, stat.mtimeMs, now)) {
    return "deferred"
  }
  const journal =
    recordedJournal?.runId === staged.runId ? recordedJournal : null

  let finalRunId = staged.runId
  const collision = await pathExists(joinRuntimePath(storageDir, finalRunId))
  if (collision) {
    finalRunId = await availableIntakeRecoveryRunId(
      storageDir,
      finalRunId,
      staged.token
    )
  }

  const recoveredAt = new Date(now).toISOString()
  let sourceValid = false
  if (journal && !collision) {
    const stagedInput = joinRuntimePath(
      stagingDirectory,
      "input",
      journal.safeFileName
    )
    try {
      const sourceStat = await fs.lstat(stagedInput)
      sourceValid =
        sourceStat.isFile() &&
        sourceStat.size === journal.fileSizeBytes &&
        (await sha256(stagedInput)) === journal.sourceSha256
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error
    }
  }

  const run = recoveredIntakeRun({
    id: finalRunId,
    journal,
    sourceValid,
    collision,
    recoveredAt,
    fallbackCreatedAt: stat.mtime.toISOString(),
    storageDir,
  })
  await writeRunToDirectory(run, stagingDirectory)
  const finalDirectory = joinRuntimePath(storageDir, finalRunId)
  await syncDirectory(stagingDirectory)
  await fs.rename(stagingDirectory, finalDirectory)
  await syncDirectory(storageDir)
  await archiveRunIntakeJournal(finalDirectory, sourceValid)
  return sourceValid ? "recovered" : "failed"
}

async function recoverCommittedRunIntake({
  directory,
  runId,
  now,
}: {
  directory: string
  runId: string
  now: number
}): Promise<IntakeRecoveryResult | null> {
  const journalPath = joinRuntimePath(directory, RUN_INTAKE_JOURNAL_FILE)
  let journalStat
  try {
    journalStat = await fs.stat(journalPath)
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return null
    throw error
  }
  const recordedJournal = await readRunIntakeJournal(directory)
  if (intakeRecoveryDeferred(recordedJournal, journalStat.mtimeMs, now)) {
    return "deferred"
  }
  const journal =
    recordedJournal?.runId === runId ? recordedJournal : null
  let sourceValid = false
  if (journal) {
    const inputPath = joinRuntimePath(
      directory,
      "input",
      journal.safeFileName
    )
    try {
      const sourceStat = await fs.lstat(inputPath)
      sourceValid =
        sourceStat.isFile() &&
        sourceStat.size === journal.fileSizeBytes &&
        (await sha256(inputPath)) === journal.sourceSha256
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error
    }
  }
  const recoveredAt = new Date(now).toISOString()
  const run = recoveredIntakeRun({
    id: runId,
    journal,
    sourceValid,
    collision: false,
    recoveredAt,
    fallbackCreatedAt: journalStat.mtime.toISOString(),
    storageDir: path.dirname(directory),
  })
  await writeRunToDirectory(run, directory)
  await archiveRunIntakeJournal(directory, sourceValid)
  return sourceValid ? "recovered" : "failed"
}

async function archiveRunIntakeJournal(
  directory: string,
  sourceValid: boolean
) {
  const journalPath = joinRuntimePath(directory, RUN_INTAKE_JOURNAL_FILE)
  if (sourceValid) {
    await fs.rm(journalPath, { force: true })
  } else {
    try {
      await fs.rename(
        journalPath,
        joinRuntimePath(directory, RUN_INTAKE_RECOVERY_FILE)
      )
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error
    }
  }
  await syncDirectory(directory)
}

function intakeRecoveryDeferred(
  journal: StoredRunIntakeJournal | null,
  modifiedAt: number,
  now: number
) {
  if (!journal) return now - modifiedAt < STORAGE_ADMISSION_STALE_MS
  const sameHost = journal.owner.hostname === os.hostname()
  const ownerAlive = sameHost ? processIsAlive(journal.owner.pid) : true
  return ownerAlive && now - modifiedAt < STORAGE_ADMISSION_STALE_MS
}

function recoveredIntakeRun({
  id,
  journal,
  sourceValid,
  collision,
  recoveredAt,
  fallbackCreatedAt,
  storageDir,
}: {
  id: string
  journal: StoredRunIntakeJournal | null
  sourceValid: boolean
  collision: boolean
  recoveredAt: string
  fallbackCreatedAt: string
  storageDir: string
}): RunRecord {
  const createdAt = journal?.createdAt ?? fallbackCreatedAt
  const inputPath = journal
    ? relativePath(
        joinRuntimePath(storageDir, id, "input", journal.safeFileName)
      )
    : undefined
  const message = sourceValid
    ? "Recovered a durably stored ECG source after an interrupted intake. Review the source and run it again; automatic processing was not resumed."
    : collision
      ? "Recovered an interrupted intake as a separate failed record because its intended run id already exists. No source from this record is eligible for digitization."
      : "Recovered an interrupted intake, but the untouched source was missing or failed its recorded size/hash check. Partial files were retained for forensic review and are not eligible for digitization."
  return {
    id,
    createdAt,
    updatedAt: recoveredAt,
    status: sourceValid ? "pending_digitizer" : "failed",
    source: "upload",
    fileName: journal?.fileName ?? "Interrupted ECG intake",
    ...(journal ? { fileSizeBytes: journal.fileSizeBytes } : {}),
    ...(sourceValid && journal
      ? {
          sourceIdentity: {
            version: 1 as const,
            algorithm: "sha256" as const,
            sha256: journal.sourceSha256,
            sizeBytes: journal.fileSizeBytes,
            recordedAt: journal.createdAt,
          },
        }
      : {}),
    localPath: relativePath(joinRuntimePath(storageDir, id)),
    message,
    ...(journal ? { storage: journal.storage } : {}),
    ...(journal?.captureQuality
      ? { captureQuality: journal.captureQuality }
      : {}),
    processing: {
      version: 1,
      state: sourceValid ? "blocked" : "failed",
      attempt: 0,
      recoveryCount: 1,
      queuedAt: createdAt,
      timeoutMs: 30 * 60 * 1_000,
      recoveredAt,
      finishedAt: recoveredAt,
      failureCode: sourceValid ? "intake_interrupted" : "intake_corrupt",
    },
    assets:
      sourceValid && inputPath && journal
        ? {
            input: {
              label: "Uploaded ECG",
              path: inputPath,
              mimeType: contentTypeForPath(inputPath),
              sizeBytes: journal.fileSizeBytes,
            },
          }
        : {},
  }
}

async function availableIntakeRecoveryRunId(
  storageDir: string,
  intendedRunId: string,
  token: string
) {
  const base = intendedRunId.slice(0, 180)
  for (let attempt = 0; attempt < 1_000; attempt += 1) {
    const suffix = attempt === 0 ? "" : `_${attempt}`
    const candidate = `${base}_intake_${token}${suffix}`
    if (!(await pathExists(joinRuntimePath(storageDir, candidate)))) return candidate
  }
  throw new Error(
    `Unable to allocate a distinct recovery record for interrupted intake ${intendedRunId}.`
  )
}

async function pathExists(target: string) {
  try {
    await fs.access(target)
    return true
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false
    throw error
  }
}

function joinRuntimePath(base: string, ...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ base, ...segments)
}

function parseRunIntakeDirectoryName(name: string) {
  const match = name.match(
    /^\.run-intake-([A-Za-z0-9_-]+)-([a-f0-9]{8}|[a-f0-9]{16})$/
  )
  if (!match || !isSafeRunId(match[1])) return null
  return { runId: match[1], token: match[2] }
}

async function readRunIntakeJournal(stagingDirectory: string) {
  try {
    const source = await fs.readFile(
      joinRuntimePath(stagingDirectory, RUN_INTAKE_JOURNAL_FILE),
      "utf8"
    )
    return parseRunIntakeJournal(source)
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return null
    throw error
  }
}

function parseRunIntakeJournal(source: string) {
  try {
    const journal = JSON.parse(source) as Partial<StoredRunIntakeJournal>
    const storage = journal.storage
    const owner = journal.owner
    if (
      journal.version !== 1 ||
      typeof journal.runId !== "string" ||
      !isSafeRunId(journal.runId) ||
      typeof journal.fileName !== "string" ||
      journal.fileName.length === 0 ||
      typeof journal.safeFileName !== "string" ||
      journal.safeFileName !== sanitizeFileName(journal.fileName) ||
      !Number.isSafeInteger(journal.fileSizeBytes) ||
      journal.fileSizeBytes! < 0 ||
      journal.fileSizeBytes! > MAX_UPLOAD_BYTES ||
      typeof journal.sourceSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(journal.sourceSha256) ||
      typeof journal.createdAt !== "string" ||
      !Number.isFinite(Date.parse(journal.createdAt)) ||
      !storage ||
      storage.version !== 1 ||
      typeof storage.admittedAt !== "string" ||
      !Number.isFinite(Date.parse(storage.admittedAt)) ||
      !Number.isSafeInteger(storage.minimumFreeBytes) ||
      storage.minimumFreeBytes < runStoragePolicy.minimumFreeBytes ||
      !Number.isSafeInteger(storage.reservedBytes) ||
      storage.reservedBytes <
        requiredRunStorageReservationBytes(journal.fileSizeBytes!) ||
      !owner ||
      typeof owner.ownerId !== "string" ||
      owner.ownerId.length === 0 ||
      !Number.isSafeInteger(owner.pid) ||
      owner.pid <= 0 ||
      typeof owner.hostname !== "string" ||
      owner.hostname.length === 0 ||
      (journal.captureQuality !== undefined &&
        (!isCaptureQualitySummary(journal.captureQuality) ||
          journal.captureQuality.outcome !== "ready" ||
          journal.captureQuality.sourceSha256 !== journal.sourceSha256))
    ) {
      return null
    }
    return journal as StoredRunIntakeJournal
  } catch {
    return null
  }
}

type StoredRunProcessingClaim = {
  version: 1
  ownerId: string
  pid: number
  hostname: string
  claimedAt: string
}

export async function tryClaimStoredRunProcessing({
  id,
  ownerId,
  staleAfterMs,
  now = Date.now(),
}: {
  id: string
  ownerId: string
  staleAfterMs: number
  now?: number
}) {
  const claimPath = path.join(runDirectory(id), JOB_CLAIM_FILE)
  const claim: StoredRunProcessingClaim = {
    version: 1,
    ownerId,
    pid: process.pid,
    hostname: os.hostname(),
    claimedAt: new Date(now).toISOString(),
  }
  await fs.mkdir(runDirectory(id), { recursive: true })

  const createClaim = async () => {
    try {
      await fs.writeFile(
        claimPath,
        `${JSON.stringify(claim, null, 2)}\n`,
        { encoding: "utf8", flag: "wx" }
      )
      return true
    } catch (error) {
      if (isNodeError(error) && error.code === "EEXIST") return false
      throw error
    }
  }

  if (await createClaim()) return true

  let existingSource: string | null = null
  let existing: StoredRunProcessingClaim | null = null
  let modifiedAt = now
  try {
    const [source, stat] = await Promise.all([
      fs.readFile(claimPath, "utf8"),
      fs.stat(claimPath),
    ])
    existingSource = source
    existing = parseStoredRunProcessingClaim(source)
    modifiedAt = stat.mtimeMs
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return createClaim()
    }
    throw error
  }

  const sameHost = existing?.hostname === os.hostname()
  const ownerAlive =
    sameHost && typeof existing?.pid === "number"
      ? processIsAlive(existing.pid)
      : true
  const stale = !ownerAlive || now - modifiedAt >= staleAfterMs
  if (!stale) return false

  try {
    const source = await fs.readFile(claimPath, "utf8")
    if (source !== existingSource) return false
    await fs.unlink(claimPath)
  } catch (error) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error
  }
  return createClaim()
}

export async function releaseStoredRunProcessingClaim(
  id: string,
  ownerId: string
) {
  const claimPath = path.join(runDirectory(id), JOB_CLAIM_FILE)
  try {
    const source = await fs.readFile(claimPath, "utf8")
    const claim = parseStoredRunProcessingClaim(source)
    if (!claim || claim.ownerId !== ownerId) return false
    await fs.unlink(claimPath)
    return true
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false
    throw error
  }
}

export async function touchStoredRun(id: string) {
  const now = new Date()
  await fs.utimes(runDirectory(id), now, now)
}

export async function purgeStaleWorkerRuns(
  olderThanMs: number,
  now = Date.now(),
  storageDir = STORAGE_DIR
) {
  let entries
  try {
    entries = await fs.readdir(storageDir, { withFileTypes: true })
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return 0
    throw error
  }

  let removed = 0
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith("neo_") || !isSafeRunId(entry.name)) {
      continue
    }
    const directory = path.join(storageDir, entry.name)
    try {
      const stat = await fs.stat(directory)
      if (now - stat.mtimeMs < olderThanMs) continue
      await fs.rm(directory, { recursive: true, force: true })
      removed += 1
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error
    }
  }
  return removed
}

export async function resolveRunAsset(id: string, key: RunAssetKey) {
  const run = await getRun(id)
  const asset = run?.assets[key]

  if (!run || !asset) {
    return null
  }

  if (key === "input" && run.source === "upload") {
    const verified = await readVerifiedRunSource(run)
    return {
      run,
      asset,
      absolutePath: verified.absolutePath,
      bytes: verified.bytes,
      contentType: asset.mimeType ?? contentTypeForPath(verified.absolutePath),
    }
  }

  if (key !== "input" && run.source === "upload") {
    const verified = await readVerifiedRunAsset(run, key)
    if (!verified) return null
    return {
      run,
      asset,
      absolutePath: verified.absolutePath,
      bytes: verified.bytes,
      contentType: asset.mimeType ?? contentTypeForPath(verified.absolutePath),
    }
  }

  const absolutePath = resolveWorkspacePath(asset.path)

  try {
    await fs.access(absolutePath)
  } catch {
    return null
  }

  return {
    run,
    asset,
    absolutePath,
    contentType: asset.mimeType ?? contentTypeForPath(absolutePath),
  }
}

export async function createInputFile({
  runId,
  fileName,
  bytes,
}: {
  runId: string
  fileName: string
  bytes: Buffer
}) {
  const inputDir = path.join(runDirectory(runId), "input")
  const safeName = sanitizeFileName(fileName)
  const absolutePath = path.join(
    /* turbopackIgnore: true */ inputDir,
    safeName
  )

  await fs.mkdir(inputDir, { recursive: true, mode: 0o700 })
  await fs.chmod(inputDir, 0o700)
  await atomicWritePrivateFile(absolutePath, bytes, true)

  return relativePath(absolutePath)
}

export function createRunId() {
  return `run_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}_${cryptoRandom()}`
}

async function listStoredRuns() {
  try {
    const entries = await fs.readdir(STORAGE_DIR, { withFileTypes: true })
    const runs = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory())
        .map((entry) => readStoredRun(entry.name))
    )

    return runs.filter((run): run is RunRecord => Boolean(run))
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return []
    }

    throw error
  }
}

async function readStoredRun(id: string): Promise<RunRecord | null> {
  if (!isSafeRunId(id)) {
    return null
  }

  try {
    const file = await fs.readFile(path.join(runDirectory(id), METADATA_FILE), "utf8")
    const run = validateDurableRun(JSON.parse(file))
    return { ...run, quantitativeEvidence: quantitativeEvidenceAvailability(run), outcomeDimensions: describeOutcome(run) }
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return null
    }

    throw error
  }
}

async function writeRun(run: RunRecord) {
  const dir = runDirectory(run.id)
  await writeRunToDirectory(run, dir)
}

async function writeRunToDirectory(run: RunRecord, dir: string) {
  validateDurableRun(run)
  const target = path.join(dir, METADATA_FILE)
  await fs.mkdir(dir, { recursive: true, mode: 0o700 })
  await fs.chmod(dir, 0o700)
  await atomicWritePrivateFile(
    target,
    `${JSON.stringify(run, null, 2)}\n`
  )
  if (isTerminalStoredRun(run)) {
    await enforcePrivateRunPermissions(dir)
  }
}

async function atomicWritePrivateFile(
  target: string,
  contents: string | Buffer,
  refuseExisting = false,
  fault?: (point: "opened" | "written" | "synced" | "renamed" | "directory_synced") => void | Promise<void>
) {
  const directory = path.dirname(target)
  const temporary = path.join(
    directory,
    `.${path.basename(target)}.${process.pid}.${cryptoRandom()}.tmp`
  )
  let handle: Awaited<ReturnType<typeof fs.open>> | undefined
  let temporaryCreated = false
  try {
    if (refuseExisting) {
      try {
        await fs.access(target)
        throw new Error(`Refusing to replace existing run file: ${relativePath(target)}`)
      } catch (error) {
        if (!isNodeError(error) || error.code !== "ENOENT") throw error
      }
    }
    handle = await fs.open(temporary, "wx", 0o600)
    temporaryCreated = true
    await fault?.("opened")
    if (typeof contents === "string") {
      await handle.writeFile(contents, "utf8")
    } else {
      await handle.writeFile(contents)
    }
    await fault?.("written")
    await handle.sync()
    await fault?.("synced")
    await handle.close()
    handle = undefined
    await fs.rename(temporary, target)
    temporaryCreated = false
    await fault?.("renamed")
    await fs.chmod(target, 0o600)
    await syncDirectory(directory)
    await fault?.("directory_synced")
  } catch (error) {
    await handle?.close().catch(() => undefined)
    if (temporaryCreated) {
      await fs.rm(temporary, { force: true }).catch(() => undefined)
    }
    throw error
  }
}

async function syncDirectory(directory: string) {
  const handle = await fs.open(directory, "r")
  try {
    await handle.sync()
  } finally {
    await handle.close()
  }
}

function isTerminalStoredRun(run: RunRecord) {
  return (
    run.status === "completed" ||
    run.status === "needs_review" ||
    run.status === "partial" ||
    run.status === "failed" ||
    run.status === "timed_out" ||
    (run.status === "pending_digitizer" && run.processing?.state === "blocked")
  )
}

async function enforcePrivateRunPermissions(directory: string) {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  await fs.chmod(directory, 0o700)
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      await enforcePrivateRunPermissions(entryPath)
    } else if (entry.isFile()) {
      await fs.chmod(entryPath, 0o600)
    }
  }
}

async function withRunMutation<T>(id: string, mutation: () => Promise<T>) {
  const previous = runMutationGates.get(id) ?? Promise.resolve()
  let release: () => void = () => undefined
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  const chain = previous.catch(() => undefined).then(() => gate)
  runMutationGates.set(id, chain)
  await previous.catch(() => undefined)
  try {
    return await mutation()
  } finally {
    release()
    if (runMutationGates.get(id) === chain) runMutationGates.delete(id)
  }
}

async function getReferenceRun(): Promise<RunRecord | null> {
  const KNOWN_TRACE_FOCUSED = KNOWN_DIGITIZATIONS[0]
  if (!KNOWN_TRACE_FOCUSED) return null
  const inputPath = KNOWN_TRACE_FOCUSED.sourcePath
  const paperPath = KNOWN_TRACE_FOCUSED.assets.paperRender
  const canonicalPath = KNOWN_TRACE_FOCUSED.assets.canonicalCsv

  if (!paperPath || !canonicalPath) {
    return null
  }

  try {
    const canonicalStat = await fs.stat(resolveWorkspacePath(canonicalPath))
    const inputStat = await fs.stat(resolveWorkspacePath(inputPath))
    const sourceSha256 = await sha256(resolveWorkspacePath(inputPath))

    return {
      id: referenceRunId(),
      createdAt: canonicalStat.mtime.toISOString(),
      updatedAt: canonicalStat.mtime.toISOString(),
      status: "completed",
      source: "reference",
      fileName: path.basename(inputPath),
      fileSizeBytes: inputStat.size,
      localPath: "output/open-ecg-digitizer",
      message: "Existing verified Open-ECG-Digitizer result from this workspace.",
      layout: KNOWN_TRACE_FOCUSED.layout,
      layoutCost: KNOWN_TRACE_FOCUSED.layoutCost,
      sampleRateHz: KNOWN_TRACE_FOCUSED.sampleRateHz,
      leadCount: KNOWN_TRACE_FOCUSED.leadCount,
      assets: {
        input: await assetFromRelativePath("Source ECG", inputPath),
        diagnostic: await assetFromRelativePath(
          "Diagnostic overlay",
          KNOWN_TRACE_FOCUSED.assets.diagnostic,
          undefined,
          sourceSha256
        ),
        paperRender: await assetFromRelativePath(
          "ECG-paper render",
          KNOWN_TRACE_FOCUSED.assets.paperRender,
          undefined,
          sourceSha256
        ),
        probability: await assetFromRelativePath(
          "Signal probability",
          KNOWN_TRACE_FOCUSED.assets.probability,
          undefined,
          sourceSha256
        ),
        canonicalCsv: await assetFromRelativePath(
          "Canonical CSV",
          KNOWN_TRACE_FOCUSED.assets.canonicalCsv,
          undefined,
          sourceSha256
        ),
        segmentsCsv: await assetFromRelativePath(
          "Compact 500 Hz CSV",
          KNOWN_TRACE_FOCUSED.assets.segmentsCsv,
          undefined,
          sourceSha256
        ),
        metadataCsv: await assetFromRelativePath(
          "Digitizer metadata",
          KNOWN_TRACE_FOCUSED.assets.metadataCsv,
          undefined,
          sourceSha256
        ),
      },
    }
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return null
    }

    throw error
  }
}

async function findKnownDigitization(inputDigest: string) {
  for (const known of KNOWN_DIGITIZATIONS) {
    try {
      const knownDigest = await sha256(resolveWorkspacePath(known.sourcePath))
      if (knownDigest === inputDigest) {
        return known
      }
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") {
        continue
      }

      throw error
    }
  }

  return null
}

async function copyKnownAssets(
  runId: string,
  digitization: KnownDigitization,
  sourceSha256: string
): Promise<Partial<Record<RunAssetKey, RunAsset>>> {
  const exportsDir = path.join(runDirectory(runId), "exports")
  await fs.mkdir(exportsDir, { recursive: true, mode: 0o700 })
  await fs.chmod(exportsDir, 0o700)

  const copiedAssets: Partial<Record<RunAssetKey, RunAsset>> = {}

  for (const [key, sourcePath] of Object.entries(digitization.assets) as [
    Exclude<RunAssetKey, "input">,
    string | undefined,
  ][]) {
    if (!sourcePath) continue

    const sourceAbsolutePath = resolveWorkspacePath(sourcePath)
    const fileName = path.basename(sourcePath)
    const destinationPath = path.join(
      /* turbopackIgnore: true */ exportsDir,
      fileName
    )

    await fs.copyFile(sourceAbsolutePath, destinationPath)
    await fs.chmod(destinationPath, 0o600)
    copiedAssets[key] = await assetFromRelativePath(
      assetLabel(key),
      relativePath(destinationPath),
      undefined,
      sourceSha256
    )
  }

  return copiedAssets
}

async function assetFromRelativePath(
  label: string,
  relativeFilePath: string | undefined,
  mimeType?: string,
  sourceSha256?: string
) {
  if (!relativeFilePath) {
    return undefined
  }

  const absolutePath = resolveWorkspacePath(relativeFilePath)
  const bytes = await fs.readFile(absolutePath)
  const recordedAt = new Date().toISOString()

  return {
    label,
    path: normalizeRelativePath(relativeFilePath),
    mimeType: mimeType ?? contentTypeForPath(absolutePath),
    sizeBytes: bytes.byteLength,
    ...(sourceSha256
      ? {
          identity: {
            version: 1 as const,
            algorithm: "sha256" as const,
            sha256: createHash("sha256").update(bytes).digest("hex"),
            sizeBytes: bytes.byteLength,
            sourceSha256,
            recordedAt,
          },
        }
      : {}),
  }
}

function assetLabel(key: RunAssetKey) {
  const labels: Record<RunAssetKey, string> = {
    input: "Uploaded ECG",
    preparedInput: "Annotation-suppressed input",
    geometryCorrectedInput: "Geometry-corrected input",
    artifactPreprocessedInput: "Artifact-specialist input",
    annotationMask: "Detected annotation mask",
    diagnostic: "Diagnostic overlay",
    paperRender: "ECG-paper render",
    probability: "Signal probability",
    canonicalCsv: "Canonical CSV",
    segmentsCsv: "Compact 500 Hz CSV",
    uncertaintyCsv: "Per-sample uncertainty CSV",
    segmentMapJson: "Segment identity and timing evidence",
    metadataCsv: "Digitizer metadata",
    provenanceJson: "Digitization provenance",
    reviewJson: "Review audit trail",
  }

  return labels[key]
}

function referenceRunId() {
  return "reference_trace_focused_1500"
}

function runDirectory(id: string) {
  if (!isSafeRunId(id)) {
    throw new Error("Invalid run id")
  }

  return path.join(/* turbopackIgnore: true */ STORAGE_DIR, id)
}

function resolveWorkspacePath(relativeFilePath: string) {
  const absolutePath = path.resolve(
    /*turbopackIgnore: true*/ ROOT_DIR,
    relativeFilePath
  )

  if (absolutePath !== ROOT_DIR && !absolutePath.startsWith(`${ROOT_DIR}${path.sep}`)) {
    throw new Error("Path escapes workspace")
  }

  return absolutePath
}

function relativePath(absolutePath: string) {
  return normalizeRelativePath(path.relative(ROOT_DIR, absolutePath))
}

function normalizeRelativePath(filePath: string) {
  return filePath.split(path.sep).join("/")
}

function isSafeRunId(id: string) {
  return /^[a-zA-Z0-9_-]+$/.test(id)
}

function cryptoRandom() {
  return randomBytes(8).toString("hex")
}

function isSha256(value: string) {
  return /^[a-f0-9]{64}$/.test(value)
}

async function sha256(filePath: string) {
  const file = await fs.readFile(filePath)
  return createHash("sha256").update(file).digest("hex")
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error
}

function processIsAlive(pid: number) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return isNodeError(error) && error.code === "EPERM"
  }
}

function parseStoredRunProcessingClaim(source: string) {
  try {
    const claim = JSON.parse(source) as Partial<StoredRunProcessingClaim>
    if (
      claim.version !== 1 ||
      typeof claim.ownerId !== "string" ||
      claim.ownerId.length === 0 ||
      !Number.isSafeInteger(claim.pid) ||
      claim.pid! <= 0 ||
      typeof claim.hostname !== "string" ||
      claim.hostname.length === 0 ||
      typeof claim.claimedAt !== "string" ||
      !Number.isFinite(Date.parse(claim.claimedAt))
    ) {
      return null
    }
    return claim as StoredRunProcessingClaim
  } catch {
    return null
  }
}

function parseStorageAdmissionClaim(source: string) {
  try {
    const claim = JSON.parse(source) as Partial<StorageAdmissionClaim>
    if (
      claim.version !== 1 ||
      typeof claim.ownerId !== "string" ||
      claim.ownerId.length === 0 ||
      !Number.isSafeInteger(claim.pid) ||
      claim.pid! <= 0 ||
      typeof claim.hostname !== "string" ||
      claim.hostname.length === 0 ||
      typeof claim.claimedAt !== "string" ||
      !Number.isFinite(Date.parse(claim.claimedAt))
    ) {
      return null
    }
    return claim as StorageAdmissionClaim
  } catch {
    return null
  }
}

export const runStorageTestUtils = {
  atomicWritePrivateFile,
  activeRunReservationBytes,
  createStoredRunFromBytesWithFault: createStoredRunFromBytesInternal,
  reconcileStoredRunIntakesUnlocked,
  tryClaimStorageAdmission,
  releaseStorageAdmissionClaim,
}
