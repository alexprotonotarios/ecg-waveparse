import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { promises as fs } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"

import type {
  AnnotationComponent,
  DigitizerCandidateSummary,
  DigitizerCandidateScoreBreakdown,
  DigitizerComputeDevice,
  DigitizerInputVariant,
  DigitizerLeadQa,
  DigitizerPanelTimingCorrection,
  DigitizerPipelineEvidence,
  DigitizerQa,
  DigitizerQaWarning,
  DigitizerReliabilitySummary,
  DigitizerSourceFidelitySummary,
  DigitizerStabilityEvidence,
  DigitizerVectorizer,
  RunAsset,
  RunAssetKey,
  RunRecord,
} from "@/lib/runs"
import {
  readVerifiedRunAsset,
  readVerifiedRunSource,
} from "@/lib/runs"
import {
  ECG_LAYOUTS as LAYOUT_ROWS,
  ECG_LEADS as LEAD_ORDER,
  LAYOUT_COLUMNS,
  PAGE_DURATION_SECONDS,
  PARTIAL_LAYOUT_LEADS,
  SAMPLE_RATE_HZ,
} from "@/lib/digitizer/domain"
import {
  candidateCapabilities,
  candidateKind,
  isBaselineSourceModelCandidate,
} from "@/lib/digitizer/candidate-capabilities"
import {
  analyzeCandidatePlanningContext,
  buildPrimaryCandidatePlan,
  candidatePlanningPhase,
  candidateScheduleReason,
  semanticLeadIdentityParameters,
  type CandidatePlanningContext,
  type DigitizerCandidateConfig,
  type NativeGridLayoutConstraint,
} from "@/lib/digitizer/candidate-plan"
import {
  parseLayoutGeometryReport,
  parseNativeExtractionReport,
  parsePreprocessingReport,
  type LayoutGeometryReport,
  type PreparedRunInput,
  type PreprocessingReport,
  type RasterCropBox,
} from "@/lib/digitizer/contracts"
import {
  DIGITIZER_POLICY,
  DIGITIZER_POLICY_EVIDENCE_BASIS,
  DIGITIZER_POLICY_ID,
  DIGITIZER_POLICY_VERSION,
} from "@/lib/digitizer/policy"
import {
  SELECTOR_CALIBRATION_PROFILE_ID,
  SELECTOR_CALIBRATION_TRAINING,
  selectorCalibrationEvidence,
} from "@/lib/digitizer/selector-calibration"
import {
  applyInputQualityPublicationGate,
  resolvePublicationDecision,
  type PublicationReasonCode,
} from "@/lib/digitizer/publication-policy"
import { normalizeSupportedEcgLayout } from "@/lib/ecg-layouts"

const execFileAsync = promisify(execFile)

import { RESOURCE_ROOT, WORKSPACE_ROOT as ROOT_DIR } from "@/lib/runtime-paths"
const DISAGREEMENT_THRESHOLD_UV = DIGITIZER_POLICY.disagreementThresholdUv
const ALIGNMENT_RADIUS_SAMPLES = DIGITIZER_POLICY.alignmentRadiusSamples
const ALIGNMENT_TIME_SCALES = DIGITIZER_POLICY.alignmentTimeScales
const ALIGNMENT_TRIM_FRACTION = DIGITIZER_POLICY.alignmentTrimFraction
const ALIGNMENT_SEARCH_MAX_SAMPLES = DIGITIZER_POLICY.alignmentSearchMaxSamples
const MIN_SCORABLE_LEAD_SAMPLES = DIGITIZER_POLICY.minimumScorableLeadSamples
const MIN_SCORABLE_LEAD_COVERAGE = DIGITIZER_POLICY.minimumScorableLeadCoverage
const MIN_PUBLISHABLE_LEAD_COVERAGE = DIGITIZER_POLICY.minimumPublishableLeadCoverage
const MIN_SOURCE_VERIFIED_BOUNDARY_COVERAGE =
  DIGITIZER_POLICY.minimumSourceVerifiedBoundaryCoverage
const MIN_REVIEWABLE_LEAD_COVERAGE = DIGITIZER_POLICY.minimumReviewableLeadCoverage
const MIN_REVIEW_LAYOUT_CONFIDENCE = DIGITIZER_POLICY.minimumReviewLayoutConfidence
const MIN_NATIVE_REVIEW_LAYOUT_CONFIDENCE =
  DIGITIZER_POLICY.minimumNativeReviewLayoutConfidence
const MIN_NATIVE_REVIEW_EVIDENCE_MEDIAN =
  DIGITIZER_POLICY.minimumNativeReviewEvidenceMedian
const MIN_NATIVE_REVIEW_EVIDENCE_P10 = DIGITIZER_POLICY.minimumNativeReviewEvidenceP10
const LOW_RESOLUTION_NATIVE_CORROBORATION_MAX_DIMENSION =
  DIGITIZER_POLICY.lowResolutionNativeMaxDimension
const LOW_RESOLUTION_NATIVE_CORROBORATION_MEDIAN_RMSE_UV =
  DIGITIZER_POLICY.lowResolutionNativeMedianRmseUv
const LOW_RESOLUTION_NATIVE_CORROBORATION_LEAD_RMSE_UV =
  DIGITIZER_POLICY.lowResolutionNativeLeadRmseUv
const LOW_RESOLUTION_NATIVE_CORROBORATION_MINIMUM_LEADS =
  DIGITIZER_POLICY.lowResolutionNativeMinimumLeads
const LOW_RESOLUTION_NATIVE_CORROBORATION_JUMP_MARGIN =
  DIGITIZER_POLICY.lowResolutionNativeJumpMargin
const MIN_GEOMETRY_CONSTRAINED_3X4_CONFIDENCE =
  DIGITIZER_POLICY.minimumGeometry3x4Confidence
const MIN_GEOMETRY_CONSTRAINED_6X2_CONFIDENCE =
  DIGITIZER_POLICY.minimumGeometry6x2Confidence
const MIN_REVIEW_LAYOUT_LEAD_FRACTION = DIGITIZER_POLICY.minimumReviewLayoutLeadFraction
const MIN_ROBUST_LEAD_AMPLITUDE_UV = DIGITIZER_POLICY.minimumRobustLeadAmplitudeUv
const CONSTRAINED_PANEL_START_MASK_FRACTION =
  DIGITIZER_POLICY.constrainedPanelStartMaskFraction
const CONSTRAINED_PANEL_END_MASK_FRACTION =
  DIGITIZER_POLICY.constrainedPanelEndMaskFraction
const MIN_NATIVE_MODEL_INPUT_DIMENSION = DIGITIZER_POLICY.minimumNativeModelInputDimension
const ANNOTATION_OUTPUT_MARGIN_FRACTION = DIGITIZER_POLICY.annotationOutputMarginFraction
const BLUE_ANNOTATION_OUTPUT_MARGIN_FRACTION =
  DIGITIZER_POLICY.blueAnnotationOutputMarginFraction
const LEAD_FUSION_CANDIDATE_ID = "lead-fusion"
const PEER_GAP_REPAIR_CANDIDATE_ID = "peer-gap-repair"
const MAX_PEER_GAP_REPAIR_SAMPLES = DIGITIZER_POLICY.maximumPeerGapRepairSamples
const SIX_BY_TWO_RHYTHM_COMPOSITE_PREFIX = "six-by-two-rhythm-composite"
const LABEL_CONFIRMED_RETRY_LAYOUTS = [
  "standard_3x4_with_r1",
  "standard_3x4_with_r2",
  "standard_3x4_with_r3",
  "cabrera_12x1",
] as const
const PAPER_RENDERER_PATH = path.join(RESOURCE_ROOT, "output", "render_digitized_paper.py")
const INPUT_PREPARER_PATH = path.join(RESOURCE_ROOT, "scripts", "prepare_ecg_input.py")
const LAYOUT_DETECTOR_PATH = path.join(RESOURCE_ROOT, "scripts", "detect_ecg_layout.py")
const NATIVE_GRID_DIGITIZER_PATH = path.join(
  /*turbopackIgnore: true*/ RESOURCE_ROOT,
  "ecg_pipeline",
  "native_grid_digitizer.py"
)
const RELIABLE_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layouts_reliable.yml"
)
const STANDARD_3X4_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layout_standard_3x4.yml"
)
const STANDARD_3X4_WITH_R1_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layout_standard_3x4_with_r1.yml"
)
const STANDARD_6X2_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layout_standard_6x2.yml"
)
const STANDARD_6X2_WITH_R1_IGNORED_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layout_standard_6x2_with_r1_ignored.yml"
)
const STANDARD_12X1_LAYOUT_CONFIG_PATH = path.join(
  RESOURCE_ROOT,
  "ecg_pipeline",
  "lead_layout_standard_12x1.yml"
)
const DIRECT_DIGITIZER_EXTENSIONS = [".png", ".jpg", ".jpeg"]
const CONVERTIBLE_DIGITIZER_EXTENSIONS = [".webp", ".tif", ".tiff"]
const NEURAL_INFERENCE_CONCURRENCY = parseBoundedInteger(
  process.env.ECG_DIGITIZER_NEURAL_CONCURRENCY,
  1,
  1,
  8
)
const NATIVE_EXTRACTION_CONCURRENCY = parseBoundedInteger(
  process.env.ECG_DIGITIZER_NATIVE_CONCURRENCY,
  1,
  1,
  4
)
const PREPROCESSING_PERMITS = parseBoundedInteger(
  process.env.ECG_DIGITIZER_PREPROCESSING_PERMITS,
  3,
  1,
  8
)
const PREPROCESSING_ATTEMPTS = [3200, 2400] as const
const PREPROCESSING_TIMEOUT_MS = 180_000

class AsyncSemaphore {
  private active = 0
  private readonly waiting: Array<{
    weight: number
    resolve: (release: () => void) => void
  }> = []

  constructor(private readonly limit: number) {}

  acquire(requestedWeight = 1): Promise<() => void> {
    const weight = Math.max(1, Math.min(this.limit, requestedWeight))
    if (this.waiting.length === 0 && this.active + weight <= this.limit) {
      this.active += weight
      return Promise.resolve(this.releaseFunction(weight))
    }
    return new Promise((resolve) => this.waiting.push({ weight, resolve }))
  }

  private releaseFunction(weight: number) {
    let released = false
    return () => {
      if (released) return
      released = true
      this.active -= weight
      while (this.waiting.length > 0) {
        const next = this.waiting[0]
        if (this.active + next.weight > this.limit) break
        this.waiting.shift()
        this.active += next.weight
        next.resolve(this.releaseFunction(next.weight))
      }
    }
  }
}

const neuralInferenceSemaphore = new AsyncSemaphore(
  NEURAL_INFERENCE_CONCURRENCY
)
const nativeExtractionSemaphore = new AsyncSemaphore(
  NATIVE_EXTRACTION_CONCURRENCY
)
const preprocessingSemaphore = new AsyncSemaphore(PREPROCESSING_PERMITS)

function parseBoundedInteger(
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number
) {
  const parsed = value ? Number.parseInt(value, 10) : fallback
  return Number.isSafeInteger(parsed) && parsed >= minimum && parsed <= maximum
    ? parsed
    : fallback
}

type CanonicalCsv = {
  leads: string[]
  rows: number[][]
}

type CandidateResult = DigitizerCandidateSummary & {
  canonicalPath?: string
  diagnosticPath?: string
  metadataPath?: string
  sourceFidelityPath?: string
  score: number
  scoreBreakdown?: DigitizerCandidateScoreBreakdown
  canonical?: CanonicalCsv
  featureCacheHit?: boolean
}

const LEAD_ALIGNMENT_RMSE_CACHE = new WeakMap<
  CandidateResult,
  WeakMap<CandidateResult, Map<string, number>>
>()

type UncertaintyRow = {
  lead: string
  leadSample: number
  timeSeconds: number
  canonicalSample: number
  valueUv: number
  reviewEstimateUv: number
  status:
    | "observed"
    | "missing"
    | "uncertain_annotation"
    | "uncertain_candidate_disagreement"
    | "uncertain_annotation_and_disagreement"
  annotationOverlap: boolean
  candidateCount: number
  candidateSpreadUv: number
}

type PublishedCandidateResult = {
  assets: Partial<Record<RunAssetKey, RunAsset>>
  reliability: DigitizerReliabilitySummary
}

type DigitizerExecutionOptions = {
  signal?: AbortSignal
  candidateMode?: "production" | "benchmark"
}

function subprocessEnvironment(): NodeJS.ProcessEnv {
  const environment = { ...process.env }
  delete environment.CUES_ECG_DIGITIZER_WORKER_SECRET
  environment.PYTHONPATH = [RESOURCE_ROOT, environment.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter)
  return environment
}

export async function digitizeRun(
  run: RunRecord,
  options: DigitizerExecutionOptions = {}
): Promise<Partial<RunRecord>> {
  const inputAsset = run.assets.input

  if (!inputAsset) {
    return {
      status: "failed",
      message: "No uploaded ECG file was found for this run.",
      updatedAt: new Date().toISOString(),
    }
  }

  let verifiedSourceSha256: string
  try {
    const verifiedSource = await readVerifiedRunSource(run)
    verifiedSourceSha256 = verifiedSource.sha256
  } catch (error) {
    return {
      status: "failed",
      message:
        error instanceof Error
          ? `Source integrity verification failed: ${error.message}`
          : "Source integrity verification failed.",
      updatedAt: new Date().toISOString(),
    }
  }

  const availabilityMessage = await checkDigitizerAvailable()
  if (availabilityMessage) {
    return {
      status: "pending_digitizer",
      message: availabilityMessage,
      updatedAt: new Date().toISOString(),
    }
  }

  let device: DigitizerComputeDevice
  try {
    device = await resolveDigitizerDevice()
  } catch (error) {
    return {
      status: "failed",
      message:
        error instanceof Error
          ? `Digitizer compute device configuration failed: ${error.message}`
          : "Digitizer compute device configuration failed.",
      updatedAt: new Date().toISOString(),
    }
  }

  let prepared: PreparedRunInput
  try {
    prepared = await prepareRunInput(run, inputAsset, options.signal)
    if (prepared.report.sourceSha256 !== verifiedSourceSha256) {
      throw new Error(
        "The source image changed after admission and before deterministic preparation completed."
      )
    }
  } catch (error) {
    return {
      status: "failed",
      message:
        error instanceof Error
          ? `Deterministic input preparation failed: ${error.message}`
          : "Deterministic input preparation failed.",
      updatedAt: new Date().toISOString(),
    }
  }

  const benchmarkMode = options.candidateMode === "benchmark"
  const geometry = await detectLayoutGeometry(prepared, options.signal)
  const planning = analyzeCandidatePlanningContext(prepared.report, geometry)
  const {
    adaptivePreprocessingEligible,
    lowResolutionNativeUpscaleTarget,
    constrainedInputVariant,
    nativeGridLayout,
    nativeGridInputVariant,
  } = planning
  const [
    nativeGridCandidate,
    preprocessedNativeGridCandidate,
    upscaledPreprocessedNativeGridCandidate,
  ] =
    nativeGridLayout
      ? await Promise.all([
          runNativeGridCandidate(
            run,
            prepared,
            geometry,
            nativeGridInputVariant,
            nativeGridLayout,
            false,
            false,
            options.signal
          ),
          benchmarkMode || adaptivePreprocessingEligible
            ? runNativeGridCandidate(
                run,
                prepared,
                geometry,
                "preprocessed",
                nativeGridLayout,
                true,
                adaptivePreprocessingEligible,
                options.signal
              )
            : Promise.resolve(undefined),
          adaptivePreprocessingEligible && lowResolutionNativeUpscaleTarget
            ? runNativeGridCandidate(
                run,
                prepared,
                geometry,
                "preprocessed",
                nativeGridLayout,
                true,
                adaptivePreprocessingEligible,
                options.signal,
                lowResolutionNativeUpscaleTarget
              )
            : Promise.resolve(undefined),
        ])
      : [undefined, undefined, undefined]
  const nativeGridCandidates = [
    nativeGridCandidate,
    preprocessedNativeGridCandidate,
    upscaledPreprocessedNativeGridCandidate,
  ].filter((candidate): candidate is CandidateResult => Boolean(candidate))
  const nativeGridStructurallyComplete = nativeGridCandidates.some(
    (candidate) =>
      candidate.sourceFidelity?.passed &&
      LEAD_ORDER.every((lead) => candidateHasPublishableLead(candidate, lead))
  )
  const candidates: CandidateResult[] = [...nativeGridCandidates]
  let productionNeuralCandidateCount = 0
  const runProductionCandidate = async (
    candidate: DigitizerCandidateConfig
  ): Promise<CandidateResult | undefined> => {
    if (
      !benchmarkMode &&
      productionNeuralCandidateCount >=
        DIGITIZER_POLICY.maximumProductionNeuralCandidates
    ) {
      return undefined
    }
    productionNeuralCandidateCount += 1
    return runCandidate(
      run,
      prepared,
      {
        ...candidate,
        planningPhase:
          candidate.planningPhase ?? (benchmarkMode ? "benchmark" : "recovery"),
        scheduleReason:
          candidate.scheduleReason ??
          (benchmarkMode
            ? "exhaustive benchmark candidate"
            : "dynamic specialist recovery within the production budget"),
      },
      options.signal
    )
  }
  const primaryPlan = buildPrimaryCandidatePlan({
    context: planning,
    geometry,
    benchmarkMode,
    device,
    annotationMasked: prepared.report.annotationMask.maskedPixels > 0,
  })
  const corePlan = primaryPlan
    .filter((candidate) => candidatePlanningPhase(candidate) === "core")
    .slice(0, DIGITIZER_POLICY.maximumProductionCoreCandidates)
  const expandedPlan = primaryPlan.filter(
    (candidate) => candidatePlanningPhase(candidate) !== "core"
  )
  for (const candidate of corePlan) {
    const result = await runProductionCandidate({
      ...candidate,
      planningPhase: candidatePlanningPhase(candidate),
      scheduleReason: candidateScheduleReason(candidate, planning),
    })
    if (result) candidates.push(result)
  }
  const coreAgreementReached = hasIndependentCompleteAgreement(candidates)
  if (benchmarkMode || !coreAgreementReached) {
    for (const candidate of expandedPlan) {
      const result = await runProductionCandidate({
        ...candidate,
        planningPhase: candidatePlanningPhase(candidate),
        scheduleReason: candidateScheduleReason(candidate, planning),
      })
      if (result) candidates.push(result)
    }
  }
  const defaultCandidate = candidates.find((candidate) =>
    isBaselineSourceModelCandidate(candidate.parameters)
  )

  if (
    benchmarkMode ||
    (!nativeGridStructurallyComplete &&
      defaultCandidate?.status === "completed" &&
      !defaultCandidate.qa?.passed)
  ) {
    const thresholdRetry = await runProductionCandidate({
      kind: "source-model",
      id: "label-thresh-005",
      label: "Retry: lower signal threshold",
      resampleSize: 1500,
      labelThresh: 0.05,
      inputVariant: "original",
      vectorizer: "probability-centroid",
      device,
    })
    if (thresholdRetry) candidates.push(thresholdRetry)
  }

  if (!benchmarkMode) {
    for (const layoutConstraint of LABEL_CONFIRMED_RETRY_LAYOUTS) {
      const hint = candidates
        .filter(
          (candidate) =>
            candidate.status === "completed" &&
            candidate.layout === layoutConstraint &&
            Number.isFinite(candidate.layoutCost) &&
            (candidate.layoutCost ?? Number.POSITIVE_INFINITY) <= 0.75
        )
        .sort(
          (a, b) =>
            (a.layoutCost ?? Number.POSITIVE_INFINITY) -
              (b.layoutCost ?? Number.POSITIVE_INFINITY) ||
            a.score - b.score
        )[0]
      if (!hint) continue
      const inputVariant =
        hint.parameters.inputVariant === "preprocessed" &&
        hint.parameters.adaptivePreprocessingEligible !== true
          ? constrainedInputVariant
          : (hint.parameters.inputVariant ?? constrainedInputVariant)
      for (const retry of [
        {
          suffix: "centroid-1500",
          resampleSize: 1500,
          vectorizer: "probability-centroid" as const,
        },
        {
          suffix: "path-2200",
          resampleSize: 2200,
          vectorizer: "dynamic-path" as const,
        },
      ]) {
        const layoutRetry = await runProductionCandidate({
          kind: "geometry-constrained",
          id: `label-confirmed-${layoutConstraint}-${retry.suffix}`,
          label: `Label-confirmed ${layoutConstraint} ${retry.vectorizer} extraction`,
          resampleSize: retry.resampleSize,
          upscaleToMaxDimension: retry.resampleSize,
          darkInkEnhancement: true,
          layoutConstraint,
          maximumLayoutCost: 0.75,
          ...(inputVariant === "preprocessed"
            ? {
                adaptivePreprocessingEligible:
                  hint.parameters.adaptivePreprocessingEligible,
              }
            : {}),
          inputVariant,
          vectorizer: retry.vectorizer,
          device,
        })
        if (layoutRetry) candidates.push(layoutRetry)
      }
    }
  }

  if (
    benchmarkMode ||
    !candidates.some(candidateHasAllScorableLeads)
  ) {
    if (geometry.layoutHint === "standard_6x2_with_r1_ignored") {
      if (benchmarkMode || !nativeGridCandidate?.sourceFidelity?.passed) {
        const cropBoxes = sixByTwoPanelCropBoxes(prepared.report, geometry)
        const fallbackGroups = [
          {
            suffix: "path-local-ink-24-1500",
            vectorizer: "dynamic-path" as const,
            darkInkEnhancement: true,
            darkInkSupportRadius: 24,
            labelThresh: undefined,
          },
          {
            suffix: "path-local-ink-40-1500",
            vectorizer: "dynamic-path" as const,
            darkInkEnhancement: true,
            darkInkSupportRadius: 40,
            labelThresh: 0.02,
          },
        ]
        for (const fallback of fallbackGroups) {
          const left = await runProductionCandidate({
              kind: "panel-component",
              id: `six-by-two-rhythm-limb-${fallback.suffix}`,
              label: "6 × 2 + rhythm left-panel extraction",
              resampleSize: 1500,
              upscaleToMaxDimension: 1500,
              darkInkEnhancement: fallback.darkInkEnhancement,
              darkInkSupportRadius: fallback.darkInkSupportRadius,
              ...(fallback.labelThresh !== undefined
                ? { labelThresh: fallback.labelThresh }
                : {}),
              layoutConstraint: "standard_6x1_limb_constrained",
              maximumLayoutCost: 1,
              cropBox: cropBoxes.left,
              inputVariant: constrainedInputVariant,
              vectorizer: fallback.vectorizer,
              device,
          })
          const right = await runProductionCandidate({
              kind: "panel-component",
              id: `six-by-two-rhythm-precordial-${fallback.suffix}`,
              label: "6 × 2 + rhythm right-panel extraction",
              resampleSize: 1500,
              upscaleToMaxDimension: 1500,
              darkInkEnhancement: fallback.darkInkEnhancement,
              darkInkSupportRadius: fallback.darkInkSupportRadius,
              ...(fallback.labelThresh !== undefined
                ? { labelThresh: fallback.labelThresh }
                : {}),
              layoutConstraint: "precordial_6x1_constrained",
              maximumLayoutCost: 1,
              cropBox: cropBoxes.right,
              inputVariant: constrainedInputVariant,
              vectorizer: fallback.vectorizer,
              device,
          })
          if (!left || !right) break
          candidates.push(left, right)
          candidates.push(
            await buildSixByTwoRhythmCompositeCandidate({
              run,
              prepared,
              left,
              right,
              suffix: fallback.suffix,
              vectorizer: fallback.vectorizer,
              darkInkEnhancement: fallback.darkInkEnhancement,
              darkInkSupportRadius: fallback.darkInkSupportRadius,
              labelThresh: fallback.labelThresh,
              device,
              signal: options.signal,
            })
          )
        }
      }
    }
  }

  if (
    !benchmarkMode &&
    !candidates.some(candidateHasAllScorableLeads) &&
    geometry.compoundPanels &&
    geometry.compoundPanels.confidence >= 0.08
  ) {
    const panels = geometry.compoundPanels
    for (const limbConvention of ["standard", "cabrera"] as const) {
      const limbConstraint =
        limbConvention === "cabrera"
          ? "cabrera_6x1_limb_constrained"
          : "standard_6x1_limb_constrained"
      const limb = await runProductionCandidate({
          kind: "panel-component",
          id: `compound-${panels.orientation}-${limbConvention}-limb`,
          label: `${limbConvention === "cabrera" ? "Cabrera" : "Standard"} limb-panel label recovery`,
          resampleSize: 2200,
          upscaleToMaxDimension: 2200,
          darkInkEnhancement: true,
          darkInkSupportRadius: 24,
          layoutConstraint: limbConstraint,
          maximumLayoutCost: 0.75,
          cropBox: panels.first,
          // A cropped component is evidence for the composite only. Its lead
          // names are not safe to publish unless the complementary panel also
          // validates its exact constrained layout.
          selectionEligible: false,
          inputVariant: constrainedInputVariant,
          vectorizer: "probability-centroid",
          device,
      })
      const precordial = await runProductionCandidate({
          kind: "panel-component",
          id: `compound-${panels.orientation}-${limbConvention}-precordial`,
          label: "Precordial-panel label recovery",
          resampleSize: 2200,
          upscaleToMaxDimension: 2200,
          darkInkEnhancement: true,
          darkInkSupportRadius: 24,
          layoutConstraint: "precordial_6x1_constrained",
          maximumLayoutCost: 0.75,
          cropBox: panels.second,
          selectionEligible: false,
          inputVariant: constrainedInputVariant,
          vectorizer: "probability-centroid",
          device,
      })
      if (!limb || !precordial) break
      candidates.push(limb, precordial)
      candidates.push(
        await buildSixByTwoRhythmCompositeCandidate({
          run,
          prepared,
          left: limb,
          right: precordial,
          suffix: `compound-${panels.orientation}-${limbConvention}`,
          vectorizer: "probability-centroid",
          darkInkEnhancement: true,
          darkInkSupportRadius: 24,
          device,
          signal: options.signal,
          orientation: panels.orientation,
          targetLayout:
            panels.orientation === "stacked"
              ? "standard_12x1"
              : "standard_6x2",
          label: `${limbConvention === "cabrera" ? "Cabrera" : "Standard"} limb/precordial panel composite`,
        })
      )
    }
  }

  if (!candidates.some(candidateHasAllScorableLeads)) {
    for (const proposal of deterministicCropProposals(prepared.report, geometry)) {
      for (const variant of [
        {
          suffix: "centroid-1500",
          label: "centroid",
          resampleSize: 1500,
          vectorizer: "probability-centroid" as const,
          darkInkEnhancement: false,
        },
        {
          suffix: "path-2200",
          label: "source-ink path",
          resampleSize: 2200,
          vectorizer: "dynamic-path" as const,
          darkInkEnhancement: true,
        },
      ]) {
        const cropCandidate = await runProductionCandidate({
          kind: "crop-proposal",
          id: `crop-${proposal.id}-${variant.suffix}`,
          label: `${proposal.label} ${variant.label} extraction`,
          resampleSize: variant.resampleSize,
          upscaleToMaxDimension: variant.resampleSize,
          ...(variant.darkInkEnhancement
            ? { darkInkEnhancement: true }
            : {}),
          cropBox: proposal.cropBox,
          selectionEligible: proposal.selectionEligible,
          inputVariant: constrainedInputVariant,
          vectorizer: variant.vectorizer,
          device,
        })
        if (cropCandidate) candidates.push(cropCandidate)
      }
    }
  }

  const completedCandidates = candidates.filter(
    (candidate) => candidate.status === "completed" && candidate.canonicalPath
  )

  if (completedCandidates.length === 0) {
    const preprocessingAssets = await preprocessingRunAssets(prepared)
    const evidence = buildPipelineEvidence({
      preprocessing: prepared.report,
      geometry,
      planning,
      primaryPlan,
      candidates,
      selectionCandidates: [],
      nativeGridCandidates,
      nativeGridStructurallyComplete,
      coreAgreementReached,
      recoveryExpanded: benchmarkMode || !coreAgreementReached,
      stability: notRequiredStabilityEvidence(),
    })
    return {
      status: "failed",
      publicationDecision: {
        policyId: DIGITIZER_POLICY_ID,
        policyVersion: DIGITIZER_POLICY_VERSION,
        outcome: "failed",
        reasonCode: "no_publishable_candidate",
      },
      message: "Open-ECG-Digitizer did not produce a usable canonical CSV.",
      digitizer: {
        engine: "Open-ECG-Digitizer",
        candidates: candidates.map(publicCandidate),
        evidence,
      },
      assets: {
        ...run.assets,
        ...preprocessingAssets,
      },
      updatedAt: new Date().toISOString(),
    }
  }

  const selectionCandidates = completedCandidates.filter(
    (candidate) =>
      candidateEligibleForSelection(candidate) ||
      hasLowResolutionNativeTraceCorroboration(
        candidate,
        completedCandidates,
        prepared.report
      )
  )
  applyCandidateDisagreementScores(selectionCandidates)
  const gapRepairCandidate = await buildPeerGapRepairCandidate(
    run,
    selectionCandidates,
    prepared.report
  )
  if (gapRepairCandidate) {
    candidates.push(gapRepairCandidate)
    completedCandidates.push(gapRepairCandidate)
    selectionCandidates.push(gapRepairCandidate)
    applyCandidateDisagreementScores(selectionCandidates)
  }
  const fusedCandidate = await buildLeadFusionCandidate(
    run,
    selectionCandidates,
    prepared.report
  )
  if (fusedCandidate) {
    candidates.push(fusedCandidate)
    completedCandidates.push(fusedCandidate)
    selectionCandidates.push(fusedCandidate)
  }
  const sourceVerifiedCandidate = selectionCandidates
    .filter(
      (candidate) =>
        candidate.sourceFidelity?.passed &&
        LEAD_ORDER.every((lead) =>
          candidateHasTrustedLead(candidate, selectionCandidates, lead)
        )
    )
    .sort(
      (a, b) =>
        candidatePublicationRank(a) - candidatePublicationRank(b) ||
        a.id.localeCompare(b.id)
    )[0]
  const adaptivePreprocessedCandidate =
    selectAdaptivePreprocessedCandidate(selectionCandidates)
  const publishableFusedCandidate =
    fusedCandidate && candidateHasAllScorableLeads(fusedCandidate)
      ? fusedCandidate
      : undefined
  const hasPrimarySelection = Boolean(
    sourceVerifiedCandidate ||
      adaptivePreprocessedCandidate ||
      publishableFusedCandidate
  )
  const safestCandidate = selectSafestCandidate(
    selectionCandidates,
    prepared.report
  )
  const reviewableCandidate =
    hasPrimarySelection || safestCandidate
      ? undefined
      : selectReviewableConstrainedCandidate(selectionCandidates)
  const partialCandidate =
    hasPrimarySelection || safestCandidate || reviewableCandidate
      ? undefined
      : selectPartialLeadCandidate(selectionCandidates)
  const decision = applyInputQualityPublicationGate(
    resolvePublicationDecision({
      sourceVerified: sourceVerifiedCandidate,
      adaptivePreprocessed: adaptivePreprocessedCandidate,
      fused: publishableFusedCandidate,
      safest: safestCandidate,
      reviewableConstrained: reviewableCandidate,
      supportedPartial: partialCandidate,
      rankCandidate: candidatePublicationRank,
    }),
    prepared.report.inputQuality?.outcome
  )
  if (decision.outcome === "failed") {
    const semanticIdentityUnconfirmed = Boolean(
      decision.reasonCode === "no_publishable_candidate" &&
        selectionCandidates.some(candidateHasAllScorableLeads) &&
        !selectionCandidates.some(candidateHasExplicitSemanticLayoutEvidence)
    )
    const failureReasonCode: PublicationReasonCode = semanticIdentityUnconfirmed
      ? "semantic_identity_unconfirmed"
      : decision.reasonCode
    const preprocessingAssets = await preprocessingRunAssets(prepared)
    const evidence = buildPipelineEvidence({
      preprocessing: prepared.report,
      geometry,
      planning,
      primaryPlan,
      candidates,
      selectionCandidates,
      nativeGridCandidates,
      nativeGridStructurallyComplete,
      coreAgreementReached,
      recoveryExpanded: benchmarkMode || !coreAgreementReached,
      stability: notRequiredStabilityEvidence(),
    })
    return {
      status: "failed",
      publicationDecision: {
        policyId: DIGITIZER_POLICY_ID,
        policyVersion: DIGITIZER_POLICY_VERSION,
        outcome: decision.outcome,
        reasonCode: failureReasonCode,
      },
      message: semanticIdentityUnconfirmed
        ? "Open-ECG-Digitizer recovered candidate traces, but no value-aware label recognizer confirmed the printed lead names and order. No quantitative CSV was published."
        : "Open-ECG-Digitizer produced candidate CSVs, but no candidate satisfied the per-lead publication requirements for coverage, gap safety, and cross-candidate agreement. No digitized output was published.",
      digitizer: {
        engine: "Open-ECG-Digitizer",
        candidates: candidates.map(publicCandidate),
        evidence,
      },
      assets: {
        ...run.assets,
        ...preprocessingAssets,
      },
      updatedAt: new Date().toISOString(),
    }
  }
  const selected = decision.candidate
  const stabilityResult = await confirmBorderlineMpsSelection({
    run,
    prepared,
    selected,
    selectionCandidates,
    decisionOutcome: decision.outcome,
    decisionReasonCode: decision.reasonCode,
    benchmarkMode,
    signal: options.signal,
  })
  candidates.push(...stabilityResult.candidates)
  const evidence = buildPipelineEvidence({
    preprocessing: prepared.report,
    geometry,
    planning,
    primaryPlan,
    candidates,
    selectionCandidates,
    nativeGridCandidates,
    nativeGridStructurallyComplete,
    coreAgreementReached,
    recoveryExpanded: benchmarkMode || !coreAgreementReached,
    selected,
    reasonCode: stabilityResult.confirmed
      ? decision.reasonCode
      : "unstable_neural_confirmation",
    stability: stabilityResult.evidence,
  })
  const publicationReasonCode: PublicationReasonCode =
    stabilityResult.confirmed
      ? decision.reasonCode
      : "unstable_neural_confirmation"
  const published = await publishSelectedCandidate(
    run,
    selected,
    candidates,
    prepared,
    geometry,
    evidence,
    decision.outcome === "needs_review",
    publicationReasonCode,
    options.signal
  )
  const preprocessingAssets = await preprocessingRunAssets(prepared)
  const publicCandidates = candidates.map((candidate) => ({
    ...publicCandidate(candidate),
    selected: candidate.id === selected.id,
  }))

  return {
    status: decision.outcome,
    publicationDecision: {
      policyId: DIGITIZER_POLICY_ID,
      policyVersion: DIGITIZER_POLICY_VERSION,
      outcome: decision.outcome,
      reasonCode: publicationReasonCode,
      candidateId: selected.id,
    },
    message: !stabilityResult.confirmed && decision.outcome === "needs_review"
      ? `The ECG Digitizer retained all 12 expected trace positions as a lower-confidence quantitative output. Repeat neural confirmation was not fully stable; compare the overlay and uncertainty file with the source before use.`
      : decision.outcome === "partial"
      ? decision.reasonCode === "input_quality_review_only"
        ? `The ECG Digitizer produced a diagnostic review view for the ${selected.layout ?? "ECG"} layout, but the source has a detected screen or glare artifact. No quantitative CSV was published.`
      : decision.partialLeadSelection
        ? `The ECG Digitizer recognized a ${selected.layout ?? "partial-lead"} source and produced a diagnostic review view. Because the expected 12 leads are not present, no quantitative CSV was published.`
        : selected.sourceFidelity?.passed
        ? `The ECG Digitizer recognized the ${selected.layout ?? "ECG"} layout and produced a source-verified diagnostic review view. This result did not satisfy the quantitative publication threshold, so no CSV was published.`
        : `The ECG Digitizer recognized the ${selected.layout ?? "ECG"} layout and produced a label-corroborated diagnostic review view. This result did not satisfy the quantitative publication threshold, so no CSV was published.`
      : selected.qa?.passed
        ? `The ECG Digitizer completed locally. Selected ${selected.label}; visual review is required.`
        : `The ECG Digitizer completed locally, but QA flagged ${selected.qa?.warnings.length ?? 0} issue(s).`,
    layout: selected.layout,
    layoutCost: selected.layoutCost,
    ...(decision.outcome === "needs_review"
      ? {
          sampleRateHz: SAMPLE_RATE_HZ,
          effectiveSampleRateHz: selected.effectiveSampleRateHz,
        }
      : {}),
    ...(validatedCalibration(geometry)
      ? {
          paperSpeedMmPerSecond:
            validatedCalibration(geometry)?.paperSpeedMmPerSecond,
          gainMmPerMv: validatedCalibration(geometry)?.gainMmPerMv,
          calibrationConfidence: validatedCalibration(geometry)?.confidence,
        }
      : {}),
    leadCount:
      decision.outcome === "needs_review"
        ? LEAD_ORDER.length
        : expectedLayoutLeads(selected.layout).length,
    selectedCandidateId: selected.id,
    digitizer: {
      engine: "Open-ECG-Digitizer",
      selectedCandidateId: selected.id,
      candidates: publicCandidates,
      evidence,
    },
    assets: {
      ...(decision.outcome === "needs_review"
        ? run.assets
        : assetsWithoutQuantitativeOutputs(run.assets)),
      ...preprocessingAssets,
      ...published.assets,
    },
    reliability: published.reliability,
    updatedAt: new Date().toISOString(),
  }
}

function assetsWithoutQuantitativeOutputs(
  assets: Partial<Record<RunAssetKey, RunAsset>>
) {
  const reviewAssets = { ...assets }
  delete reviewAssets.canonicalCsv
  delete reviewAssets.segmentsCsv
  delete reviewAssets.uncertaintyCsv
  delete reviewAssets.paperRender
  return reviewAssets
}

export async function persistPaperRenderForRun(
  run: RunRecord,
  options: DigitizerExecutionOptions = {}
) {
  const segmentsAsset = run.assets.segmentsCsv
  if (!segmentsAsset) return null
  if (run.source !== "upload") {
    throw new Error(
      "Only locally admitted uploads can persist a new derived paper render."
    )
  }

  const verifiedSegments = await readVerifiedRunAsset(run, "segmentsCsv")
  if (!verifiedSegments) return null
  const verifiedUncertainty = run.assets.uncertaintyCsv
    ? await readVerifiedRunAsset(run, "uncertaintyCsv")
    : null
  const exportsDir = path.join(runDirectory(run.id), "exports")
  const stem = path.parse(segmentsAsset.path).name.replace(/_segments_500hz$/, "")
  const outputPath = path.join(exportsDir, `${stem}_paper_render.png`)

  await fs.mkdir(exportsDir, { recursive: true, mode: 0o700 })
  const verifiedInputDir = await fs.mkdtemp(
    path.join(exportsDir, ".verified-paper-input-")
  )
  try {
    await fs.chmod(verifiedInputDir, 0o700)
    const csvPath = path.join(verifiedInputDir, "segments.csv")
    await fs.writeFile(csvPath, verifiedSegments.bytes, { mode: 0o600 })
    const uncertaintyPath = verifiedUncertainty
      ? path.join(verifiedInputDir, "uncertainty.csv")
      : undefined
    if (uncertaintyPath && verifiedUncertainty) {
      await fs.writeFile(uncertaintyPath, verifiedUncertainty.bytes, {
        mode: 0o600,
      })
    }
    await renderPaperFromSegmentsCsv({
      csvPath,
      outputPath,
      uncertaintyPath,
      layout: run.layout,
      effectiveSampleRateHz: run.effectiveSampleRateHz,
      paperSpeedMmPerSecond: run.paperSpeedMmPerSecond,
      gainMmPerMv: run.gainMmPerMv,
      signal: options.signal,
    })
  } finally {
    await fs.rm(verifiedInputDir, { recursive: true, force: true })
  }

  return assetFromAbsolutePath(
    "ECG-paper render",
    outputPath,
    verifiedSegments.identity.sourceSha256
  )
}

async function checkDigitizerAvailable() {
  const openEcgDir = openEcgDirPath()
  const pythonPath = openEcgPythonPath()
  const requiredPaths = [
    openEcgDir,
    pythonPath,
    path.join(openEcgDir, "weights", "unet_weights_07072025.pt"),
    path.join(openEcgDir, "weights", "lead_name_unet_weights_07072025.pt"),
    RELIABLE_LAYOUT_CONFIG_PATH,
    STANDARD_3X4_LAYOUT_CONFIG_PATH,
    STANDARD_3X4_WITH_R1_LAYOUT_CONFIG_PATH,
    STANDARD_6X2_LAYOUT_CONFIG_PATH,
    STANDARD_6X2_WITH_R1_IGNORED_LAYOUT_CONFIG_PATH,
    STANDARD_12X1_LAYOUT_CONFIG_PATH,
  ]

  for (const requiredPath of requiredPaths) {
    try {
      await fs.access(requiredPath)
    } catch {
      return "Input saved locally. Open-ECG-Digitizer is not set up under .external/open-ecg-digitizer, so automatic extraction was skipped."
    }
  }

  return null
}

async function runCandidate(
  run: RunRecord,
  prepared: PreparedRunInput,
  candidate: DigitizerCandidateConfig,
  signal?: AbortSignal
): Promise<CandidateResult> {
  const startedAt = performance.now()
  const inputPath = candidateInputPath(prepared, candidate.inputVariant)
  const candidateDir = path.join(
    /* turbopackIgnore: true */ runDirectory(run.id),
    "candidates",
    candidate.id
  )
  const candidateInputDir = path.join(
    /* turbopackIgnore: true */ runDirectory(run.id),
    "candidate-inputs",
    candidate.id
  )
  const configPath = path.join(candidateDir, "open_ecg_config.yml")
  const localPath = relativePath(candidateDir)

  try {
    await fs.mkdir(candidateDir, { recursive: true })
    const preparedInput = await prepareCandidateInput(
      inputPath,
      candidateInputDir,
      signal,
      candidate.upscaleToMaxDimension,
      candidate.cropBox
    )
    const featureCachePath = await neuralFeatureCachePath(
      run,
      preparedInput.inputPath,
      candidate.resampleSize,
      candidate.device
    )
    const featureCacheHit = await pathExists(featureCachePath)
    const configText = openEcgConfig({
      inputDir: preparedInput.inputDir,
      outputDir: candidateDir,
      resampleSize: candidate.resampleSize,
      labelThresh: candidate.labelThresh,
      vectorizer: candidate.vectorizer,
      device: candidate.device,
      darkInkEnhancement: candidate.darkInkEnhancement,
      darkInkSupportRadius: candidate.darkInkSupportRadius,
      layoutConstraint: candidate.layoutConstraint,
      featureCachePath,
    })

    await fs.writeFile(configPath, configText, "utf8")
    const releaseNeuralInference = await neuralInferenceSemaphore.acquire()
    try {
      await execFileAsync(
        openEcgPythonPath(),
        ["-m", "src.digitize", "--config", configPath],
        {
          cwd: openEcgDirPath(),
          env: subprocessEnvironment(),
          maxBuffer: 32 * 1024 * 1024,
          signal,
          timeout: 180_000,
        }
      )
    } finally {
      releaseNeuralInference()
    }
    await fs.writeFile(configPath, configText, "utf8")

    const canonicalPath = path.join(
      /* turbopackIgnore: true */ candidateDir,
      `${preparedInput.stem}_timeseries_canonical.csv`
    )
    const diagnosticPath = path.join(
      /* turbopackIgnore: true */ candidateDir,
      `${preparedInput.stem}.png`
    )
    const metadataPath = path.join(
      /* turbopackIgnore: true */ candidateDir,
      "digitization_metadata.csv"
    )

    const rawCanonical = await readCanonicalCsv(canonicalPath)
    const metadata = await readDigitizerMetadata(metadataPath)
    const minimumGeometryConfidence =
      candidate.layoutConstraint === "standard_6x2" ||
      candidate.layoutConstraint === "standard_6x2_with_r1_ignored"
        ? MIN_GEOMETRY_CONSTRAINED_6X2_CONFIDENCE
        : MIN_GEOMETRY_CONSTRAINED_3X4_CONFIDENCE
    const geometryConfirmedLayoutFallback = Boolean(
      candidate.geometryConfirmedLayout &&
        normalizeSupportedEcgLayout(candidate.layoutConstraint) &&
        (candidate.geometryLayoutConfidence ?? 0) >=
          minimumGeometryConfidence &&
        (!metadata.layout || metadata.layout === "Unknown layout")
    )
    if (
      candidate.layoutConstraint &&
      ((!geometryConfirmedLayoutFallback &&
        !Number.isFinite(metadata.layoutCost)) ||
        (candidate.maximumLayoutCost !== undefined &&
          (metadata.layoutCost ?? Number.POSITIVE_INFINITY) >
            candidate.maximumLayoutCost))
    ) {
      throw new Error(
        `The constrained ${candidate.layoutConstraint} layout did not have sufficient lead-label evidence.`
      )
    }
    const effectiveLayout = geometryConfirmedLayoutFallback
      ? candidate.layoutConstraint
      : metadata.layout
    const effectiveLayoutCost = geometryConfirmedLayoutFallback
      ? 1 - (candidate.geometryLayoutConfidence ?? 0)
      : metadata.layoutCost
    const timingNormalization = normalizeCanonicalPanelTiming(
      rawCanonical,
      effectiveLayout
    )
    const canonical = timingNormalization.canonical
    if (timingNormalization.corrections.length > 0) {
      const rawCanonicalPath = path.join(
        candidateDir,
        `${preparedInput.stem}_timeseries_canonical_raw.csv`
      )
      await fs.copyFile(canonicalPath, rawCanonicalPath)
      await writeCanonicalCsv(canonical, canonicalPath)
    }
    const qa = evaluateQa(canonical, effectiveLayout)
    const effectiveSampleRateHz = estimateEffectiveSampleRateHz(
      prepared.report,
      candidate.resampleSize
    )
    const scoreBreakdown = scoreCandidate(
      qa,
      effectiveLayout,
      effectiveLayoutCost,
      candidate,
      prepared.report
    )
    const score = Object.values(scoreBreakdown).reduce(
      (total, value) => total + value,
      0
    )
    qa.score = score
    qa.scoreBreakdown = scoreBreakdown

    return {
      id: candidate.id,
      label: candidate.label,
      localPath,
      status: "completed",
      parameters: parametersForCandidate(candidate),
      layout: effectiveLayout,
      layoutCost: effectiveLayoutCost,
      effectiveSampleRateHz,
      panelTimingCorrections: timingNormalization.corrections,
      runtimeMs: performance.now() - startedAt,
      qa,
      canonicalPath,
      diagnosticPath,
      metadataPath,
      canonical,
      score,
      scoreBreakdown,
      featureCacheHit,
    }
  } catch (error) {
    return {
      id: candidate.id,
      label: candidate.label,
      localPath,
      status: "failed",
      parameters: parametersForCandidate(candidate),
      runtimeMs: performance.now() - startedAt,
      message:
        error instanceof Error
          ? error.message
          : "Digitizer output could not be inspected.",
      score: Number.POSITIVE_INFINITY,
    }
  }
}

function candidateInputPath(
  prepared: PreparedRunInput,
  inputVariant: DigitizerInputVariant
) {
  if (inputVariant === "annotation-masked") return prepared.preparedPath
  if (inputVariant === "preprocessed") return prepared.enhancedPath
  if (inputVariant === "geometry-corrected") {
    return prepared.geometryCorrectedPath
  }
  if (inputVariant === "artifact-preprocessed") {
    return prepared.artifactPreprocessedPath
  }
  return prepared.workingSourcePath
}

async function runNativeGridCandidate(
  run: RunRecord,
  prepared: PreparedRunInput,
  geometry: LayoutGeometryReport,
  inputVariant: DigitizerInputVariant,
  layoutConstraint: NativeGridLayoutConstraint,
  usePreparedEvidence = false,
  adaptivePreprocessingEligible = false,
  signal?: AbortSignal,
  upscaleToMaxDimension?: number
): Promise<CandidateResult> {
  const baseId = usePreparedEvidence
    ? "preprocessed-native-grid-path"
    : layoutConstraint === "standard_3x4"
      ? "three-by-four-native-grid-path"
    : layoutConstraint === "standard_3x4_with_r1"
      ? "three-by-four-rhythm-native-grid-path"
      : layoutConstraint === "standard_12x1"
        ? "twelve-row-native-grid-path"
      : layoutConstraint === "row_local_compound_12lead"
        ? "compound-row-local-native-grid-path"
      : layoutConstraint === "row_local_labeled_6x2"
        ? "labeled-six-by-two-row-local-native-grid-path"
      : layoutConstraint === "standard_6x2"
        ? "six-by-two-native-grid-path"
      : "six-by-two-rhythm-native-grid-path"
  const id = upscaleToMaxDimension
    ? `${baseId}-upscaled-${upscaleToMaxDimension}`
    : baseId
  const label = usePreparedEvidence
    ? upscaleToMaxDimension
      ? `Colour/grid preprocessed native extraction (${upscaleToMaxDimension}px deterministic scale)`
      : "Colour/grid preprocessed native extraction"
    : "Native grid-aware deterministic extraction"
  const publishedLayoutConstraint =
    layoutConstraint === "row_local_compound_12lead"
      ? "standard_6x2"
      : layoutConstraint === "row_local_labeled_6x2"
        ? "standard_6x2"
        : layoutConstraint
  const candidateDir = path.join(
    /* turbopackIgnore: true */ runDirectory(run.id),
    "candidates",
    id
  )
  const localPath = relativePath(candidateDir)
  const startedAt = performance.now()
  const parameters: DigitizerCandidateConfig = {
    kind: "native-grid",
    id,
    label,
    resampleSize:
      prepared.report.workingImage?.width ?? prepared.report.source.width,
    ...(upscaleToMaxDimension ? { upscaleToMaxDimension } : {}),
    layoutConstraint,
    ...(normalizeQaLayout(publishedLayoutConstraint) ===
      normalizeQaLayout(geometry.layoutHint ?? undefined) &&
    (geometry.confidence ?? 0) >= MIN_REVIEW_LAYOUT_CONFIDENCE
      ? {
          geometryConfirmedLayout: true,
          geometryLayoutConfidence: geometry.confidence,
        }
      : {}),
    ...semanticLeadIdentityParameters(geometry),
    ...(usePreparedEvidence ? { adaptivePreprocessingEligible } : {}),
    inputVariant,
    vectorizer: "native-grid-path",
    device: "cpu",
    planningPhase: "core",
    scheduleReason: "deterministic native grid and source-pixel extraction",
  }

  try {
    await fs.rm(candidateDir, { force: true, recursive: true })
    await fs.mkdir(candidateDir, { recursive: true })
    const extractionPath = usePreparedEvidence
      ? prepared.preparedPath
      : candidateInputPath(prepared, inputVariant)
    const fidelitySourcePath =
      inputVariant === "geometry-corrected"
        ? prepared.geometryCorrectedPath
        : prepared.workingSourcePath
    const commandArguments = [
      NATIVE_GRID_DIGITIZER_PATH,
      "--input",
      extractionPath,
      "--source",
      fidelitySourcePath,
      "--output-dir",
      candidateDir,
    ]
    if (usePreparedEvidence) {
      commandArguments.push("--evidence", prepared.traceProbabilityPath)
    }
    if (upscaleToMaxDimension) {
      commandArguments.push(
        "--upscale-max-dimension",
        String(upscaleToMaxDimension)
      )
    }
    if (
      layoutConstraint === "standard_6x2_with_r1_ignored" &&
      geometry.rhythmLeadValidation?.passed === true &&
      geometry.rhythmLeadValidation.semanticIdentityConfirmed === true &&
      geometry.rhythmLeadValidation.lead === "II"
    ) {
      commandArguments.push("--verified-rhythm-lead", "II")
    }
    const releaseNativeExtraction = await nativeExtractionSemaphore.acquire()
    let stdout: string
    try {
      const result = await execFileAsync(
        openEcgPythonPath(),
        commandArguments,
        {
          cwd: RESOURCE_ROOT,
          env: subprocessEnvironment(),
          maxBuffer: 4 * 1024 * 1024,
          signal,
          timeout: 60_000,
        }
      )
      stdout = result.stdout
    } finally {
      releaseNativeExtraction()
    }
    const result = parseNativeExtractionReport(
      stdout.trim(),
      publishedLayoutConstraint
    )
    const sourceFidelity = prepared.report.workingImage
      ? {
          ...result.sourceFidelity,
          boundedSourceRaster: {
            method: prepared.report.workingImage.method,
            originalWidth: prepared.report.source.width,
            originalHeight: prepared.report.source.height,
            workingWidth: prepared.report.workingImage.width,
            workingHeight: prepared.report.workingImage.height,
            scaleX: prepared.report.workingImage.scaleX,
            scaleY: prepared.report.workingImage.scaleY,
            morphologyReconstructed: false as const,
          },
        }
      : result.sourceFidelity

    const canonicalPath = path.join(
      candidateDir,
      "native_grid_timeseries_canonical.csv"
    )
    const diagnosticPath = path.join(candidateDir, "native_grid_overlay.png")
    const metadataPath = path.join(
      /* turbopackIgnore: true */ candidateDir,
      "digitization_metadata.csv"
    )
    const sourceFidelityPath = path.join(candidateDir, "source_fidelity.json")
    const extractedCanonical = await readCanonicalCsv(canonicalPath)
    const canonical =
      nativeCanonicalRequiresBoundarySuppression(
        publishedLayoutConstraint,
        sourceFidelity
      )
        ? suppressConstrainedPanelBoundaries(extractedCanonical)
        : extractedCanonical
    if (canonical !== extractedCanonical) {
      await fs.copyFile(
        canonicalPath,
        path.join(candidateDir, "native_grid_timeseries_canonical_raw.csv")
      )
      await writeCanonicalCsv(canonical, canonicalPath)
    }
    const qa = evaluateQa(canonical, result.layout)
    const scoreBreakdown = scoreCandidate(
      qa,
      result.layout,
      result.layoutCost,
      parameters,
      prepared.report
    )
    const score = Object.values(scoreBreakdown).reduce(
      (total, value) => total + value,
      0
    )
    qa.score = score
    qa.scoreBreakdown = scoreBreakdown

    return {
      id,
      label,
      localPath,
      status: "completed",
      parameters: parametersForCandidate(parameters),
      layout: result.layout,
      layoutCost: result.layoutCost,
      effectiveSampleRateHz: result.effectiveSampleRateHz,
      panelTimingCorrections: [],
      sourceFidelity,
      runtimeMs: performance.now() - startedAt,
      qa,
      canonicalPath,
      diagnosticPath,
      metadataPath,
      sourceFidelityPath,
      canonical,
      score,
      scoreBreakdown,
    }
  } catch (error) {
    return {
      id,
      label,
      localPath,
      status: "failed",
      parameters: parametersForCandidate(parameters),
      runtimeMs: performance.now() - startedAt,
      message:
        error instanceof Error
          ? error.message
          : "Native grid extraction could not be inspected.",
      score: Number.POSITIVE_INFINITY,
    }
  }
}

function nativeCanonicalRequiresBoundarySuppression(
  layout: string,
  sourceFidelity: DigitizerSourceFidelitySummary
) {
  return (
    layout === "standard_6x2" &&
    sourceFidelity.sourcePanelTimingDetected !== true
  )
}

async function prepareRunInput(
  run: RunRecord,
  inputAsset: RunAsset,
  signal?: AbortSignal
): Promise<PreparedRunInput> {
  const sourcePath = resolveWorkspacePath(inputAsset.path)
  const preprocessingDir = path.join(
    /* turbopackIgnore: true */ runDirectory(run.id),
    "preprocessing"
  )
  const workingSourcePath = path.join(preprocessingDir, "working_source.png")
  const preparedPath = path.join(preprocessingDir, "annotation_suppressed.png")
  const maskPath = path.join(preprocessingDir, "annotation_mask.png")
  const enhancedPath = path.join(preprocessingDir, "enhanced.png")
  const backgroundPath = path.join(
    preprocessingDir,
    "background_flattened.png"
  )
  const traceProbabilityPath = path.join(
    preprocessingDir,
    "trace_probability.png"
  )
  const gridProbabilityPath = path.join(
    preprocessingDir,
    "grid_probability.png"
  )
  const exclusionMaskPath = path.join(
    preprocessingDir,
    "exclusion_mask.png"
  )
  const geometryCorrectedPath = path.join(
    preprocessingDir,
    "geometry_corrected.png"
  )
  const artifactPreprocessedPath = path.join(
    preprocessingDir,
    "artifact_preprocessed.png"
  )
  const reportPath = path.join(preprocessingDir, "provenance.json")
  const { stdout: inspectionStdout } = await execFileAsync(
    openEcgPythonPath(),
    [INPUT_PREPARER_PATH, "--input", sourcePath, "--inspect"],
    {
      cwd: RESOURCE_ROOT,
      env: subprocessEnvironment(),
      maxBuffer: 1024 * 1024,
      signal,
      timeout: 15_000,
    }
  )
  const inspection = JSON.parse(inspectionStdout) as {
    width?: number
    height?: number
    pixelCount?: number
  }
  if (
    !Number.isFinite(inspection.width) ||
    !Number.isFinite(inspection.height) ||
    !Number.isFinite(inspection.pixelCount)
  ) {
    throw new Error("The source image metadata probe returned invalid dimensions.")
  }
  const sourceMegapixels = (inspection.pixelCount ?? 0) / 1_000_000
  const preprocessingWeight =
    sourceMegapixels >= 25 ? 3 : sourceMegapixels >= 12 ? 2 : 1
  const release = await preprocessingSemaphore.acquire(preprocessingWeight)
  const attempts: NonNullable<PreprocessingReport["execution"]>["attempts"] = []
  let report: PreprocessingReport | undefined
  try {
    for (const maxWorkingLongEdgePixels of PREPROCESSING_ATTEMPTS) {
      await fs.rm(preprocessingDir, { force: true, recursive: true })
      await fs.mkdir(preprocessingDir, { recursive: true })
      const startedAt = performance.now()
      try {
        await execFileAsync(
          openEcgPythonPath(),
          [
            INPUT_PREPARER_PATH,
            "--input",
            sourcePath,
            "--max-working-long-edge",
            String(maxWorkingLongEdgePixels),
            "--prepared",
            preparedPath,
            "--working-source",
            workingSourcePath,
            "--mask",
            maskPath,
            "--enhanced",
            enhancedPath,
            "--background",
            backgroundPath,
            "--trace-probability",
            traceProbabilityPath,
            "--grid-probability",
            gridProbabilityPath,
            "--exclusion-mask",
            exclusionMaskPath,
            "--geometry-corrected",
            geometryCorrectedPath,
            "--artifact-preprocessed",
            artifactPreprocessedPath,
            "--report",
            reportPath,
          ],
          {
            cwd: RESOURCE_ROOT,
            env: subprocessEnvironment(),
            maxBuffer: 4 * 1024 * 1024,
            signal,
            timeout: PREPROCESSING_TIMEOUT_MS,
          }
        )
        attempts.push({
          maxWorkingLongEdgePixels,
          runtimeMs: Math.round(performance.now() - startedAt),
          succeeded: true,
        })
        report = parsePreprocessingReport(
          await fs.readFile(reportPath, "utf8")
        )
        break
      } catch (error) {
        const detail = preprocessingFailureDetail(error)
        attempts.push({
          maxWorkingLongEdgePixels,
          runtimeMs: Math.round(performance.now() - startedAt),
          succeeded: false,
          error: detail,
        })
        if (signal?.aborted) throw error
      }
    }
    if (!report) {
      const failurePath = path.join(
        preprocessingDir,
        "preprocessing_failure.json"
      )
      await fs.writeFile(
        failurePath,
        `${JSON.stringify(
          {
            version: 1,
            sourceMegapixels,
            preprocessingWeight,
            attempts,
          },
          null,
          2
        )}\n`,
        "utf8"
      )
      throw new Error(
        `resource-bounded preprocessing failed after ${attempts.length} attempt(s): ${attempts.at(-1)?.error ?? "unknown error"}`
      )
    }
    report.execution = {
      attempts,
      preprocessingPermits: preprocessingWeight,
      sourceMegapixels,
    }
    await fs.writeFile(
      reportPath,
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8"
    )
  } finally {
    release()
  }

  return {
    sourcePath,
    workingSourcePath,
    preparedPath,
    enhancedPath,
    backgroundPath,
    traceProbabilityPath,
    gridProbabilityPath,
    exclusionMaskPath,
    geometryCorrectedPath,
    artifactPreprocessedPath,
    maskPath,
    reportPath,
    report,
  }
}

function preprocessingFailureDetail(error: unknown) {
  if (!error || typeof error !== "object") {
    return String(error)
  }
  const record = error as {
    message?: unknown
    code?: unknown
    signal?: unknown
    stderr?: unknown
  }
  const stderr =
    typeof record.stderr === "string"
      ? record.stderr.trim().slice(-2000)
      : ""
  return [
    typeof record.message === "string" ? record.message : "subprocess failed",
    record.code !== undefined ? `code=${String(record.code)}` : "",
    record.signal !== undefined ? `signal=${String(record.signal)}` : "",
    stderr ? `stderr=${stderr}` : "",
  ]
    .filter(Boolean)
    .join("; ")
}

async function detectLayoutGeometry(
  prepared: PreparedRunInput,
  signal?: AbortSignal
): Promise<LayoutGeometryReport> {
  const workingGeometry = {
    ...(await detectLayoutGeometryAtPath(prepared.preparedPath, signal)),
    detectedInputVariant:
      prepared.report.annotationMask.maskedPixels > 0
        ? ("annotation-masked" as const)
        : ("original" as const),
    coordinateSpace: "working" as const,
  }
  if (prepared.report.geometryCorrection?.applied !== true) {
    return workingGeometry
  }

  const correctedGeometry = {
    ...(await detectLayoutGeometryAtPath(
      prepared.geometryCorrectedPath,
      signal
    )),
    detectedInputVariant: "geometry-corrected" as const,
    coordinateSpace: "geometry-corrected" as const,
  }
  const sourceConfidence = workingGeometry.confidence ?? 0
  const correctedConfidence = correctedGeometry.confidence ?? 0
  const correctedIsUsable = Boolean(
    correctedGeometry.layoutHint &&
      correctedConfidence >= 0.2 &&
      (workingGeometry.layoutHint === undefined ||
        workingGeometry.layoutHint === null ||
        correctedGeometry.layoutHint === workingGeometry.layoutHint ||
        correctedConfidence >= sourceConfidence + 0.2) &&
      correctedConfidence + 0.15 >= sourceConfidence
  )
  return correctedIsUsable
    ? correctedGeometry
    : workingGeometry
}

async function detectLayoutGeometryAtPath(
  inputPath: string,
  signal?: AbortSignal
): Promise<LayoutGeometryReport> {
  try {
    const { stdout } = await execFileAsync(
      openEcgPythonPath(),
      [
        LAYOUT_DETECTOR_PATH,
        "--input",
        inputPath,
      ],
      {
        cwd: RESOURCE_ROOT,
        env: subprocessEnvironment(),
        maxBuffer: 1024 * 1024,
        signal,
        timeout: 30_000,
      }
    )
    return attachSemanticLeadEvidence(parseLayoutGeometryReport(stdout))
  } catch (error) {
    return {
      detectionFailure: {
        method: "layout-detector-subprocess-v1",
        message: preprocessingFailureDetail(error).slice(0, 2_000),
      },
    }
  }
}

function attachSemanticLeadEvidence(
  geometry: LayoutGeometryReport
): LayoutGeometryReport {
  const expectedLeads = expectedLayoutLeads(geometry.layoutHint ?? undefined)
  const normalizedLayout = normalizeQaLayout(geometry.layoutHint ?? undefined)
  const expectedOrder = normalizedLayout?.startsWith("cabrera")
    ? "cabrera"
    : "standard"
  const semanticRecognitionConfirmed = (
    validation:
      | LayoutGeometryReport["leadLabelValidation"]
      | LayoutGeometryReport["precordialLabelValidation"]
      | LayoutGeometryReport["limbLabelValidation"]
      | undefined
  ) => {
    const recognition = validation?.semanticRecognition
    return Boolean(
      validation?.passed === true &&
        validation.order === expectedOrder &&
        validation.semanticIdentityConfirmed === true &&
        recognition?.passed === true &&
        recognition.semanticIdentityConfirmed === true &&
        recognition.order === expectedOrder &&
        recognition.method === validation.method &&
        recognition.confidence > 0 &&
        recognition.failureReasons.length === 0 &&
        recognition.expectedLabels?.length === expectedLeads.length &&
        recognition.expectedLabels.every(
          (lead, index) => lead === expectedLeads[index]
        ) &&
        recognition.recognizedLabels?.length === expectedLeads.length &&
        recognition.recognizedLabels.every(
          (lead, index) => lead === expectedLeads[index]
        )
    )
  }
  const labelValidated = Boolean(
    semanticRecognitionConfirmed(geometry.leadLabelValidation)
  )
  const pairedLimbLabels = Boolean(
    semanticRecognitionConfirmed(geometry.limbLabelValidation)
  )
  const pairedPanelLabels =
    semanticRecognitionConfirmed(geometry.precordialLabelValidation) &&
    pairedLimbLabels
  const passed = Boolean(
    expectedLeads.length > 0 &&
      (labelValidated || pairedPanelLabels)
  )
  const reasons = passed
    ? [
        labelValidated
          ? "explicit-lead-label-sequence"
          : "paired-limb-and-precordial-labels",
      ]
    : [
        expectedLeads.length === 0
          ? "unsupported-or-unknown-layout"
          : "lead-label-values-and-order-not-confirmed",
      ]
  return {
    ...geometry,
    semanticLeadEvidence: {
      passed,
      confidence: passed ? geometry.confidence ?? 0 : 0,
      method: "pre-extraction-lead-label-and-layout-gate-v1",
      visibleLeads: passed ? expectedLeads : [],
      reasons,
    },
  }
}

function validatedCalibration(geometry: LayoutGeometryReport) {
  const calibration = geometry.calibration
  if (
    calibration?.detected !== true ||
    (calibration.confidence ?? 0) < 0.35 ||
    ![25, 50].includes(calibration.paperSpeedMmPerSecond ?? 0) ||
    !Number.isFinite(calibration.gainMmPerMv) ||
    (calibration.gainMmPerMv ?? 0) < 5 ||
    (calibration.gainMmPerMv ?? 0) > 20
  ) {
    return undefined
  }
  return calibration as Required<
    Pick<
      NonNullable<LayoutGeometryReport["calibration"]>,
      "paperSpeedMmPerSecond" | "gainMmPerMv" | "confidence"
    >
  > & NonNullable<LayoutGeometryReport["calibration"]>
}

function sixByTwoPanelCropBoxes(
  preprocessing: PreprocessingReport,
  geometry: LayoutGeometryReport
): { left: RasterCropBox; right: RasterCropBox } {
  const { width, height } = geometryRasterDimensions(preprocessing, geometry)
  const centers = geometry.rowCenters ?? []
  const primaryBottom =
    centers.length >= 7
      ? clamp(
          Math.round((centers[5] + centers[6]) / 2),
          1,
          height
        )
      : Math.round(height * 0.84)
  const overlap = Math.max(4, Math.round(width * 0.025))
  const split = Math.round(width / 2)
  return {
    left: {
      left: 0,
      top: 0,
      right: Math.min(width, split + overlap),
      bottom: primaryBottom,
    },
    right: {
      left: Math.max(0, split - overlap),
      top: 0,
      right: width,
      bottom: primaryBottom,
    },
  }
}

function deterministicCropProposals(
  preprocessing: PreprocessingReport,
  geometry: LayoutGeometryReport
) {
  const { width, height } = geometryRasterDimensions(preprocessing, geometry)
  const proposals: Array<{
    id: string
    label: string
    cropBox: RasterCropBox
    selectionEligible: boolean
  }> = []
  const content = validCropBox(geometry.contentBox, width, height)
  const contentAreaFraction = content
    ? ((content.right - content.left) * (content.bottom - content.top)) /
      (width * height)
    : 1
  if (
    content &&
    (geometry.contentConfidence ?? 0) >= 0.5 &&
    contentAreaFraction <= 0.92
  ) {
    proposals.push({
      id: "content",
      label: "Grid/content-cropped",
      cropBox: content,
      selectionEligible: true,
    })
  }

  const base = content ?? { left: 0, top: 0, right: width, bottom: height }
  const baseWidth = base.right - base.left
  const baseHeight = base.bottom - base.top
  const aspectRatio = baseWidth / Math.max(baseHeight, 1)
  if (aspectRatio < 1.35 && baseHeight >= 480) {
    const overlap = Math.max(8, Math.round(baseHeight * 0.04))
    const middle = Math.round((base.top + base.bottom) / 2)
    proposals.push(
      {
        id: "upper-panel",
        label: "Upper composite panel",
        cropBox: {
          left: base.left,
          top: base.top,
          right: base.right,
          bottom: Math.min(base.bottom, middle + overlap),
        },
        selectionEligible: false,
      },
      {
        id: "lower-panel",
        label: "Lower composite panel",
        cropBox: {
          left: base.left,
          top: Math.max(base.top, middle - overlap),
          right: base.right,
          bottom: base.bottom,
        },
        selectionEligible: false,
      }
    )
  } else if (aspectRatio > 3.2 && baseWidth >= 900) {
    const overlap = Math.max(8, Math.round(baseWidth * 0.025))
    const middle = Math.round((base.left + base.right) / 2)
    proposals.push(
      {
        id: "left-panel",
        label: "Left composite panel",
        cropBox: {
          left: base.left,
          top: base.top,
          right: Math.min(base.right, middle + overlap),
          bottom: base.bottom,
        },
        selectionEligible: false,
      },
      {
        id: "right-panel",
        label: "Right composite panel",
        cropBox: {
          left: Math.max(base.left, middle - overlap),
          top: base.top,
          right: base.right,
          bottom: base.bottom,
        },
        selectionEligible: false,
      }
    )
  }

  const unique = new Map<string, (typeof proposals)[number]>()
  for (const proposal of proposals) {
    const crop = validCropBox(proposal.cropBox, width, height)
    if (!crop) continue
    const key = `${crop.left}:${crop.top}:${crop.right}:${crop.bottom}`
    unique.set(key, { ...proposal, cropBox: crop })
  }
  return [...unique.values()]
}

function geometryRasterDimensions(
  preprocessing: PreprocessingReport,
  geometry: LayoutGeometryReport
) {
  if (geometry.coordinateSpace === "geometry-corrected") {
    return {
      width:
        preprocessing.geometryCorrection?.outputWidth ??
        preprocessing.workingImage?.width ??
        preprocessing.source.width,
      height:
        preprocessing.geometryCorrection?.outputHeight ??
        preprocessing.workingImage?.height ??
        preprocessing.source.height,
    }
  }
  return {
    width: preprocessing.workingImage?.width ?? preprocessing.source.width,
    height: preprocessing.workingImage?.height ?? preprocessing.source.height,
  }
}

function validCropBox(
  cropBox: RasterCropBox | null | undefined,
  width: number,
  height: number
) {
  if (!cropBox) return undefined
  const crop = {
    left: clamp(Math.round(cropBox.left), 0, width - 1),
    top: clamp(Math.round(cropBox.top), 0, height - 1),
    right: clamp(Math.round(cropBox.right), 1, width),
    bottom: clamp(Math.round(cropBox.bottom), 1, height),
  }
  if (
    crop.right - crop.left < 320 ||
    crop.bottom - crop.top < 160 ||
    crop.right <= crop.left ||
    crop.bottom <= crop.top
  ) {
    return undefined
  }
  return crop
}

async function buildSixByTwoRhythmCompositeCandidate({
  run,
  prepared,
  left,
  right,
  suffix,
  vectorizer,
  darkInkEnhancement,
  darkInkSupportRadius,
  labelThresh,
  device,
  signal,
  orientation = "side_by_side",
  targetLayout = "standard_6x2_with_r1_ignored",
  label: suppliedLabel,
}: {
  run: RunRecord
  prepared: PreparedRunInput
  left: CandidateResult
  right: CandidateResult
  suffix: string
  vectorizer: DigitizerVectorizer
  darkInkEnhancement: boolean
  darkInkSupportRadius?: number
  labelThresh?: number
  device: DigitizerComputeDevice
  signal?: AbortSignal
  orientation?: "side_by_side" | "stacked"
  targetLayout?: "standard_6x2_with_r1_ignored" | "standard_6x2" | "standard_12x1"
  label?: string
}): Promise<CandidateResult> {
  const id = `${SIX_BY_TWO_RHYTHM_COMPOSITE_PREFIX}-${suffix}`
  const label = suppliedLabel ?? "6 × 2 + rhythm panel-composite extraction"
  const compositeOrientation = orientation ?? "side_by_side"
  const candidateDir = path.join(runDirectory(run.id), "candidates", id)
  const localPath = relativePath(candidateDir)
  const startedAt = performance.now()
  const leftInputVariant = left.parameters.inputVariant ?? "original"
  const rightInputVariant = right.parameters.inputVariant ?? "original"
  const compositeInputVariant =
    leftInputVariant === rightInputVariant
      ? leftInputVariant
      : [leftInputVariant, rightInputVariant].every(
            isAnnotationSafeInputVariant
          )
        ? "annotation-masked"
        : "original"
  const parameters: DigitizerCandidateConfig = {
    kind: "panel-composite",
    id,
    label,
    resampleSize: 1500,
    upscaleToMaxDimension: 1500,
    ...(darkInkEnhancement ? { darkInkEnhancement: true } : {}),
    ...(darkInkSupportRadius ? { darkInkSupportRadius } : {}),
    ...(labelThresh !== undefined ? { labelThresh } : {}),
    layoutConstraint: targetLayout,
    inputVariant: compositeInputVariant,
    vectorizer,
    device,
  }

  if (
    left.status !== "completed" ||
    right.status !== "completed" ||
    !left.canonical ||
    !right.canonical ||
    !left.diagnosticPath ||
    !right.diagnosticPath ||
    !left.parameters.layoutConstraint ||
    left.layout !== left.parameters.layoutConstraint ||
    !right.parameters.layoutConstraint ||
    right.layout !== right.parameters.layoutConstraint
  ) {
    return {
      id,
      label,
      localPath,
      status: "failed",
      parameters: parametersForCandidate(parameters),
      runtimeMs: performance.now() - startedAt,
      message:
        "The paired limb and precordial panel extractions did not both satisfy constrained layout evidence.",
      score: Number.POSITIVE_INFINITY,
    }
  }

  const layout = targetLayout
  const canonical =
    compositeOrientation === "stacked"
      ? suppressFullWidthPanelBoundaries(
          combineStackedPanelCanonicals(left.canonical, right.canonical)
        )
      : suppressConstrainedPanelBoundaries(
          combineSixByTwoPanelCanonicals(left.canonical, right.canonical)
        )
  const qa = evaluateQa(canonical, layout)
  const layoutCosts = [left.layoutCost, right.layoutCost].filter(
    (cost): cost is number => Number.isFinite(cost)
  )
  const layoutCost =
    layoutCosts.length > 0 ? Math.max(...layoutCosts) : undefined
  const scoreBreakdown = scoreCandidate(
    qa,
    layout,
    layoutCost,
    parameters,
    prepared.report
  )
  const score = Object.values(scoreBreakdown).reduce(
    (total, value) => total + value,
    0
  )
  qa.score = score
  qa.scoreBreakdown = scoreBreakdown

  const canonicalPath = path.join(
    candidateDir,
    "panel_composite_timeseries_canonical.csv"
  )
  const diagnosticPath = path.join(candidateDir, "panel_composite.png")
  const metadataPath = path.join(candidateDir, "digitization_metadata.csv")
  await fs.rm(candidateDir, { force: true, recursive: true })
  await fs.mkdir(candidateDir, { recursive: true })
  await writeCanonicalCsv(canonical, canonicalPath)
  await stitchDiagnosticOverlays(
    left.diagnosticPath,
    right.diagnosticPath,
    diagnosticPath,
    compositeOrientation,
    signal
  )
  await fs.writeFile(
    metadataPath,
    [
      "file_path,matching_cost,is_flipped,lead_layout",
      `panel_composite,${layoutCost ?? ""},False,${layout}`,
      "",
    ].join("\n"),
    "utf8"
  )

  return {
    id,
    label,
    localPath,
    leadSources: Object.fromEntries(
      LEAD_ORDER.map((lead) => [
        lead,
        LAYOUT_COLUMNS.standard_6x2[lead] === 0 ? left.id : right.id,
      ])
    ),
    status: "completed",
    parameters: parametersForCandidate(parameters),
    layout,
    layoutCost,
    effectiveSampleRateHz: Math.min(
      left.effectiveSampleRateHz ?? SAMPLE_RATE_HZ,
      right.effectiveSampleRateHz ?? SAMPLE_RATE_HZ
    ),
    panelTimingCorrections: [],
    runtimeMs:
      (left.runtimeMs ?? 0) +
      (right.runtimeMs ?? 0) +
      (performance.now() - startedAt),
    qa,
    canonicalPath,
    diagnosticPath,
    metadataPath,
    canonical,
    score,
    scoreBreakdown,
  }
}

async function stitchDiagnosticOverlays(
  firstPath: string,
  secondPath: string,
  outputPath: string,
  orientation: "side_by_side" | "stacked",
  signal?: AbortSignal
) {
  const script = [
    "from PIL import Image",
    "import sys",
    "with Image.open(sys.argv[1]) as first_source, Image.open(sys.argv[2]) as second_source:",
    "    first = first_source.convert('RGB')",
    "    second = second_source.convert('RGB')",
    "    if sys.argv[4] == 'stacked':",
    "        canvas = Image.new('RGB', (max(first.width, second.width), first.height + second.height), 'white')",
    "        canvas.paste(first, (0, 0))",
    "        canvas.paste(second, (0, first.height))",
    "    else:",
    "        canvas = Image.new('RGB', (first.width + second.width, max(first.height, second.height)), 'white')",
    "        canvas.paste(first, (0, 0))",
    "        canvas.paste(second, (first.width, 0))",
    "    canvas.save(sys.argv[3], format='PNG')",
  ].join("\n")
  await execFileAsync(
    openEcgPythonPath(),
    ["-c", script, firstPath, secondPath, outputPath, orientation],
    {
      cwd: RESOURCE_ROOT,
      env: subprocessEnvironment(),
      maxBuffer: 1024 * 1024,
      signal,
      timeout: 30_000,
    }
  )
}

function combineSixByTwoPanelCanonicals(
  left: CanonicalCsv,
  right: CanonicalCsv
): CanonicalCsv {
  const panelSamples = (SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS) / 2
  const rows = Array.from(
    { length: panelSamples * 2 },
    () => LEAD_ORDER.map(() => Number.NaN)
  )

  for (const lead of LEAD_ORDER) {
    const source =
      LAYOUT_COLUMNS.standard_6x2[lead] === 0 ? left : right
    const sourceIndex = source.leads.indexOf(lead)
    const targetIndex = LEAD_ORDER.indexOf(lead)
    if (sourceIndex === -1 || targetIndex === -1) continue

    const sourceValues = source.rows.map((row) => row[sourceIndex])
    const values = resamplePanelValues(
      sourceValues,
      0,
      Math.max(0, sourceValues.length - 1),
      panelSamples
    )
    const targetStart =
      LAYOUT_COLUMNS.standard_6x2[lead] === 0 ? 0 : panelSamples
    values.forEach((value, sample) => {
      rows[targetStart + sample][targetIndex] = value
    })
  }

  return {
    leads: [...LEAD_ORDER],
    rows,
  }
}

function combineStackedPanelCanonicals(
  limb: CanonicalCsv,
  precordial: CanonicalCsv
): CanonicalCsv {
  const rows = Array.from(
    { length: SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS },
    () => LEAD_ORDER.map(() => Number.NaN)
  )
  for (const lead of LEAD_ORDER) {
    const source = lead.startsWith("V") ? precordial : limb
    const sourceIndex = source.leads.indexOf(lead)
    const targetIndex = LEAD_ORDER.indexOf(lead)
    if (sourceIndex === -1 || targetIndex === -1) continue
    const sourceValues = source.rows.map((row) => row[sourceIndex])
    const values = resamplePanelValues(
      sourceValues,
      0,
      Math.max(0, sourceValues.length - 1),
      rows.length
    )
    values.forEach((value, sample) => {
      rows[sample][targetIndex] = value
    })
  }
  return { leads: [...LEAD_ORDER], rows }
}

function suppressFullWidthPanelBoundaries(canonical: CanonicalCsv): CanonicalCsv {
  const rows = canonical.rows.map((row) => [...row])
  const startMaskSamples = Math.ceil(
    rows.length * CONSTRAINED_PANEL_START_MASK_FRACTION
  )
  const endMaskSamples = Math.ceil(
    rows.length * CONSTRAINED_PANEL_END_MASK_FRACTION
  )
  for (const lead of LEAD_ORDER) {
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) continue
    for (let sample = 0; sample < startMaskSamples; sample += 1) {
      rows[sample][leadIndex] = Number.NaN
    }
    for (
      let sample = rows.length - endMaskSamples;
      sample < rows.length;
      sample += 1
    ) {
      rows[sample][leadIndex] = Number.NaN
    }
  }
  return { leads: [...canonical.leads], rows }
}

function suppressConstrainedPanelBoundaries(
  canonical: CanonicalCsv
): CanonicalCsv {
  const panelSamples = canonical.rows.length / 2
  const startMaskSamples = Math.ceil(
    panelSamples * CONSTRAINED_PANEL_START_MASK_FRACTION
  )
  const endMaskSamples = Math.ceil(
    panelSamples * CONSTRAINED_PANEL_END_MASK_FRACTION
  )
  const rows = canonical.rows.map((row) => [...row])

  for (const lead of LEAD_ORDER) {
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) continue
    const panelStart =
      LAYOUT_COLUMNS.standard_6x2[lead] === 0 ? 0 : panelSamples
    const panelEnd = panelStart + panelSamples
    for (
      let sample = panelStart;
      sample < panelStart + startMaskSamples;
      sample += 1
    ) {
      rows[sample][leadIndex] = Number.NaN
    }
    for (
      let sample = panelEnd - endMaskSamples;
      sample < panelEnd;
      sample += 1
    ) {
      rows[sample][leadIndex] = Number.NaN
    }
  }

  return {
    leads: [...canonical.leads],
    rows,
  }
}

async function preprocessingRunAssets(
  prepared: PreparedRunInput
): Promise<Partial<Record<RunAssetKey, RunAsset>>> {
  return {
    preparedInput: await assetFromAbsolutePath(
      "Colour/grid preprocessed input",
      prepared.enhancedPath,
      prepared.report.sourceSha256
    ),
    geometryCorrectedInput: await assetFromAbsolutePath(
      "Geometry-corrected input",
      prepared.geometryCorrectedPath,
      prepared.report.sourceSha256
    ),
    artifactPreprocessedInput: await assetFromAbsolutePath(
      "Artifact-specialist input",
      prepared.artifactPreprocessedPath,
      prepared.report.sourceSha256
    ),
    annotationMask: await assetFromAbsolutePath(
      "Detected annotation mask",
      prepared.maskPath,
      prepared.report.sourceSha256
    ),
    provenanceJson: await assetFromAbsolutePath(
      "Digitization provenance",
      prepared.reportPath,
      prepared.report.sourceSha256
    ),
  }
}

async function prepareCandidateInput(
  inputPath: string,
  inputDir: string,
  signal?: AbortSignal,
  upscaleToMaxDimension?: number,
  cropBox?: RasterCropBox
) {
  const extension = path.extname(inputPath).toLowerCase()
  const stem = safeOutputStem(path.parse(inputPath).name)

  await fs.rm(inputDir, { force: true, recursive: true })
  await fs.mkdir(inputDir, { recursive: true })

  if (upscaleToMaxDimension || cropBox) {
    const fileName = `${stem}.png`
    const preparedInputPath = path.join(
      /* turbopackIgnore: true */ inputDir,
      fileName
    )
    await convertImageToPng(
      inputPath,
      preparedInputPath,
      signal,
      upscaleToMaxDimension,
      cropBox
    )
    return { inputDir, stem, inputPath: preparedInputPath }
  }

  if (DIRECT_DIGITIZER_EXTENSIONS.includes(extension)) {
    const fileName = `${stem}${extension}`
    const preparedInputPath = path.join(
      /* turbopackIgnore: true */ inputDir,
      fileName
    )
    await fs.copyFile(inputPath, preparedInputPath)
    return { inputDir, stem, inputPath: preparedInputPath }
  }

  if (CONVERTIBLE_DIGITIZER_EXTENSIONS.includes(extension)) {
    const fileName = `${stem}.png`
    const preparedInputPath = path.join(
      /* turbopackIgnore: true */ inputDir,
      fileName
    )
    await convertImageToPng(inputPath, preparedInputPath, signal)
    return { inputDir, stem, inputPath: preparedInputPath }
  }

  throw new Error(`Unsupported ECG image extension for digitization: ${extension}`)
}

async function neuralFeatureCachePath(
  run: RunRecord,
  inputPath: string,
  resampleSize: number,
  device: DigitizerComputeDevice
) {
  const digest = createHash("sha256")
    .update("open-ecg-segmentation-cache-v1\0")
    .update(await fs.readFile(inputPath))
    .update(`\0${resampleSize}\0${device}`)
    .digest("hex")
  const cacheDirectory = path.join(runDirectory(run.id), "neural-cache")
  await fs.mkdir(cacheDirectory, { recursive: true })
  return path.join(cacheDirectory, `${digest}.pt`)
}

async function pathExists(filePath: string) {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

async function convertImageToPng(
  inputPath: string,
  outputPath: string,
  signal?: AbortSignal,
  upscaleToMaxDimension?: number,
  cropBox?: RasterCropBox
) {
  const script = [
    "from PIL import Image",
    "import sys",
    "with Image.open(sys.argv[1]) as image:",
    "    image = image.convert('RGB')",
    "    target = int(sys.argv[3]) if len(sys.argv) > 3 else 0",
    "    crop = tuple(map(int, sys.argv[4:8])) if len(sys.argv) >= 8 else None",
    "    if crop:",
    "        image = image.crop(crop)",
    "    if target > 0 and max(image.size) < target:",
    "        scale = target / max(image.size)",
    "        size = tuple(max(1, round(value * scale)) for value in image.size)",
    "        image = image.resize(size, Image.Resampling.LANCZOS)",
    "    image.save(sys.argv[2], format='PNG')",
  ].join("\n")

  await execFileAsync(
    openEcgPythonPath(),
    [
      "-c",
      script,
      inputPath,
      outputPath,
      String(upscaleToMaxDimension ?? 0),
      ...(cropBox
        ? [
            String(cropBox.left),
            String(cropBox.top),
            String(cropBox.right),
            String(cropBox.bottom),
          ]
        : []),
    ],
    {
      env: subprocessEnvironment(),
      maxBuffer: 4 * 1024 * 1024,
      signal,
      timeout: 60_000,
    }
  )
}

async function publishSelectedCandidate(
  run: RunRecord,
  selected: CandidateResult,
  candidates: CandidateResult[],
  prepared: PreparedRunInput,
  geometry: LayoutGeometryReport,
  pipelineEvidence: DigitizerPipelineEvidence,
  quantitativeOutput: boolean,
  publicationReasonCode: PublicationReasonCode,
  signal?: AbortSignal
): Promise<PublishedCandidateResult> {
  if (
    !selected.canonical ||
    !selected.canonicalPath ||
    !selected.diagnosticPath ||
    !selected.metadataPath
  ) {
    throw new Error("The selected candidate is missing required output files.")
  }

  const exportsDir = path.join(runDirectory(run.id), "exports")
  await fs.rm(exportsDir, { force: true, recursive: true })
  await fs.mkdir(exportsDir, { recursive: true })

  const stem = path.parse(selected.canonicalPath).name.replace(
    /_timeseries_canonical$/,
    ""
  )
  const diagnosticPath = path.join(exportsDir, `${stem}.png`)
  const canonicalPath = path.join(exportsDir, `${stem}_timeseries_canonical.csv`)
  const metadataPath = path.join(exportsDir, "digitization_metadata.csv")
  const segmentsPath = path.join(exportsDir, `${stem}_segments_500hz.csv`)
  const uncertaintyPath = path.join(exportsDir, `${stem}_uncertainty.csv`)
  const paperRenderPath = path.join(exportsDir, `${stem}_paper_render.png`)

  const publishedCanonical = suppressAnnotatedSamples(
    suppressNearFlatLeads(
      sanitizeCanonicalForLayout(selected.canonical, selected.layout),
      selected.layout
    ),
    prepared.report,
    selected.layout,
    selected.panelTimingCorrections,
    selected.parameters
  )
  const uncertaintyRows = buildUncertaintyRows(
    selected,
    candidates,
    prepared.report,
    publishedCanonical
  )
  updateQaWithUncertainty(selected, uncertaintyRows)
  const reliability = reliabilitySummary(
    selected,
    uncertaintyRows,
    prepared.report,
    publicationReasonCode
  )

  await fs.copyFile(selected.diagnosticPath, diagnosticPath)
  await writePublishedDigitizerMetadata(selected, metadataPath)
  if (quantitativeOutput) {
    await writeCanonicalCsv(publishedCanonical, canonicalPath)
    await writeCompactSegmentsCsv(canonicalPath, segmentsPath, selected.layout)
    await writeUncertaintyCsv(uncertaintyRows, uncertaintyPath)
    await renderPaperFromSegmentsCsv({
      csvPath: segmentsPath,
      outputPath: paperRenderPath,
      uncertaintyPath,
      layout: selected.layout,
      effectiveSampleRateHz: selected.effectiveSampleRateHz,
      paperSpeedMmPerSecond:
        validatedCalibration(geometry)?.paperSpeedMmPerSecond,
      gainMmPerMv: validatedCalibration(geometry)?.gainMmPerMv,
      signal,
    })
  }
  await writeDigitizationProvenance({
    path: prepared.reportPath,
    preprocessing: prepared.report,
    selected,
    candidates,
    reliability,
    calibration: geometry.calibration,
    pipelineEvidence,
  })

  return {
    assets: {
      diagnostic: await assetFromAbsolutePath(
        "Diagnostic overlay",
        diagnosticPath,
        prepared.report.sourceSha256
      ),
      ...(quantitativeOutput
        ? {
            paperRender: await assetFromAbsolutePath(
              "ECG-paper render",
              paperRenderPath,
              prepared.report.sourceSha256
            ),
            canonicalCsv: await assetFromAbsolutePath(
              "Canonical CSV",
              canonicalPath,
              prepared.report.sourceSha256
            ),
            segmentsCsv: await assetFromAbsolutePath(
              "Compact 500 Hz CSV",
              segmentsPath,
              prepared.report.sourceSha256
            ),
            uncertaintyCsv: await assetFromAbsolutePath(
              "Sample-level uncertainty",
              uncertaintyPath,
              prepared.report.sourceSha256
            ),
          }
        : {}),
      metadataCsv: await assetFromAbsolutePath(
        "Digitizer metadata",
        metadataPath,
        prepared.report.sourceSha256
      ),
    },
    reliability,
  }
}

function sanitizeCanonicalForLayout(
  canonical: CanonicalCsv,
  layout?: string
): CanonicalCsv {
  const rows = canonical.rows.map((row) => [...row])
  const expectedLeads = new Set(expectedLayoutLeads(layout))
  const normalizedLayout = normalizeQaLayout(layout)
  const layoutColumns = normalizedLayout
    ? LAYOUT_COLUMNS[normalizedLayout]
    : undefined
  if (!layoutColumns || expectedLeads.size === 0) {
    return { leads: [...canonical.leads], rows }
  }

  const columnCount = Math.max(...Object.values(layoutColumns)) + 1
  const segmentSamples = Math.floor(rows.length / columnCount)
  for (const lead of LEAD_ORDER) {
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) continue
    if (!expectedLeads.has(lead)) {
      for (const row of rows) row[leadIndex] = Number.NaN
      continue
    }
    if (lead === "II" && layout?.toLowerCase().includes("with_r1")) {
      continue
    }
    const column = layoutColumns[lead]
    const start = column * segmentSamples
    const end =
      column === columnCount - 1 ? rows.length : start + segmentSamples
    for (let sample = 0; sample < rows.length; sample += 1) {
      if (sample < start || sample >= end) {
        rows[sample][leadIndex] = Number.NaN
      }
    }
  }
  return { leads: [...canonical.leads], rows }
}

function suppressNearFlatLeads(canonical: CanonicalCsv, layout?: string) {
  const rows = canonical.rows.map((row) => [...row])
  const normalizedLayout = normalizeQaLayout(layout)
  const layoutColumns = normalizedLayout
    ? LAYOUT_COLUMNS[normalizedLayout]
    : undefined
  if (!layoutColumns) return { leads: [...canonical.leads], rows }

  const columnCount = Math.max(...Object.values(layoutColumns)) + 1
  const segmentSamples = Math.floor(rows.length / columnCount)
  for (const lead of expectedLayoutLeads(layout)) {
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) continue
    const fullWidthRhythm =
      lead === "II" && Boolean(layout?.toLowerCase().includes("with_r1"))
    const column = layoutColumns[lead]
    const start = fullWidthRhythm ? 0 : column * segmentSamples
    const end = fullWidthRhythm
      ? rows.length
      : column === columnCount - 1
        ? rows.length
        : start + segmentSamples
    const finiteValues = rows
      .slice(start, end)
      .map((row) => row[leadIndex])
      .filter(Number.isFinite)
      .sort((a, b) => a - b)
    if (finiteValues.length < MIN_SCORABLE_LEAD_SAMPLES) continue
    const robustAmplitudeUv =
      percentile(finiteValues, 0.99) - percentile(finiteValues, 0.01)
    if (robustAmplitudeUv >= MIN_ROBUST_LEAD_AMPLITUDE_UV) continue
    for (const row of rows) row[leadIndex] = Number.NaN
  }
  return { leads: [...canonical.leads], rows }
}

async function renderPaperFromSegmentsCsv({
  csvPath,
  outputPath,
  uncertaintyPath,
  layout,
  effectiveSampleRateHz,
  paperSpeedMmPerSecond,
  gainMmPerMv,
  signal,
}: {
  csvPath: string
  outputPath: string
  uncertaintyPath?: string
  layout?: string
  effectiveSampleRateHz?: number
  paperSpeedMmPerSecond?: number
  gainMmPerMv?: number
  signal?: AbortSignal
}) {
  await execFileAsync(
    openEcgPythonPath(),
    [
      PAPER_RENDERER_PATH,
      "--csv",
      csvPath,
      "--output",
      outputPath,
      ...(uncertaintyPath
        ? [
            "--uncertainty",
            uncertaintyPath,
          ]
        : []),
      "--layout",
      layout ?? "standard_6x2",
      "--title",
      "",
      ...(effectiveSampleRateHz
        ? [
            "--effective-sample-rate",
            String(effectiveSampleRateHz),
          ]
        : []),
      ...(paperSpeedMmPerSecond
        ? ["--paper-speed", String(paperSpeedMmPerSecond)]
        : []),
      ...(gainMmPerMv ? ["--gain", String(gainMmPerMv)] : []),
    ],
    {
      cwd: RESOURCE_ROOT,
      env: subprocessEnvironment(),
      maxBuffer: 4 * 1024 * 1024,
      signal,
      timeout: 60_000,
    }
  )
}

function applyCandidateDisagreementScores(candidates: CandidateResult[]) {
  for (const candidate of candidates) {
    if (!candidate.canonical || !candidate.scoreBreakdown) continue

    const disagreements: number[] = []
    for (const peer of candidates) {
      if (
        peer.id === candidate.id ||
        !peer.canonical ||
        candidateCropKey(peer) !== candidateCropKey(candidate) ||
        normalizeQaLayout(peer.layout) !== normalizeQaLayout(candidate.layout)
      ) {
        continue
      }

      for (const lead of LEAD_ORDER) {
        const rmseUv = candidateLeadAlignmentRmse(candidate, peer, lead)
        if (Number.isFinite(rmseUv)) {
          disagreements.push(rmseUv)
        }
      }
    }

    const typicalDisagreementUv = median(disagreements)
    const penalty = Number.isFinite(typicalDisagreementUv)
      ? Math.round(typicalDisagreementUv * 25)
      : 0
    candidate.scoreBreakdown.candidateDisagreementPenalty = penalty
    candidate.score = Object.values(candidate.scoreBreakdown).reduce(
      (total, value) => total + value,
      0
    )
    if (candidate.qa) {
      candidate.qa.score = candidate.score
      candidate.qa.scoreBreakdown = candidate.scoreBreakdown
    }
  }
}

async function buildPeerGapRepairCandidate(
  run: RunRecord,
  candidates: CandidateResult[],
  preprocessing: PreprocessingReport
): Promise<CandidateResult | undefined> {
  const compatible = largestCompatibleCandidateGroup(
    candidates.filter((candidate) => !isDerivedCandidate(candidate))
  )
  const reference = compatible
    .filter(
      (candidate) =>
        candidate.canonical &&
        candidate.canonicalPath &&
        candidate.diagnosticPath &&
        candidate.metadataPath &&
        isConservativeCandidate(candidate) &&
        Boolean(normalizeSupportedEcgLayout(candidate.layout)) &&
        LEAD_ORDER.every((lead) =>
          candidateHasSemanticLeadIdentity(candidate, compatible, lead)
        ) &&
        LEAD_ORDER.every((lead) => candidateHasScorableLead(candidate, lead))
    )
    .sort(
      (a, b) =>
        candidateMissingSamples(a) - candidateMissingSamples(b) ||
        candidateErrorCount(a) - candidateErrorCount(b) ||
        a.score - b.score
    )[0]
  if (
    !reference?.canonical ||
    !reference.canonicalPath ||
    !reference.diagnosticPath ||
    !reference.metadataPath ||
    !reference.layout
  ) {
    return undefined
  }

  const canonical: CanonicalCsv = {
    leads: [...reference.canonical.leads],
    rows: reference.canonical.rows.map((row) => [...row]),
  }
  const leadSources: Record<string, string> = {}
  let repairedSamples = 0

  for (const lead of LEAD_ORDER) {
    const referenceSegment = canonicalLeadSegment(
      reference.canonical,
      reference.layout,
      lead
    )
    const outputSegment = canonicalLeadSegment(canonical, reference.layout, lead)
    const leadIndex = canonical.leads.indexOf(lead)
    if (!referenceSegment || !outputSegment || leadIndex === -1) continue

    const alignedPeers = compatible.flatMap((peer) => {
      if (
        peer.id === reference.id ||
        isDerivedCandidate(peer) ||
        !candidateHasQuantitativeTimingEvidence(peer) ||
        !peer.canonical ||
        !isConservativeCandidate(peer) ||
        normalizeQaLayout(peer.layout) !== normalizeQaLayout(reference.layout) ||
        !candidateHasScorableLead(peer, lead) ||
        !candidateHasSemanticLeadIdentity(peer, compatible, lead)
      ) {
        return []
      }
      const peerSegment = canonicalLeadSegment(peer.canonical, peer.layout, lead)
      if (!peerSegment) return []
      const alignment = alignSeries(referenceSegment.values, peerSegment.values)
      if (
        !Number.isFinite(alignment.rmseUv) ||
        alignment.rmseUv > DISAGREEMENT_THRESHOLD_UV
      ) {
        return []
      }
      return [
        {
          id: peer.id,
          values: alignment.values,
          sourceVerified: peer.sourceFidelity?.passed === true,
        },
      ]
    })
    const repaired = repairShortPeerSupportedGaps(
      referenceSegment.values,
      alignedPeers
    )
    repaired.values.forEach((value, index) => {
      canonical.rows[outputSegment.start + index][leadIndex] = value
    })
    repairedSamples += repaired.repairedSamples
    leadSources[lead] =
      repaired.supportingCandidateIds.length > 0
        ? `${reference.id}+${repaired.supportingCandidateIds.join("+")}`
        : reference.id
  }

  if (repairedSamples === 0) return undefined
  const qa = evaluateQa(canonical, reference.layout)
  const config: DigitizerCandidateConfig = {
    kind: "peer-gap-repair",
    id: PEER_GAP_REPAIR_CANDIDATE_ID,
    label: "Peer-supported short-gap repair",
    resampleSize: reference.parameters.resampleSize,
    inputVariant: reference.parameters.inputVariant ?? "original",
    vectorizer: reference.parameters.vectorizer ?? "probability-centroid",
    device: reference.parameters.device ?? "cpu",
    ...(reference.parameters.cropBox
      ? { cropBox: reference.parameters.cropBox }
      : {}),
    planningPhase: "recovery",
    scheduleReason: "derived short-gap repair supported by agreeing peers",
  }
  const scoreBreakdown = scoreCandidate(
    qa,
    reference.layout,
    reference.layoutCost,
    config,
    preprocessing
  )
  const score = Object.values(scoreBreakdown).reduce(
    (total, value) => total + value,
    0
  )
  qa.score = score
  qa.scoreBreakdown = scoreBreakdown

  const candidateDir = path.join(
    runDirectory(run.id),
    "candidates",
    PEER_GAP_REPAIR_CANDIDATE_ID
  )
  const canonicalPath = path.join(
    candidateDir,
    "peer_gap_repair_timeseries_canonical.csv"
  )
  const diagnosticPath = path.join(candidateDir, "peer_gap_repair.png")
  const metadataPath = path.join(candidateDir, "digitization_metadata.csv")
  await fs.rm(candidateDir, { force: true, recursive: true })
  await fs.mkdir(candidateDir, { recursive: true })
  await writeCanonicalCsv(canonical, canonicalPath)
  await fs.copyFile(reference.diagnosticPath, diagnosticPath)
  await fs.copyFile(reference.metadataPath, metadataPath)

  return {
    id: PEER_GAP_REPAIR_CANDIDATE_ID,
    label: config.label,
    localPath: relativePath(candidateDir),
    leadSources,
    status: "completed",
    parameters: parametersForCandidate(config),
    layout: reference.layout,
    layoutCost: reference.layoutCost,
    effectiveSampleRateHz: reference.effectiveSampleRateHz,
    panelTimingCorrections: reference.panelTimingCorrections,
    runtimeMs: 0,
    qa,
    canonicalPath,
    diagnosticPath,
    metadataPath,
    canonical,
    score,
    scoreBreakdown,
  }
}

function repairShortPeerSupportedGaps(
  reference: number[],
  peers: Array<{
    id: string
    values: number[]
    sourceVerified: boolean
  }>
) {
  const values = [...reference]
  const supportingCandidateIds = new Set<string>()
  let repairedSamples = 0
  const gaps = contiguousGaps(reference.map(Number.isFinite)).filter(
    (gap) =>
      gap.start > 0 &&
      gap.end < reference.length &&
      gap.length <= MAX_PEER_GAP_REPAIR_SAMPLES
  )

  for (const gap of gaps) {
    const supporters = peers.filter((peer) => {
      const supportsGap = peer.values
        .slice(gap.start, gap.end)
        .every(Number.isFinite)
      if (!supportsGap) return false
      const left = peer.values[gap.start - 1]
      const right = peer.values[gap.end]
      return (
        Number.isFinite(left) &&
        Number.isFinite(right) &&
        Math.abs(left - reference[gap.start - 1]) <=
          DISAGREEMENT_THRESHOLD_UV &&
        Math.abs(right - reference[gap.end]) <= DISAGREEMENT_THRESHOLD_UV
      )
    })
    if (
      !supporters.some((peer) => peer.sourceVerified) &&
      supporters.length < 2
    ) {
      continue
    }

    const replacements: number[] = []
    let supported = true
    for (let index = gap.start; index < gap.end; index += 1) {
      const sampleValues = supporters
        .map((peer) => peer.values[index])
        .filter(Number.isFinite)
      const minimum = Math.min(...sampleValues)
      const maximum = Math.max(...sampleValues)
      if (
        sampleValues.length === 0 ||
        maximum - minimum > DISAGREEMENT_THRESHOLD_UV
      ) {
        supported = false
        break
      }
      replacements.push(median(sampleValues))
    }
    if (!supported) continue
    replacements.forEach((value, offset) => {
      values[gap.start + offset] = value
    })
    repairedSamples += gap.length
    supporters.forEach((peer) => supportingCandidateIds.add(peer.id))
  }

  return {
    values,
    repairedSamples,
    supportingCandidateIds: [...supportingCandidateIds].sort(),
  }
}

function isDerivedCandidate(candidate: CandidateResult) {
  return candidateCapabilities(candidate.parameters).derived
}

async function buildLeadFusionCandidate(
  run: RunRecord,
  candidates: CandidateResult[],
  preprocessing: PreprocessingReport
): Promise<CandidateResult | undefined> {
  const compatible = largestCompatibleCandidateGroup(candidates)
  if (compatible.length === 0) return undefined

  const reference = [...compatible].sort((a, b) => {
    const scorableDelta =
      candidateScorableLeadCount(b) - candidateScorableLeadCount(a)
    return scorableDelta || a.score - b.score
  })[0]
  if (
    !reference.canonical ||
    !reference.layout ||
    !reference.diagnosticPath ||
    !reference.metadataPath
  ) {
    return undefined
  }

  const leadSources: Record<string, CandidateResult> = {}
  const leadCorroborators: Record<string, string[]> = {}
  const fullWidthRhythmLeads = new Set<string>()
  for (const lead of LEAD_ORDER) {
    const trustedCandidates = compatible.filter((candidate) =>
      candidateHasTrustedLead(candidate, compatible, lead)
    )
    const fullWidthRhythmCandidates = trustedCandidates.filter((candidate) =>
      candidateHasCorroboratedFullWidthRhythmLead(
        candidate,
        compatible,
        lead
      )
    )
    const source = selectLeadSourceCandidate(
      fullWidthRhythmCandidates.length > 0
        ? fullWidthRhythmCandidates
        : trustedCandidates,
      preprocessing,
      reference.layout,
      lead
    )
    if (!source) return undefined
    leadSources[lead] = source
    if (fullWidthRhythmCandidates.some((candidate) => candidate.id === source.id)) {
      fullWidthRhythmLeads.add(lead)
    }
    leadCorroborators[lead] = trustedCandidates
      .map((candidate) => candidate.id)
      .sort()
  }

  const canonical = fuseLeadCandidates(
    reference.canonical,
    reference.layout,
    leadSources,
    fullWidthRhythmLeads
  )
  const qa = evaluateQa(canonical, reference.layout)
  const sourceCandidates = [...new Set(Object.values(leadSources))]
  const device =
    sourceCandidates.find((candidate) => candidate.parameters.device)?.parameters
      .device ?? "cpu"
  const sourceInputVariants = [
    ...new Set(
      sourceCandidates.map(
        (candidate) => candidate.parameters.inputVariant ?? "original"
      )
    ),
  ]
  const inputVariant =
    sourceInputVariants.length === 1
      ? sourceInputVariants[0]
      : sourceInputVariants.every(isAnnotationSafeInputVariant)
        ? "annotation-masked"
        : "original"
  const vectorizer = sourceCandidates.every(
    (candidate) => candidate.parameters.vectorizer === "dynamic-path"
  )
    ? "dynamic-path"
    : "probability-centroid"
  const resampleSize = Math.max(
    ...sourceCandidates.map((candidate) => candidate.parameters.resampleSize)
  )
  const fusionConfig: DigitizerCandidateConfig = {
    kind: "lead-fusion",
    id: LEAD_FUSION_CANDIDATE_ID,
    label: "Per-lead consensus extraction",
    resampleSize,
    inputVariant,
    vectorizer,
    device,
    ...(reference.parameters.cropBox
      ? { cropBox: reference.parameters.cropBox }
      : {}),
    planningPhase: "recovery",
    scheduleReason: "per-lead fusion of independently trusted candidates",
  }
  const scoreBreakdown = scoreCandidate(
    qa,
    reference.layout,
    reference.layoutCost,
    fusionConfig,
    preprocessing
  )
  const score = Object.values(scoreBreakdown).reduce(
    (total, value) => total + value,
    0
  )
  qa.score = score
  qa.scoreBreakdown = scoreBreakdown

  const candidateDir = path.join(
    runDirectory(run.id),
    "candidates",
    LEAD_FUSION_CANDIDATE_ID
  )
  const canonicalPath = path.join(
    candidateDir,
    "lead_fusion_timeseries_canonical.csv"
  )
  const diagnosticPath = path.join(candidateDir, "lead_fusion.png")
  const metadataPath = path.join(candidateDir, "digitization_metadata.csv")
  await fs.rm(candidateDir, { force: true, recursive: true })
  await fs.mkdir(candidateDir, { recursive: true })
  await writeCanonicalCsv(canonical, canonicalPath)
  await fs.copyFile(reference.diagnosticPath, diagnosticPath)
  await fs.copyFile(reference.metadataPath, metadataPath)

  const fusedResult: CandidateResult = {
    id: LEAD_FUSION_CANDIDATE_ID,
    label: fusionConfig.label,
    localPath: relativePath(candidateDir),
    leadSources: Object.fromEntries(
      LEAD_ORDER.map((lead) => [lead, leadSources[lead].id])
    ),
    leadCorroborators,
    status: "completed",
    parameters: parametersForCandidate(fusionConfig),
    layout: reference.layout,
    layoutCost: reference.layoutCost,
    effectiveSampleRateHz: Math.min(
      ...sourceCandidates.map(
        (candidate) => candidate.effectiveSampleRateHz ?? SAMPLE_RATE_HZ
      )
    ),
    panelTimingCorrections: reference.panelTimingCorrections,
    runtimeMs: 0,
    qa,
    canonicalPath,
    diagnosticPath,
    metadataPath,
    canonical,
    score,
    scoreBreakdown,
  }
  const disagreementPenalty = fusionDisagreementPenalty(
    buildUncertaintyRows(
      fusedResult,
      [...candidates, fusedResult],
      preprocessing
    )
  )
  scoreBreakdown.candidateDisagreementPenalty += disagreementPenalty
  fusedResult.score = Object.values(scoreBreakdown).reduce(
    (total, value) => total + value,
    0
  )
  qa.score = fusedResult.score

  return fusedResult
}

function fusionDisagreementPenalty(
  rows: Array<Pick<UncertaintyRow, "status">>
) {
  const disagreementSamples = rows.filter((row) =>
    row.status.includes("disagreement")
  ).length
  return (
    disagreementSamples *
    DIGITIZER_POLICY.fusionDisagreementPenaltyPerSample
  )
}

function largestCompatibleCandidateGroup(candidates: CandidateResult[]) {
  const groups = new Map<string, CandidateResult[]>()
  for (const candidate of candidates) {
    if (!candidate.canonical) continue
    const layout = normalizeQaLayout(candidate.layout)
    if (!layout) continue
    const key = `${layout}:${candidate.canonical.rows.length}:${candidateCropKey(candidate)}`
    const group = groups.get(key) ?? []
    group.push(candidate)
    groups.set(key, group)
  }
  return [...groups.values()].sort((a, b) => {
    const leadDelta =
      Math.max(...b.map(candidateScorableLeadCount)) -
      Math.max(...a.map(candidateScorableLeadCount))
    const layoutCostDelta =
      minimumFiniteLayoutCost(a) - minimumFiniteLayoutCost(b)
    return leadDelta || layoutCostDelta || b.length - a.length
  })[0] ?? []
}

function minimumFiniteLayoutCost(candidates: CandidateResult[]) {
  const costs = candidates
    .map((candidate) => candidate.layoutCost)
    .filter((cost): cost is number => Number.isFinite(cost))
  return costs.length > 0 ? Math.min(...costs) : Number.POSITIVE_INFINITY
}

function candidateScorableLeadCount(candidate: CandidateResult) {
  return LEAD_ORDER.filter((lead) => candidateHasScorableLead(candidate, lead))
    .length
}

function candidateHasScorableLead(candidate: CandidateResult, lead: string) {
  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = leadQa?.expectedSamples ?? 0
  const hasNearFlatError = (candidate.qa?.warnings ?? []).some(
    (warning) =>
      warning.lead === lead &&
      warning.severity === "error" &&
      warning.message.includes("near-flat")
  )
  return (
    expectedSamples > 0 &&
    (leadQa?.finiteSamples ?? 0) >=
      minimumScorableLeadSamples(expectedSamples) &&
    !hasNearFlatError
  )
}

function candidateHasStructurallyCompleteLead(
  candidate: CandidateResult,
  lead: string
) {
  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = leadQa?.expectedSamples ?? 0
  const minimumSamples = Math.min(
    expectedSamples,
    Math.max(
      MIN_SCORABLE_LEAD_SAMPLES,
      Math.ceil(expectedSamples * MIN_PUBLISHABLE_LEAD_COVERAGE)
    )
  )
  const hasBlockingLeadError = (candidate.qa?.warnings ?? []).some(
    (warning) => warning.lead === lead && warning.severity === "error"
  )
  return (
    expectedSamples > 0 &&
    (leadQa?.finiteSamples ?? 0) >= minimumSamples &&
    !hasBlockingLeadError
  )
}

function candidateHasSourceVerifiedBoundaryLead(
  candidate: CandidateResult,
  lead: string
) {
  if (
    !candidate.canonical ||
    candidate.sourceFidelity?.passed !== true ||
    !candidateHasValidatedBoundaryLayout(candidate)
  ) {
    return false
  }

  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = leadQa?.expectedSamples ?? 0
  const minimumSamples = Math.min(
    expectedSamples,
    Math.max(
      MIN_SCORABLE_LEAD_SAMPLES,
      Math.ceil(expectedSamples * MIN_SOURCE_VERIFIED_BOUNDARY_COVERAGE)
    )
  )
  const segment = canonicalLeadSegment(
    candidate.canonical,
    candidate.layout,
    lead
  )
  const hasBlockingLeadError = (candidate.qa?.warnings ?? []).some(
    (warning) => warning.lead === lead && warning.severity === "error"
  )
  const fidelityLead = candidate.sourceFidelity.leadMetrics[lead]
  if (
    !segment ||
    expectedSamples <= 0 ||
    (leadQa?.finiteSamples ?? 0) < minimumSamples ||
    (leadQa?.outsideSegmentSamples ?? 0) > 0 ||
    hasBlockingLeadError ||
    candidate.sourceFidelity.minimumCoverage <
      MIN_SOURCE_VERIFIED_BOUNDARY_COVERAGE ||
    (candidate.sourceFidelity.unsafeExcursionCount ?? 0) > 0 ||
    (fidelityLead?.unsafeExcursionCount ?? 0) > 0
  ) {
    return false
  }

  const finite = segment.values.map(Number.isFinite)
  const firstFinite = finite.indexOf(true)
  const lastFinite = finite.lastIndexOf(true)
  return (
    firstFinite >= 0 &&
    finite
      .slice(firstFinite, lastFinite + 1)
      .every((sampleIsFinite) => sampleIsFinite)
  )
}

function candidateHasValidatedBoundaryLayout(candidate: CandidateResult) {
  const fidelity = candidate.sourceFidelity
  if (!fidelity) return false

  if (fidelity.method === "row-local-compound-label-and-algebra-v1") {
    return (
      fidelity.leadOrderValidation?.passed === true &&
      fidelity.precordialLabelValidation?.passed === true
    )
  }
  if (
    fidelity.method === "row-local-twelve-lead-unverified-v2" ||
    fidelity.method === "row-local-labeled-six-by-two-algebra-validated-v1"
  ) {
    return (
      fidelity.leadOrderValidation?.passed === true &&
      (fidelity.method !==
        "row-local-labeled-six-by-two-algebra-validated-v1" ||
        fidelity.precordialLabelValidation?.passed === true)
    )
  }
  if (
    fidelity.method === "row-local-sequential-label-validated-v2" ||
    fidelity.method === "row-local-sequential-label-validated-v3"
  ) {
    return (
      fidelity.leadOrderValidation?.passed === true &&
      fidelity.leadLabelValidation?.passed === true
    )
  }
  if (fidelity.method === "row-local-calibration-anchored-six-by-two-v1") {
    return fidelity.layoutConfidence >= 0.8
  }
  if (
    fidelity.method ===
    "rhythm-anchored-label-validated-three-by-four-v1"
  ) {
    return Boolean(
      fidelity.leadLabelValidation?.passed === true &&
        fidelity.leadLabelValidation.order === "standard" &&
        (fidelity.rhythmAnchorCount ?? 0) >= 3 &&
        fidelity.rhythmTracePassed === true &&
        fidelity.layoutConfidence >= 0.35
    )
  }

  return Boolean(
    candidate.layout?.includes("with_r1") &&
      (fidelity.rhythmAnchorCount ?? 0) >= 3 &&
      fidelity.rhythmTracePassed === true &&
      fidelity.layoutConfidence >= 0.75
  )
}

function candidateHasPublishableLead(
  candidate: CandidateResult,
  lead: string
) {
  return (
    candidateHasStructurallyCompleteLead(candidate, lead) ||
    candidateHasSourceVerifiedBoundaryLead(candidate, lead)
  )
}

function candidateHasExplicitSemanticLayoutEvidence(
  candidate: CandidateResult
) {
  const layout = normalizeQaLayout(candidate.layout)
  if (!layout || expectedLayoutLeads(layout).length === 0) {
    return false
  }
  const expectedOrder = layout.startsWith("cabrera")
    ? "cabrera"
    : "standard"
  const proofLayout = normalizeQaLayout(
    candidate.parameters.semanticLeadIdentityLayout
  )
  if (
    candidate.parameters.semanticLeadIdentityConfirmed === true &&
    candidate.parameters.semanticLeadIdentityOrder === expectedOrder &&
    Boolean(candidate.parameters.semanticLeadIdentityMethod) &&
    (!proofLayout || proofLayout === layout)
  ) {
    return true
  }
  // This function represents explicit printed-label identity only. Bounded
  // layout-inferred identity is evaluated separately so provenance and the UI
  // cannot mistake a provisional assignment for OCR confirmation.
  const fidelity = candidate.sourceFidelity
  if (!fidelity) return false
  if (layout === "standard_6x1_limb" || layout === "cabrera_6x1_limb") {
    return Boolean(
      fidelity.leadOrderValidation?.passed === true &&
        fidelity.leadOrderValidation.selectedOrder === expectedOrder
    )
  }
  return false
}

function candidateHasProvisionalLayoutLeadIdentity(
  candidate: CandidateResult
) {
  const layout = normalizeQaLayout(candidate.layout)
  const constrainedLayout = normalizeQaLayout(
    candidate.parameters.layoutConstraint
  )
  const fidelity = candidate.sourceFidelity
  return Boolean(
    layout &&
      !candidateHasExplicitSemanticLayoutEvidence(candidate) &&
      expectedLayoutLeads(layout).length === LEAD_ORDER.length &&
      constrainedLayout === layout &&
      candidate.parameters.vectorizer === "native-grid-path" &&
      candidate.parameters.geometryConfirmedLayout === true &&
      (candidate.parameters.geometryLayoutConfidence ?? 0) >=
        MIN_REVIEW_LAYOUT_CONFIDENCE &&
      fidelity &&
      fidelity.layoutConfidence >= MIN_REVIEW_LAYOUT_CONFIDENCE
  )
}

function candidateHasSemanticLeadIdentity(
  candidate: CandidateResult,
  peers: CandidateResult[],
  lead: string
) {
  if (!expectedLayoutLeads(candidate.layout).includes(lead)) return false
  if (candidateHasExplicitSemanticLayoutEvidence(candidate)) return true
  if (candidateHasProvisionalLayoutLeadIdentity(candidate)) return true
  if (candidateInheritsProvisionalGeometryLeadIdentity(candidate, peers)) {
    return true
  }
  if (!candidate.canonical) return false
  return peers.some((anchor) => {
    if (
      anchor.id === candidate.id ||
      !anchor.canonical ||
      !candidateHasExplicitSemanticLayoutEvidence(anchor) ||
      candidateCropKey(anchor) !== candidateCropKey(candidate) ||
      normalizeQaLayout(anchor.layout) !== normalizeQaLayout(candidate.layout)
    ) {
      return false
    }
    return (
      candidateLeadAlignmentRmse(candidate, anchor, lead) <=
      DISAGREEMENT_THRESHOLD_UV
    )
  })
}

function candidateInheritsProvisionalGeometryLeadIdentity(
  candidate: CandidateResult,
  peers: CandidateResult[]
) {
  if (
    !candidate.canonical ||
    (candidate.parameters.inputVariant !== "geometry-corrected" &&
      candidate.parameters.inputVariant !== "artifact-preprocessed" &&
      candidate.parameters.inputVariant !== "preprocessed")
  ) {
    return false
  }
  return peers.some(
    (anchor) =>
      anchor.id !== candidate.id &&
      anchor.parameters.inputVariant === "geometry-corrected" &&
      candidateHasProvisionalLayoutLeadIdentity(anchor) &&
      candidateCropKey(anchor) === candidateCropKey(candidate) &&
      normalizeQaLayout(anchor.layout) === normalizeQaLayout(candidate.layout)
  )
}

function candidateHasTrustedLead(
  candidate: CandidateResult,
  peers: CandidateResult[],
  lead: string
) {
  const candidateCapabilitiesSummary = candidateCapabilities(
    candidate.parameters
  )
  if (
    !candidate.canonical ||
    !candidateHasQuantitativeTimingEvidence(candidate) ||
    (candidateCapabilitiesSummary.nativeGrid &&
      candidate.sourceFidelity?.passed !== true) ||
    !candidateHasPublishableLead(candidate, lead) ||
    !candidateHasSemanticLeadIdentity(candidate, peers, lead)
  ) {
    return false
  }
  const reference = canonicalLeadSegment(
    candidate.canonical,
    candidate.layout,
    lead
  )
  if (!reference) return false
  const candidateIsPermissive =
    typeof candidate.parameters.labelThresh === "number" &&
    candidate.parameters.labelThresh < 0.1

  return peers.some((peer) => {
    const peerCapabilitiesSummary = candidateCapabilities(peer.parameters)
    const peerIsPermissive =
      typeof peer.parameters.labelThresh === "number" &&
      peer.parameters.labelThresh < 0.1
    if (
      peer.id === candidate.id ||
      isDerivedCandidate(peer) ||
      !candidateHasQuantitativeTimingEvidence(peer) ||
      !candidatePathsAreIndependent(candidate, peer) ||
      (peerCapabilitiesSummary.nativeGrid &&
        peer.sourceFidelity?.passed !== true) ||
      !peer.canonical ||
      candidateCropKey(peer) !== candidateCropKey(candidate) ||
      normalizeQaLayout(peer.layout) !== normalizeQaLayout(candidate.layout) ||
      (candidateIsPermissive && peerIsPermissive) ||
      !candidateHasStructurallyCompleteLead(peer, lead) ||
      !candidateHasSemanticLeadIdentity(peer, peers, lead)
    ) {
      return false
    }
    return (
      candidateLeadAlignmentRmse(candidate, peer, lead) <=
      DISAGREEMENT_THRESHOLD_UV
    )
  })
}

function candidateHasQuantitativeTimingEvidence(candidate: CandidateResult) {
  return (
    candidate.sourceFidelity?.sourceTimingInference
      ?.quantitativeCalibrationConfirmed !== false
  )
}

function candidatePathsAreIndependent(
  candidate: CandidateResult,
  peer: CandidateResult
) {
  const candidateCapabilitiesSummary = candidateCapabilities(
    candidate.parameters
  )
  const peerCapabilitiesSummary = candidateCapabilities(peer.parameters)
  // Transforming an image or changing the vectorizer does not make two paths
  // epistemically independent when both still rely on the same neural signal
  // model. Quantitative publication requires agreement between a deterministic
  // native-grid trace and a neural extraction; neural-only agreement remains
  // useful for diagnostics and uncertainty, but cannot establish trust.
  return (
    candidateCapabilitiesSummary.nativeGrid !==
    peerCapabilitiesSummary.nativeGrid
  )
}

function candidateHasAllTrustedLeads(
  candidate: CandidateResult,
  peers: CandidateResult[]
) {
  return LEAD_ORDER.every((lead) =>
    candidateHasTrustedLead(candidate, peers, lead)
  )
}

function hasIndependentCompleteAgreement(candidates: CandidateResult[]) {
  const completed = candidates.filter(
    (candidate) =>
      candidate.status === "completed" &&
      candidateEligibleForSelection(candidate) &&
      !isDerivedCandidate(candidate)
  )
  return completed.some((candidate) =>
    candidateHasAllTrustedLeads(candidate, completed)
  )
}

function notRequiredStabilityEvidence(): DigitizerStabilityEvidence {
  return {
    required: false,
    reasons: [],
    outcome: "not-required",
    repeatCandidateIds: [],
    cpuCandidateIds: [],
    repeatAgreementLeadCount: LEAD_ORDER.length,
    confirmedLeadCount: LEAD_ORDER.length,
  }
}

function buildPipelineEvidence({
  preprocessing,
  geometry,
  planning,
  primaryPlan,
  candidates,
  selectionCandidates,
  nativeGridCandidates,
  nativeGridStructurallyComplete,
  coreAgreementReached,
  recoveryExpanded,
  selected,
  reasonCode,
  stability,
}: {
  preprocessing: PreprocessingReport
  geometry: LayoutGeometryReport
  planning: CandidatePlanningContext
  primaryPlan: DigitizerCandidateConfig[]
  candidates: CandidateResult[]
  selectionCandidates: CandidateResult[]
  nativeGridCandidates: CandidateResult[]
  nativeGridStructurallyComplete: boolean
  coreAgreementReached: boolean
  recoveryExpanded: boolean
  selected?: CandidateResult
  reasonCode?: PublicationReasonCode
  stability: DigitizerStabilityEvidence
}): DigitizerPipelineEvidence {
  const executedIds = new Set(candidates.map((candidate) => candidate.id))
  const escalationReasons = [
    "complete publication requires independent morphology corroboration",
  ]
  if (nativeGridCandidates.length === 0) {
    escalationReasons.push("no deterministic native candidate was available")
  } else if (!nativeGridStructurallyComplete) {
    escalationReasons.push(
      "deterministic native candidates were not structurally publishable"
    )
  }
  if (planning.artifactPreprocessingEligible) {
    escalationReasons.push("artifact-specific deterministic preprocessing was eligible")
  }
  if (planning.geometryCorrectionApplied) {
    escalationReasons.push("page geometry correction was applied")
  }

  const rankedAlternatives = selected
    ? selectionCandidates
        .filter(
          (candidate) =>
            candidate.id !== selected.id && Number.isFinite(candidate.score)
        )
        .sort(
          (a, b) =>
            candidatePublicationRank(a) - candidatePublicationRank(b) ||
            a.id.localeCompare(b.id)
        )
    : []
  const strongestAlternative = rankedAlternatives[0]
  const neuralCandidates = candidates.filter(
    (candidate) =>
      candidate.parameters.vectorizer !== "native-grid-path" &&
      candidate.featureCacheHit !== undefined
  )
  const selectedCalibration = selected
    ? selectorCalibrationEvidence(
        selected.layout,
        {
          id: selected.id,
          kind: selected.parameters.kind,
          capabilities: selected.parameters.capabilities,
          inputVariant: selected.parameters.inputVariant ?? "original",
          vectorizer:
            selected.parameters.vectorizer ?? "probability-centroid",
        },
        preprocessing
      )
    : undefined

  return {
    version: 1,
    selectorCalibrationProfileId: SELECTOR_CALIBRATION_PROFILE_ID,
    neuralEscalated: primaryPlan.some(
      (candidate) =>
        candidatePlanningPhase(candidate) === "core" &&
        executedIds.has(candidate.id)
    ),
    escalationReasons,
    coreAgreementReached,
    recoveryExpanded,
    candidatePlan: primaryPlan.map((candidate) => ({
      candidateId: candidate.id,
      phase: candidatePlanningPhase(candidate),
      reason: candidateScheduleReason(candidate, planning),
      executed: executedIds.has(candidate.id),
    })),
    ...(selected && reasonCode
      ? {
          selection: {
            selectedCandidateId: selected.id,
            ...(strongestAlternative
              ? { strongestAlternativeId: strongestAlternative.id }
              : {}),
            selectedSource: isDerivedCandidate(selected)
              ? ("fused" as const)
              : selected.parameters.vectorizer === "native-grid-path"
                ? ("native" as const)
                : ("neural" as const),
            reasonCode,
            calibration: {
              source: selectedCalibration?.source ?? "unavailable",
              ...(selectedCalibration?.artifactContext
                ? { artifactContext: selectedCalibration.artifactContext }
                : {}),
              confidence:
                selectedCalibration?.calibrationConfidence ?? "uncalibrated",
              trainingCaseCount:
                selectedCalibration?.trainingCaseCount ?? 0,
              trainingPipelineCommit:
                selectedCalibration?.trainingPipelineCommit ?? "unknown",
              calibrationContractVersion:
                selectedCalibration?.calibrationContractVersion ?? 0,
              requiredCalibrationContractVersion:
                selectedCalibration?.requiredCalibrationContractVersion ?? 0,
              priorsApplied: selectedCalibration?.priorsApplied ?? false,
              heldoutUsed: selectedCalibration?.heldoutUsed ?? false,
              exactCandidateCalibrated:
                selectedCalibration?.exactCandidateCalibrated ?? false,
              outOfDomainReasons:
                selectedCalibration?.outOfDomainReasons ?? [
                  "selector-calibration-unavailable",
                ],
            },
            semanticLeadIdentity: {
              passed: expectedLayoutLeads(selected.layout).every((lead) =>
                candidateHasSemanticLeadIdentity(
                  selected,
                  selectionCandidates,
                  lead
                )
              ),
              explicitlyAnchored:
                candidateHasExplicitSemanticLayoutEvidence(selected),
              inferredFromLayout:
                candidateHasProvisionalLayoutLeadIdentity(selected),
              requiresManualVerification:
                candidateHasProvisionalLayoutLeadIdentity(selected),
              expectedLeads: expectedLayoutLeads(selected.layout),
              sourceCandidateIds: selected.leadCorroborators
                ? [
                    ...new Set(
                      Object.values(selected.leadCorroborators).flat()
                    ),
                  ].sort()
                : selected.leadSources
                ? [
                    ...new Set(
                      Object.values(selected.leadSources).flatMap((source) =>
                        source.split("+")
                      )
                    ),
                  ].sort()
                : [selected.id],
            },
            quantitativeOutputEligible: [
              "source_verified",
              "adaptive_preprocessed",
              "lead_fusion",
              "peer_trusted",
              "reviewable_constrained",
              "unstable_neural_confirmation",
            ].includes(reasonCode),
            ...(strongestAlternative && Number.isFinite(selected.score)
              ? {
                  scoreGap: Math.abs(
                    candidatePublicationRank(strongestAlternative) -
                      candidatePublicationRank(selected)
                  ),
                }
              : {}),
          },
        }
      : {}),
    stability,
    layoutDetection: {
      outcome: geometry.layoutHint
        ? "detected"
        : geometry.detectionFailure
          ? "failed"
          : "not-detected",
      ...(geometry.detectedInputVariant
        ? { inputVariant: geometry.detectedInputVariant }
        : {}),
      ...(geometry.coordinateSpace
        ? { coordinateSpace: geometry.coordinateSpace }
        : {}),
      ...(geometry.layoutHint ? { layoutHint: geometry.layoutHint } : {}),
      ...(Number.isFinite(geometry.confidence)
        ? { confidence: geometry.confidence }
        : {}),
      ...(geometry.detectionFailure
        ? { failure: geometry.detectionFailure }
        : {}),
      ...(() => {
        const recognition =
          geometry.leadLabelValidation?.semanticRecognition ??
          geometry.limbLabelValidation?.semanticRecognition ??
          geometry.precordialLabelValidation?.semanticRecognition
        return recognition
          ? {
              semanticRecognition: {
                passed: recognition.passed,
                method: recognition.method,
                confidence: recognition.confidence,
                recognizedLabels: recognition.recognizedLabels ?? [],
                failureReasons: recognition.failureReasons,
                ...(recognition.engine
                  ? { engine: recognition.engine }
                  : {}),
              },
            }
          : {}
      })(),
    },
    neuralFeatureCache: {
      hits: neuralCandidates.filter((candidate) => candidate.featureCacheHit)
        .length,
      misses: neuralCandidates.filter(
        (candidate) => candidate.featureCacheHit === false
      ).length,
    },
  }
}

async function confirmBorderlineMpsSelection({
  run,
  prepared,
  selected,
  selectionCandidates,
  decisionOutcome,
  decisionReasonCode,
  benchmarkMode,
  signal,
}: {
  run: RunRecord
  prepared: PreparedRunInput
  selected: CandidateResult
  selectionCandidates: CandidateResult[]
  decisionOutcome: "needs_review" | "partial"
  decisionReasonCode: PublicationReasonCode
  benchmarkMode: boolean
  signal?: AbortSignal
}): Promise<{
  confirmed: boolean
  candidates: CandidateResult[]
  evidence: DigitizerStabilityEvidence
}> {
  if (
    benchmarkMode ||
    !selected.canonical ||
    !stabilityRequiredForPublicationOutcome(decisionOutcome)
  ) {
    return {
      confirmed: true,
      candidates: [],
      evidence: notRequiredStabilityEvidence(),
    }
  }

  const assignments = mpsConfirmationAssignments(
    selected,
    selectionCandidates
  )
  if (assignments.size === 0) {
    return {
      confirmed: true,
      candidates: [],
      evidence: notRequiredStabilityEvidence(),
    }
  }

  const rankedAlternatives = selectionCandidates
    .filter(
      (candidate) =>
        candidate.id !== selected.id &&
        candidate.status === "completed" &&
        Number.isFinite(candidate.score)
    )
    .sort(
      (a, b) =>
        candidatePublicationRank(a) - candidatePublicationRank(b) ||
        a.id.localeCompare(b.id)
    )
  const scoreGap =
    rankedAlternatives.length > 0 && Number.isFinite(selected.score)
      ? Math.abs(
          candidatePublicationRank(rankedAlternatives[0]) -
            candidatePublicationRank(selected)
        )
      : Infinity
  const assignedAgreementRmse = [...assignments.entries()].flatMap(
    ([candidate, leads]) =>
      leads.map((lead) => candidateLeadAlignmentRmse(selected, candidate, lead))
  )
  const typicalAgreementRmse = median(
    assignedAgreementRmse.filter(Number.isFinite)
  )
  const reasons: string[] = []
  if (decisionReasonCode === "source_verified") {
    reasons.push("source-verified publication depends on neural corroboration")
  }
  if (scoreGap <= DIGITIZER_POLICY.borderlineSelectionScoreGap) {
    reasons.push(`top candidate score gap was ${Math.round(scoreGap)}`)
  }
  if (
    Number.isFinite(typicalAgreementRmse) &&
    typicalAgreementRmse >=
      DISAGREEMENT_THRESHOLD_UV - DIGITIZER_POLICY.borderlineAgreementMarginUv
  ) {
    reasons.push(
      `typical corroboration error was ${typicalAgreementRmse.toFixed(1)} µV`
    )
  }
  if (reasons.length === 0) {
    return {
      confirmed: true,
      candidates: [],
      evidence: notRequiredStabilityEvidence(),
    }
  }

  const repeatedCandidates: CandidateResult[] = []
  const repeatConfirmedLeads = new Set<string>()
  const deterministicLeadCount = LEAD_ORDER.length - new Set(
    [...assignments.values()].flat()
  ).size
  for (const [source, leads] of assignments) {
    const repeat = await runCandidate(
      run,
      prepared,
      stabilityCandidateConfig(source, "mps-repeat", "mps"),
      signal
    )
    repeatedCandidates.push(repeat)
    if (!repeat.canonical) continue
    for (const lead of leads) {
      if (
        candidateLeadAlignmentRmse(selected, repeat, lead) <=
        DISAGREEMENT_THRESHOLD_UV
      ) {
        repeatConfirmedLeads.add(lead)
      }
    }
  }
  const repeatAgreementLeadCount =
    deterministicLeadCount + repeatConfirmedLeads.size
  if (
    repeatAgreementLeadCount >=
    DIGITIZER_POLICY.minimumMpsRepeatAgreementLeads
  ) {
    return {
      confirmed: true,
      candidates: repeatedCandidates,
      evidence: {
        required: true,
        reasons,
        outcome: "confirmed-repeat",
        repeatCandidateIds: repeatedCandidates.map((candidate) => candidate.id),
        cpuCandidateIds: [],
        repeatAgreementLeadCount,
        confirmedLeadCount: repeatAgreementLeadCount,
      },
    }
  }

  const cpuCandidates: CandidateResult[] = []
  const cpuConfirmedLeads = new Set(repeatConfirmedLeads)
  for (const [source, leads] of assignments) {
    const unresolvedLeads = leads.filter(
      (lead) => !repeatConfirmedLeads.has(lead)
    )
    if (unresolvedLeads.length === 0) continue
    const cpu = await runCandidate(
      run,
      prepared,
      stabilityCandidateConfig(source, "cpu-confirmation", "cpu"),
      signal
    )
    cpuCandidates.push(cpu)
    if (!cpu.canonical) continue
    for (const lead of unresolvedLeads) {
      if (
        candidateLeadAlignmentRmse(selected, cpu, lead) <=
        DISAGREEMENT_THRESHOLD_UV
      ) {
        cpuConfirmedLeads.add(lead)
      }
    }
  }
  const confirmedLeadCount = deterministicLeadCount + cpuConfirmedLeads.size
  const confirmed =
    confirmedLeadCount >= DIGITIZER_POLICY.minimumMpsRepeatAgreementLeads
  return {
    confirmed,
    candidates: [...repeatedCandidates, ...cpuCandidates],
    evidence: {
      required: true,
      reasons,
      outcome: confirmed ? "confirmed-cpu" : "unstable",
      repeatCandidateIds: repeatedCandidates.map((candidate) => candidate.id),
      cpuCandidateIds: cpuCandidates.map((candidate) => candidate.id),
      repeatAgreementLeadCount,
      confirmedLeadCount,
    },
  }
}

function stabilityRequiredForPublicationOutcome(
  outcome: "needs_review" | "partial"
) {
  return outcome === "needs_review"
}

function mpsConfirmationAssignments(
  selected: CandidateResult,
  candidates: CandidateResult[]
) {
  const eligibleMpsCandidates = candidates
    .filter(
      (candidate) =>
        candidate.status === "completed" &&
        candidate.parameters.device === "mps" &&
        candidate.parameters.vectorizer !== "native-grid-path" &&
        !isDerivedCandidate(candidate) &&
        isConservativeCandidate(candidate) &&
        candidate.canonical &&
        candidateEligibleForSelection(candidate) &&
        candidateCropKey(candidate) === candidateCropKey(selected) &&
        normalizeQaLayout(candidate.layout) === normalizeQaLayout(selected.layout)
    )
    .sort(
      (a, b) =>
        candidatePublicationRank(a) - candidatePublicationRank(b) ||
        a.id.localeCompare(b.id)
    )
  const assignments = new Map<CandidateResult, string[]>()
  for (const lead of LEAD_ORDER) {
    const declaredSourceId = selected.leadSources?.[lead]
    const declaredSource = declaredSourceId
      ? eligibleMpsCandidates.find((candidate) => candidate.id === declaredSourceId)
      : undefined
    const source =
      declaredSource ??
      (eligibleMpsCandidates.includes(selected) &&
      candidateHasStructurallyCompleteLead(selected, lead)
        ? selected
        : eligibleMpsCandidates.find(
            (candidate) =>
              candidateHasStructurallyCompleteLead(candidate, lead) &&
              candidateLeadAlignmentRmse(selected, candidate, lead) <=
                DISAGREEMENT_THRESHOLD_UV
          ))
    if (!source) continue
    const leads = assignments.get(source) ?? []
    leads.push(lead)
    assignments.set(source, leads)
  }
  return assignments
}

function stabilityCandidateConfig(
  source: CandidateResult,
  suffix: string,
  device: DigitizerComputeDevice
): DigitizerCandidateConfig {
  const parameters = source.parameters
  return {
    kind: candidateKind(parameters),
    id: `stability-${safeOutputStem(source.id)}-${suffix}`,
    label: `${source.label} (${suffix.replaceAll("-", " ")})`,
    resampleSize: parameters.resampleSize,
    ...(parameters.upscaleToMaxDimension
      ? { upscaleToMaxDimension: parameters.upscaleToMaxDimension }
      : {}),
    ...(parameters.darkInkEnhancement
      ? { darkInkEnhancement: true }
      : {}),
    ...(parameters.darkInkSupportRadius
      ? { darkInkSupportRadius: parameters.darkInkSupportRadius }
      : {}),
    ...(typeof parameters.labelThresh === "number"
      ? { labelThresh: parameters.labelThresh }
      : {}),
    ...(parameters.layoutConstraint
      ? { layoutConstraint: parameters.layoutConstraint }
      : {}),
    ...(parameters.geometryConfirmedLayout
      ? { geometryConfirmedLayout: true }
      : {}),
    ...(typeof parameters.geometryLayoutConfidence === "number"
      ? { geometryLayoutConfidence: parameters.geometryLayoutConfidence }
      : {}),
    ...(parameters.semanticLeadIdentityConfirmed
      ? { semanticLeadIdentityConfirmed: true }
      : {}),
    ...(parameters.semanticLeadIdentityMethod
      ? { semanticLeadIdentityMethod: parameters.semanticLeadIdentityMethod }
      : {}),
    ...(parameters.semanticLeadIdentityOrder
      ? { semanticLeadIdentityOrder: parameters.semanticLeadIdentityOrder }
      : {}),
    ...(parameters.semanticLeadIdentityLayout
      ? { semanticLeadIdentityLayout: parameters.semanticLeadIdentityLayout }
      : {}),
    ...(typeof parameters.adaptivePreprocessingEligible === "boolean"
      ? {
          adaptivePreprocessingEligible:
            parameters.adaptivePreprocessingEligible,
        }
      : {}),
    ...(typeof parameters.artifactPreprocessingEligible === "boolean"
      ? {
          artifactPreprocessingEligible:
            parameters.artifactPreprocessingEligible,
        }
      : {}),
    ...(typeof parameters.maximumLayoutCost === "number"
      ? { maximumLayoutCost: parameters.maximumLayoutCost }
      : {}),
    ...(parameters.cropBox ? { cropBox: parameters.cropBox } : {}),
    selectionEligible: false,
    inputVariant: parameters.inputVariant ?? "original",
    vectorizer: parameters.vectorizer ?? "probability-centroid",
    device,
    planningPhase: "recovery",
    scheduleReason: "borderline neural stability confirmation",
  }
}

function selectLeadSourceCandidate(
  candidates: CandidateResult[],
  preprocessing: PreprocessingReport,
  layout: string,
  lead: string
) {
  let pool = candidates.filter(
    (candidate) =>
      candidate.canonical &&
      normalizeQaLayout(candidate.layout) === normalizeQaLayout(layout) &&
      candidateHasScorableLead(candidate, lead)
  )
  if (pool.length === 0) return undefined

  if (preprocessing.annotationMask.maskedPixels > 0) {
    const annotationSafe = pool.filter(
      (candidate) => isAnnotationSafeInputVariant(
        candidate.parameters.inputVariant
      )
    )
    if (annotationSafe.length > 0) pool = annotationSafe
  }

  const conservative = pool.filter(
    (candidate) =>
      candidate.parameters.labelThresh === undefined ||
      candidate.parameters.labelThresh >= 0.1
  )
  if (conservative.length > 0) pool = conservative

  // A source-verified native trace may be trusted with explicit edge gaps,
  // but those gaps are a publication exception rather than a reason to
  // replace a complete, independently corroborated model trace. Keep the
  // native path as the identity/source anchor and select morphology from the
  // complete pool when one exists.
  const structurallyComplete = pool.filter((candidate) =>
    candidateHasStructurallyCompleteLead(candidate, lead)
  )
  if (structurallyComplete.length > 0) pool = structurallyComplete

  // A label-anchored 12x1 native path remains valuable independent source
  // evidence when it crosses a near-vertical ink stroke. It is not necessarily
  // the best morphology source, however: one state per raster column cannot
  // represent the sub-column ordering of a jump spanning more than a quarter
  // row. Once an independently trusted non-aliased candidate exists, retain
  // the native trace as the corroborator instead of copying its lossy samples
  // into the fused output.
  const topologyResolved = pool.filter(
    (candidate) => !candidateHasNativeRasterColumnSamplingRisk(candidate, lead)
  )
  if (topologyResolved.length > 0) pool = topologyResolved

  const sourceMaxDimension = Math.max(
    preprocessing.source.width,
    preprocessing.source.height
  )
  if (sourceMaxDimension < MIN_NATIVE_MODEL_INPUT_DIMENSION) {
    const modelScaleSafe = pool.filter(
      (candidate) =>
        (candidate.parameters.upscaleToMaxDimension ?? 0) >=
        MIN_NATIVE_MODEL_INPUT_DIMENSION
    )
    if (modelScaleSafe.length > 0) pool = modelScaleSafe
  }

  const nativeCentroidAvailable = pool.some(
    (candidate) =>
      !candidate.parameters.upscaleToMaxDimension &&
      candidate.parameters.vectorizer !== "dynamic-path"
  )
  const maximumResampleSize = Math.max(
    ...pool.map((candidate) => candidate.parameters.resampleSize)
  )

  const consensusScores = new Map(
    pool.map((candidate) => [
      candidate.id,
      leadConsensusSelectionScore(
        candidate,
        pool,
        lead,
        nativeCentroidAvailable,
        maximumResampleSize
      ),
    ])
  )
  return [...pool].sort((a, b) => {
    const scoreA = consensusScores.get(a.id) ?? Number.POSITIVE_INFINITY
    const scoreB = consensusScores.get(b.id) ?? Number.POSITIVE_INFINITY
    return scoreA - scoreB || a.score - b.score || a.id.localeCompare(b.id)
  })[0]
}

function candidateHasNativeRasterColumnSamplingRisk(
  candidate: CandidateResult,
  lead: string
) {
  const fidelity = candidate.sourceFidelity
  const leadFidelity = fidelity?.leadMetrics[lead]
  return Boolean(
    candidateCapabilities(candidate.parameters).nativeGrid &&
      candidate.parameters.vectorizer === "native-grid-path" &&
      fidelity?.inkConnectedTransitionRecovery?.method ===
        "near-continuous-vertical-source-ink-v1" &&
      (leadFidelity?.largeJumpCount ?? 0) > 0
  )
}

function leadConsensusSelectionScore(
  candidate: CandidateResult,
  peers: CandidateResult[],
  lead: string,
  nativeCentroidAvailable: boolean,
  maximumResampleSize: number
) {
  if (!candidate.canonical) return Number.POSITIVE_INFINITY
  const reference = canonicalLeadSegment(
    candidate.canonical,
    candidate.layout,
    lead
  )
  if (!reference) return Number.POSITIVE_INFINITY

  const disagreements = peers.flatMap((peer) => {
    if (peer.id === candidate.id || !peer.canonical) return []
    const rmseUv = candidateLeadAlignmentRmse(candidate, peer, lead)
    return Number.isFinite(rmseUv) ? [rmseUv] : []
  })
  const consensusPenalty = disagreements.length > 0 ? median(disagreements) : 0
  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = Math.max(leadQa?.expectedSamples ?? 0, 1)
  const missingPenalty =
    ((leadQa?.missingSamples ?? expectedSamples) / expectedSamples) * 40
  const gapPenalty =
    ((leadQa?.maxGapSamples ?? expectedSamples) / expectedSamples) * 10
  const upscalePenalty =
    nativeCentroidAvailable && candidate.parameters.upscaleToMaxDimension
      ? 12
      : 0
  const pathPenalty =
    nativeCentroidAvailable &&
    candidate.parameters.vectorizer === "dynamic-path"
      ? 6
      : 0
  const resolutionPenalty =
    nativeCentroidAvailable &&
    !candidate.parameters.upscaleToMaxDimension &&
    candidate.parameters.vectorizer !== "dynamic-path"
      ? Math.max(
          0,
          (maximumResampleSize - candidate.parameters.resampleSize) / 100
        )
      : 0
  const sourceInkFidelityBonus = candidate.parameters.darkInkEnhancement
    ? -30
    : 0

  return (
    consensusPenalty +
    missingPenalty +
    gapPenalty +
    upscalePenalty +
    pathPenalty +
    resolutionPenalty +
    sourceInkFidelityBonus
  )
}

function fuseLeadCandidates(
  reference: CanonicalCsv,
  layout: string,
  leadSources: Record<string, CandidateResult>,
  fullWidthRhythmLeads: ReadonlySet<string> = new Set()
): CanonicalCsv {
  const rows = reference.rows.map((row) =>
    row.map(() => Number.NaN)
  )
  const fused: CanonicalCsv = {
    leads: [...reference.leads],
    rows,
  }

  for (const lead of LEAD_ORDER) {
    const source = leadSources[lead]
    if (!source?.canonical) continue
    const leadIndex = fused.leads.indexOf(lead)
    if (leadIndex === -1) continue
    if (fullWidthRhythmLeads.has(lead)) {
      const sourceLeadIndex = source.canonical.leads.indexOf(lead)
      if (sourceLeadIndex === -1) continue
      const sampleCount = Math.min(rows.length, source.canonical.rows.length)
      for (let sample = 0; sample < sampleCount; sample += 1) {
        rows[sample][leadIndex] = source.canonical.rows[sample][sourceLeadIndex]
      }
      continue
    }
    const sourceSegment = canonicalLeadSegment(
      source.canonical,
      source.layout,
      lead
    )
    const targetSegment = canonicalLeadSegment(fused, layout, lead)
    if (!sourceSegment || !targetSegment) continue

    const sampleCount = Math.min(
      sourceSegment.values.length,
      targetSegment.values.length
    )
    for (let sample = 0; sample < sampleCount; sample += 1) {
      rows[targetSegment.start + sample][leadIndex] =
        sourceSegment.values[sample]
    }
  }

  return fused
}

function candidateHasCorroboratedFullWidthRhythmLead(
  candidate: CandidateResult,
  peers: CandidateResult[],
  lead: string
) {
  if (
    candidate.layout !== "standard_3x4_with_r1" ||
    lead !== "II" ||
    !candidate.canonical ||
    !candidateHasTrustedLead(candidate, peers, lead)
  ) {
    return false
  }
  const leadIndex = candidate.canonical.leads.indexOf(lead)
  if (leadIndex === -1 || candidate.canonical.rows.length === 0) return false
  const values = candidate.canonical.rows.map((row) => row[leadIndex])
  if (
    values.filter(Number.isFinite).length <
    Math.ceil(values.length * MIN_PUBLISHABLE_LEAD_COVERAGE)
  ) {
    return false
  }

  return peers.some((peer) => {
    if (
      peer.id === candidate.id ||
      isDerivedCandidate(peer) ||
      !candidatePathsAreIndependent(candidate, peer) ||
      !candidateHasQuantitativeTimingEvidence(peer) ||
      !peer.canonical ||
      peer.layout !== candidate.layout ||
      candidateCropKey(peer) !== candidateCropKey(candidate) ||
      !candidateHasSemanticLeadIdentity(peer, peers, lead)
    ) {
      return false
    }
    const peerCapabilities = candidateCapabilities(peer.parameters)
    if (peerCapabilities.nativeGrid && peer.sourceFidelity?.passed !== true) {
      return false
    }
    const peerLeadIndex = peer.canonical.leads.indexOf(lead)
    if (peerLeadIndex === -1 || peer.canonical.rows.length !== values.length) {
      return false
    }
    const peerValues = peer.canonical.rows.map((row) => row[peerLeadIndex])
    if (
      peerValues.filter(Number.isFinite).length <
      Math.ceil(peerValues.length * MIN_PUBLISHABLE_LEAD_COVERAGE)
    ) {
      return false
    }
    return alignSeries(values, peerValues).rmseUv <= DISAGREEMENT_THRESHOLD_UV
  })
}

function buildUncertaintyRows(
  selected: CandidateResult,
  candidates: CandidateResult[],
  preprocessing: PreprocessingReport,
  publishedCanonical = selected.canonical
): UncertaintyRow[] {
  if (!publishedCanonical) return []

  const comparableCandidates = candidates.filter(
    (candidate) =>
      candidateEligibleForSelection(candidate) &&
      (candidate.id === selected.id || !isDerivedCandidate(candidate)) &&
      candidate.canonical &&
      candidateCropKey(candidate) === candidateCropKey(selected) &&
      normalizeQaLayout(candidate.layout) === normalizeQaLayout(selected.layout) &&
      (!selected.sourceFidelity?.passed || candidate.sourceFidelity?.passed)
  )
  const annotationRanges = annotationRangesByLead(
    preprocessing,
    selected.layout,
    publishedCanonical.rows.length,
    selected.panelTimingCorrections,
    selected.parameters
  )
  const result: UncertaintyRow[] = []

  for (const lead of LEAD_ORDER) {
    const publishedSegment = canonicalLeadSegment(
      publishedCanonical,
      selected.layout,
      lead
    )
    const reviewSegment = selected.canonical
      ? canonicalLeadSegment(selected.canonical, selected.layout, lead)
      : publishedSegment
    if (!publishedSegment || !reviewSegment) continue

    const alignedSeries = comparableCandidates.flatMap((candidate) => {
      if (
        candidate.id !== selected.id &&
        !candidateHasStructurallyCompleteLead(candidate, lead)
      ) {
        return []
      }
      const segment = candidate.canonical
        ? canonicalLeadSegment(candidate.canonical, candidate.layout, lead)
        : null
      if (!segment) return []
      if (candidate.id === selected.id) {
        return [{ values: reviewSegment.values }]
      }
      return [{ values: alignSeries(reviewSegment.values, segment.values).values }]
    })

    publishedSegment.values.forEach((selectedValue, leadSample) => {
      const canonicalSample = publishedSegment.start + leadSample
      const candidateValues = alignedSeries
        .map((series) => series.values[leadSample])
        .filter(Number.isFinite)
      const candidateSpreadUv =
        candidateValues.length > 1
          ? Math.max(...candidateValues) - Math.min(...candidateValues)
          : 0
      const annotationOverlap = (annotationRanges[lead] ?? []).some(
        ({ start, end }) =>
          canonicalSample >= start && canonicalSample < end
      )
      const disagreement =
        candidateValues.length > 1 &&
        candidateSpreadUv >= DISAGREEMENT_THRESHOLD_UV
      let status: UncertaintyRow["status"] = "observed"

      if (annotationOverlap && disagreement) {
        status = "uncertain_annotation_and_disagreement"
      } else if (annotationOverlap) {
        status = "uncertain_annotation"
      } else if (!Number.isFinite(selectedValue)) {
        status = "missing"
      } else if (disagreement) {
        status = "uncertain_candidate_disagreement"
      }

      result.push({
        lead,
        leadSample,
        timeSeconds: leadSample / SAMPLE_RATE_HZ,
        canonicalSample,
        valueUv: selectedValue,
        reviewEstimateUv: annotationOverlap
          ? reviewSegment.values[leadSample]
          : Number.NaN,
        status,
        annotationOverlap,
        candidateCount: candidateValues.length,
        candidateSpreadUv,
      })
    })
  }

  return result
}

function updateQaWithUncertainty(
  selected: CandidateResult,
  uncertaintyRows: UncertaintyRow[]
) {
  if (!selected.qa) return

  let uncertainTotal = 0
  for (const lead of LEAD_ORDER) {
    const rows = uncertaintyRows.filter((row) => row.lead === lead)
    const spreads = rows
      .map((row) => row.candidateSpreadUv)
      .filter(Number.isFinite)
      .sort((a, b) => a - b)
    const uncertainSamples = rows.filter((row) =>
      row.status.startsWith("uncertain_")
    ).length
    uncertainTotal += uncertainSamples

    const leadQa = selected.qa.leads[lead]
    if (!leadQa) continue
    leadQa.candidateSpreadUvP95 = percentile(spreads, 0.95)
    leadQa.candidateSpreadUvMax = spreads.at(-1) ?? 0
    leadQa.uncertainSamples = uncertainSamples

    if (uncertainSamples > 0) {
      selected.qa.warnings.push({
        lead,
        severity: "warning",
        message: `${lead} has ${uncertainSamples} sample(s) flagged for annotation overlap or candidate disagreement.`,
      })
    }
  }

  if (uncertainTotal > 0) {
    selected.qa.summary = `${selected.qa.summary} ${uncertainTotal} sample(s) carry explicit uncertainty flags.`
  }
}

function reliabilitySummary(
  selected: CandidateResult,
  uncertaintyRows: UncertaintyRow[],
  preprocessing: PreprocessingReport,
  publicationReasonCode: PublicationReasonCode
): DigitizerReliabilitySummary {
  const inputQualityOutcome = preprocessing.inputQuality?.outcome
  const confidenceReasons: string[] = []
  if (publicationReasonCode === "reviewable_constrained") {
    confidenceReasons.push(
      "All 12 expected trace positions were recoverable, but the result did not meet the strongest independent-agreement or source-verification threshold."
    )
  }
  if (publicationReasonCode === "unstable_neural_confirmation") {
    confidenceReasons.push(
      "Repeated neural extraction was not fully stable; the recoverable result was retained for source comparison."
    )
  }
  if (
    !candidateHasExplicitSemanticLayoutEvidence(selected) &&
    candidateHasProvisionalLayoutLeadIdentity(selected)
  ) {
    confidenceReasons.push(
      "Printed lead labels were not fully confirmed by OCR. Lead identities are provisionally assigned from the detected standard layout; verify every label against the source before acceptance."
    )
  }
  if (inputQualityOutcome && inputQualityOutcome !== "acceptable") {
    confidenceReasons.push(
      "The source is below one or more preferred acquisition-fidelity targets; effective resolution and uncertainty should be reviewed."
    )
  }
  if (selected.sourceFidelity && !selected.sourceFidelity.passed) {
    confidenceReasons.push(
      "Direct pixel-to-trace source verification did not pass for the selected candidate."
    )
  }
  if (selected.qa && !selected.qa.passed) {
    confidenceReasons.push(
      `Lead-level QA reported ${selected.qa.warnings.length} warning(s).`
    )
  }
  return {
    version: 1,
    originalPreserved: true,
    morphologyReconstructed: false,
    sourceSha256: preprocessing.sourceSha256,
    sourceWidth: preprocessing.source.width,
    sourceHeight: preprocessing.source.height,
    annotationMaskedPixels: preprocessing.annotationMask.maskedPixels,
    annotationMaskedFraction: preprocessing.annotationMask.maskedFraction,
    annotationComponents: preprocessing.annotationMask.components,
    nominalSampleRateHz: SAMPLE_RATE_HZ,
    effectiveSampleRateHz:
      selected.effectiveSampleRateHz ?? SAMPLE_RATE_HZ,
    uncertainSampleCount: uncertaintyRows.filter((row) =>
      row.status.startsWith("uncertain_")
    ).length,
    missingSampleCount: uncertaintyRows.filter(
      (row) => !Number.isFinite(row.valueUv)
    ).length,
    confidence: confidenceReasons.length > 0 ? "lower" : "standard",
    confidenceReasons,
    ...(inputQualityOutcome ? { inputQualityOutcome } : {}),
    inputQualityReasons: preprocessing.inputQuality?.reasons ?? [],
    reviewRequired: true,
  }
}

async function writeUncertaintyCsv(
  uncertaintyRows: UncertaintyRow[],
  outputPath: string
) {
  const lines = [
    [
      "lead",
      "lead_sample",
      "t_s_at_500hz",
      "canonical_sample",
      "value_uv",
      "review_estimate_uv",
      "status",
      "annotation_overlap",
      "candidate_count",
      "candidate_spread_uv",
    ].join(","),
  ]

  for (const row of uncertaintyRows) {
    lines.push(
      [
        row.lead,
        row.leadSample,
        row.timeSeconds,
        row.canonicalSample,
        Number.isFinite(row.valueUv) ? row.valueUv : "",
        Number.isFinite(row.reviewEstimateUv) ? row.reviewEstimateUv : "",
        row.status,
        row.annotationOverlap ? 1 : 0,
        row.candidateCount,
        row.candidateSpreadUv,
      ].join(",")
    )
  }

  await fs.writeFile(outputPath, `${lines.join("\n")}\n`, "utf8")
}

async function writeDigitizationProvenance({
  path: outputPath,
  preprocessing,
  selected,
  candidates,
  reliability,
  calibration,
  pipelineEvidence,
}: {
  path: string
  preprocessing: PreprocessingReport
  selected: CandidateResult
  candidates: CandidateResult[]
  reliability: DigitizerReliabilitySummary
  calibration?: LayoutGeometryReport["calibration"]
  pipelineEvidence: DigitizerPipelineEvidence
}) {
  const provenance = {
    version: 5,
    generatedAt: new Date().toISOString(),
    input: {
      sourceSha256: preprocessing.sourceSha256,
      width: preprocessing.source.width,
      height: preprocessing.source.height,
      format: preprocessing.source.format,
      originalPreserved: true,
    },
    preprocessing: {
      workingImage: preprocessing.workingImage
        ? {
            ...preprocessing.workingImage,
            file: "preprocessing/working_source.png",
            morphologyReconstructed: false,
          }
        : undefined,
      inputQuality: preprocessing.inputQuality,
      execution: preprocessing.execution,
      annotationMask: preprocessing.annotationMask,
      preparedImage: preprocessing.preparedImage,
      evidenceMaps: preprocessing.evidenceMaps
        ? {
            ...preprocessing.evidenceMaps,
            files: {
              enhanced: "preprocessing/enhanced.png",
              backgroundFlattened:
                "preprocessing/background_flattened.png",
              traceProbability:
                "preprocessing/trace_probability.png",
              gridProbability: "preprocessing/grid_probability.png",
              exclusionMask: "preprocessing/exclusion_mask.png",
            },
          }
        : undefined,
      adaptivePreprocessing: preprocessing.adaptivePreprocessing,
      geometryCorrection: preprocessing.geometryCorrection,
      artifactPreprocessing: preprocessing.artifactPreprocessing,
    },
    digitization: {
      engine: "Open-ECG-Digitizer",
      nominalSampleRateHz: SAMPLE_RATE_HZ,
      publicationPolicy: {
        id: DIGITIZER_POLICY_ID,
        version: DIGITIZER_POLICY_VERSION,
        evidenceBasis: DIGITIZER_POLICY_EVIDENCE_BASIS,
      },
      pipelineEvidence,
      selectorCalibrationProfileId: SELECTOR_CALIBRATION_PROFILE_ID,
      selectorCalibrationTraining: SELECTOR_CALIBRATION_TRAINING,
      selectedCandidateId: selected.id,
      selectedLayout: selected.layout,
      selectedLayoutCost: selected.layoutCost,
      selectedEffectiveSampleRateHz: selected.effectiveSampleRateHz,
      selectedComputeDevice: selected.parameters.device,
      calibration: calibration ?? {
        method: "not-detected",
        detected: false,
        confidence: 0,
      },
      panelTimingNormalization: {
        method: "layout-column affine expansion from shared dense lead support",
        corrections: selected.panelTimingCorrections ?? [],
        rawCandidateCsvPreserved:
          (selected.panelTimingCorrections?.length ?? 0) > 0,
      },
      selectionPolicy: {
        minimumFiniteSamplesPerLead: MIN_SCORABLE_LEAD_SAMPLES,
        minimumFiniteCoveragePerLead: MIN_SCORABLE_LEAD_COVERAGE,
        minimumPublishableCoveragePerLead:
          MIN_PUBLISHABLE_LEAD_COVERAGE,
        minimumSourceVerifiedBoundaryCoveragePerLead:
          MIN_SOURCE_VERIFIED_BOUNDARY_COVERAGE,
        minimumReviewOnlyCoveragePerLead:
          MIN_REVIEWABLE_LEAD_COVERAGE,
        minimumNativeGridLayoutConfidenceForReviewOnly:
          MIN_REVIEW_LAYOUT_CONFIDENCE,
        minimumGeometryConstrained3x4Confidence:
          MIN_GEOMETRY_CONSTRAINED_3X4_CONFIDENCE,
        minimumGeometryConstrained6x2Confidence:
          MIN_GEOMETRY_CONSTRAINED_6X2_CONFIDENCE,
        minimumNativeGridReviewableLeadFractionForReviewOnly:
          MIN_REVIEW_LAYOUT_LEAD_FRACTION,
        requireCandidateAgreementWithinUv: DISAGREEMENT_THRESHOLD_UV,
        candidateAlignmentTimeScales: ALIGNMENT_TIME_SCALES,
        candidateAlignmentTrimFraction: ALIGNMENT_TRIM_FRACTION,
        candidateAlignmentSearchMaximumSamples:
          ALIGNMENT_SEARCH_MAX_SAMPLES,
        maximumPeerSupportedGapRepairSamples:
          MAX_PEER_GAP_REPAIR_SAMPLES,
        peerGapRepairRequires:
          "Two aligned conservative peers, or one source-pixel-verified peer, with boundary and per-sample agreement.",
        rejectLeadScopedQaErrorsForPublishableOutput: true,
        allowSourceVerifiedBoundaryLimitedPublication: true,
        sourceVerifiedBoundaryLimitedPublicationRequires:
          "Contiguous source-backed samples, no internal gap, no outside-panel samples, no lead QA error, and no unsafe excursion.",
        reviewOnlyOutputMayRetainExplicitMissingIntervals: true,
        abstainWhenNoCandidateQualifies: true,
        normalizeSharedInterPanelWhitespace: true,
        publishDetectedAnnotationOverlapAsMissing: true,
        useRhythmStripAsSoftQrsTimingEvidence: true,
        crossLeadEventCorroborationMinimumOtherLeads: 2,
        excludeVerticalArtifactsFromCrossLeadEventCorroboration: true,
        preserveIndependentLeadMorphology: true,
        rejectUncorroboratedLargeExcursions: true,
        recoverRejectedExcursionsOnlyFromConservativeSourceInkPath: true,
        publishRejectedExcursionsAsMissing: true,
        lowResolutionNativeCorroborationRequires:
          "Review-only retention requires agreement between the untouched and prepared native traces; a deterministic upscaled trace is retained as a third audit view, and effective sampling remains capped to source resolution.",
        annotationReviewVisualization:
          "Dashed selected-candidate estimate in shaded annotation intervals; excluded from quantitative signal CSVs.",
        annotationOutputMarginFraction: {
          default: ANNOTATION_OUTPUT_MARGIN_FRACTION,
          blue: BLUE_ANNOTATION_OUTPUT_MARGIN_FRACTION,
        },
        constrainedPanelBoundaryMaskFraction: {
          start: CONSTRAINED_PANEL_START_MASK_FRACTION,
          end: CONSTRAINED_PANEL_END_MASK_FRACTION,
        },
        minimumRobustLeadAmplitudeUv: MIN_ROBUST_LEAD_AMPLITUDE_UV,
      },
      candidates: candidates.map((candidate) => ({
        ...publicCandidate(candidate),
        selected: candidate.id === selected.id,
      })),
    },
    uncertainty: {
      candidateAlignmentRadiusSamples: ALIGNMENT_RADIUS_SAMPLES,
      disagreementThresholdUv: DISAGREEMENT_THRESHOLD_UV,
      annotationMapping:
        "Source-coordinate annotation boxes were transformed into the selected working/corrected/cropped candidate raster before mapping to the nearest supported lead panel with a horizontal safety margin.",
      ...reliability,
    },
  }

  await fs.writeFile(
    outputPath,
    `${JSON.stringify(provenance, finiteJsonNumber, 2)}\n`,
    "utf8"
  )
}

function annotationRangesByLead(
  preprocessing: PreprocessingReport,
  layout: string | undefined,
  canonicalLength: number,
  timingCorrections: DigitizerPanelTimingCorrection[] = [],
  candidateParameters?: Pick<
    DigitizerCandidateSummary["parameters"],
    "inputVariant" | "cropBox"
  >
) {
  const normalizedLayout = normalizeQaLayout(layout)
  const layoutRows = normalizedLayout ? LAYOUT_ROWS[normalizedLayout] : undefined
  const ranges: Record<string, { start: number; end: number }[]> = {}
  if (!layoutRows) return ranges

  const columnCount = Math.max(...layoutRows.rows.map((row) => row.length))
  const annotationFrame = annotationCandidateFrame(
    preprocessing,
    candidateParameters
  )
  for (const sourceComponent of preprocessing.annotationMask.components) {
    const component = annotationFrame.map(sourceComponent)
    if (!component) continue
    const marginFraction =
      component.dominantColor === "blue"
        ? BLUE_ANNOTATION_OUTPUT_MARGIN_FRACTION
        : ANNOTATION_OUTPUT_MARGIN_FRACTION
    const horizontalMargin = Math.max(
      6,
      Math.round(annotationFrame.width * marginFraction)
    )
    const centerX = (component.x0 + component.x1) / 2
    const centerY = (component.y0 + component.y1) / 2
    const column = clamp(
      Math.floor((centerX / annotationFrame.width) * columnCount),
      0,
      columnCount - 1
    )
    const row = layoutRows.rows.reduce(
      (best, _entry, index) => {
        const rowCenter =
          ((index + 0.5) / layoutRows.pageRows) *
          annotationFrame.height
        const distance = Math.abs(centerY - rowCenter)
        return distance < best.distance ? { index, distance } : best
      },
      { index: 0, distance: Number.POSITIVE_INFINITY }
    ).index
    const lead = layoutRows.rows[row]?.[column]
    if (!lead) continue

    let start = clamp(
      Math.floor(
        ((component.x0 - horizontalMargin) /
          annotationFrame.width) *
          canonicalLength
      ),
      0,
      canonicalLength
    )
    let end = clamp(
      Math.ceil(
        ((component.x1 + horizontalMargin) /
          annotationFrame.width) *
          canonicalLength
      ),
      0,
      canonicalLength
    )
    const correction = timingCorrections.find(
      (candidate) => candidate.column === column
    )
    if (correction) {
      const segmentSamples = Math.floor(canonicalLength / columnCount)
      const segmentStart = column * segmentSamples
      const segmentEnd =
        column === columnCount - 1
          ? canonicalLength
          : segmentStart + segmentSamples
      start = clamp(
        Math.floor(
          segmentStart +
            (start - segmentStart - correction.sourceStart) *
              correction.scale
        ),
        segmentStart,
        segmentEnd
      )
      end = clamp(
        Math.ceil(
          segmentStart +
            (end - segmentStart - correction.sourceStart) *
              correction.scale
        ),
        segmentStart,
        segmentEnd
      )
    }
    ;(ranges[lead] ??= []).push({ start, end })
  }

  return ranges
}

function annotationCandidateFrame(
  preprocessing: PreprocessingReport,
  candidateParameters?: Pick<
    DigitizerCandidateSummary["parameters"],
    "inputVariant" | "cropBox"
  >
) {
  const workingWidth =
    preprocessing.workingImage?.width ?? preprocessing.source.width
  const workingHeight =
    preprocessing.workingImage?.height ?? preprocessing.source.height
  const scaleX = workingWidth / Math.max(preprocessing.source.width, 1)
  const scaleY = workingHeight / Math.max(preprocessing.source.height, 1)
  const corrected =
    (candidateParameters?.inputVariant === "geometry-corrected" ||
      candidateParameters?.inputVariant === "artifact-preprocessed") &&
    preprocessing.geometryCorrection?.applied === true
  const transform = corrected
    ? preprocessing.geometryCorrection?.transform
    : undefined
  const baseWidth = corrected
    ? preprocessing.geometryCorrection?.outputWidth ?? workingWidth
    : workingWidth
  const baseHeight = corrected
    ? preprocessing.geometryCorrection?.outputHeight ?? workingHeight
    : workingHeight
  const crop = candidateParameters?.cropBox
  const width = crop ? crop.right - crop.left : baseWidth
  const height = crop ? crop.bottom - crop.top : baseHeight

  const mapPoint = (x: number, y: number) => {
    const workingX = x * scaleX
    const workingY = y * scaleY
    if (!transform || transform.length !== 3) {
      return { x: workingX, y: workingY }
    }
    const denominator =
      transform[2][0] * workingX +
      transform[2][1] * workingY +
      transform[2][2]
    if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-9) {
      return { x: workingX, y: workingY }
    }
    return {
      x:
        (transform[0][0] * workingX +
          transform[0][1] * workingY +
          transform[0][2]) /
        denominator,
      y:
        (transform[1][0] * workingX +
          transform[1][1] * workingY +
          transform[1][2]) /
        denominator,
    }
  }

  return {
    width: Math.max(1, width),
    height: Math.max(1, height),
    map(component: AnnotationComponent) {
      const corners = [
        mapPoint(component.x0, component.y0),
        mapPoint(component.x1, component.y0),
        mapPoint(component.x1, component.y1),
        mapPoint(component.x0, component.y1),
      ]
      const xValues = corners.map((point) => point.x - (crop?.left ?? 0))
      const yValues = corners.map((point) => point.y - (crop?.top ?? 0))
      const x0 = Math.max(0, Math.min(...xValues))
      const y0 = Math.max(0, Math.min(...yValues))
      const x1 = Math.min(width, Math.max(...xValues))
      const y1 = Math.min(height, Math.max(...yValues))
      if (x1 <= x0 || y1 <= y0) return null
      return { ...component, x0, y0, x1, y1 }
    },
  }
}

function suppressAnnotatedSamples(
  canonical: CanonicalCsv,
  preprocessing: PreprocessingReport,
  layout: string | undefined,
  timingCorrections: DigitizerPanelTimingCorrection[] = [],
  candidateParameters?: Pick<
    DigitizerCandidateSummary["parameters"],
    "inputVariant" | "cropBox"
  >
) {
  const rows = canonical.rows.map((row) => [...row])
  const ranges = annotationRangesByLead(
    preprocessing,
    layout,
    canonical.rows.length,
    timingCorrections,
    candidateParameters
  )

  for (const [lead, leadRanges] of Object.entries(ranges)) {
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) continue

    for (const { start, end } of leadRanges) {
      for (let sample = start; sample < end; sample += 1) {
        if (rows[sample]) rows[sample][leadIndex] = Number.NaN
      }
    }
  }

  return {
    leads: [...canonical.leads],
    rows,
  }
}

function canonicalLeadSegment(
  canonical: CanonicalCsv,
  layout: string | undefined,
  lead: string
) {
  const leadIndex = canonical.leads.indexOf(lead)
  if (leadIndex === -1) return null

  const normalizedLayout = normalizeQaLayout(layout)
  const layoutColumns = normalizedLayout
    ? LAYOUT_COLUMNS[normalizedLayout]
    : undefined
  if (!layoutColumns || !(lead in layoutColumns)) return null

  const columnCount = Math.max(...Object.values(layoutColumns)) + 1
  const segmentSamples = Math.floor(canonical.rows.length / columnCount)
  const column = layoutColumns[lead] ?? 0
  const start = column * segmentSamples
  const end =
    column === columnCount - 1
      ? canonical.rows.length
      : start + segmentSamples

  return {
    start,
    end,
    values: canonical.rows
      .slice(start, end)
      .map((row) => row[leadIndex]),
  }
}

function normalizeCanonicalPanelTiming(
  canonical: CanonicalCsv,
  layout: string | undefined
): {
  canonical: CanonicalCsv
  corrections: DigitizerPanelTimingCorrection[]
} {
  const normalizedLayout = normalizeQaLayout(layout)
  const layoutColumns = normalizedLayout
    ? LAYOUT_COLUMNS[normalizedLayout]
    : undefined
  if (!layoutColumns) return { canonical, corrections: [] }

  const columnCount = Math.max(...Object.values(layoutColumns)) + 1
  if (columnCount <= 1) return { canonical, corrections: [] }

  const segmentSamples = Math.floor(canonical.rows.length / columnCount)
  if (segmentSamples < 2) return { canonical, corrections: [] }

  const rows = canonical.rows.map((row) => [...row])
  const corrections: DigitizerPanelTimingCorrection[] = []

  for (let column = 0; column < columnCount; column += 1) {
    const segmentStart = column * segmentSamples
    const segmentEnd =
      column === columnCount - 1
        ? canonical.rows.length
        : segmentStart + segmentSamples
    const targetSamples = segmentEnd - segmentStart
    const supports = Object.entries(layoutColumns)
      .filter(([, leadColumn]) => leadColumn === column)
      .flatMap(([lead]) => {
        const leadIndex = canonical.leads.indexOf(lead)
        if (leadIndex === -1) return []
        const values = canonical.rows
          .slice(segmentStart, segmentEnd)
          .map((row) => row[leadIndex])
        const finiteIndices = values.flatMap((value, index) =>
          Number.isFinite(value) ? [index] : []
        )
        if (finiteIndices.length === 0) return []
        const first = finiteIndices[0]
        const last = finiteIndices.at(-1) ?? first
        const span = last - first + 1
        return [{
          lead,
          leadIndex,
          values,
          first,
          last,
          span,
          density: finiteIndices.length / span,
        }]
      })
    const references = supports.filter(
      (support) =>
        support.span >= targetSamples * 0.9 &&
        support.density >= 0.9
    )
    if (references.length < 2) continue

    const sourceStart = Math.min(...references.map((support) => support.first))
    const sourceEnd = Math.max(...references.map((support) => support.last))
    const sourceIntervals = sourceEnd - sourceStart
    const targetIntervals = targetSamples - 1
    if (sourceIntervals <= 0) continue

    const scale = targetIntervals / sourceIntervals
    const boundaryTolerance = Math.ceil(targetSamples * 0.05)
    const startSpread =
      Math.max(...references.map((support) => support.first)) - sourceStart
    const endSpread =
      sourceEnd - Math.min(...references.map((support) => support.last))
    if (
      scale < 1.005 ||
      scale > 1.2 ||
      startSpread > boundaryTolerance ||
      endSpread > boundaryTolerance
    ) {
      continue
    }

    for (const support of supports) {
      const normalizedValues = resamplePanelValues(
        support.values,
        sourceStart,
        sourceEnd,
        targetSamples
      )
      normalizedValues.forEach((value, index) => {
        rows[segmentStart + index][support.leadIndex] = value
      })
    }
    corrections.push({
      column,
      sourceStart,
      sourceEnd,
      scale,
      referenceLeads: references.map((support) => support.lead),
    })
  }

  return {
    canonical: {
      leads: [...canonical.leads],
      rows,
    },
    corrections,
  }
}

function resamplePanelValues(
  values: number[],
  sourceStart: number,
  sourceEnd: number,
  targetSamples: number
) {
  const output = Array.from({ length: targetSamples }, () => Number.NaN)
  if (targetSamples < 2 || sourceEnd <= sourceStart) return output

  const scale = (targetSamples - 1) / (sourceEnd - sourceStart)
  let runStart: number | null = null

  for (let index = 0; index <= values.length; index += 1) {
    const finite = index < values.length && Number.isFinite(values[index])
    if (finite && runStart === null) runStart = index
    if (finite || runStart === null) continue

    const runEnd = index - 1
    if (runStart === runEnd) {
      const targetIndex = Math.round((runStart - sourceStart) * scale)
      if (targetIndex >= 0 && targetIndex < targetSamples) {
        output[targetIndex] = values[runStart]
      }
      runStart = null
      continue
    }

    const targetStart = Math.max(
      0,
      Math.ceil((runStart - sourceStart) * scale - 1e-9)
    )
    const targetEnd = Math.min(
      targetSamples - 1,
      Math.floor((runEnd - sourceStart) * scale + 1e-9)
    )
    for (
      let targetIndex = targetStart;
      targetIndex <= targetEnd;
      targetIndex += 1
    ) {
      const sourceIndex = sourceStart + targetIndex / scale
      const lower = Math.max(runStart, Math.floor(sourceIndex))
      const upper = Math.min(runEnd, Math.ceil(sourceIndex))
      if (lower === upper) {
        output[targetIndex] = values[lower]
      } else {
        const weight = sourceIndex - lower
        output[targetIndex] =
          values[lower] * (1 - weight) + values[upper] * weight
      }
    }
    runStart = null
  }

  return output
}

function alignSeries(reference: number[], comparison: number[]) {
  let best = {
    shift: 0,
    timeScale: 1,
    offsetUv: 0,
    rmseUv: Number.POSITIVE_INFINITY,
  }

  const searchStride = Math.max(
    1,
    Math.ceil(reference.length / ALIGNMENT_SEARCH_MAX_SAMPLES)
  )

  for (const timeScale of ALIGNMENT_TIME_SCALES) {
    for (
      let shift = -ALIGNMENT_RADIUS_SAMPLES;
      shift <= ALIGNMENT_RADIUS_SAMPLES;
      shift += 1
    ) {
      const metrics = alignmentMetrics(
        reference,
        comparison,
        timeScale,
        shift,
        searchStride
      )
      if (metrics && metrics.rmseUv < best.rmseUv) {
        best = { shift, timeScale, ...metrics }
      }
    }
  }

  const fullMetrics = alignmentMetrics(
    reference,
    comparison,
    best.timeScale,
    best.shift,
    1
  )
  if (fullMetrics) {
    best = { ...best, ...fullMetrics }
  }

  const values = reference.map((_, index) => {
    const value = sampleAffineSeries(
      comparison,
      index,
      reference.length,
      best.timeScale,
      best.shift
    )
    return Number.isFinite(value) ? value - best.offsetUv : Number.NaN
  })
  return { ...best, values }
}

function candidateLeadAlignmentRmse(
  first: CandidateResult,
  second: CandidateResult,
  lead: string
) {
  const [referenceCandidate, comparisonCandidate] =
    first.id.localeCompare(second.id) <= 0
      ? [first, second]
      : [second, first]
  let comparisonCache = LEAD_ALIGNMENT_RMSE_CACHE.get(referenceCandidate)
  if (!comparisonCache) {
    comparisonCache = new WeakMap()
    LEAD_ALIGNMENT_RMSE_CACHE.set(referenceCandidate, comparisonCache)
  }
  let leadCache = comparisonCache.get(comparisonCandidate)
  if (!leadCache) {
    leadCache = new Map()
    comparisonCache.set(comparisonCandidate, leadCache)
  }
  const cached = leadCache.get(lead)
  if (cached !== undefined) return cached

  const reference = referenceCandidate.canonical
    ? canonicalLeadSegment(
        referenceCandidate.canonical,
        referenceCandidate.layout,
        lead
      )
    : null
  const comparison = comparisonCandidate.canonical
    ? canonicalLeadSegment(
        comparisonCandidate.canonical,
        comparisonCandidate.layout,
        lead
      )
    : null
  const rmseUv =
    reference && comparison
      ? alignSeries(reference.values, comparison.values).rmseUv
      : Number.POSITIVE_INFINITY
  leadCache.set(lead, rmseUv)
  return rmseUv
}

function alignmentMetrics(
  reference: number[],
  comparison: number[],
  timeScale: number,
  shift: number,
  stride: number
) {
  const differences: number[] = []
  let inspectedSamples = 0
  for (let index = 0; index < reference.length; index += stride) {
    inspectedSamples += 1
    const a = reference[index]
    const b = sampleAffineSeries(
      comparison,
      index,
      reference.length,
      timeScale,
      shift
    )
    if (Number.isFinite(a) && Number.isFinite(b)) {
      differences.push(b - a)
    }
  }
  if (
    differences.length <
    Math.min(inspectedSamples, Math.max(20, Math.ceil(inspectedSamples * 0.6)))
  ) {
    return undefined
  }

  const offsetUv = median(differences)
  const residuals = differences
    .map((difference) => difference - offsetUv)
    .sort((a, b) => Math.abs(a) - Math.abs(b))
  const retainedCount = Math.max(
    1,
    Math.ceil(residuals.length * (1 - ALIGNMENT_TRIM_FRACTION))
  )
  let squaredError = 0
  for (let index = 0; index < retainedCount; index += 1) {
    squaredError += residuals[index] ** 2
  }
  return {
    offsetUv,
    rmseUv: Math.sqrt(squaredError / retainedCount),
  }
}

function sampleAffineSeries(
  values: number[],
  referenceIndex: number,
  referenceLength: number,
  timeScale: number,
  shift: number
) {
  const referenceCenter = (referenceLength - 1) / 2
  const comparisonIndex =
    referenceCenter +
    (referenceIndex - referenceCenter) * timeScale +
    shift
  const lower = Math.floor(comparisonIndex)
  const upper = Math.ceil(comparisonIndex)
  if (lower < 0 || upper >= values.length) return Number.NaN
  const lowerValue = values[lower]
  const upperValue = values[upper]
  if (!Number.isFinite(lowerValue) || !Number.isFinite(upperValue)) {
    return Number.NaN
  }
  if (lower === upper) return lowerValue
  const weight = comparisonIndex - lower
  return lowerValue * (1 - weight) + upperValue * weight
}

function median(values: number[]) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (finite.length === 0) return Number.NaN
  const midpoint = Math.floor(finite.length / 2)
  return finite.length % 2 === 0
    ? (finite[midpoint - 1] + finite[midpoint]) / 2
    : finite[midpoint]
}

function percentile(sortedValues: number[], fraction: number) {
  if (sortedValues.length === 0) return 0
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(sortedValues.length * fraction) - 1)
  )
  return sortedValues[index]
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}

function finiteJsonNumber(_key: string, value: unknown) {
  return typeof value === "number" && !Number.isFinite(value) ? null : value
}

function evaluateQa(canonical: CanonicalCsv, layout?: string): DigitizerQa {
  const warnings: DigitizerQaWarning[] = []
  const leads: Record<string, DigitizerLeadQa> = {}
  const qaLayout = normalizeQaLayout(layout)
  const layoutColumns = qaLayout ? LAYOUT_COLUMNS[qaLayout] : undefined
  const configuredColumnCount = qaLayout
    ? Math.max(...Object.values(layoutColumns ?? { only: 0 })) + 1
    : 0
  const columnCount = configuredColumnCount || 1
  const segmentSamples = Math.floor(canonical.rows.length / columnCount)

  if (!layoutColumns) {
    warnings.push({
      severity: "warning",
      message: layout
        ? `Lead QA does not know the ${layout} layout.`
        : "Lead QA could not read the detected layout.",
    })
  }

  for (const lead of LEAD_ORDER) {
    if (layoutColumns && !(lead in layoutColumns)) {
      leads[lead] = {
        expectedSamples: 0,
        finiteSamples: 0,
        missingSamples: 0,
        maxGapSamples: 0,
        outsideSegmentSamples: 0,
      }
      continue
    }
    const leadIndex = canonical.leads.indexOf(lead)
    if (leadIndex === -1) {
      leads[lead] = {
        finiteSamples: 0,
        missingSamples: segmentSamples,
        maxGapSamples: segmentSamples,
        outsideSegmentSamples: 0,
      }
      warnings.push({
        lead,
        severity: "error",
        message: `${lead} is missing from the canonical CSV.`,
      })
      continue
    }

    const expectedColumn = layoutColumns?.[lead] ?? 0
    const start = expectedColumn * segmentSamples
    const end =
      expectedColumn === columnCount - 1
        ? canonical.rows.length
        : start + segmentSamples
    const values = canonical.rows.map((row) => row[leadIndex])
    const finite = values.map((value) => Number.isFinite(value))
    const segmentFinite = finite.slice(start, end)
    const finiteSamples = segmentFinite.filter(Boolean).length
    const expectedSamples = end - start
    const missingSamples = expectedSamples - finiteSamples
    const finiteSegmentValues = values
      .slice(start, end)
      .filter(Number.isFinite)
      .sort((a, b) => a - b)
    const robustAmplitudeUv =
      finiteSegmentValues.length > 0
        ? percentile(finiteSegmentValues, 0.99) -
          percentile(finiteSegmentValues, 0.01)
        : 0
    const isFullWidthRhythmLead =
      lead === "II" && Boolean(layout?.toLowerCase().includes("with_r1"))
    const outsideSegmentSamples = layoutColumns && !isFullWidthRhythmLead
      ? finite.filter((value, index) => value && (index < start || index >= end)).length
      : 0
    const gaps = contiguousGaps(segmentFinite)
    const maxGapSamples = gaps.reduce((max, gap) => Math.max(max, gap.length), 0)

    leads[lead] = {
      expectedSamples,
      finiteSamples,
      missingSamples,
      maxGapSamples,
      outsideSegmentSamples,
    }

    if (missingSamples > 0) {
      warnings.push({
        lead,
        severity: "warning",
        message: `${lead} has ${missingSamples} missing sample(s) in its expected panel.`,
      })
    }

    const minimumScorableSamples = minimumScorableLeadSamples(expectedSamples)
    if (finiteSamples < minimumScorableSamples) {
      warnings.push({
        lead,
        severity: "error",
        message: `${lead} has only ${finiteSamples} of ${expectedSamples} expected samples; at least ${minimumScorableSamples} (${Math.round(MIN_SCORABLE_LEAD_COVERAGE * 100)}%) are required for selection.`,
      })
    }
    if (
      finiteSamples >= minimumScorableSamples &&
      robustAmplitudeUv < MIN_ROBUST_LEAD_AMPLITUDE_UV
    ) {
      warnings.push({
        lead,
        severity: "error",
        message: `${lead} is near-flat (${robustAmplitudeUv.toFixed(1)} µV robust amplitude); finite placeholder samples are not treated as an observed ECG trace.`,
      })
    }

    const longGaps = gaps.filter((gap) => gap.length >= 10)
    for (const gap of longGaps.slice(0, 2)) {
      const isPanelTransition =
        gap.start < expectedSamples * 0.15 ||
        gap.end > expectedSamples * 0.85
      warnings.push({
        lead,
        severity: isPanelTransition ? "warning" : "error",
        message: `${lead} has a ${gap.length}-sample ${isPanelTransition ? "panel-transition" : "internal"} gap near sample ${start + gap.start}.`,
      })
    }

    if (outsideSegmentSamples > 0) {
      warnings.push({
        lead,
        severity: "warning",
        message: `${lead} has ${outsideSegmentSamples} finite sample(s) outside its expected segment.`,
      })
    }
  }

  const passed = warnings.every((warning) => warning.severity !== "error")
  const errorCount = warnings.filter((warning) => warning.severity === "error").length
  const warningCount = warnings.length - errorCount

  return {
    passed,
    summary: passed
      ? warningCount > 0
        ? `Lead QA found no internal-gap errors; ${warningCount} coverage warning(s) still require review.`
        : "Lead QA passed with complete expected segments."
      : `${errorCount} lead QA error(s) need visual review.`,
    warnings,
    leads,
  }
}

function normalizeQaLayout(layout?: string) {
  return (
    normalizeSupportedEcgLayout(layout) ??
    Object.keys(PARTIAL_LAYOUT_LEADS).find(
      (candidate) =>
        layout === candidate || layout?.startsWith(`${candidate}_`)
    ) ??
    layout
  )
}

function scoreCandidate(
  qa: DigitizerQa,
  layout: string | undefined,
  layoutCost: number | undefined,
  candidate: DigitizerCandidateConfig,
  preprocessing: PreprocessingReport
): DigitizerCandidateScoreBreakdown {
  const errorCount = qa.warnings.filter((warning) => warning.severity === "error").length
  const warningCount = qa.warnings.length - errorCount
  const missingSamples = Object.values(qa.leads).reduce(
    (total, lead) => total + lead.missingSamples,
    0
  )
  const maxGap = Object.values(qa.leads).reduce(
    (max, lead) => Math.max(max, lead.maxGapSamples),
    0
  )

  return {
    errorPenalty: errorCount * 100_000,
    warningPenalty: warningCount * 10_000,
    missingPenalty: missingSamples * 10,
    gapPenalty: maxGap,
    layoutPenalty: (layoutCost ?? 1) * 100,
    annotationRiskPenalty:
      preprocessing.annotationMask.maskedPixels > 0 &&
      candidate.inputVariant === "original"
        ? 25_000
        : 0,
    permissiveThresholdPenalty:
      typeof candidate.labelThresh === "number" &&
      candidate.labelThresh < 0.1
        ? 250_000
        : 0,
    candidateDisagreementPenalty: 0,
    layoutCalibrationPenalty: selectorCalibrationEvidence(
      layout,
      candidate,
      preprocessing
    ).penalty,
  }
}

function candidatePublicationRank(candidate: Pick<
  CandidateResult,
  "score" | "scoreBreakdown"
>) {
  const warningPenalty = candidate.scoreBreakdown?.warningPenalty ?? 0
  return (
    candidate.score -
    warningPenalty +
    warningPenalty * DIGITIZER_POLICY.publicationWarningPenaltyWeight
  )
}

function selectSafestCandidate(
  completed: CandidateResult[],
  preprocessing: PreprocessingReport
) {
  const scorable = completed.filter((candidate) =>
    candidateHasAllTrustedLeads(candidate, completed)
  )
  if (scorable.length === 0) return undefined

  const conservative = scorable.filter(
    (candidate) =>
      candidate.parameters.labelThresh === undefined ||
      candidate.parameters.labelThresh >= 0.1
  )
  const pool = conservative.length > 0 ? conservative : scorable
  const totalExpectedSamples = Math.max(
    ...pool.map((candidate) =>
      Object.values(candidate.qa?.leads ?? {}).reduce(
        (total, lead) => total + (lead.expectedSamples ?? 0),
        0
      )
    ),
    1
  )
  const minimumMissing = Math.min(
    ...pool.map((candidate) => candidateMissingSamples(candidate))
  )
  const minimumErrors = Math.min(
    ...pool.map((candidate) => candidateErrorCount(candidate))
  )

  const conservativeCoverageFailed =
    minimumMissing > totalExpectedSamples * 0.05
  const viablePool = conservativeCoverageFailed
    ? scorable
    : pool.filter(
        (candidate) =>
          candidateErrorCount(candidate) <= minimumErrors + 1 &&
          candidateMissingSamples(candidate) <=
            minimumMissing + Math.max(25, totalExpectedSamples * 0.01)
      )

  const annotationSafe =
    preprocessing.annotationMask.maskedPixels > 0
      ? viablePool.filter(
          (candidate) => isAnnotationSafeInputVariant(
            candidate.parameters.inputVariant
          )
        )
      : []
  const finalPool =
    annotationSafe.length > 0
      ? annotationSafe
      : viablePool.length > 0
        ? viablePool
        : pool

  return [...finalPool].sort(
    (a, b) =>
      candidatePublicationRank(a) - candidatePublicationRank(b) ||
      a.id.localeCompare(b.id)
  )[0]
}

function isAnnotationSafeInputVariant(
  inputVariant: DigitizerInputVariant | undefined
) {
  return (
    inputVariant === "annotation-masked" ||
    inputVariant === "preprocessed" ||
    inputVariant === "geometry-corrected" ||
    inputVariant === "artifact-preprocessed"
  )
}

function candidateEligibleForSelection(candidate: CandidateResult) {
  const capabilities = candidateCapabilities(candidate.parameters)
  if (!capabilities.selectionEligible) return false
  if (
    candidate.parameters.inputVariant === "preprocessed" &&
    candidate.parameters.adaptivePreprocessingEligible !== true &&
    candidate.parameters.geometryConfirmedLayout !== true
  ) {
    return false
  }
  if (
    candidate.parameters.inputVariant === "artifact-preprocessed" &&
    candidate.parameters.artifactPreprocessingEligible !== true
  ) {
    return false
  }
  if (capabilities.nativeGrid && capabilities.preprocessed) {
    return candidate.sourceFidelity?.passed === true
  }
  return true
}

function hasLowResolutionNativeTraceCorroboration(
  candidate: CandidateResult,
  completed: CandidateResult[],
  preprocessing: PreprocessingReport
) {
  const sourceMaxDimension = Math.max(
    preprocessing.source.width,
    preprocessing.source.height
  )
  const fidelity = candidate.sourceFidelity
  const capabilities = candidateCapabilities(candidate.parameters)
  if (
    sourceMaxDimension > LOW_RESOLUTION_NATIVE_CORROBORATION_MAX_DIMENSION ||
    !capabilities.nativeGrid ||
    !capabilities.preprocessed ||
    candidate.parameters.inputVariant !== "preprocessed" ||
    candidate.parameters.adaptivePreprocessingEligible !== true ||
    candidate.parameters.vectorizer !== "native-grid-path" ||
    fidelity?.passed !== false ||
    !candidate.canonical ||
    !normalizeSupportedEcgLayout(candidate.layout) ||
    fidelity.layoutConfidence < MIN_NATIVE_REVIEW_LAYOUT_CONFIDENCE ||
    fidelity.minimumCoverage < MIN_REVIEWABLE_LEAD_COVERAGE ||
    (fidelity.unsupportedLargeJumpCount ?? Number.POSITIVE_INFINITY) >
      (fidelity.maximumUnsupportedLargeJumps ?? Number.NEGATIVE_INFINITY) +
        LOW_RESOLUTION_NATIVE_CORROBORATION_JUMP_MARGIN ||
    !LEAD_ORDER.every((lead) => candidateHasReviewableLead(candidate, lead))
  ) {
    return false
  }

  const matchingPeers = completed.filter((peer) => {
    const peerFidelity = peer.sourceFidelity
    if (
      peer.id === candidate.id ||
      peer.parameters.vectorizer !== "native-grid-path" ||
      !peer.canonical ||
      normalizeQaLayout(peer.layout) !== normalizeQaLayout(candidate.layout) ||
      candidateCropKey(peer) !== candidateCropKey(candidate) ||
      !peerFidelity ||
      peerFidelity.layoutConfidence < MIN_NATIVE_REVIEW_LAYOUT_CONFIDENCE ||
      peerFidelity.minimumCoverage < MIN_REVIEWABLE_LEAD_COVERAGE ||
      (peerFidelity.unsupportedLargeJumpCount ?? Number.POSITIVE_INFINITY) >
        (peerFidelity.maximumUnsupportedLargeJumps ?? Number.NEGATIVE_INFINITY) ||
      !LEAD_ORDER.every((lead) => candidateHasReviewableLead(peer, lead))
    ) {
      return false
    }

    const leadRmse = LEAD_ORDER.map((lead) =>
      candidateLeadAlignmentRmse(candidate, peer, lead)
    )
    if (!leadRmse.every(Number.isFinite)) return false
    return Boolean(
      median(leadRmse) <= LOW_RESOLUTION_NATIVE_CORROBORATION_MEDIAN_RMSE_UV &&
      leadRmse.filter(
        (rmseUv) =>
          rmseUv <= LOW_RESOLUTION_NATIVE_CORROBORATION_LEAD_RMSE_UV
      ).length >= LOW_RESOLUTION_NATIVE_CORROBORATION_MINIMUM_LEADS
    )
  })
  return (
    matchingPeers.some(
      (peer) => peer.parameters.inputVariant !== "preprocessed"
    )
  )
}

function selectAdaptivePreprocessedCandidate(
  completed: CandidateResult[]
) {
  return completed
    .filter(
      (candidate) =>
        candidateKind(candidate.parameters) === "adaptive-preprocessed" &&
        candidate.parameters.vectorizer === "dynamic-path" &&
        candidate.parameters.adaptivePreprocessingEligible === true &&
        candidate.qa?.passed === true &&
        LEAD_ORDER.every((lead) =>
          candidateHasStructurallyCompleteLead(candidate, lead)
        ) &&
        candidateHasAllTrustedLeads(candidate, completed)
    )
    .sort(
      (a, b) =>
        candidatePublicationRank(a) - candidatePublicationRank(b) ||
        a.id.localeCompare(b.id)
    )[0]
}

function selectReviewableConstrainedCandidate(
  completed: CandidateResult[]
) {
  const reviewable = completed.filter(
    (candidate) =>
      Boolean(normalizeSupportedEcgLayout(candidate.layout)) &&
      LEAD_ORDER.every((lead) =>
        candidateHasMinimumCoverage(candidate, lead, MIN_REVIEWABLE_LEAD_COVERAGE) &&
        candidateHasSemanticLeadIdentity(candidate, completed, lead)
      )
  )
  const corroborated = reviewable.filter(
    (candidate) =>
      candidate.sourceFidelity?.passed ||
      hasSupportedReviewLayoutConstraint(candidate) ||
      hasNativeGridLayoutCorroboration(candidate, completed) ||
      hasConservativeLayoutPeer(candidate, completed)
  )
  const conservative = corroborated.filter(isConservativeCandidate)
  if (conservative.length === 0) return undefined
  const peerCorroborated = new Set(
    conservative
      .filter((candidate) =>
        hasCompleteConservativeLayoutPeer(candidate, completed)
      )
      .map((candidate) => candidate.id)
  )
  // A native path can cover every sample while losing the order of steep
  // strokes inside one raster column. Keep it as source corroboration, but
  // prefer another eligible extraction with the same scorable-lead count.
  // This does not discard the native candidate when no alternative exists.
  const columnSamplingRisk = new Set(
    conservative
      .filter((candidate) =>
        LEAD_ORDER.some((lead) =>
          candidateHasNativeRasterColumnSamplingRisk(candidate, lead)
        )
      )
      .map((candidate) => candidate.id)
  )
  // This native method enforces strict source-ink continuity and rhythm
  // anchoring. Prefer its confirmed source time over agreement between
  // correlated model outputs. Other native modes use different fidelity gates.
  const strictSourceTiming = new Set(
    conservative
      .filter((candidate) =>
        candidate.parameters.vectorizer === "native-grid-path" &&
        candidate.qa?.passed === true &&
        candidate.sourceFidelity?.passed === true &&
        candidate.sourceFidelity.sourcePanelTimingDetected === true &&
        candidate.sourceFidelity.method ===
          "rhythm-anchored-connected-multievidence-native-path-v6"
      )
      .map((candidate) => candidate.id)
  )
  return [...conservative].sort(
    (a, b) =>
      candidateScorableLeadCount(b) - candidateScorableLeadCount(a) ||
      Number(columnSamplingRisk.has(a.id)) -
        Number(columnSamplingRisk.has(b.id)) ||
      candidateStructurallyCompleteLeadCount(b) -
        candidateStructurallyCompleteLeadCount(a) ||
      Number(strictSourceTiming.has(b.id)) -
        Number(strictSourceTiming.has(a.id)) ||
      Number(peerCorroborated.has(b.id)) -
        Number(peerCorroborated.has(a.id)) ||
      candidatePublicationRank(a) - candidatePublicationRank(b) ||
      a.id.localeCompare(b.id)
  )[0]
}

function selectPartialLeadCandidate(completed: CandidateResult[]) {
  const reviewable = completed.filter((candidate) => {
    const expectedLeads = expectedLayoutLeads(candidate.layout)
    return (
      isSupportedPartialLayout(candidate.layout) &&
      expectedLeads.length >= 3 &&
      expectedLeads.every((lead) =>
        candidateHasMinimumCoverage(
          candidate,
          lead,
          MIN_REVIEWABLE_LEAD_COVERAGE
        ) && candidateHasSemanticLeadIdentity(candidate, completed, lead)
      )
    )
  })
  const corroborated = reviewable.filter(
    (candidate) =>
      candidate.sourceFidelity?.passed ||
      hasConservativeLayoutPeer(candidate, completed)
  )
  const conservative = corroborated.filter(isConservativeCandidate)
  if (conservative.length === 0) return undefined
  const peerCorroborated = new Set(
    conservative
      .filter((candidate) =>
        hasCompleteConservativeLayoutPeer(candidate, completed)
      )
      .map((candidate) => candidate.id)
  )
  return [...conservative].sort(
    (a, b) =>
      expectedLayoutLeads(b.layout).length -
        expectedLayoutLeads(a.layout).length ||
      Number(peerCorroborated.has(b.id)) -
        Number(peerCorroborated.has(a.id)) ||
      candidateExpectedLeadCoverage(b) - candidateExpectedLeadCoverage(a) ||
      a.score - b.score ||
      a.id.localeCompare(b.id)
  )[0]
}

function isConservativeCandidate(candidate: CandidateResult) {
  return (
    candidate.parameters.labelThresh === undefined ||
    candidate.parameters.labelThresh >= 0.1
  )
}

function hasConservativeLayoutPeer(
  candidate: CandidateResult,
  completed: CandidateResult[]
) {
  if (!isConservativeCandidate(candidate) || !candidate.canonical) return false
  const layout = normalizeQaLayout(candidate.layout)
  if (!layout || !LAYOUT_COLUMNS[layout]) return false
  const expectedLeads = expectedLayoutLeads(candidate.layout)
  if (expectedLeads.length === 0) return false
  return completed.some(
    (peer) => {
      if (
        peer.id === candidate.id ||
        isDerivedCandidate(peer) ||
        !isConservativeCandidate(peer) ||
        !peer.canonical ||
        candidateCropKey(peer) !== candidateCropKey(candidate) ||
        normalizeQaLayout(peer.layout) !== layout ||
        !expectedLeads.every((lead) =>
          candidateHasMinimumCoverage(
            peer,
            lead,
            MIN_REVIEWABLE_LEAD_COVERAGE
          )
        )
      ) {
        return false
      }
      return expectedLeads.every(
        (lead) =>
          candidateLeadAlignmentRmse(candidate, peer, lead) <=
          DISAGREEMENT_THRESHOLD_UV
      )
    }
  )
}

function hasCompleteConservativeLayoutPeer(
  candidate: CandidateResult,
  completed: CandidateResult[]
) {
  const expectedLeads = expectedLayoutLeads(candidate.layout)
  return (
    expectedLeads.length > 0 &&
    expectedLeads.every((lead) =>
      candidateHasStructurallyCompleteLead(candidate, lead)
    ) &&
    hasConservativeLayoutPeer(candidate, completed)
  )
}

function candidateCropKey(candidate: CandidateResult) {
  const crop = candidate.parameters.cropBox
  return crop
    ? `${crop.left}:${crop.top}:${crop.right}:${crop.bottom}`
    : "full-source"
}

function expectedLayoutLeads(layout?: string) {
  const normalized = normalizeQaLayout(layout)
  if (!normalized) return []
  if (PARTIAL_LAYOUT_LEADS[normalized]) {
    return [...PARTIAL_LAYOUT_LEADS[normalized]]
  }
  return normalizeSupportedEcgLayout(layout) ? [...LEAD_ORDER] : []
}

function isSupportedPartialLayout(layout?: string) {
  const normalized = normalizeQaLayout(layout)
  return Boolean(normalized && PARTIAL_LAYOUT_LEADS[normalized])
}

function candidateExpectedLeadCoverage(candidate: CandidateResult) {
  const coverages = expectedLayoutLeads(candidate.layout).map((lead) => {
    const qa = candidate.qa?.leads[lead]
    return qa?.expectedSamples
      ? qa.finiteSamples / qa.expectedSamples
      : 0
  })
  return coverages.length > 0 ? Math.min(...coverages) : 0
}

function candidateHasMinimumCoverage(
  candidate: CandidateResult,
  lead: string,
  fraction: number
) {
  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = leadQa?.expectedSamples ?? 0
  return (
    expectedSamples > 0 &&
    (leadQa?.finiteSamples ?? 0) >= Math.ceil(expectedSamples * fraction)
  )
}

function candidateStructurallyCompleteLeadCount(candidate: CandidateResult) {
  return expectedLayoutLeads(candidate.layout).filter((lead) =>
    candidateHasStructurallyCompleteLead(candidate, lead)
  ).length
}

function hasSupportedReviewLayoutConstraint(candidate: CandidateResult) {
  const constraint = candidate.parameters.layoutConstraint
  const layoutMatchesConstraint =
    (constraint === "standard_6x2" ||
      constraint === "standard_6x2_with_r1_ignored" ||
      constraint === "standard_3x4" ||
      constraint === "standard_3x4_with_r1" ||
      constraint === "standard_3x4_with_r2" ||
      constraint === "standard_3x4_with_r3" ||
      constraint === "cabrera_12x1" ||
      constraint === "standard_12x1") &&
    candidate.layout === constraint
  if (!layoutMatchesConstraint) return false

  const fidelity = candidate.sourceFidelity
  if (
    candidate.parameters.vectorizer === "native-grid-path" &&
    fidelity &&
    !fidelity.passed
  ) {
    if (
      (fidelity.method === "row-local-sequential-label-validated-v2" ||
        fidelity.method === "row-local-sequential-label-validated-v3") &&
      fidelity.leadLabelValidation?.passed === true &&
      fidelity.leadLabelValidation.order === "standard" &&
      fidelity.leadOrderValidation?.passed === true &&
      fidelity.leadOrderValidation.selectedOrder === "standard"
    ) {
      const leadMetrics = Object.values(fidelity.leadMetrics)
      const weakLeads = leadMetrics.filter(
        (lead) => lead.evidenceMedian < 0.5
      )
      const strongLeads = leadMetrics.filter(
        (lead) => lead.evidenceMedian >= 0.5
      )
      return (
        fidelity.layoutConfidence >= 0.8 &&
        fidelity.evidenceMedian >= 0.9 &&
        fidelity.minimumCoverage >= 0.85 &&
        (fidelity.minimumHomeRowFraction ?? 0) >= 0.75 &&
        (fidelity.unsupportedLargeJumpCount ?? Number.POSITIVE_INFINITY) <=
          (fidelity.maximumUnsupportedLargeJumps ?? Number.NEGATIVE_INFINITY) &&
        weakLeads.length === 1 &&
        weakLeads[0].evidenceMedian >= 0.1 &&
        (weakLeads[0].homeRowFraction ?? 0) >= 0.9 &&
        weakLeads[0].p95NativeJumpPixels <= 10 &&
        (weakLeads[0].unsupportedLargeJumpCount ?? Number.POSITIVE_INFINITY) <=
          1 &&
        strongLeads.length === LEAD_ORDER.length - 1 &&
        strongLeads.every(
          (lead) =>
            lead.evidenceMedian >= 0.9 &&
            lead.evidenceP10 >= 0.6 &&
            (lead.homeRowFraction ?? 1) >= 0.8
        )
      )
    }
    if (
      candidate.layout === "standard_3x4_with_r1" &&
      fidelity.method.startsWith("rhythm-anchored-connected-") &&
      candidate.qa?.passed === true &&
      fidelity.layoutConfidence >= 0.9 &&
      fidelity.evidenceMedian >= 0.95 &&
      fidelity.minimumCoverage >= 0.9 &&
      (fidelity.minimumHomeRowFraction ?? 0) >= 0.95 &&
      (fidelity.rhythmAnchorCount ?? 0) >= 6 &&
      fidelity.rhythmTracePassed === true &&
      (fidelity.unsafeExcursionCount ?? 0) === 0 &&
      (fidelity.unsupportedLargeJumpCount ?? Number.POSITIVE_INFINITY) <=
        (fidelity.maximumUnsupportedLargeJumps ?? Number.NEGATIVE_INFINITY) &&
      Object.values(fidelity.leadMetrics).every(
        (lead) =>
          lead.evidenceMedian >= 0.95 &&
          (lead.homeRowFraction ?? 0) >= 0.95 &&
          (lead.unsafeExcursionCount ?? 0) === 0
      )
    ) {
      return true
    }
    return (
      fidelity.layoutConfidence >= MIN_NATIVE_REVIEW_LAYOUT_CONFIDENCE &&
      fidelity.evidenceMedian >= MIN_NATIVE_REVIEW_EVIDENCE_MEDIAN &&
      fidelity.evidenceP10 >= MIN_NATIVE_REVIEW_EVIDENCE_P10 &&
      fidelity.minimumCoverage >= MIN_REVIEWABLE_LEAD_COVERAGE &&
      (fidelity.minimumHomeRowFraction ?? 1) >= 0.55
    )
  }
  return true
}

function hasNativeGridLayoutCorroboration(
  candidate: CandidateResult,
  completed: CandidateResult[]
) {
  if (!candidate.layout || !normalizeSupportedEcgLayout(candidate.layout)) {
    return false
  }

  const minimumReviewableLeads = Math.ceil(
    LEAD_ORDER.length * MIN_REVIEW_LAYOUT_LEAD_FRACTION
  )
  return completed.some((peer) => {
    const sourceFidelity = peer.sourceFidelity
    return (
      peer.id !== candidate.id &&
      peer.parameters.vectorizer === "native-grid-path" &&
      peer.parameters.layoutConstraint === peer.layout &&
      peer.layout === candidate.layout &&
      Boolean(sourceFidelity) &&
      (sourceFidelity?.layoutConfidence ?? 0) >=
        MIN_REVIEW_LAYOUT_CONFIDENCE &&
      LEAD_ORDER.filter((lead) => candidateHasReviewableLead(peer, lead))
        .length >= minimumReviewableLeads
    )
  })
}

function candidateHasReviewableLead(
  candidate: CandidateResult,
  lead: string
) {
  const leadQa = candidate.qa?.leads[lead]
  const expectedSamples = leadQa?.expectedSamples ?? 0
  return (
    expectedSamples > 0 &&
    (leadQa?.finiteSamples ?? 0) >=
      Math.ceil(expectedSamples * MIN_REVIEWABLE_LEAD_COVERAGE) &&
    (leadQa?.outsideSegmentSamples ?? 0) === 0
  )
}

function candidateHasAllScorableLeads(candidate: CandidateResult) {
  return LEAD_ORDER.every((lead) => candidateHasScorableLead(candidate, lead))
}

function minimumScorableLeadSamples(expectedSamples: number) {
  return Math.min(
    expectedSamples,
    Math.max(
      MIN_SCORABLE_LEAD_SAMPLES,
      Math.ceil(expectedSamples * MIN_SCORABLE_LEAD_COVERAGE)
    )
  )
}

function candidateMissingSamples(candidate: CandidateResult) {
  return Object.values(candidate.qa?.leads ?? {}).reduce(
    (total, lead) => total + lead.missingSamples,
    0
  )
}

function candidateErrorCount(candidate: CandidateResult) {
  return (
    candidate.qa?.warnings.filter((warning) => warning.severity === "error")
      .length ?? Number.POSITIVE_INFINITY
  )
}

function estimateEffectiveSampleRateHz(
  preprocessing: PreprocessingReport,
  resampleSize: number
) {
  const { width, height } = preprocessing.source
  const landscapeWidth = Math.min(Math.max(width, height), resampleSize)
  return Math.min(
    SAMPLE_RATE_HZ,
    landscapeWidth / PAGE_DURATION_SECONDS
  )
}

function contiguousGaps(values: boolean[]) {
  const gaps: { start: number; end: number; length: number }[] = []
  let start: number | null = null

  values.forEach((finite, index) => {
    if (!finite && start === null) {
      start = index
    }

    if ((finite || index === values.length - 1) && start !== null) {
      const end = finite ? index : index + 1
      gaps.push({ start, end, length: end - start })
      start = null
    }
  })

  return gaps
}

async function readCanonicalCsv(filePath: string): Promise<CanonicalCsv> {
  const text = await fs.readFile(filePath, "utf8")
  const lines = text.trim().split(/\r?\n/)
  const leads = lines[0]?.split(",") ?? []

  if (leads.length === 0) {
    throw new Error("Canonical CSV has no header.")
  }

  const rows = lines.slice(1).map((line) =>
    line.split(",").map((cell) => {
      return parseCanonicalCell(cell)
    })
  )

  return { leads, rows }
}

function parseCanonicalCell(cell: string) {
  if (cell.trim() === "") return Number.NaN
  const value = Number(cell)
  return Number.isFinite(value) ? value : Number.NaN
}

async function writeCanonicalCsv(canonical: CanonicalCsv, outputPath: string) {
  const lines = [canonical.leads.join(",")]
  for (const row of canonical.rows) {
    lines.push(
      row.map((value) => (Number.isFinite(value) ? String(value) : "")).join(",")
    )
  }
  await fs.writeFile(outputPath, `${lines.join("\n")}\n`, "utf8")
}

async function readDigitizerMetadata(filePath: string) {
  const text = await fs.readFile(filePath, "utf8")
  const [headerLine, dataLine] = text.trim().split(/\r?\n/)
  const headers = headerLine.split(",")
  const data = dataLine.split(",")
  const record = Object.fromEntries(headers.map((header, index) => [header, data[index]]))
  const layoutCost = Number(record.matching_cost)

  return {
    layout: record.lead_layout || undefined,
    layoutCost: Number.isFinite(layoutCost) ? layoutCost : undefined,
  }
}

async function writePublishedDigitizerMetadata(
  selected: CandidateResult,
  outputPath: string
) {
  if (!selected.metadataPath) {
    throw new Error("The selected candidate is missing digitizer metadata.")
  }
  if (!selected.parameters.geometryConfirmedLayout || !selected.layout) {
    await fs.copyFile(selected.metadataPath, outputPath)
    return
  }

  const text = await fs.readFile(selected.metadataPath, "utf8")
  const [headerLine, dataLine] = text.trim().split(/\r?\n/)
  const headers = headerLine.split(",")
  const data = dataLine.split(",")
  const layoutIndex = headers.indexOf("lead_layout")
  const costIndex = headers.indexOf("matching_cost")
  if (layoutIndex === -1 || costIndex === -1) {
    throw new Error("Digitizer metadata is missing layout fields.")
  }
  data[layoutIndex] = selected.layout
  data[costIndex] = String(selected.layoutCost ?? 1)
  await fs.writeFile(
    outputPath,
    `${headers.join(",")}\n${data.join(",")}\n`,
    "utf8"
  )
}

async function writeCompactSegmentsCsv(
  canonicalPath: string,
  outputPath: string,
  layout?: string
) {
  const canonical = await readCanonicalCsv(canonicalPath)
  const compactColumns = LEAD_ORDER.map((lead) => {
    if (
      lead === "II" &&
      layout?.toLowerCase().includes("with_r1")
    ) {
      const leadIndex = canonical.leads.indexOf(lead)
      if (leadIndex !== -1) {
        const fullWidth = canonical.rows.map((row) => row[leadIndex])
        const normalizedLayout = normalizeQaLayout(layout)
        const layoutColumns = normalizedLayout
          ? LAYOUT_COLUMNS[normalizedLayout]
          : undefined
        const columnCount = layoutColumns
          ? Math.max(...Object.values(layoutColumns)) + 1
          : 1
        const segmentSamples = Math.floor(
          canonical.rows.length / columnCount
        )
        if (
          fullWidth
            .slice(segmentSamples)
            .some((value) => Number.isFinite(value))
        ) {
          return fullWidth
        }
      }
    }
    const segment = canonicalLeadSegment(canonical, layout, lead)
    if (segment) return segment.values

    const index = canonical.leads.indexOf(lead)
    if (index === -1) return []
    return canonical.rows.map((row) => row[index])
  })
  const maxRows = compactColumns.reduce(
    (max, column) => Math.max(max, column.length),
    0
  )
  const rows = [`sample,t_s_at_500hz,${LEAD_ORDER.join(",")}`]

  for (let sample = 0; sample < maxRows; sample += 1) {
    const values = compactColumns.map((column) =>
      Number.isFinite(column[sample]) ? String(column[sample]) : ""
    )
    rows.push(`${sample},${sample / SAMPLE_RATE_HZ},${values.join(",")}`)
  }

  await fs.writeFile(outputPath, `${rows.join("\n")}\n`, "utf8")
}

function openEcgConfig({
  inputDir,
  outputDir,
  resampleSize,
  labelThresh,
  vectorizer,
  device,
  darkInkEnhancement,
  darkInkSupportRadius,
  layoutConstraint,
  featureCachePath,
}: {
  inputDir: string
  outputDir: string
  resampleSize: number
  labelThresh?: number
  vectorizer: DigitizerVectorizer
  device: DigitizerComputeDevice
  darkInkEnhancement?: boolean
  darkInkSupportRadius?: number
  layoutConstraint?: string
  featureCachePath?: string
}) {
  const signalExtractorKwargs =
    typeof labelThresh === "number"
      ? `\n          label_thresh: ${labelThresh}`
      : " {}"
  const isConstrainedSixRowPanel =
    layoutConstraint?.includes("_6x1_") ?? false
  const signalExtractorClass = isConstrainedSixRowPanel
    ? vectorizer === "dynamic-path"
      ? "ecg_pipeline.fixed_row_signal_extractor.FixedRowPathSignalExtractor"
      : "ecg_pipeline.fixed_row_signal_extractor.FixedRowCentroidSignalExtractor"
    : vectorizer === "dynamic-path"
      ? "ecg_pipeline.reliable_signal_extractor.ReliableSignalExtractor"
      : "src.model.signal_extractor.SignalExtractor"

  const inferenceWrapperClass =
    "ecg_pipeline.fidelity_inference_wrapper.FidelityInferenceWrapper"
  const fidelityKwargs = darkInkEnhancement
    ? isConstrainedSixRowPanel
      ? `
    dark_ink_threshold: 0.35
    dark_ink_strength: 0.75
    dark_ink_support_radius: ${darkInkSupportRadius ?? 128}`
      : `
    dark_ink_strength: 0.5
    dark_ink_support_radius: ${darkInkSupportRadius ?? 48}`
    : `
    dark_ink_strength: 0.0`
  const forcedLayoutKwarg =
    darkInkEnhancement && layoutConstraint
      ? `\n    forced_layout_substring: ${JSON.stringify(layoutConstraint)}`
      : ""
  const leadLabelThresholdKwarg =
    darkInkEnhancement && isConstrainedSixRowPanel
      ? "\n    lead_label_threshold: 0.5"
      : ""
  const featureCacheKwarg = featureCachePath
    ? `\n    feature_cache_path: ${JSON.stringify(featureCachePath)}`
    : ""

  return `MODEL:
  class_path: '${inferenceWrapperClass}'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: '${signalExtractorClass}'
        KWARGS:${signalExtractorKwargs}

      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250

      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1

      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2

      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]
          alpha: 0.85

      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.3

      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: ${JSON.stringify(
          layoutConfigPathForConstraint(layoutConstraint)
        )}
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: '${device}'
          possibly_flipped: false

    device: '${device}'
    resample_size: ${resampleSize}
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false
    ${fidelityKwargs}${forcedLayoutKwarg}${leadLabelThresholdKwarg}${featureCacheKwarg}

DATA:
  images_path: ${JSON.stringify(inputDir)}
  image_extensions: [${DIRECT_DIGITIZER_EXTENSIONS.map((extension) => JSON.stringify(extension)).join(", ")}]
  output_path: ${JSON.stringify(outputDir)}
  save_mode: 'all'
  layout_should_include_substring: ${
    layoutConstraint ? JSON.stringify(layoutConstraint) : "null"
  }
  clear_output_dir_if_exists: true
`
}

function layoutConfigPathForConstraint(layoutConstraint?: string) {
  if (!layoutConstraint) return "src/config/lead_layouts_all.yml"
  if (layoutConstraint === "standard_3x4") {
    return STANDARD_3X4_LAYOUT_CONFIG_PATH
  }
  if (layoutConstraint === "standard_3x4_with_r1") {
    return STANDARD_3X4_WITH_R1_LAYOUT_CONFIG_PATH
  }
  if (layoutConstraint === "standard_6x2") {
    return STANDARD_6X2_LAYOUT_CONFIG_PATH
  }
  if (layoutConstraint === "standard_6x2_with_r1_ignored") {
    return STANDARD_6X2_WITH_R1_IGNORED_LAYOUT_CONFIG_PATH
  }
  if (layoutConstraint === "standard_12x1") {
    return STANDARD_12X1_LAYOUT_CONFIG_PATH
  }
  return RELIABLE_LAYOUT_CONFIG_PATH
}

function parametersForCandidate(candidate: DigitizerCandidateConfig) {
  return {
    kind: candidateKind(candidate),
    capabilities: candidateCapabilities(candidate),
    resampleSize: candidate.resampleSize,
    ...(candidate.upscaleToMaxDimension
      ? { upscaleToMaxDimension: candidate.upscaleToMaxDimension }
      : {}),
    ...(candidate.darkInkEnhancement ? { darkInkEnhancement: true } : {}),
    ...(candidate.darkInkSupportRadius
      ? { darkInkSupportRadius: candidate.darkInkSupportRadius }
      : {}),
    ...(typeof candidate.labelThresh === "number"
      ? { labelThresh: candidate.labelThresh }
      : {}),
    ...(candidate.layoutConstraint
      ? { layoutConstraint: candidate.layoutConstraint }
      : {}),
    ...(candidate.geometryConfirmedLayout
      ? { geometryConfirmedLayout: true }
      : {}),
    ...(typeof candidate.geometryLayoutConfidence === "number"
      ? { geometryLayoutConfidence: candidate.geometryLayoutConfidence }
      : {}),
    ...(candidate.semanticLeadIdentityConfirmed
      ? { semanticLeadIdentityConfirmed: true }
      : {}),
    ...(candidate.semanticLeadIdentityMethod
      ? { semanticLeadIdentityMethod: candidate.semanticLeadIdentityMethod }
      : {}),
    ...(candidate.semanticLeadIdentityOrder
      ? { semanticLeadIdentityOrder: candidate.semanticLeadIdentityOrder }
      : {}),
    ...(candidate.semanticLeadIdentityLayout
      ? { semanticLeadIdentityLayout: candidate.semanticLeadIdentityLayout }
      : {}),
    ...(typeof candidate.adaptivePreprocessingEligible === "boolean"
      ? {
          adaptivePreprocessingEligible:
            candidate.adaptivePreprocessingEligible,
        }
      : {}),
    ...(typeof candidate.artifactPreprocessingEligible === "boolean"
      ? {
          artifactPreprocessingEligible:
            candidate.artifactPreprocessingEligible,
        }
      : {}),
    ...(typeof candidate.maximumLayoutCost === "number"
      ? { maximumLayoutCost: candidate.maximumLayoutCost }
      : {}),
    ...(candidate.cropBox ? { cropBox: candidate.cropBox } : {}),
    ...(typeof candidate.selectionEligible === "boolean"
      ? { selectionEligible: candidate.selectionEligible }
      : {}),
    inputVariant: candidate.inputVariant,
    vectorizer: candidate.vectorizer,
    device: candidate.device,
    ...(candidate.planningPhase
      ? { planningPhase: candidate.planningPhase }
      : {}),
    ...(candidate.scheduleReason
      ? { scheduleReason: candidate.scheduleReason }
      : {}),
  }
}

function selectDigitizerDevice(
  configuredValue: string | undefined,
  mpsAvailable: boolean
): DigitizerComputeDevice {
  const preference = configuredValue?.trim().toLowerCase() || "auto"
  if (!["auto", "cpu", "mps"].includes(preference)) {
    throw new Error(
      "ECG_DIGITIZER_DEVICE must be one of auto, cpu, or mps."
    )
  }
  if (preference === "cpu") return "cpu"
  if (preference === "mps" && !mpsAvailable) {
    throw new Error(
      "ECG_DIGITIZER_DEVICE=mps was requested, but this PyTorch runtime does not expose an available MPS backend."
    )
  }
  return mpsAvailable ? "mps" : "cpu"
}

async function resolveDigitizerDevice(): Promise<DigitizerComputeDevice> {
  const configuredValue = process.env.ECG_DIGITIZER_DEVICE
  const preference = configuredValue?.trim().toLowerCase() || "auto"
  if (preference === "cpu") return "cpu"
  if (!["auto", "mps"].includes(preference)) {
    return selectDigitizerDevice(configuredValue, false)
  }

  try {
    const { stdout } = await execFileAsync(
      openEcgPythonPath(),
      [
        "-c",
        "import torch; print('1' if torch.backends.mps.is_available() else '0')",
      ],
      {
        cwd: openEcgDirPath(),
        env: subprocessEnvironment(),
        maxBuffer: 1024 * 1024,
        timeout: 15_000,
      }
    )
    return selectDigitizerDevice(configuredValue, stdout.trim() === "1")
  } catch (error) {
    if (preference === "mps") {
      throw new Error(
        `MPS availability probe failed: ${
          error instanceof Error ? error.message : "unknown error"
        }`
      )
    }
    return "cpu"
  }
}

function safeOutputStem(stem: string) {
  return stem.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "ecg-upload"
}

function openEcgDirPath() {
  return path.resolve(
    /* turbopackIgnore: true */ RESOURCE_ROOT,
    process.env.OPEN_ECG_DIGITIZER_DIR ??
      [".external", "open-ecg-digitizer"].join(path.sep)
  )
}

function openEcgPythonPath() {
  return path.join(openEcgDirPath(), ".venv", "bin", "python")
}

function publicCandidate(candidate: CandidateResult): DigitizerCandidateSummary {
  return {
    id: candidate.id,
    label: candidate.label,
    localPath: candidate.localPath,
    workingFilesRetained: true,
    leadSources: candidate.leadSources,
    leadCorroborators: candidate.leadCorroborators,
    status: candidate.status,
    parameters: candidate.parameters,
    layout: candidate.layout,
    layoutCost: candidate.layoutCost,
    effectiveSampleRateHz: candidate.effectiveSampleRateHz,
    panelTimingCorrections: candidate.panelTimingCorrections,
    sourceFidelity: candidate.sourceFidelity,
    runtimeMs: candidate.runtimeMs,
    score: candidate.score,
    scoreBreakdown: candidate.scoreBreakdown,
    featureCacheHit: candidate.featureCacheHit,
    qa: candidate.qa,
    message: candidate.message,
  }
}

async function assetFromAbsolutePath(
  label: string,
  absolutePath: string,
  sourceSha256: string
): Promise<RunAsset> {
  const bytes = await fs.readFile(absolutePath)

  return {
    label,
    path: relativePath(absolutePath),
    mimeType: contentTypeForPath(absolutePath),
    sizeBytes: bytes.byteLength,
    identity: {
      version: 1,
      algorithm: "sha256",
      sha256: createHash("sha256").update(bytes).digest("hex"),
      sizeBytes: bytes.byteLength,
      sourceSha256,
      recordedAt: new Date().toISOString(),
    },
  }
}

function contentTypeForPath(filePath: string) {
  const ext = path.extname(filePath).toLowerCase()

  if (ext === ".png") return "image/png"
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg"
  if (ext === ".webp") return "image/webp"
  if (ext === ".tif" || ext === ".tiff") return "image/tiff"
  if (ext === ".csv") return "text/csv; charset=utf-8"
  if (ext === ".json") return "application/json; charset=utf-8"

  return "application/octet-stream"
}

function runDirectory(id: string) {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
    throw new Error("Invalid run id")
  }

  return path.join(
    /* turbopackIgnore: true */ ROOT_DIR,
    "storage",
    "runs",
    id
  )
}

function resolveWorkspacePath(relativeFilePath: string) {
  const absolutePath = path.resolve(
    /* turbopackIgnore: true */ ROOT_DIR,
    relativeFilePath
  )

  if (absolutePath !== ROOT_DIR && !absolutePath.startsWith(`${ROOT_DIR}${path.sep}`)) {
    throw new Error("Path escapes workspace")
  }

  return absolutePath
}

function relativePath(absolutePath: string) {
  return path.relative(ROOT_DIR, absolutePath).split(path.sep).join("/")
}

export const digitizerTestUtils = {
  AsyncSemaphore,
  alignSeries,
  annotationRangesByLead,
  assetsWithoutQuantitativeOutputs,
  attachSemanticLeadEvidence,
  buildUncertaintyRows,
  candidateEligibleForSelection,
  candidatePublicationRank,
  hasLowResolutionNativeTraceCorroboration,
  selectAdaptivePreprocessedCandidate,
  candidateHasAllScorableLeads,
  candidateHasAllTrustedLeads,
  candidateHasCorroboratedFullWidthRhythmLead,
  candidateHasPublishableLead,
  candidateHasExplicitSemanticLayoutEvidence,
  candidateInheritsProvisionalGeometryLeadIdentity,
  candidateHasProvisionalLayoutLeadIdentity,
  candidateHasNativeRasterColumnSamplingRisk,
  candidateHasSemanticLeadIdentity,
  candidateHasSourceVerifiedBoundaryLead,
  candidateHasValidatedBoundaryLayout,
  candidateHasStructurallyCompleteLead,
  candidateHasTrustedLead,
  candidatePathsAreIndependent,
  canonicalLeadSegment,
  combineSixByTwoPanelCanonicals,
  combineStackedPanelCanonicals,
  deterministicCropProposals,
  evaluateQa,
  estimateEffectiveSampleRateHz,
  fuseLeadCandidates,
  fusionDisagreementPenalty,
  geometryRasterDimensions,
  largestCompatibleCandidateGroup,
  layoutConfigPathForConstraint,
  labelConfirmedRetryLayouts: LABEL_CONFIRMED_RETRY_LAYOUTS,
  mpsConfirmationAssignments,
  normalizeCanonicalPanelTiming,
  nativeCanonicalRequiresBoundarySuppression,
  openEcgConfig,
  parseCanonicalCell,
  repairShortPeerSupportedGaps,
  sanitizeCanonicalForLayout,
  selectLeadSourceCandidate,
  selectReviewableConstrainedCandidate,
  selectPartialLeadCandidate,
  suppressConstrainedPanelBoundaries,
  suppressFullWidthPanelBoundaries,
  validatedCalibration,
  suppressNearFlatLeads,
  selectDigitizerDevice,
  selectSafestCandidate,
  stabilityCandidateConfig,
  stabilityRequiredForPublicationOutcome,
  suppressAnnotatedSamples,
}
