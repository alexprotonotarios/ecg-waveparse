import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { existsSync } from "node:fs"
import { promises as fs } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"

import {
  DIGITIZER_POLICY_ID,
  DIGITIZER_POLICY_VERSION,
} from "@/lib/digitizer/policy"
import type {
  DigitizerLeadQa,
  DigitizerQaWarning,
  RunAsset,
  RunRecord,
} from "@/lib/runs"
import { RESOURCE_ROOT, WORKSPACE_ROOT } from "@/lib/runtime-paths"

const execFileAsync = promisify(execFile)

function resolveNeuralPythonPath(): string {
  const oedDir = process.env.OPEN_ECG_DIGITIZER_DIR?.trim()
  if (oedDir) {
    const candidate = path.join(oedDir, ".venv", "bin", "python")
    if (existsSync(candidate)) return candidate
  }
  return path.join(WORKSPACE_ROOT, ".external/open-ecg-digitizer/.venv/bin/python")
}

function resolveCleanroomScriptPath(): string {
  const packaged = path.join(RESOURCE_ROOT, "scripts/digitize_cleanroom.py")
  if (existsSync(packaged)) return packaged
  return path.join(WORKSPACE_ROOT, "scripts/digitize_cleanroom.py")
}

function resolveCleanroomWeights(): { rectifier: string; decoder: string } | null {
  const roots = [
    process.env.WAVEPARSE_CLEANROOM_WEIGHTS_DIR?.trim(),
    process.env.OPEN_ECG_DIGITIZER_DIR?.trim()
      ? path.join(process.env.OPEN_ECG_DIGITIZER_DIR.trim(), "cleanroom")
      : undefined,
    path.join(WORKSPACE_ROOT, ".external/waveparse-cleanroom-v1"),
    "/Volumes/SeagatePort/alex/WaveParse-CleanRoom-v1",
  ].filter((x): x is string => Boolean(x))

  const rectNames = [
    "grid-rectifier/waveparse-grid-rectifier-v3-fp16.pt",
    "waveparse-grid-rectifier-v3-fp16.pt",
    "grid-rectifier/waveparse-grid-rectifier-v2-fp16.pt",
    "waveparse-grid-rectifier-v2-fp16.pt",
  ]
  const decNames = [
    "lead-decoder/waveparse-lead-decoder-v1-fp16.pt",
    "waveparse-lead-decoder-v1-fp16.pt",
  ]

  let rectifier: string | null = null
  let decoder: string | null = null
  for (const r of roots) {
    if (!rectifier) {
      for (const rel of rectNames) {
        const p = path.join(r, rel)
        if (existsSync(p)) {
          rectifier = p
          break
        }
      }
    }
    if (!decoder) {
      for (const rel of decNames) {
        const p = path.join(r, rel)
        if (existsSync(p)) {
          decoder = p
          break
        }
      }
    }
  }
  return rectifier && decoder ? { rectifier, decoder } : null
}

const NEURAL_SCRIPT_PATH = path.join(WORKSPACE_ROOT, "scripts/digitize_ecg.py")
const STAGE0_WEIGHT_PATH = path.join(
  WORKSPACE_ROOT,
  ".external/dense-grid-frontend-20261109-v1/weight/stage0-last.checkpoint.pth"
)
const STAGE1_WEIGHT_PATH = path.join(
  WORKSPACE_ROOT,
  ".external/dense-grid-frontend-20261109-v1/weight/stage1-last.checkpoint.pth"
)
const COAT_VGG_WEIGHT_PATH = path.join(
  WORKSPACE_ROOT,
  ".external/heatmap-decoder-20261115-v1/assets/FINAL_01-19__21-15-59.746642_ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611_ep3_step70007_val_SNR28.006985.ckpt"
)
const STAGE2_WEIGHT_PATH = path.join(
  WORKSPACE_ROOT,
  ".external/dense-grid-frontend-20261109-v1/decoder-review-v1/weight/stage2-00005810.checkpoint.pth"
)

type NeuralMetadata = {
  engine: string
  layout?: string
  inputShapeHW: [number, number]
  calibration?: {
    paperSpeedMmPerSecond: number
    gainMmPerMv: number
    effectiveSampleRateHz: number
    calibrationSource: string
    confidence: number
    conflict: boolean
    warnings: string[]
  }
  decoder: {
    decoder: string
    device: string
    clippedLeadsRescued?: string[]
  }
  clinicalQuality: {
    disposition: "usable" | "usable_with_warnings" | "review_required"
    warnings: string[]
    geometry?: {
      stage0Inliers: number
      gridObserved: number
      gridTotal: number
      gridFraction: number
      orientationRot90: number
    }
    physicsSelfChecks?: {
      shortVsLongII: { rmseUv: number | null; corr: number | null; shiftSamples: number | null }
      einthovenIPlusIIIvsII: { rmseUv: number | null; corr: number | null; shiftSamples: number | null }
      goldbergerSumRmsUv: number | null
    }
    perLead?: Record<
      string,
      {
        coverage: number
        meanConfidence: number
        p05Confidence: number
        gapIntervalsSec: [number, number][]
        lowConfidenceIntervalsSec: [number, number][]
        tallAmplitudeRescued: boolean
      }
    >
  }
  uncertaintySummary?: {
    uncertainSampleCount: number
    missingSampleCount: number
  }
  rectSeconds: number
  decodeSeconds: number
  totalSeconds: number
}

export type NeuralDecoderMode = "cleanroom" | "coat_vgg" | "stage2"

export function configuredNeuralDecoderMode(): NeuralDecoderMode | null {
  const raw = (process.env.WAVEPARSE_NEURAL_ENGINE ?? "auto").trim().toLowerCase()
  if (raw === "off" || raw === "false" || raw === "0" || raw === "legacy") {
    return null
  }
  const pyPath = resolveNeuralPythonPath()
  if (!existsSync(pyPath)) {
    return null
  }
  const cleanroomScript = resolveCleanroomScriptPath()
  const cleanroomWeights = resolveCleanroomWeights()
  if (raw === "cleanroom") {
    return existsSync(cleanroomScript) && cleanroomWeights ? "cleanroom" : null
  }
  if (raw === "auto" && existsSync(cleanroomScript) && cleanroomWeights) {
    return "cleanroom"
  }
  if (!existsSync(NEURAL_SCRIPT_PATH)) {
    return null
  }
  if (!existsSync(STAGE0_WEIGHT_PATH) || !existsSync(STAGE1_WEIGHT_PATH)) {
    return null
  }
  if (raw === "stage2") {
    return existsSync(STAGE2_WEIGHT_PATH) ? "stage2" : null
  }
  if (existsSync(COAT_VGG_WEIGHT_PATH)) {
    return "coat_vgg"
  }
  if (existsSync(STAGE2_WEIGHT_PATH)) {
    return "stage2"
  }
  return null
}

async function signedAsset(
  label: string,
  absolutePath: string,
  mimeType: string,
  sourceSha256: string
): Promise<RunAsset> {
  const bytes = await fs.readFile(absolutePath)
  const rel = path.relative(WORKSPACE_ROOT, absolutePath).split(path.sep).join("/")
  return {
    label,
    path: rel,
    mimeType,
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

/**
 * Attempt end-to-end neural 3x4 digitization (DenseGrid TPS + CoaT/VGG19 or Stage 2).
 *
 * Returns `null` when the engine is disabled/unavailable OR when the image is not a
 * standard 3x4+rhythm-II layout (exit code 3), allowing the caller to fall back cleanly
 * to Open-ECG-Digitizer (e.g. for 6x2 scans).
 */
export async function tryNeuralDigitizeRun(
  run: RunRecord,
  verifiedSourceSha256: string,
  options: { signal?: AbortSignal } = {}
): Promise<Partial<RunRecord> | null> {
  const decoderMode = configuredNeuralDecoderMode()
  if (!decoderMode || !run.assets.input) {
    return null
  }

  const sourcePath = path.resolve(WORKSPACE_ROOT, run.assets.input.path)
  const exportsDir = path.join(WORKSPACE_ROOT, "storage", "runs", run.id, "exports")
  await fs.mkdir(exportsDir, { recursive: true })

  const startedAt = performance.now()
  const pyPath = resolveNeuralPythonPath()
  const cleanroomWeights = decoderMode === "cleanroom" ? resolveCleanroomWeights() : null
  const args =
    decoderMode === "cleanroom"
      ? [
          resolveCleanroomScriptPath(),
          sourcePath,
          "--out-dir",
          exportsDir,
          "--device",
          process.env.ECG_DIGITIZER_DEVICE ?? "auto",
          ...(cleanroomWeights
            ? ["--rectifier-weights", cleanroomWeights.rectifier, "--decoder-weights", cleanroomWeights.decoder]
            : []),
          "--export-wfdb",
          "--export-json",
        ]
      : [
          NEURAL_SCRIPT_PATH,
          sourcePath,
          "--out-dir",
          exportsDir,
          "--decoder",
          decoderMode,
          "--layout",
          path.extname(sourcePath).toLowerCase() === ".pdf" ? "auto" : "3x4",
          "--export-wfdb",
          "--export-json",
        ]

  try {
    await execFileAsync(pyPath, args, {
      cwd: WORKSPACE_ROOT,
      signal: options.signal,
      maxBuffer: 10 * 1024 * 1024,
      env: {
        ...process.env,
        PYTHONPATH: [RESOURCE_ROOT, WORKSPACE_ROOT, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        PYTORCH_ENABLE_MPS_FALLBACK: "1",
        MPLBACKEND: "Agg",
      },
    })
  } catch (error) {
    const code =
      typeof error === "object" && error !== null && "code" in error
        ? (error as { code?: number | string }).code
        : undefined
    // Exit code 3 = explicit LayoutNotSupportedError (e.g., 6x2 scan) -> clean fallback
    if (code === 3) {
      return null
    }
    // Any abort must propagate; unexpected errors also fall back gracefully so uploads are never bricked
    if (options.signal?.aborted) {
      throw error
    }
    return null
  }

  const metadataPath = path.join(exportsDir, "metadata.json")
  const rawMeta = await fs.readFile(metadataPath, "utf8")
  const meta = JSON.parse(rawMeta) as NeuralMetadata
  const [sourceHeight, sourceWidth] = meta.inputShapeHW
  const cq = meta.clinicalQuality
  const layout = meta.layout ?? "standard_3x4_with_r1"
  const candidateId =
    meta.engine === "waveparse_vector_pdf_lossless"
      ? "waveparse-vector-pdf-lossless"
      : decoderMode === "cleanroom"
        ? "waveparse-cleanroom-mit-v1"
        : decoderMode === "coat_vgg"
          ? "neural-densegrid-coat-vgg19"
          : "neural-densegrid-stage2"
  const candidateLabel =
    meta.engine === "waveparse_vector_pdf_lossless"
      ? "WaveParse Lossless Vector PDF (MIT)"
      : decoderMode === "cleanroom"
        ? "WaveParse Clean-Room MIT (Universal Any-Layout)"
        : decoderMode === "coat_vgg"
          ? "DenseGrid TPS + CoaT/VGG19 (1024px)"
          : "DenseGrid TPS + Stage 2 (fast)"

  // Map clinical quality warnings & per-lead coverage into WaveParse QA schema
  const warnings: DigitizerQaWarning[] = cq.warnings.map((w) => ({
    severity: cq.disposition === "review_required" ? "error" : "warning",
    message: w,
  }))
  const qaLeads: Record<string, DigitizerLeadQa> = {}
  for (const [lead, info] of Object.entries(cq.perLead ?? {})) {
    const expected = lead === "II" ? 5000 : 1250
    const finiteSamples = Math.round(info.coverage * expected)
    const missingSamples = Math.max(0, expected - finiteSamples)
    qaLeads[lead] = {
      expectedSamples: expected,
      finiteSamples,
      missingSamples,
      maxGapSamples: 0,
      outsideSegmentSamples: 0,
    }
  }

  const shortRmse = cq.physicsSelfChecks?.shortVsLongII.rmseUv ?? null
  const einthovenRmse = cq.physicsSelfChecks?.einthovenIPlusIIIvsII.rmseUv ?? null
  const gridFrac = cq.geometry?.gridFraction ?? 1
  const physicsSummary = [
    shortRmse !== null ? `Short-vs-Long II ${shortRmse.toFixed(1)} µV` : null,
    einthovenRmse !== null ? `Einthoven ${einthovenRmse.toFixed(1)} µV` : null,
    cq.geometry ? `grid ${cq.geometry.gridObserved}/${cq.geometry.gridTotal}` : meta.engine,
  ]
    .filter(Boolean)
    .join(", ")

  const qaPassed = cq.disposition !== "review_required"
  const runtimeMs = Math.round(performance.now() - startedAt)

  const [diagnostic, paperRender, canonicalCsv, segmentsCsv, uncertaintyCsv, provenanceJson] =
    await Promise.all([
      signedAsset(
        "Untouched-source trace overlay",
        path.join(exportsDir, "source-overlay.png"),
        "image/png",
        verifiedSourceSha256
      ),
      signedAsset(
        "ECG-paper render",
        path.join(exportsDir, "paper-render.png"),
        "image/png",
        verifiedSourceSha256
      ),
      signedAsset(
        "Canonical CSV",
        path.join(exportsDir, "canonical-uv.csv"),
        "text/csv; charset=utf-8",
        verifiedSourceSha256
      ),
      signedAsset(
        "Compact 500 Hz CSV",
        path.join(exportsDir, "segments-500hz.csv"),
        "text/csv; charset=utf-8",
        verifiedSourceSha256
      ),
      signedAsset(
        "Sample-level uncertainty",
        path.join(exportsDir, "uncertainty-500hz.csv"),
        "text/csv; charset=utf-8",
        verifiedSourceSha256
      ),
      signedAsset(
        "Digitization metadata & clinical quality",
        metadataPath,
        "application/json; charset=utf-8",
        verifiedSourceSha256
      ),
    ])

  const wfdbHeaPath = path.join(exportsDir, "digitized_ecg.hea")
  const wfdbDatPath = path.join(exportsDir, "digitized_ecg.dat")
  const downstreamJsonPath = path.join(exportsDir, "downstream-ecg.json")
  const [wfdbHeader, wfdbSignal, downstreamJson] = await Promise.all([
    existsSync(wfdbHeaPath)
      ? signedAsset("PhysioNet WFDB Header (.hea)", wfdbHeaPath, "text/plain; charset=utf-8", verifiedSourceSha256)
      : Promise.resolve(undefined),
    existsSync(wfdbDatPath)
      ? signedAsset("PhysioNet WFDB Signal (.dat)", wfdbDatPath, "application/octet-stream", verifiedSourceSha256)
      : Promise.resolve(undefined),
    existsSync(downstreamJsonPath)
      ? signedAsset("Downstream Medical JSON", downstreamJsonPath, "application/json; charset=utf-8", verifiedSourceSha256)
      : Promise.resolve(undefined),
  ])

  const confidenceReasons: string[] = []
  if (cq.warnings.length > 0) {
    confidenceReasons.push(...cq.warnings)
  }

  const paperSpeed = meta.calibration?.paperSpeedMmPerSecond ?? 25
  const gainMmPerMv = meta.calibration?.gainMmPerMv ?? 10
  const effectiveSampleRateHz = meta.calibration?.effectiveSampleRateHz ?? 500
  const calibrationConfidence = Number(
    (meta.calibration?.confidence ?? gridFrac).toFixed(4)
  )

  return {
    status: "needs_review",
    publicationDecision: {
      policyId: DIGITIZER_POLICY_ID,
      policyVersion: DIGITIZER_POLICY_VERSION,
      outcome: "needs_review",
      reasonCode: cq.disposition === "usable" ? "source_verified" : "reviewable_constrained",
      candidateId,
    },
    message:
      cq.disposition === "usable"
        ? `WaveParse Neural (${candidateLabel}) completed locally (${paperSpeed} mm/s, ${gainMmPerMv} mm/mV, ${physicsSummary}). Visual verification against the source overlay is required before clinical use.`
        : `WaveParse Neural (${candidateLabel}) completed with ${cq.warnings.length} clinical quality warning(s) (${paperSpeed} mm/s, ${gainMmPerMv} mm/mV, ${physicsSummary}). Compare the source overlay before clinical use.`,
    layout,
    layoutCost: Number((1 - gridFrac).toFixed(4)),
    sampleRateHz: 500,
    effectiveSampleRateHz,
    paperSpeedMmPerSecond: paperSpeed,
    gainMmPerMv,
    calibrationConfidence,
    leadCount: 12,
    selectedCandidateId: candidateId,
    digitizer: {
      engine:
        meta.engine === "waveparse_vector_pdf_lossless"
          ? "WaveParse-Vector-PDF-MIT"
          : decoderMode === "cleanroom"
            ? "WaveParse-CleanRoom-MIT"
            : "WaveParse-Neural-3x4",
      selectedCandidateId: candidateId,
      candidates: [
        {
          id: candidateId,
          label: candidateLabel,
          selected: true,
          status: "completed",
          parameters: {
            resampleSize: decoderMode === "stage2" ? 1696 : 1024,
            rotateOnResample: true,
            inputVariant: "original",
            planningPhase: "core",
          },
          layout,
          layoutCost: Number((1 - gridFrac).toFixed(4)),
          effectiveSampleRateHz,
          runtimeMs,
          qa: {
            passed: qaPassed,
            summary: `${cq.disposition} (${paperSpeed} mm/s, ${gainMmPerMv} mm/mV, ${physicsSummary})`,
            warnings,
            leads: qaLeads,
          },
        },
      ],
    },
    reliability: {
      version: 1,
      originalPreserved: true,
      morphologyReconstructed: false,
      sourceSha256: verifiedSourceSha256,
      sourceWidth,
      sourceHeight,
      annotationMaskedPixels: 0,
      annotationMaskedFraction: 0,
      annotationComponents: [],
      nominalSampleRateHz: 500,
      effectiveSampleRateHz,
      uncertainSampleCount: meta.uncertaintySummary?.uncertainSampleCount ?? 0,
      missingSampleCount: meta.uncertaintySummary?.missingSampleCount ?? 0,
      confidence: cq.disposition === "usable" ? "standard" : "lower",
      ...(confidenceReasons.length > 0 ? { confidenceReasons } : {}),
      reviewRequired: true,
    },
    assets: {
      ...run.assets,
      diagnostic,
      paperRender,
      canonicalCsv,
      segmentsCsv,
      uncertaintyCsv,
      ...(wfdbHeader ? { wfdbHeader } : {}),
      ...(wfdbSignal ? { wfdbSignal } : {}),
      ...(downstreamJson ? { downstreamJson } : {}),
      provenanceJson,
    },
    updatedAt: new Date().toISOString(),
  }
}
