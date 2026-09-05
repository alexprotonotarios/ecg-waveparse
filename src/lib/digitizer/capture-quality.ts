import captureQualityDocument from "../../../config/capture-quality.v1.json"

import type { PreprocessingReport } from "@/lib/digitizer/contracts"

export type CaptureMethod = "camera" | "upload"
export type CaptureCheckStatus = "pass" | "warning" | "fail"

export type CaptureQualityCheck = {
  id:
    | "resolution"
    | "focus"
    | "contrast"
    | "framing"
    | "perspective"
    | "glare"
    | "source"
  label: string
  status: CaptureCheckStatus
  detail: string
  guidance?: string
}

export type CaptureQualitySummary = {
  version: 1
  policyId: string
  checkedAt: string
  captureMethod: CaptureMethod
  outcome: "ready" | "retake"
  sourceSha256: string
  source: {
    width: number
    height: number
    format?: string
  }
  metrics: {
    sharpnessLaplacianVariance?: number
    perspectiveStrength?: number
    rotationDegrees?: number
    pageAreaFraction?: number
    glareFraction?: number
    chromaMoireScore?: number
  }
  checks: CaptureQualityCheck[]
}

export type ClientCaptureMetrics = {
  width: number
  height: number
  sharpnessLaplacianVariance: number
  contrastRange: number
  luminanceP05: number
  luminanceP50: number
  luminanceP95: number
  inkWidthFraction: number
  inkHeightFraction: number
}

export type ClientCaptureAssessment = {
  outcome: "ready" | "retake" | "server-check-required"
  metrics?: ClientCaptureMetrics
  checks: CaptureQualityCheck[]
}

export const CAPTURE_QUALITY_POLICY_ID = captureQualityDocument.profileId
export const CAPTURE_QUALITY_THRESHOLDS = Object.freeze(
  captureQualityDocument.thresholds
)

const CAPTURE_CHECK_IDS = new Set<CaptureQualityCheck["id"]>([
  "resolution",
  "focus",
  "contrast",
  "framing",
  "perspective",
  "glare",
  "source",
])
const CAPTURE_CHECK_STATUSES = new Set<CaptureCheckStatus>([
  "pass",
  "warning",
  "fail",
])

export function isCaptureQualitySummary(
  value: unknown
): value is CaptureQualitySummary {
  if (!isRecord(value)) return false
  const source = value.source
  const metrics = value.metrics
  const checks = value.checks
  if (
    value.version !== 1 ||
    typeof value.policyId !== "string" ||
    value.policyId.length === 0 ||
    typeof value.checkedAt !== "string" ||
    !Number.isFinite(Date.parse(value.checkedAt)) ||
    (value.captureMethod !== "camera" && value.captureMethod !== "upload") ||
    (value.outcome !== "ready" && value.outcome !== "retake") ||
    typeof value.sourceSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.sourceSha256) ||
    !isRecord(source) ||
    !Number.isSafeInteger(source.width) ||
    Number(source.width) <= 0 ||
    !Number.isSafeInteger(source.height) ||
    Number(source.height) <= 0 ||
    (source.format !== undefined && typeof source.format !== "string") ||
    !isRecord(metrics) ||
    !captureMetricsAreValid(metrics) ||
    !Array.isArray(checks) ||
    checks.length === 0
  ) {
    return false
  }
  const validChecks = checks.every((check) => {
    if (!isRecord(check)) return false
    return (
      typeof check.id === "string" &&
      CAPTURE_CHECK_IDS.has(check.id as CaptureQualityCheck["id"]) &&
      typeof check.label === "string" &&
      check.label.length > 0 &&
      typeof check.status === "string" &&
      CAPTURE_CHECK_STATUSES.has(check.status as CaptureCheckStatus) &&
      typeof check.detail === "string" &&
      check.detail.length > 0 &&
      (check.guidance === undefined || typeof check.guidance === "string")
    )
  })
  if (!validChecks) return false
  const hasFailure = checks.some(
    (check) => isRecord(check) && check.status === "fail"
  )
  return value.outcome === "retake" ? hasFailure : !hasFailure
}

export function evaluatePreprocessingCaptureQuality(
  report: PreprocessingReport,
  captureMethod: CaptureMethod,
  checkedAt = new Date().toISOString()
): CaptureQualitySummary {
  const inputQuality = report.inputQuality
  const adaptive = report.adaptivePreprocessing
  const artifacts = report.artifactPreprocessing
  const geometry = report.geometryCorrection
  const longEdge = Math.max(report.source.width, report.source.height)
  const shortEdge = Math.min(report.source.width, report.source.height)
  const aspectRatio = longEdge / Math.max(shortEdge, 1)
  const cameraShortEdge =
    aspectRatio >= CAPTURE_QUALITY_THRESHOLDS.highAspectRatioThreshold
      ? CAPTURE_QUALITY_THRESHOLDS.minimumCameraHighAspectShortEdgePixels
      : CAPTURE_QUALITY_THRESHOLDS.minimumCameraShortEdgePixels
  const minimumLongEdge =
    captureMethod === "camera"
      ? CAPTURE_QUALITY_THRESHOLDS.minimumCameraLongEdgePixels
      : (inputQuality?.minimumQuantitativeLongEdgePixels ?? 1500)
  const minimumShortEdge =
    captureMethod === "camera"
      ? cameraShortEdge
      : effectiveUploadShortEdge(report, aspectRatio)
  const checks: CaptureQualityCheck[] = []

  checks.push(
    longEdge >= minimumLongEdge && shortEdge >= minimumShortEdge
      ? passing(
          "resolution",
          "Native resolution",
          `${report.source.width} × ${report.source.height} px preserves enough source detail.`
        )
      : warning(
          "resolution",
          "Native resolution",
          `${report.source.width} × ${report.source.height} px is below the ${minimumLongEdge} × ${minimumShortEdge} px acquisition floor.`,
          captureMethod === "camera"
            ? "Move closer and retake with the rear camera at full resolution."
            : "Use the original scan or export rather than a screenshot or compressed copy."
        )
  )

  const sharpness = adaptive?.sharpnessLaplacianVariance
  const focusPassed = Boolean(
    adaptive &&
      adaptive.blurRejected !== true &&
      (adaptive.traceDominantSmoothSource === true ||
        sharpness === undefined ||
        sharpness >=
          CAPTURE_QUALITY_THRESHOLDS.minimumCaptureSharpnessLaplacianVariance)
  )
  checks.push(
    focusPassed
      ? passing(
          "focus",
          "Focus and motion",
          sharpness === undefined
            ? "No supported blur warning was detected."
            : `Edge sharpness ${sharpness.toFixed(0)} passed the capture floor.`
        )
      : warning(
          "focus",
          "Focus and motion",
          `Edge sharpness ${Number(sharpness ?? 0).toFixed(0)} indicates lost trace detail.`,
          "Clean the lens, brace the phone, tap the ECG to focus, and retake."
        )
  )

  const perspectiveStrength = geometry?.perspectiveStrength ?? 0
  const rotationDegrees = Math.abs(geometry?.rotationDegrees ?? 0)
  const perspectivePassed =
    perspectiveStrength <
      CAPTURE_QUALITY_THRESHOLDS.maximumCapturePerspectiveStrength &&
    rotationDegrees < CAPTURE_QUALITY_THRESHOLDS.maximumCaptureRotationDegrees
  checks.push(
    perspectivePassed
      ? passing(
          "perspective",
          "Camera alignment",
          "The ECG is sufficiently square to the camera."
        )
      : warning(
          "perspective",
          "Camera alignment",
          `Detected ${formatPercent(perspectiveStrength)} perspective distortion and ${rotationDegrees.toFixed(1)}° rotation.`,
          "Place the ECG flat and hold the camera directly above it with all four edges parallel to the guide."
        )
  )

  const pageArea = geometry?.pageAreaFraction
  const framingPassed =
    pageArea === undefined ||
    geometry?.pageBoundaryCrop !== true ||
    pageArea >= CAPTURE_QUALITY_THRESHOLDS.minimumDetectedPageAreaFraction
  checks.push(
    framingPassed
      ? passing(
          "framing",
          "Page framing",
          pageArea === undefined
            ? "No undersized page boundary was detected."
            : `The ECG occupies ${formatPercent(pageArea)} of the frame.`
        )
      : warning(
          "framing",
          "Page framing",
          `The ECG occupies only ${formatPercent(pageArea ?? 0)} of the frame.`,
          "Move closer while keeping the entire ECG, labels, and calibration pulse visible."
        )
  )

  checks.push(
    artifacts?.glareLikely !== true
      ? passing("glare", "Lighting", "No supported glare pattern was detected.")
      : warning(
          "glare",
          "Lighting",
          `Glare covers ${formatPercent(artifacts.glareFraction)} of the corrected page.`,
          "Turn off flash and use diffuse light from both sides before retaking."
        )
  )

  checks.push(
    artifacts?.screenArtifactLikely !== true
      ? passing(
          "source",
          "Original source",
          "No screen moiré pattern was detected."
        )
      : warning(
          "source",
          "Original source",
          "A screen or display moiré pattern was detected.",
          "Photograph the paper ECG or import the original image/PDF export instead of photographing a screen."
        )
  )

  if (inputQuality?.quantitativeEligible === false) {
    replaceOrAppendWarning(
      checks,
      "resolution",
      "Native source information",
      "The source is below the preferred native-information target; extraction will continue with lower-confidence fidelity reporting.",
      "For higher fidelity, retake from the original paper at full resolution without digital zoom."
    )
  }

  return {
    version: 1,
    policyId: CAPTURE_QUALITY_POLICY_ID,
    checkedAt,
    captureMethod,
    outcome: checks.some((check) => check.status === "fail")
      ? "retake"
      : "ready",
    sourceSha256: report.sourceSha256,
    source: report.source,
    metrics: {
      ...(sharpness === undefined
        ? {}
        : { sharpnessLaplacianVariance: sharpness }),
      ...(geometry
        ? {
            perspectiveStrength,
            rotationDegrees: geometry.rotationDegrees,
            ...(pageArea === undefined ? {} : { pageAreaFraction: pageArea }),
          }
        : {}),
      ...(artifacts
        ? {
            glareFraction: artifacts.glareFraction,
            chromaMoireScore: artifacts.chromaMoireScore,
          }
        : {}),
    },
    checks,
  }
}

export function evaluateClientCaptureMetrics(
  metrics: ClientCaptureMetrics,
  captureMethod: CaptureMethod
): ClientCaptureAssessment {
  const longEdge = Math.max(metrics.width, metrics.height)
  const shortEdge = Math.min(metrics.width, metrics.height)
  const aspectRatio = longEdge / Math.max(shortEdge, 1)
  const minimumLongEdge =
    captureMethod === "camera"
      ? CAPTURE_QUALITY_THRESHOLDS.minimumCameraLongEdgePixels
      : 1500
  const minimumShortEdge =
    captureMethod === "camera"
      ? aspectRatio >= CAPTURE_QUALITY_THRESHOLDS.highAspectRatioThreshold
        ? CAPTURE_QUALITY_THRESHOLDS.minimumCameraHighAspectShortEdgePixels
        : CAPTURE_QUALITY_THRESHOLDS.minimumCameraShortEdgePixels
      : aspectRatio >= CAPTURE_QUALITY_THRESHOLDS.highAspectRatioThreshold
        ? 450
        : 700
  const checks: CaptureQualityCheck[] = []

  checks.push(
    longEdge >= minimumLongEdge && shortEdge >= minimumShortEdge
      ? passing(
          "resolution",
          "Native resolution",
          `${metrics.width} × ${metrics.height} px`
        )
      : warning(
          "resolution",
          "Native resolution",
          `${metrics.width} × ${metrics.height} px is below the capture floor.`,
          captureMethod === "camera"
            ? "Move closer and use the rear camera at full resolution."
            : "Choose the original scan or image export."
        )
  )

  const sharpnessStatus: CaptureCheckStatus =
    metrics.sharpnessLaplacianVariance <
    CAPTURE_QUALITY_THRESHOLDS.clientSevereBlurLaplacianVariance
      ? "warning"
      : metrics.sharpnessLaplacianVariance <
          CAPTURE_QUALITY_THRESHOLDS.clientMinimumSharpnessLaplacianVariance
        ? "warning"
        : "pass"
  checks.push({
    id: "focus",
    label: "Focus and motion",
    status: sharpnessStatus,
    detail: `Live sharpness ${metrics.sharpnessLaplacianVariance.toFixed(0)}.`,
    ...(sharpnessStatus === "pass"
      ? {}
      : {
          guidance:
            "Hold still, tap the ECG to focus, and wait for the paper grid to look crisp.",
        }),
  })

  const contrastStatus: CaptureCheckStatus =
    metrics.contrastRange <
    CAPTURE_QUALITY_THRESHOLDS.clientSevereContrastRange
      ? "warning"
      : metrics.contrastRange <
          CAPTURE_QUALITY_THRESHOLDS.clientMinimumContrastRange
        ? "warning"
        : "pass"
  checks.push({
    id: "contrast",
    label: "Light and contrast",
    status: contrastStatus,
    detail: `Luminance range ${metrics.contrastRange.toFixed(0)} levels.`,
    ...(contrastStatus === "pass"
      ? {}
      : {
          guidance:
            "Use brighter, even light without flash and avoid casting a shadow over the ECG.",
        }),
  })

  const framingPassed =
    metrics.inkWidthFraction >=
      CAPTURE_QUALITY_THRESHOLDS.clientMinimumInkWidthFraction &&
    metrics.inkHeightFraction >=
      CAPTURE_QUALITY_THRESHOLDS.clientMinimumInkHeightFraction
  checks.push(
    framingPassed
      ? passing(
          "framing",
          "ECG framing",
          "Trace content fills the capture guide."
        )
      : warning(
          "framing",
          "ECG framing",
          "The ECG trace area is too small or incomplete in the frame.",
          "Move closer while keeping every lead label and the calibration pulse visible."
        )
  )

  return {
    // These browser measurements describe likely fidelity, not whether a
    // visible ECG trace is recoverable. The server and extractor make that
    // decision from the submitted still; proxy warnings must not block it.
    outcome: checks.some((check) => check.status === "fail")
      ? "retake"
      : "ready",
    metrics,
    checks,
  }
}

export async function analyzeCaptureFile(
  file: File,
  captureMethod: CaptureMethod
): Promise<ClientCaptureAssessment> {
  if (typeof createImageBitmap !== "function") {
    return { outcome: "server-check-required", checks: [] }
  }
  let bitmap: ImageBitmap | undefined
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" })
    const scale = Math.min(
      1,
      CAPTURE_QUALITY_THRESHOLDS.clientAnalysisLongEdgePixels /
        Math.max(bitmap.width, bitmap.height)
    )
    const width = Math.max(1, Math.round(bitmap.width * scale))
    const height = Math.max(1, Math.round(bitmap.height * scale))
    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext("2d", { willReadFrequently: true })
    if (!context) {
      return { outcome: "server-check-required", checks: [] }
    }
    context.drawImage(bitmap, 0, 0, width, height)
    const metrics = captureMetricsFromImageData(
      context.getImageData(0, 0, width, height),
      bitmap.width,
      bitmap.height
    )
    return evaluateClientCaptureMetrics(metrics, captureMethod)
  } catch {
    return { outcome: "server-check-required", checks: [] }
  } finally {
    bitmap?.close()
  }
}

export function captureMetricsFromImageData(
  image: ImageData,
  sourceWidth = image.width,
  sourceHeight = image.height
): ClientCaptureMetrics {
  const { width, height, data } = image
  const luminance = new Uint8Array(width * height)
  const histogram = new Uint32Array(256)
  for (let pixel = 0, offset = 0; pixel < luminance.length; pixel += 1, offset += 4) {
    const value = Math.round(
      data[offset] * 0.2126 +
        data[offset + 1] * 0.7152 +
        data[offset + 2] * 0.0722
    )
    luminance[pixel] = value
    histogram[value] += 1
  }
  const p05 = histogramPercentile(histogram, luminance.length, 0.05)
  const p50 = histogramPercentile(histogram, luminance.length, 0.5)
  const p95 = histogramPercentile(histogram, luminance.length, 0.95)
  let laplacianCount = 0
  let laplacianMean = 0
  let laplacianM2 = 0
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const index = y * width + x
      const laplacian =
        luminance[index] * 4 -
        luminance[index - 1] -
        luminance[index + 1] -
        luminance[index - width] -
        luminance[index + width]
      laplacianCount += 1
      const delta = laplacian - laplacianMean
      laplacianMean += delta / laplacianCount
      laplacianM2 += delta * (laplacian - laplacianMean)
    }
  }
  const inkThreshold = Math.max(40, Math.min(215, p50 - 14))
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      if (luminance[y * width + x] >= inkThreshold) continue
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
  }
  return {
    width: sourceWidth,
    height: sourceHeight,
    sharpnessLaplacianVariance:
      laplacianCount > 1 ? laplacianM2 / (laplacianCount - 1) : 0,
    contrastRange: p95 - p05,
    luminanceP05: p05,
    luminanceP50: p50,
    luminanceP95: p95,
    inkWidthFraction: maxX >= minX ? (maxX - minX + 1) / width : 0,
    inkHeightFraction: maxY >= minY ? (maxY - minY + 1) / height : 0,
  }
}

function effectiveUploadShortEdge(
  report: PreprocessingReport,
  aspectRatio: number
) {
  const input = report.inputQuality as
    | (NonNullable<PreprocessingReport["inputQuality"]> & {
        minimumQuantitativeHighAspectShortEdgePixels?: number
        quantitativeHighAspectRatioThreshold?: number
        effectiveMinimumQuantitativeShortEdgePixels?: number
      })
    | undefined
  if (input?.effectiveMinimumQuantitativeShortEdgePixels !== undefined) {
    return input.effectiveMinimumQuantitativeShortEdgePixels
  }
  return aspectRatio >= (input?.quantitativeHighAspectRatioThreshold ?? 2.2)
    ? (input?.minimumQuantitativeHighAspectShortEdgePixels ?? 450)
    : (input?.minimumQuantitativeShortEdgePixels ?? 700)
}

function passing(
  id: CaptureQualityCheck["id"],
  label: string,
  detail: string
): CaptureQualityCheck {
  return { id, label, status: "pass", detail }
}

function warning(
  id: CaptureQualityCheck["id"],
  label: string,
  detail: string,
  guidance: string
): CaptureQualityCheck {
  return { id, label, status: "warning", detail, guidance }
}

function replaceOrAppendWarning(
  checks: CaptureQualityCheck[],
  id: CaptureQualityCheck["id"],
  label: string,
  detail: string,
  guidance: string
) {
  const index = checks.findIndex((check) => check.id === id)
  const qualityWarning = warning(id, label, detail, guidance)
  if (index === -1) checks.push(qualityWarning)
  else checks[index] = qualityWarning
}

function histogramPercentile(
  histogram: Uint32Array,
  total: number,
  fraction: number
) {
  const target = total * fraction
  let cumulative = 0
  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value]
    if (cumulative >= target) return value
  }
  return 255
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function captureMetricsAreValid(metrics: Record<string, unknown>) {
  return [
    "sharpnessLaplacianVariance",
    "perspectiveStrength",
    "rotationDegrees",
    "pageAreaFraction",
    "glareFraction",
    "chromaMoireScore",
  ].every(
    (key) =>
      metrics[key] === undefined ||
      (typeof metrics[key] === "number" && Number.isFinite(metrics[key]))
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}
