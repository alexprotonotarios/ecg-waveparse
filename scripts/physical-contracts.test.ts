import assert from "node:assert/strict"
import test from "node:test"
import { parseLayoutGeometryReport, parseNativeExtractionReport, parsePreprocessingReport } from "../src/lib/digitizer/contracts"
import { EvidenceContractError, sourceTimingEvidenceState } from "../src/lib/digitizer/physical-contracts"
import { bindPrintedCalibrationSource, calibrationPublicationBlock } from "../src/lib/digitizer/calibration-settings"
import type { PrintedSettingsReport } from "../src/lib/digitizer/calibration-settings"

const lead = { evidenceMedian: 1, evidenceP10: 0.9, coverage: 0.98, sourceStartPixel: 10, sourceEndPixel: 800, maxNativeJumpPixels: 20, p95NativeJumpPixels: 2 }
const native = { layout: "standard_6x2", layoutCost: 0.05, effectiveSampleRateHz: 250,
  sourceFidelity: { passed: true, method: "test-source-pixel", pixelsPerMm: 10, layoutConfidence: 0.9, evidenceMedian: 1, evidenceP10: 0.9, minimumCoverage: 0.98, leadMetrics: { I: lead } } }
const preprocessing = { version: 7, sourceSha256: "a".repeat(64), source: { width: 2000, height: 1000 },
  workingImage: { width: 1000, height: 500, scaleX: 0.5, scaleY: 0.5, method: "bounded", maxLongEdgePixels: 1000 },
  annotationMask: { maskedPixels: 4, maskedFraction: 0.000002, components: [{ x0: 1500, y0: 700, x1: 1502, y1: 702, pixels: 4, fillFraction: 1, dominantColor: "blue" }] },
  evidenceMaps: { method: "test", backgroundKernelPixels: 1 }, adaptivePreprocessing: { method: "test", eligible: false, reasons: [] },
  inputQuality: { outcome: "acceptable", quantitativeEligible: true, originalLongEdgePixels: 2000, originalShortEdgePixels: 1000, minimumQuantitativeLongEdgePixels: 1500, minimumQuantitativeShortEdgePixels: 700, workingScale: 0.5, reasons: [] },
  geometryCorrection: { applied: false, method: "identity", confidence: 1, outputWidth: 1000, outputHeight: 500, transform: [[1,0,0],[0,1,0],[0,0,1]], morphologyReconstructed: false } }

test("printed nondefault settings cannot inherit unresolved default physical units", () => {
  for (const values of [{gain: [5], speed: []}, {gain: [20], speed: [25]}, {gain: [10], speed: [50]}]) {
    const printedSettings = {values} as PrintedSettingsReport
    assert.equal(calibrationPublicationBlock({detected: false, printedSettings}), "unsupported_calibration")
    assert.equal(calibrationPublicationBlock({printedSettings}), "unsupported_calibration")
    assert.equal(calibrationPublicationBlock({detected: true, confidence: .9, gainMmPerMv: values.gain[0] ?? 10,
      paperSpeedMmPerSecond: values.speed[0] ?? 25, printedSettings}), undefined)
  }
  assert.equal(calibrationPublicationBlock({detected: false, printedSettings: {values: {gain: [10], speed: [25]}} as PrintedSettingsReport}), undefined)
  assert.equal(calibrationPublicationBlock({detected: true, confidence: .2, paperSpeedMmPerSecond: 25, gainMmPerMv: 20}), "unsupported_calibration")
  assert.equal(calibrationPublicationBlock({detected: true, confidence: .9, paperSpeedMmPerSecond: 25, gainMmPerMv: 7.5}), "unsupported_calibration")
})

test("horizontal scale refinement must reference consistent calibrated row measurements", () => {
  const calibration = {method: "grid-pulse", detected: true, confidence: .9, pixelsPerMmX: 6,
    pixelsPerMmY: 6, paperSpeedMmPerSecond: 25, gainMmPerMv: 10, gridScaleMmX: 1, rowPixelsPerMmX: [6,6,6,6,6,6],
    horizontalScaleEvidence: {version: 1, method: "consistent-row-grid-median-v1", priorPixelsPerMmX: 6.2, rowCount: 6, relativeRange: 0}}
  assert.equal(parseLayoutGeometryReport(JSON.stringify({calibration})).calibration?.pixelsPerMmX, 6)
  for (const patch of [{pixelsPerMmX: 6.2}, {detected: false}, {confidence: .2}, {rowPixelsPerMmX: [5,5,6,6,7,7]},
    {horizontalScaleEvidence: {...calibration.horizontalScaleEvidence, rowCount: 12}},
    {horizontalScaleEvidence: {...calibration.horizontalScaleEvidence, relativeRange: .01}},
    {horizontalScaleEvidence: {...calibration.horizontalScaleEvidence, priorPixelsPerMmX: 12}}]) {
    invalid(() => parseLayoutGeometryReport(JSON.stringify({calibration: {...calibration, ...patch}})), "invalid_horizontal_scale_evidence")
  }
})

function invalid(fn: () => unknown, issue?: string) {
  assert.throws(fn, (e: unknown) => e instanceof EvidenceContractError && e.code === "invalid_evidence_contract" && (!issue || e.issue === issue))
}

test("native reports reject invalid physical rates and malformed per-lead evidence", () => {
  assert.equal(parseNativeExtractionReport(JSON.stringify(native), native.layout).effectiveSampleRateHz, 250)
  for (const rate of [0, -1, null, "500"]) invalid(() => parseNativeExtractionReport(JSON.stringify({ ...native, effectiveSampleRateHz: rate }), native.layout))
  for (const change of [{ coverage: 1.01 }, { evidenceMedian: -0.1 }, { sourceEndPixel: 1 }, { maxNativeJumpPixels: null }, { recoveredSampleCount: 1.5 }]) {
    invalid(() => parseNativeExtractionReport(JSON.stringify({ ...native, sourceFidelity: { ...native.sourceFidelity, leadMetrics: { I: { ...lead, ...change } } } }), native.layout))
  }
  for (const leadMetrics of [{ XYZ: lead }, [], {}]) invalid(() => parseNativeExtractionReport(JSON.stringify({ ...native, sourceFidelity: { ...native.sourceFidelity, leadMetrics } }), native.layout))
  invalid(() => parseNativeExtractionReport(JSON.stringify({ ...native, sourceFidelity: { ...native.sourceFidelity, sourceTimingInference: { quantitativeCalibrationConfirmed: true } } }), native.layout), "unsubstantiated_timing_confirmation")
})

test("old timing booleans have explicit compatibility meaning", () => {
  assert.equal(sourceTimingEvidenceState(undefined), "not_applicable")
  assert.equal(sourceTimingEvidenceState({}), "unresolved")
  assert.equal(sourceTimingEvidenceState({ quantitativeCalibrationConfirmed: false }), "unresolved")
  assert.equal(sourceTimingEvidenceState({ quantitativeCalibrationConfirmed: true }), "verified")
})

test("source annotation coordinates survive downsampling and malformed geometry is refused", () => {
  assert.equal(parsePreprocessingReport(JSON.stringify(preprocessing)).source.width, 2000)
  invalid(() => parsePreprocessingReport(JSON.stringify({ ...preprocessing, sourceSha256: "z".repeat(64) })))
  invalid(() => parsePreprocessingReport(JSON.stringify({ ...preprocessing, annotationMask: { ...preprocessing.annotationMask, maskedPixels: 2000001 } })), "invalid_mask_counts")
  invalid(() => parsePreprocessingReport(JSON.stringify({ ...preprocessing, geometryCorrection: { ...preprocessing.geometryCorrection, transform: [[1,0,0],[1,0,0],[0,0,1]] } })), "invalid_transform")
})

test("layout confidence, crop bounds, units and report versions fail closed", () => {
  const layout = { version: 1, image: { width: 1000, height: 500 }, contentBox: { left: 0, top: 0, right: 900, bottom: 400 } }
  assert.equal(parseLayoutGeometryReport(JSON.stringify(layout)).evidenceStatus?.coordinateBounds, "verified")
  assert.equal(parseLayoutGeometryReport("{}").evidenceStatus?.calibration, "unresolved")
  assert.equal(parseLayoutGeometryReport("{}").evidenceStatus?.coordinateBounds, "unresolved")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({ ...layout, confidence: 2 })))
  invalid(() => parseLayoutGeometryReport(JSON.stringify({ ...layout, version: 2 })))
  invalid(() => parseLayoutGeometryReport(JSON.stringify({ ...layout, contentBox: { ...layout.contentBox, right: 1001 } })), "invalid_crop")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({ calibration: { method: "test", detected: true, confidence: 1 } })), "unsubstantiated_calibration")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({ calibration: { method: "test", detected: false, confidence: 0, pixelsPerMmX: -10 } })), "invalid_physical_scale")
  assert.equal(parseLayoutGeometryReport(JSON.stringify({ calibration: { method: "not-detected", detected: false, confidence: 0, gridSpacingXPixels: null, gridSpacingYPixels: null } })).evidenceStatus?.calibration, "unresolved")
})

test("printed calibration conflicts cannot be concealed by a serialized decision flag", () => {
  const calibration = { method: "grid-pulse", detected: true, confidence: .9, pixelsPerMmX: 4, pixelsPerMmY: 4,
    paperSpeedMmPerSecond: 25, gainMmPerMv: 10,
    printedSettings: { version: 1, method: "local-setting-token-ocr-v1", state: "recognized", coordinateSpace: "original",
      sourceSize: {width: 4000, height: 2000}, confidenceMinimum: .85, sourceSha256: "a".repeat(64), values: {speed: [50], gain: []},
      observations: [{kind: "speed", value: 50, units: "mm/s", confidence: .99, sourceBox: {left: 0, top: 0, right: 300, bottom: 50}}] },
    reconciliation: {version: 1, state: "conflict", quantitativeBlocked: true, compared: ["speed"],
      reasons: ["printed-speed-disagrees-with-pulse-grid"], pulseAssumptions: {durationSeconds: .2, amplitudeMv: 1}, acquisitionSampleRateHz: null} }
  const parsed = parseLayoutGeometryReport(JSON.stringify({calibration}))
  assert.equal(parsed.evidenceStatus?.calibration, "unresolved")
  assert.equal(calibrationPublicationBlock(parsed.calibration), "calibration_conflict")
  bindPrintedCalibrationSource(parsed.calibration?.printedSettings,{sha256:"a".repeat(64),width:4000,height:2000})
  invalid(() => bindPrintedCalibrationSource(parsed.calibration?.printedSettings,{sha256:"b".repeat(64),width:4000,height:2000}),"calibration_source_identity_mismatch")
  invalid(() => bindPrintedCalibrationSource(parsed.calibration?.printedSettings,{sha256:"a".repeat(64),width:2000,height:4000}),"calibration_source_coordinate_mismatch")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({calibration: {...calibration,
    reconciliation: {...calibration.reconciliation, quantitativeBlocked: false}}})), "invalid_calibration_reconciliation")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({calibration: {...calibration,
    printedSettings: {...calibration.printedSettings, values: {speed:[25],gain:[]}}}})), "invalid_calibration_reconciliation")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({calibration: {...calibration,
    reconciliation: null}})), "invalid_calibration_reconciliation")
  invalid(() => parseLayoutGeometryReport(JSON.stringify({calibration: {...calibration,
    printedSettings: {...calibration.printedSettings, sourceSha256: "bad"}}})), "invalid_calibration_reconciliation")
})
