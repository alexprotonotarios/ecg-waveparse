import type { RunAsset, RunAssetKey, RunRecord } from "@/lib/runs"
import { validateSegmentEvidence, type SegmentEvidenceMap } from "@/lib/digitizer/segment-evidence"
import { EvidenceContractError } from "@/lib/digitizer/physical-contracts"

type LocalAsset = RunAsset & { absolutePath?: string }
export type InterpreterEvidence = {
  version: 1
  runId: string
  sourceSha256: string
  original: LocalAsset
  reconstruction: { state: "available_for_source_review" | "unavailable"; canonicalSha256: string | null; assets: Partial<Record<RunAssetKey, LocalAsset>> }
  segments: SegmentEvidenceMap["segments"]
  omittedSourceSegments: SegmentEvidenceMap["omittedSourceSegments"]
  review: RunRecord["review"] | null
  limitations: string[]
  sourceInspectionAvailable: true
  generatedSamplesPermitted: false
  networkTransmission: "not_performed"
  clinicalValidationUse: false
}

/** Called only after the runner verifies source and derived asset identities. */
export function interpreterEvidence(run: RunRecord, assets: Partial<Record<RunAssetKey, LocalAsset>>, segmentMap?: unknown): InterpreterEvidence {
  const original = assets.input
  if (!original || !run.sourceIdentity || (original.identity && original.identity.sha256 !== run.sourceIdentity.sha256)) {
    throw new EvidenceContractError("missing_original_identity", "Interpreter evidence requires the verified original source.")
  }
  const canonicalSha256 = assets.canonicalCsv?.identity?.sha256 ?? null
  const map = segmentMap === undefined ? undefined : validateSegmentEvidence(segmentMap, {sourceSha256:run.sourceIdentity.sha256,canonicalSha256:canonicalSha256 ?? ""})
  const complete = Boolean(canonicalSha256 && map && assets.segmentsCsv && assets.uncertaintyCsv && assets.provenanceJson)
  const limitations = ["Human source review is required for any research measurement.",
    "Panel position does not establish acquisition simultaneity; export sampling does not establish acquisition sampling.",
    "Source-region bands are inferred navigation aids; calibrated trace support must be checked against the original.",
    "Missing or ambiguous intervals remain unavailable; a downstream model cannot supply observed samples."]
  if (!complete) limitations.push("The complete quantitative evidence bundle is unavailable; inspect the original image directly.")
  const keys: RunAssetKey[] = ["canonicalCsv", "segmentsCsv", "uncertaintyCsv", "segmentMapJson", "provenanceJson", "diagnostic", "paperRender", "reviewJson"]
  const boundOriginal: LocalAsset = {...original,identity:original.identity ?? {
    version:1,algorithm:"sha256",sha256:run.sourceIdentity.sha256,sizeBytes:run.sourceIdentity.sizeBytes,
    sourceSha256:run.sourceIdentity.sha256,recordedAt:run.sourceIdentity.recordedAt,
  }}
  return { version:1,runId:run.id,sourceSha256:run.sourceIdentity.sha256,original:boundOriginal,
    reconstruction:{state:complete ? "available_for_source_review" : "unavailable",canonicalSha256,
      assets:Object.fromEntries(keys.flatMap(key=>assets[key] ? [[key,assets[key]]] : []))},
    segments:map?.segments ?? [],omittedSourceSegments:map?.omittedSourceSegments ?? [],review:run.review ?? null,
    limitations,sourceInspectionAvailable:true,generatedSamplesPermitted:false,networkTransmission:"not_performed",clinicalValidationUse:false }
}

export type MeasurementEvidenceReference = {
  runId: string; sourceSha256: string; canonicalSha256: string; immutableSegmentId: string;
  startSample: number; endSample: number;
}

/** Reference validation only; this API neither diagnoses nor computes a measurement. */
export function validateMeasurementReference(bundle: InterpreterEvidence, reference: MeasurementEvidenceReference) {
  const segment = bundle.segments.find(item=>item.immutableId===reference.immutableSegmentId)
  const fail = (): never => { throw new EvidenceContractError("unsupported_measurement_reference", "Measurement must reference this exact source, signal and available segment interval.") }
  if (bundle.reconstruction.state!=="available_for_source_review" || reference.runId!==bundle.runId || reference.sourceSha256!==bundle.sourceSha256 ||
      reference.canonicalSha256!==bundle.reconstruction.canonicalSha256 || !segment || !Number.isSafeInteger(reference.startSample) || !Number.isSafeInteger(reference.endSample) ||
      reference.startSample<0 || reference.endSample<=reference.startSample || reference.endSample>segment.sampleCount) return fail()
  if (segment.support.some(span=>span.status==="missing" && span.startSample<reference.endSample && span.endSample>reference.startSample)) return fail()
  return { ...reference, units:"uV" as const,exportSampleRateHz:500,calibrationReviewRequired:true,uncertaintyReviewRequired:true,
    acquisitionSimultaneity:"unresolved" as const,clinicalValidationUse:false as const }
}
