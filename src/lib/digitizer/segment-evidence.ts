import { createHash } from "node:crypto"
import { ECG_LEADS, SAMPLE_RATE_HZ } from "@/lib/digitizer/domain"
import { EvidenceContractError, positivePhysicalValue, type EvidenceState } from "@/lib/digitizer/physical-contracts"
import type { SourceRegion } from "@/lib/digitizer/geometry-evidence"
import { validateIntervalLineage, type IntervalLineage } from "@/lib/digitizer/lineage"

export type SourceTimeConversionEvidence = {
  version: 1
  method: "source_grid_time_publication_v1"
  evidenceAsset: "sourceTimeEvidence"
  lineageFile: "lineage.csv"
  lineageSha256: string
  coordinateSha256: string
  inputCanonicalSha256: string
  inputUncertaintySha256: string
  inputSegmentMapSha256: string
  sourceOriginMark: string
  sourceMidpointXY: [number, number]
  originPaperXmm: number
  originLocusPaperXmmRange: [number, number]
  inkFootprintPaperXmmRange: [number, number]
  geometricTimingShiftMsRange: [number, number]
}

export type SegmentDescriptor = {
  lead: string
  panelIndex: number
  rowIndex: number
  canonicalStartSample: number
  sampleCount: number
  role: "panel" | "rhythm"
  identityState: EvidenceState
  identityMethod: string
  candidateId: string
  lineageState: "direct_candidate" | "candidate_summary_only" | "interval_recorded" | "source_time_conversion_recorded"
  intervalLineage?: IntervalLineage[]
  sourceTimeConversion?: SourceTimeConversionEvidence
  sourceLabel?: string
  polarity?: 1 | -1
  sourceCrop?: SourceRegion
  support: Array<{ startSample: number; endSample: number; status: "returned" | "missing" }>
}

export type SegmentEvidenceMap = {
  version: 1
  runId: string
  sourceSha256: string
  canonicalSha256: string
  units: "uV"
  exportSampleRateHz: number
  sourceAcquisitionSampleRateHz: null
  canonicalSampleCount: number
  endpointConvention: "half_open"
  calibrationReference: "provenanceJson#/digitization/calibration"
  segments: Array<SegmentDescriptor & {
    segmentId: string
    immutableId: string
    polarity: 1 | -1
    displayStartSeconds: number
    displayEndSeconds: number
    localTimeOriginSeconds: 0
    compactStartSample: 0
    sourceLabel: string
    acquisitionTime: { state: "unresolved"; startSeconds: null; simultaneousGroupId: null }
    sourceCrop: SourceRegion
    sourceMappingReference: "provenanceJson#/preprocessing"
  }>
  omittedSourceSegments: Array<{ lead: string; role: "panel" | "rhythm"; reason: string; segmentId?: string; immutableId?: string }>
}

export function signalSupport(values: readonly number[]): SegmentDescriptor["support"] {
  const result: SegmentDescriptor["support"] = []
  values.forEach((value, sample) => {
    const status = Number.isFinite(value) ? "returned" : "missing"
    const previous = result.at(-1)
    if (previous?.status === status) previous.endSample = sample + 1
    else result.push({ startSample: sample, endSample: sample + 1, status })
  })
  return result
}

export function buildSegmentEvidence(input: {
  runId: string; sourceSha256: string; canonicalSha256: string; canonicalSampleCount: number;
  descriptors: SegmentDescriptor[]; omittedSourceSegments?: SegmentEvidenceMap["omittedSourceSegments"]
}): SegmentEvidenceMap {
  const result: SegmentEvidenceMap = {
    version: 1, runId: input.runId, sourceSha256: input.sourceSha256, canonicalSha256: input.canonicalSha256,
    units: "uV", exportSampleRateHz: SAMPLE_RATE_HZ, sourceAcquisitionSampleRateHz: null,
    canonicalSampleCount: input.canonicalSampleCount, endpointConvention: "half_open",
    calibrationReference: "provenanceJson#/digitization/calibration",
    omittedSourceSegments: (input.omittedSourceSegments ?? []).map((segment,index) => {
      const segmentId = `${segment.lead}:${segment.role}:omitted:${index}`
      return {...segment,segmentId,immutableId:createHash("sha256").update(`${input.runId}:${input.sourceSha256}:${input.canonicalSha256}:${segmentId}`).digest("hex")}
    }),
    segments: input.descriptors.map(segment => {
      const segmentId = `${segment.lead}:${segment.role}:${segment.panelIndex}`
      return { ...segment, segmentId,
        immutableId: createHash("sha256").update(`${input.runId}:${input.sourceSha256}:${input.canonicalSha256}:${segmentId}`).digest("hex"),
        sourceLabel: segment.sourceLabel ?? segment.lead, polarity: segment.polarity ?? 1,
        displayStartSeconds: segment.canonicalStartSample / SAMPLE_RATE_HZ,
        displayEndSeconds: (segment.canonicalStartSample + segment.sampleCount) / SAMPLE_RATE_HZ,
        localTimeOriginSeconds: 0, compactStartSample: 0,
        acquisitionTime: { state: "unresolved", startSeconds: null, simultaneousGroupId: null },
        sourceCrop: segment.sourceCrop ?? { state: "unresolved", reason: "A lead-specific crop in original-source coordinates has not been independently verified." },
        sourceMappingReference: "provenanceJson#/preprocessing",
      }
    }),
  }
  return validateSegmentEvidence(result)
}

export function validateSegmentEvidence(value: unknown, identities?: { sourceSha256: string; canonicalSha256: string }): SegmentEvidenceMap {
  const fail = (issue: string): never => { throw new EvidenceContractError(issue, `Invalid segment evidence: ${issue}.`) }
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail("invalid_segment_map")
  const map = value as SegmentEvidenceMap
  if (map.version !== 1 || map.units !== "uV" || map.endpointConvention !== "half_open" ||
      typeof map.runId !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(map.runId) ||
      !/^[a-f0-9]{64}$/.test(map.sourceSha256) || !/^[a-f0-9]{64}$/.test(map.canonicalSha256) ||
      !positivePhysicalValue(map.exportSampleRateHz) || !Number.isSafeInteger(map.canonicalSampleCount) || map.canonicalSampleCount <= 0 ||
      map.sourceAcquisitionSampleRateHz !== null || !Array.isArray(map.segments) || !map.segments.length || !Array.isArray(map.omittedSourceSegments)) return fail("invalid_segment_header")
  if (identities && (map.sourceSha256 !== identities.sourceSha256 || map.canonicalSha256 !== identities.canonicalSha256)) return fail("segment_artifact_identity_mismatch")
  const ids = new Set<string>()
  const leads = new Set<string>()
  for (const segment of map.segments) {
    if (!segment || !(ECG_LEADS as readonly string[]).includes(segment.lead) || ids.has(segment.segmentId) || leads.has(segment.lead) ||
        !["panel", "rhythm"].includes(segment.role) || !["verified", "inferred", "unresolved", "not_applicable"].includes(segment.identityState) ||
        typeof segment.identityMethod !== "string" || !segment.identityMethod || typeof segment.sourceLabel !== "string" || !segment.sourceLabel ||
        typeof segment.candidateId !== "string" || !segment.candidateId ||
        !["direct_candidate", "candidate_summary_only", "interval_recorded", "source_time_conversion_recorded"].includes(segment.lineageState) || ![1, -1].includes(segment.polarity)) return fail("invalid_segment_identity")
    ids.add(segment.segmentId); leads.add(segment.lead)
    if (segment.segmentId !== `${segment.lead}:${segment.role}:${segment.panelIndex}` ||
        segment.immutableId !== createHash("sha256").update(`${map.runId}:${map.sourceSha256}:${map.canonicalSha256}:${segment.segmentId}`).digest("hex")) return fail("invalid_immutable_segment_id")
    for (const key of ["panelIndex", "rowIndex", "canonicalStartSample", "sampleCount"] as const) {
      if (!Number.isSafeInteger(segment[key]) || segment[key] < 0) return fail("invalid_segment_samples")
    }
    if (!segment.sampleCount || segment.canonicalStartSample + segment.sampleCount > map.canonicalSampleCount ||
        segment.displayStartSeconds !== segment.canonicalStartSample / map.exportSampleRateHz ||
        segment.displayEndSeconds !== (segment.canonicalStartSample + segment.sampleCount) / map.exportSampleRateHz ||
        segment.compactStartSample !== 0 || segment.localTimeOriginSeconds !== 0 ||
        segment.acquisitionTime?.state !== "unresolved" || segment.acquisitionTime.startSeconds !== null || segment.acquisitionTime.simultaneousGroupId !== null) return fail("inconsistent_segment_timing")
    if (!Array.isArray(segment.support)) return fail("invalid_segment_support")
    let next = 0
    for (const span of segment.support) {
      if (!span || !Number.isSafeInteger(span.startSample) || !Number.isSafeInteger(span.endSample) || span.startSample !== next ||
          span.endSample <= next || span.endSample > segment.sampleCount || !["returned", "missing"].includes(span.status)) return fail("invalid_segment_support")
      next = span.endSample
    }
    if (next !== segment.sampleCount) return fail("incomplete_segment_support")
    if (segment.intervalLineage) validateIntervalLineage(segment.intervalLineage,segment.sampleCount)
    if (segment.lineageState==="interval_recorded" && !segment.intervalLineage) return fail("missing_interval_lineage")
    if (segment.lineageState === "source_time_conversion_recorded") {
      const proof = segment.sourceTimeConversion
      const pair = (value: unknown): value is [number, number] => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite)
      // The same source boundary can be evaluated through different polygon
      // subdivisions. Permit only arithmetic roundoff when comparing extents;
      // retain the measured bounds exactly and do not alter the time origin.
      const containsBound = (lower: number, upper: number) => lower <= upper ||
        lower - upper <= 8 * Number.EPSILON * Math.max(1, Math.abs(lower), Math.abs(upper))
      const mark = segment.panelIndex === 0 ? `row${segment.rowIndex}:pulse` : `row${segment.rowIndex}:separator${segment.panelIndex}`
      if (!proof || segment.intervalLineage !== undefined || proof.version !== 1 || proof.method !== "source_grid_time_publication_v1" ||
          proof.evidenceAsset !== "sourceTimeEvidence" || proof.lineageFile !== "lineage.csv" || proof.sourceOriginMark !== mark ||
          ![proof.lineageSha256, proof.coordinateSha256, proof.inputCanonicalSha256, proof.inputUncertaintySha256, proof.inputSegmentMapSha256].every(v => typeof v === "string" && /^[a-f0-9]{64}$/.test(v)) ||
          !pair(proof.sourceMidpointXY) || !pair(proof.originLocusPaperXmmRange) || !pair(proof.inkFootprintPaperXmmRange) ||
          !pair(proof.geometricTimingShiftMsRange) || !Number.isFinite(proof.originPaperXmm) ||
          !(proof.originLocusPaperXmmRange[0] <= proof.originPaperXmm && proof.originPaperXmm <= proof.originLocusPaperXmmRange[1]) ||
          !(containsBound(proof.inkFootprintPaperXmmRange[0], proof.originLocusPaperXmmRange[0]) && containsBound(proof.originLocusPaperXmmRange[1], proof.inkFootprintPaperXmmRange[1])) ||
          proof.geometricTimingShiftMsRange[0] !== (proof.originPaperXmm - proof.originLocusPaperXmmRange[1]) * 40 ||
          proof.geometricTimingShiftMsRange[1] !== (proof.originPaperXmm - proof.originLocusPaperXmmRange[0]) * 40 || map.exportSampleRateHz !== 500) return fail("invalid_source_time_lineage")
    } else if (segment.sourceTimeConversion !== undefined) return fail("source_time_lineage_state_mismatch")
    if (!segment.sourceCrop || !["inferred", "unresolved"].includes(segment.sourceCrop.state)) return fail("invalid_source_region")
    if (segment.sourceCrop.state === "inferred") {
      const region = segment.sourceCrop
      if (region.coordinateSpace !== "original" || region.traceSupportVerified !== false || !Array.isArray(region.polygon) || region.polygon.length !== 4 ||
          !region.polygon.every(point => point && Number.isFinite(point.x) && Number.isFinite(point.y)) || !region.bounds ||
          !Object.values(region.bounds ?? {}).every(Number.isFinite) ||
          !(region.bounds.left >= 0 && region.bounds.top >= 0) ||
          !(region.bounds.right > region.bounds.left && region.bounds.bottom > region.bounds.top)) return fail("invalid_source_region")
    } else if (!segment.sourceCrop.reason) return fail("missing_source_region_reason")
  }
  for (const [index,segment] of map.omittedSourceSegments.entries()) {
    if (!segment || !(ECG_LEADS as readonly string[]).includes(segment.lead) || !["panel","rhythm"].includes(segment.role) ||
        typeof segment.reason !== "string" || !segment.reason) return fail("invalid_omitted_source_segment")
    // Earlier v1 maps recorded omissions without immutable IDs; keep them readable.
    if (segment.segmentId !== undefined || segment.immutableId !== undefined) {
      const id = `${segment.lead}:${segment.role}:omitted:${index}`
      if (segment.segmentId !== id || segment.immutableId !== createHash("sha256").update(`${map.runId}:${map.sourceSha256}:${map.canonicalSha256}:${id}`).digest("hex")) return fail("invalid_omitted_segment_identity")
    }
  }
  return map
}
