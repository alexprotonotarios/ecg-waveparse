import assert from "node:assert/strict"
import test from "node:test"
import { buildSegmentEvidence, signalSupport, validateSegmentEvidence, type SegmentDescriptor } from "../src/lib/digitizer/segment-evidence"

const descriptor: SegmentDescriptor = { lead: "V4", panelIndex: 3, rowIndex: 0, canonicalStartSample: 3750, sampleCount: 1250, role: "panel", identityState: "verified", identityMethod: "synthetic-label-ground-truth", candidateId: "test", lineageState: "direct_candidate", support: signalSupport(Array.from({length: 1250}, (_, i) => i >= 200 && i < 250 ? NaN : Math.sin(i / 10))) }
const input = { runId: "run_segment_fixture", sourceSha256: "a".repeat(64), canonicalSha256: "b".repeat(64), canonicalSampleCount: 5000, descriptors: [descriptor] }

test("segment map round-trips canonical and compact sample coordinates without closing gaps", () => {
  const map = buildSegmentEvidence(input)
  assert.deepEqual(validateSegmentEvidence(JSON.parse(JSON.stringify(map))), map)
  const segment = map.segments[0]
  assert.equal(segment.displayStartSeconds, 7.5)
  assert.equal(segment.displayEndSeconds, 10)
  assert.equal(segment.sampleCount, 1250)
  assert.equal(segment.acquisitionTime.state, "unresolved")
  for (let local = 0; local < segment.sampleCount; local++) {
    const canonical = segment.canonicalStartSample + local
    assert.equal(canonical - segment.canonicalStartSample, local)
    const status = segment.support.find(span => local >= span.startSample && local < span.endSample)?.status
    assert.equal(status, local >= 200 && local < 250 ? "missing" : "returned")
  }
  assert.equal(map.sourceAcquisitionSampleRateHz, null)
})

test("segment evidence refuses changed identities, timings, duplicated leads and missing spans", () => {
  const map = buildSegmentEvidence(input)
  const reject = (mutate: (value: typeof map) => void) => {
    const changed = structuredClone(map); mutate(changed)
    assert.throws(() => validateSegmentEvidence(changed), /Invalid segment evidence/)
  }
  reject(v => { v.canonicalSha256 = "c".repeat(64) })
  reject(v => { v.segments[0].displayStartSeconds = 0 })
  reject(v => { v.segments.push(structuredClone(v.segments[0])) })
  reject(v => { v.segments[0].support[1].startSample = 201 })
  assert.throws(() => validateSegmentEvidence(map, { sourceSha256: "c".repeat(64), canonicalSha256: input.canonicalSha256 }), /segment_artifact_identity_mismatch/)
})

test("repeated source segments and Cabrera polarity remain explicit", () => {
  const map = buildSegmentEvidence({ ...input, descriptors: [{ ...descriptor, lead: "aVR", sourceLabel: "-aVR", polarity: -1 }], omittedSourceSegments: [{ lead: "II", role: "rhythm", reason: "Not exported by this adapter" }] })
  assert.equal(map.segments[0].sourceLabel, "-aVR")
  assert.equal(map.segments[0].polarity, -1)
  assert.equal(map.omittedSourceSegments[0].lead, "II")
  assert.match(map.omittedSourceSegments[0].immutableId!, /^[a-f0-9]{64}$/)
  assert.notEqual(map.omittedSourceSegments[0].immutableId, map.segments[0].immutableId)
  const legacy=structuredClone(map)
  delete legacy.omittedSourceSegments[0].immutableId
  delete legacy.omittedSourceSegments[0].segmentId
  assert.doesNotThrow(()=>validateSegmentEvidence(legacy))
  assert.notEqual(map.segments[0].immutableId, buildSegmentEvidence({ ...input, runId: "run_second_fixture" }).segments[0].immutableId)
})

test("malformed nested evidence consistently returns contract errors",()=>{
  const original=buildSegmentEvidence(input)
  const variants: Array<(map: typeof original)=>void> = [
    map=>{map.segments[0].support=[null] as never},
    map=>{map.segments[0].sourceCrop={state:"inferred",coordinateSpace:"original",traceSupportVerified:false,polygon:[null,null,null,null]} as never},
    map=>{map.segments[0].sourceCrop={state:"inferred",coordinateSpace:"original",traceSupportVerified:false,polygon:[{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}]} as never},
    map=>{map.segments[0].intervalLineage=[{startSample:0,endSample:1250,operation:"selected_samples",contributors:[null]}] as never},
    map=>{map.omittedSourceSegments=[null] as never},
    map=>{map.segments[0].candidateId=1 as never},
    map=>{map.segments[0].lineageState="verified" as never},
  ]
  for (const mutate of variants) {
    const changed=structuredClone(original);mutate(changed)
    assert.throws(()=>validateSegmentEvidence(changed),error=>error instanceof Error && "code" in error && error.code==="invalid_evidence_contract")
  }
})
