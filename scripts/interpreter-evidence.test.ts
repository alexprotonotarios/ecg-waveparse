import assert from "node:assert/strict"
import test from "node:test"
import { interpreterEvidence, validateMeasurementReference } from "../src/lib/digitizer/interpreter-evidence"
import { buildSegmentEvidence, signalSupport } from "../src/lib/digitizer/segment-evidence"
import type { RunRecord, RunAsset } from "../src/lib/runs"

const asset=(sha:string)=>({label:"test",path:"synthetic",identity:{version:1,algorithm:"sha256",sha256:sha}} as RunAsset)
const source="a".repeat(64),canonical="b".repeat(64)
const run={id:"run_evidence",status:"failed",sourceIdentity:{sha256:source},assets:{input:asset(source)}} as RunRecord

test("abstention retains a source inspection path and cannot become a numerical measurement",()=>{
  const bundle=interpreterEvidence(run,run.assets)
  assert.equal(bundle.sourceInspectionAvailable,true)
  assert.equal(bundle.reconstruction.state,"unavailable")
  assert.equal(bundle.generatedSamplesPermitted,false)
  assert.equal(bundle.networkTransmission,"not_performed")
  const legacyInput={label:"source",path:"synthetic"}
  const legacy=interpreterEvidence({...run,assets:{input:legacyInput}},{input:legacyInput})
  assert.equal(legacy.original.identity?.sha256,source)
  assert.throws(()=>interpreterEvidence(run,{input:asset("c".repeat(64))}),/verified original/)
  assert.throws(()=>validateMeasurementReference(bundle,{runId:run.id,sourceSha256:source,canonicalSha256:canonical,immutableSegmentId:"unknown",startSample:0,endSample:1}),/Measurement/)
})

test("interpreter references bind exact source, canonical signal, segment and available interval",()=>{
  const map=buildSegmentEvidence({runId:run.id,sourceSha256:source,canonicalSha256:canonical,canonicalSampleCount:5000,descriptors:[{lead:"I",panelIndex:0,rowIndex:0,canonicalStartSample:0,sampleCount:5,role:"panel",identityState:"verified",identityMethod:"fixture",candidateId:"fixture",lineageState:"direct_candidate",support:signalSupport([1,2,NaN,3,4])}]})
  const assets={...run.assets,canonicalCsv:asset(canonical),segmentsCsv:asset("c".repeat(64)),uncertaintyCsv:asset("d".repeat(64)),segmentMapJson:asset("e".repeat(64)),provenanceJson:asset("f".repeat(64))}
  const bundle=interpreterEvidence(run,assets,map)
  const reference={runId:run.id,sourceSha256:source,canonicalSha256:canonical,immutableSegmentId:map.segments[0].immutableId,startSample:0,endSample:2}
  assert.equal(validateMeasurementReference(bundle,reference).calibrationReviewRequired,true)
  assert.throws(()=>validateMeasurementReference(bundle,{...reference,endSample:4}),/Measurement/)
  assert.throws(()=>validateMeasurementReference(bundle,{...reference,sourceSha256:"c".repeat(64)}),/Measurement/)
})
