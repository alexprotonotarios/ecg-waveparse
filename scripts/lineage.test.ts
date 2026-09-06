import assert from "node:assert/strict"
import test from "node:test"
import { directContributor, intervalLineage, validateIntervalLineage } from "../src/lib/digitizer/lineage"
import { digitizerTestUtils } from "../src/lib/digitizer"

test("lineage separates selected samples, missing intervals and aligned peer medians",()=>{
  const values=[1,2,NaN,4,5,6]
  const peer={...directContributor("peer",2500,12),alignment:{referenceLength:6,timeScale:1,shiftSamples:2,subtractOffsetUv:8}}
  const spans=intervalLineage(values,directContributor("base",0,6),[{startSample:3,endSample:5,operation:"aligned_peer_median",contributors:[peer]}])
  assert.deepEqual(spans.map(s=>[s.startSample,s.endSample,s.operation]),[[0,2,"selected_samples"],[2,3,"missing"],[3,5,"aligned_peer_median"],[5,6,"selected_samples"]])
  assert.deepEqual(spans[1].contributors,[])
  assert.equal(spans[2].contributors[0].alignment.subtractOffsetUv,8)
  const changed=structuredClone(spans);changed[2].contributors[0].alignment.shiftSamples=100
  assert.throws(()=>validateIntervalLineage(changed,6),/Lineage/)
})

test("peer repair records exactly the repaired intervals without changing existing values",()=>{
  const original=Array.from({length:100},(_,i)=>i>=40 && i<44 ? NaN : 50)
  const result=digitizerTestUtils.repairShortPeerSupportedGaps(original,[{id:"peer",values:Array(100).fill(50),sourceVerified:true}])
  assert.equal(result.repairedSamples,4)
  assert.deepEqual(result.intervals,[{startSample:40,endSample:44,candidateIds:["peer"]}])
  assert(result.values.every(Number.isFinite))
  assert(original.slice(40,44).every(Number.isNaN))
})
