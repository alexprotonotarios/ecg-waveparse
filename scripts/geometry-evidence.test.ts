import assert from "node:assert/strict"
import test from "node:test"
import { calibrationEvidence, inverseTransform, segmentSourceRegion, sourceTransformChain, transformPoint } from "../src/lib/digitizer/geometry-evidence"
import type { PreprocessingReport } from "../src/lib/digitizer/contracts"

const preprocessing = { sourceSha256: "a".repeat(64), source: { width: 4000, height: 2000 }, workingImage: {width: 2000, height: 800},
  geometryCorrection: { applied: true, transform: [[1, .05, 20], [.02, 1, -10], [.00001, .00002, 1]], outputWidth: 2100, outputHeight: 900 } } as PreprocessingReport

test("original, anisotropic working, projective and cropped coordinates round-trip", () => {
  const chain = sourceTransformChain(preprocessing, {inputVariant: "geometry-corrected", cropBox: {left: 100, top: 50, right: 1900, bottom: 850}})
  assert.deepEqual(chain.candidateSize, {width: 1800, height: 800})
  let maxError = 0
  for (let x=0; x<=4000; x+=200) for (let y=0; y<=2000; y+=100) {
    const mapped = transformPoint(chain.originalToCandidate, {x,y})
    const roundTrip = transformPoint(chain.candidateToOriginal, mapped)
    maxError = Math.max(maxError, Math.hypot(x-roundTrip.x, y-roundTrip.y))
  }
  assert(maxError < 1e-8, String(maxError))
  assert.equal(chain.waveformResamplingPerformed, false)
  assert.deepEqual(transformPoint(inverseTransform([[1,0,20000],[0,1,-20000],[0,0,1]]), {x: 20001,y:-19999}), {x:1,y:1})
  assert.throws(() => inverseTransform([[1,0,0],[1,0,0],[0,0,1]]), /invertible/)
  assert.throws(() => inverseTransform([[1e300,0,0],[0,1e300,0],[0,0,1e300]]), error =>
    error instanceof Error && "code" in error && error.code === "invalid_evidence_contract")
  assert.throws(() => transformPoint([[1,0,0],[0,1,0],[1,0,0]], {x:0,y:1}), /projective/)
})

test("source regions are review bands with original coordinates and explicit missing mappings", () => {
  const region = segmentSourceRegion(preprocessing, {image:{width:2000,height:800}, coordinateSpace:"working", rowCenters:[100,300,500,700]}, 1, 1, 2, false)
  assert.equal(region.state, "inferred")
  if (region.state === "inferred") {
    assert.deepEqual(region.bounds, {left:2000,top:500,right:4000,bottom:1000})
    assert.equal(region.traceSupportVerified, false)
  }
  assert.equal(segmentSourceRegion(preprocessing, {}, 1, 1, 2, false).state, "unresolved")
  assert.equal(segmentSourceRegion(preprocessing, {image:{width:500,height:800},rowCenters:[100,300]}, 1, 1, 2, false).state, "unresolved")
})

test("pulse inference never becomes printed settings or acquisition sampling evidence", () => {
  for (const paperSpeedMmPerSecond of [25,50]) for (const gainMmPerMv of [5,10,20]) {
    const evidence = calibrationEvidence({detected:true,method:"grid-pulse",confidence:.9,paperSpeedMmPerSecond,gainMmPerMv,pixelsPerMmX:8,pixelsPerMmY:5})
    assert.equal(evidence.speed.value, paperSpeedMmPerSecond)
    assert.equal(evidence.gain.value, gainMmPerMv)
    assert.equal(evidence.physicalUnitsState, "inferred")
    assert.equal(evidence.printedSettings.state, "unresolved")
    assert.equal(evidence.acquisitionSampleRateHz, null)
  }
  assert.equal(calibrationEvidence(undefined).speed.value, null)
})
