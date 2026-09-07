import assert from "node:assert/strict"
import test from "node:test"
import parameterContract from "../config/selector-parameter-contract.v1.json"
import { profileParameterCompatibility, selectorParameterSignature } from "../src/lib/digitizer/selector-feature-identity"

test("calibration identity rejects a reused candidate name with changed numerical parameters", () => {
  const candidate = { id: "base", kind: "source-model", inputVariant: "original", vectorizer: "probability-centroid", resampleSize: 1500 }
  const profile = { parameterContractVersion: 1, parameterKeys: parameterContract.keys, parameterSignatures: { "standard_6x2/base": [selectorParameterSignature(candidate)] } }
  assert.equal(profileParameterCompatibility(profile, "standard_6x2", candidate), "compatible")
  for (const change of [{ resampleSize: 2200 }, { labelThresh: 0.01 }, { vectorizer: "native-grid-path" }, { kind: "native-grid" }, { id: "renamed" }]) {
    assert.equal(profileParameterCompatibility(profile, "standard_6x2", { ...candidate, ...change }), "incompatible")
  }
  assert.equal(profileParameterCompatibility({ ...profile, parameterContractVersion: 2 }, "standard_6x2", candidate), "incompatible")
  assert.equal(profileParameterCompatibility(undefined, "standard_6x2", candidate), "historical_unverified")
  assert.equal(profileParameterCompatibility(profile, "standard_6x2", {...candidate, gainMmPerMv: 10}), "compatible")
  for (const gainMmPerMv of [5, 20, 0, NaN]) {
    assert.equal(profileParameterCompatibility(profile, "standard_6x2", {...candidate, gainMmPerMv}), "incompatible")
    assert.equal(profileParameterCompatibility(undefined, "standard_6x2", {...candidate, gainMmPerMv}), "incompatible")
  }
})

test("Python JSON numeric spelling and object-key order preserve parameter identity", () => {
  const candidate = {id:"base", labelThresh:1e-7, darkInkSupportRadius:1, capabilities:{"Z":true,"a-b":false,"aB":true}}
  const signature = selectorParameterSignature(candidate).replace('"darkInkSupportRadius":1,', '"darkInkSupportRadius":1.0,').replace('1e-7', '1e-07')
  const profile = { parameterContractVersion:1,parameterKeys:parameterContract.keys,parameterSignatures:{"standard_6x2/base":[signature]} }
  assert.equal(profileParameterCompatibility(profile,"standard_6x2",candidate),"compatible")
  assert.equal(profileParameterCompatibility(profile,"standard_6x2",{...candidate,labelThresh:2e-7}),"incompatible")
  assert.equal(profileParameterCompatibility({...profile,parameterSignatures:{"standard_6x2/base":["invalid JSON"]}},"standard_6x2",candidate),"incompatible")
})
