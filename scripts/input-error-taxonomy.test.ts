import assert from "node:assert/strict"
import test from "node:test"
import {sourceInputFailure} from "../src/lib/digitizer/capture-preflight"

test("malformed or unsupported input is distinct from unavailable infrastructure",()=>{
  for (const issue of ["unsupported_raster_format","source_dimensions_exceed_budget","multiple_frames_require_explicit_selection"]) {
    const result=sourceInputFailure({stderr:`Traceback with private-source-label\nValueError: invalid_input: ${issue}`})
    assert.equal(result?.code,"invalid_input")
    assert.equal(result?.issue,issue)
    assert(!result?.message.includes("private-source-label"))
  }
  assert.equal(sourceInputFailure({stderr:"PIL.UnidentifiedImageError: patient-file"})?.issue,"malformed_or_oversized_raster")
  assert.equal(sourceInputFailure({code:"ENOENT",stderr:"python was not available"}),undefined)
})
