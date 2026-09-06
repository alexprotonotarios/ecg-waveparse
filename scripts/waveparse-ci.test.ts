import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

test("package verification cannot skip a PR because of its changed paths", () => {
  const workflow = readFileSync(new URL("../.github/workflows/waveparse-packages.yml", import.meta.url), "utf8")
  const triggers = workflow.split(/^permissions:/m)[0]
  assert.match(triggers, /^  pull_request:\s*$/m)
  assert.doesNotMatch(triggers, /^\s+(?:paths|paths-ignore|branches-ignore):/m)
  // Both execution tiers are unconditional; the aggregate must run even after failure.
  assert.match(workflow, /^  contracts:\n    name: Fast package contracts/m)
  assert.match(workflow, /^  package:\n    needs: contracts/m)
  assert.match(workflow, /os: \[ubuntu-24\.04, macos-15\]/)
  const [executionJobs, aggregate] = workflow.split(/^  platform-equivalence:\n/m)
  assert.doesNotMatch(executionJobs, /^    if:/m)
  assert.ok(aggregate)
  assert.match(aggregate, /^    name: Supported platform equivalence$/m)
  assert.match(aggregate, /^    if: \$\{\{ always\(\) \}\}$/m)
  assert.match(aggregate, /^    needs: \[contracts, package\]$/m)
  assert.match(aggregate, /needs\.contracts\.result != 'success' \|\| needs\.package\.result != 'success'/)
  assert.match(aggregate, /run: exit 1/)
  assert.match(aggregate, /scripts\/platform_evidence\.py compare/)
  assert.match(workflow, /Freeze identical synthetic inputs for both supported platforms/)
  assert.equal(workflow.match(/name: waveparse-frozen-fixtures/g)?.length, 2)
})
