import test from "node:test"
import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { createRequire } from "node:module"

const require = createRequire(import.meta.url)
const { Digitizer, WaveParseError } = require("../packages/javascript/dist/index.cjs")

test("packaged runner isolates concurrent workspaces and rejects unsafe identifiers", async () => {
  const workspace = await mkdtemp(path.join(tmpdir(), "waveparse-test-"))
  try {
    const a = new Digitizer({ workspaceDir: path.join(workspace, "a") })
    const b = new Digitizer({ workspaceDir: path.join(workspace, "b") })
    assert.deepEqual(await Promise.all([a.getRun("run_absent"), b.getRun("run_absent")]), [null, null])
    await assert.rejects(a.getRun("../../input"), { code: "invalid_request" })
    await assert.rejects(a.review("run_absent", { decision: "rejected", reviewer: "test", notes: "absent" }), { code: "not_found" })
    await assert.rejects(a.digitize("absent.png", { device: "cuda" }), { code: "invalid_request" })
    await assert.rejects(a.digitize("absent.png", { timeoutMs: 1 }), { code: "invalid_request" })
    const controller = new AbortController(); controller.abort()
    await assert.rejects(a.digitize("absent.png", { signal: controller.signal }), { code: "cancelled" })
    assert.throws(() => new Digitizer({}), WaveParseError)
  } finally { await rm(workspace, { recursive: true, force: true }) }
})
