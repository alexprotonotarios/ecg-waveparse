// Real synthetic-image extraction through an installed npm package. No patient data.
import { createRequire } from 'node:module';
import { promises as fs } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
const [consumer, input, runtimeDir, workspaceDir, existingRunId] = process.argv.slice(2);
if (!workspaceDir) throw new Error('Usage: node waveparse-inference-smoke.mjs CONSUMER_DIR INPUT RUNTIME_DIR WORKSPACE_DIR [EXISTING_RUN_ID]');
const require = createRequire(path.resolve(consumer, 'package.json'));
const { Digitizer } = require('ecg-waveparse');
const digitizer = new Digitizer({ workspaceDir, runtimeDir });
const result = existingRunId ? await digitizer.getRun(existingRunId) : await digitizer.digitize(input, { device: process.env.WAVEPARSE_SMOKE_DEVICE || 'cpu' });
assert.equal(result.sourceIdentity.sha256, createHash('sha256').update(await fs.readFile(input)).digest('hex'));
assert.equal(result.status, 'needs_review', result.message);
assert.equal(result.layout, 'standard_6x2');
assert(result.assets.canonicalCsv);
const hash = createHash('sha256').update(await fs.readFile(result.assets.canonicalCsv.absolutePath)).digest('hex');
assert.equal((await digitizer.getRun(result.id)).assets.canonicalCsv.identity.sha256, hash);
await assert.rejects(digitizer.review(result.id, { decision: 'accepted', reviewer: 'synthetic-test', notes: 'No visual review performed.' }));
const rejected = await digitizer.review(result.id, { decision: 'rejected', reviewer: 'synthetic-test', notes: 'Engineering test only; not accepted for quantitative use.' });
assert.equal(rejected.review.decision, 'rejected');
assert.equal(rejected.retention.policy, 'lean-final-evidence-v2');
assert.equal(rejected.retention.reviewArtifactsRetained, true);
for (const key of ['canonicalCsv', 'segmentsCsv', 'uncertaintyCsv', 'segmentMapJson']) {
  assert.equal(rejected.assets[key].identity.sha256, result.assets[key].identity.sha256);
}
const original = await fs.readFile(rejected.assets.input.absolutePath);
await fs.writeFile(rejected.assets.input.absolutePath, Buffer.from('deliberately corrupted synthetic test copy'));
try { await assert.rejects(digitizer.getRun(result.id)); }
finally { await fs.writeFile(rejected.assets.input.absolutePath, original); }
const controller = new AbortController();
let cancelledRunId;
let cancelRequestedAt;
await assert.rejects(digitizer.digitize(input, { signal: controller.signal, onProgress: ({runId}) => { cancelledRunId = runId; cancelRequestedAt = performance.now(); controller.abort(); } }), { code: 'cancelled' });
const cancellationLatencyMs = performance.now() - cancelRequestedAt;
assert(cancellationLatencyMs < 30_000, 'Post-admission cancellation exceeded the cleanup budget');
console.log(JSON.stringify({ cancellationLatencyMs, cancellationBoundary: 'immediately_after_source_admission' }));
assert(cancelledRunId);
const cancelled = await digitizer.getRun(cancelledRunId);
assert.equal(cancelled.status, 'failed');
assert.equal(cancelled.processing.state, 'failed');
assert(cancelled.assets.input);
const abandoned = { ...cancelled, status: 'running', processing: { ...cancelled.processing, state: 'running', ownerId: 'waveparse-2147483647-test' } };
delete abandoned.schemaVersion; delete abandoned.waveparse;
const runDir = path.join(workspaceDir, cancelled.localPath);
await fs.writeFile(path.join(runDir, 'metadata.json'), JSON.stringify(abandoned));
await fs.writeFile(path.join(runDir, '.digitizer-job-claim.json'), JSON.stringify({ version: 1, ownerId: abandoned.processing.ownerId, pid: 2147483647, hostname: os.hostname(), claimedAt: new Date().toISOString() }), { mode: 0o600 });
assert.equal((await digitizer.getRun(cancelledRunId)).processing.failureCode, 'worker_failure');
console.log(JSON.stringify({ status: result.status, layout: result.layout, canonicalSha256: hash, runId: result.id }));
