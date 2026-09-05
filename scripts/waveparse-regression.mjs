// Run deterministic engineering fixtures through an installed package, never the
// source checkout. Scores describe waveform reconstruction, not diagnosis.
import { promises as fs } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';

const { values } = parseArgs({ options: {
  consumer: { type: 'string' }, runtime: { type: 'string' }, python: { type: 'string' },
  manifest: { type: 'string', multiple: true }, output: { type: 'string' },
  device: { type: 'string', default: 'cpu' }, 'timeout-ms': { type: 'string', default: '1800000' },
}});
for (const required of ['consumer', 'runtime', 'python', 'manifest', 'output']) {
  assert(values[required], `--${required} is required`);
}
assert(['cpu', 'mps'].includes(values.device), '--device must be cpu or mps');
const timeoutMs = Number(values['timeout-ms']);
assert(Number.isSafeInteger(timeoutMs) && timeoutMs >= 60000 && timeoutMs <= 7200000);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(values.output);
await fs.mkdir(output); // Require fresh evidence; never overwrite a prior campaign.
const sha = data => createHash('sha256').update(data).digest('hex');
const hashFile = async file => sha(await fs.readFile(file));
const exec = promisify(execFile);
const require = createRequire(path.resolve(values.consumer, 'package.json'));
const entry = require.resolve('ecg-waveparse');
const packageRoot = path.resolve(path.dirname(entry), '..');
assert(!packageRoot.startsWith(path.join(root, 'packages') + path.sep), 'Use an installed package outside the source checkout');
const { Digitizer, version } = require('ecg-waveparse');
const payloadPath = path.join(packageRoot, 'runtime/payload-manifest.json');
const payload = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
async function verifyPayload() {
  for (const [file, expected] of Object.entries(payload.files)) {
    assert.equal(await hashFile(path.join(packageRoot, 'runtime', file)), expected, file);
  }
}
await verifyPayload();
const cases = [];
const manifests = [];
for (const inputManifest of values.manifest) {
  const manifestPath = path.resolve(inputManifest);
  const bytes = await fs.readFile(manifestPath);
  const manifest = JSON.parse(bytes);
  assert.equal(manifest.version, 1);
  manifests.push({ path: manifestPath, sha256: sha(bytes), suiteId: manifest.suiteId });
  for (const item of manifest.cases) {
    assert.equal(item.provenance?.clinicalValidationUse, false, 'Only nonclinical engineering fixtures are admitted');
    assert.equal(item.provenance?.waveformOrigin, 'deterministic engineering fixture');
    assert.equal(item.provenance?.quantitativeTruth, true);
    assert.equal(item.provenance?.acquisition, 'synthetic_render');
    assert(!String(item.split).includes('locked'), 'Locked validation is not part of this regression');
    const artifacts = {};
    for (const key of ['image', 'truth', 'annotations']) {
      const absolutePath = path.resolve(path.dirname(manifestPath), item[key].path);
      assert.equal(await hashFile(absolutePath), item[key].sha256, `${item.caseId} ${key}`);
      artifacts[key] = absolutePath;
    }
    const id = `${manifest.suiteId}__${item.caseId}`;
    assert(/^[a-zA-Z0-9_-]+$/.test(id) && !cases.some(c => c.id === id));
    cases.push({ id, item, artifacts });
  }
}
const report = {
  schemaVersion: 1, startedAt: new Date().toISOString(), clinicalValidationUse: false,
  description: 'Deterministic engineering regression; one synthetic waveform family, no patient cohort.',
  platform: { os: os.platform(), architecture: os.arch(), release: os.release(), node: process.version, device: values.device },
  package: { version, entrySha256: await hashFile(entry), payloadManifestSha256: await hashFile(payloadPath), runtimeManifestSha256: await hashFile(path.join(packageRoot, 'runtime/config/waveparse-runtime.json')) },
  scorer: { version: 5, scriptSha256: await hashFile(path.join(root, 'ecg_benchmark/scoring.py')), maxAlignmentMs: 40 },
  manifests, denominator: cases.length, cases: [],
};
const save = async () => {
  await fs.writeFile(path.join(output, 'report.json.tmp'), JSON.stringify(report, null, 2) + '\n');
  await fs.rename(path.join(output, 'report.json.tmp'), path.join(output, 'report.json'));
};
await save();
const digitizer = new Digitizer({ workspaceDir: path.join(output, 'workspace'), runtimeDir: path.resolve(values.runtime) });
for (const { id, item, artifacts } of cases) {
  console.log(`Starting ${id}`);
  const started = performance.now();
  const record = { id, layout: item.layout, degradation: item.degradation.kind, sourceSha256: item.image.sha256, truthSha256: item.truth.sha256 };
  try {
    const run = await digitizer.digitize(artifacts.image, { device: values.device, timeoutMs,
      onProgress: event => { record.runId = event.runId; console.log(`${id}: admitted ${event.runId}`); },
    });
    assert.equal(run.sourceIdentity.sha256, item.image.sha256);
    assert.equal(run.waveparse.runtimeManifestSha256, report.package.runtimeManifestSha256);
    record.status = run.status;
    record.detectedLayout = run.layout;
    record.publicationDecision = run.publicationDecision;
    record.selectedCandidateId = run.selectedCandidateId;
    record.message = run.message;
    record.candidateOutcomes = (run.digitizer?.candidates || []).map(c => ({ id: c.id, status: c.status, runtimeMs: c.runtimeMs }));
    record.effectiveSampleRateHz = run.reliability?.effectiveSampleRateHz;
    record.reviewRequired = run.reliability?.reviewRequired;
    if (run.assets.canonicalCsv) {
      const candidate = run.assets.canonicalCsv.absolutePath;
      record.canonicalSha256 = await hashFile(candidate);
      assert.equal(record.canonicalSha256, run.assets.canonicalCsv.identity.sha256);
      const scorePath = path.join(output, `${id}.score.json`);
      const args = [path.join(root, 'scripts/score_digitization.py'), '--truth', artifacts.truth, '--candidate', candidate,
        '--truth-rate', String(item.sampleRateHz), '--candidate-rate', '500', '--annotations', artifacts.annotations,
        '--case-id', id, '--output', scorePath];
      if (run.assets.uncertaintyCsv) args.push('--uncertainty', run.assets.uncertaintyCsv.absolutePath);
      await exec(values.python, args, { cwd: root, maxBuffer: 4 * 1024 * 1024 });
      const score = JSON.parse(await fs.readFile(scorePath, 'utf8'));
      record.score = score.summary;
      record.outcome = score.summary.completeExpectedLeads ? 'quantitative_needs_review' : 'scoring_failure';
    } else {
      // A policy refusal is a terminal abstention even though its UI status is
      // "failed". Timeouts, unavailable runtimes and crashes remain failures.
      record.outcome = run.publicationDecision && !run.processing?.failureCode
        ? 'abstention' : 'runtime_failure';
    }
    // Integrity-checked reload exercises persistence for every terminal outcome.
    assert.equal((await digitizer.getRun(run.id)).status, run.status);
  } catch (error) {
    record.outcome = 'runtime_failure';
    record.error = { code: error.code, message: error.message, runId: error.runId };
  }
  record.runtimeMs = Math.round(performance.now() - started);
  report.cases.push(record);
  await save();
  console.log(`${id}: ${record.outcome}; ${record.runtimeMs} ms`);
}
await verifyPayload();
for (const manifest of manifests) assert.equal(await hashFile(manifest.path), manifest.sha256);
for (const { item, artifacts } of cases) for (const key of ['image', 'truth', 'annotations']) assert.equal(await hashFile(artifacts[key]), item[key].sha256);
const quantitative = report.cases.filter(c => c.outcome === 'quantitative_needs_review');
const mean = key => quantitative.length ? quantitative.reduce((sum, c) => sum + c.score[key], 0) / quantitative.length : null;
const sampleCount = quantitative.reduce((sum, c) => sum + c.score.comparedSamples, 0);
report.summary = {
  quantitative: quantitative.length, abstentions: report.cases.filter(c => c.outcome === 'abstention').length,
  failures: report.cases.filter(c => !['abstention', 'quantitative_needs_review'].includes(c.outcome)).length,
  quantitativeYield: quantitative.length / cases.length, comparedSamples: sampleCount,
  globalRmseUv: sampleCount ? Math.sqrt(quantitative.reduce((sum, c) => sum + c.score.globalRmseUv ** 2 * c.score.comparedSamples, 0) / sampleCount) : null,
  meanCaseCorrelation: mean('macroMeanCorrelation'), meanCaseCoverage: mean('macroMeanCoverage'),
  totalRuntimeMs: report.cases.reduce((sum, c) => sum + c.runtimeMs, 0),
};
report.finishedAt = new Date().toISOString();
await save();
console.log(JSON.stringify(report.summary));
if (report.summary.failures) process.exitCode = 1;
