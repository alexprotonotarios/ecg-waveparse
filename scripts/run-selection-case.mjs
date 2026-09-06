// Local research harness. It calls the unmodified production core and retains
// working candidates for offline scoring; nothing is installed as a new policy.
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import assert from 'node:assert/strict';

const { values } = parseArgs({ options: {
  protocol: { type: 'string' }, case: { type: 'string' }, profile: { type: 'string' },
  output: { type: 'string' }, runtime: { type: 'string' }, resources: { type: 'string' },
}});
for (const key of ['protocol', 'case', 'profile', 'output', 'runtime', 'resources']) assert(values[key]);
assert(['production', 'benchmark'].includes(values.profile));
const output = path.resolve(values.output);
await fs.mkdir(output, { recursive: false, mode: 0o700 });
process.env.WAVEPARSE_WORKSPACE_ROOT = path.join(output, 'workspace');
process.env.WAVEPARSE_RESOURCE_ROOT = path.resolve(values.resources);
process.env.OPEN_ECG_DIGITIZER_DIR = path.resolve(values.runtime, 'engine');
process.env.ECG_DIGITIZER_DEVICE = 'mps';
const load = async file => { const imported = await import(file); return imported.default ?? imported; };
const { digitizeRun } = await load('../src/lib/digitizer.ts');
const { createStoredRunFromBytes, saveRun } = await load('../src/lib/runs.ts');
const { preflightCaptureBytes } = await load('../src/lib/digitizer/capture-preflight.ts');
const protocol = JSON.parse(await fs.readFile(values.protocol, 'utf8'));
const member = protocol.membership.find(m => m.caseId === values.case);
assert(member);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const source = path.resolve(path.dirname(values.protocol), member.image);
const bytes = await fs.readFile(source);
assert.equal(hash(bytes), member.imageSha256);
const stages = {};
const started = performance.now();
let phase = performance.now();
const captureQuality = await preflightCaptureBytes({ fileName: path.basename(source), bytes, captureMethod: 'upload' });
stages.capturePreflightMs = performance.now() - phase;
let run = await createStoredRunFromBytes({ id: `run_${randomUUID().replaceAll('-', '')}`, fileName: path.basename(source), bytes, captureQuality });
await fs.mkdir(path.join(output, 'workspace', run.localPath), { recursive: true });
await fs.writeFile(path.join(output, '.keep-artifacts.json'), JSON.stringify({ version: 1, reason: 'Explicit retained selection experiment; preserve candidates and denominator.' }));
const controller = new AbortController();
const deadline = setTimeout(() => controller.abort(new Error('experiment deadline')), protocol.maximumCaseSeconds * 1000);
let failure;
try {
  const patch = await digitizeRun(run, { candidateMode: values.profile, signal: controller.signal,
    onStage: (stage, duration) => { stages[stage] = duration; } });
  run = { ...run, ...patch, assets: { ...run.assets, ...patch.assets } };
} catch (error) {
  failure = { type: error.name, message: error.message };
  run = { ...run, status: 'failed', message: error.message };
} finally {
  clearTimeout(deadline);
  if (controller.signal.aborted) run = { ...run, status: 'timed_out', assets: { input: run.assets.input } };
  await saveRun(run);
}
const candidates = [];
for (const candidate of run.digitizer?.candidates ?? []) {
  const row = { ...candidate };
  if (candidate.localPath) {
    const directory = path.resolve(output, 'workspace', candidate.localPath);
    assert(directory.startsWith(path.join(output, 'workspace') + path.sep));
    const files = await fs.readdir(directory).catch(error => { if (error.code === 'ENOENT') return []; throw error; });
    const csvs = files.filter(name => name.endsWith('_timeseries_canonical.csv'));
    if (csvs.length === 1) row.canonicalPath = path.relative(output, path.join(directory, csvs[0]));
  }
  candidates.push(row);
}
const result = { version: 1, clinicalValidationUse: false, caseId: member.caseId, profile: values.profile,
  sourceSha256: hash(bytes), sourceCore: 'Unmodified current TypeScript core; candidate retention is a harness operation.',
  protocolSha256: hash(await fs.readFile(values.protocol)), totalRuntimeMs: performance.now() - started,
  stages, timedOut: controller.signal.aborted, failure, runId: run.id, status: run.status,
  selectedCandidateId: run.selectedCandidateId, publicationDecision: run.publicationDecision,
  selectedAssets: Object.fromEntries(Object.entries(run.assets).map(([key, asset]) => [key, {
    path: path.relative(output, path.resolve(output, 'workspace', asset.path)), sha256: asset.identity?.sha256,
  }])), candidates, pipelineEvidence: run.digitizer?.evidence };
await fs.writeFile(path.join(output, 'case-result.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ caseId: member.caseId, profile: values.profile, status: run.status, candidateCount: candidates.length, runtimeMs: result.totalRuntimeMs }));
