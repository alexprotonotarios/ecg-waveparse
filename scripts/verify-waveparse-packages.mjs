import { promises as fs } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = await fs.mkdtemp(path.join(tmpdir(), 'waveparse-consumer-'));
const python = process.env.WAVEPARSE_PYTHON || path.join(root, '.external/open-ecg-digitizer/.venv/bin/python');
const run = (command, args, options = {}) => execFileSync(command, args, { cwd: scratch, stdio: 'inherit', ...options });
try {
  await fs.writeFile(path.join(scratch, 'package.json'), '{"private":true,"type":"module"}');
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--cache', path.join(scratch, 'cache'), path.join(root, 'dist/ecg-waveparse-0.1.0.tgz')]);
  run(python, ['-m', 'venv', path.join(scratch, 'venv')]);
  const consumerPython = path.join(scratch, 'venv/bin/python');
  run(consumerPython, ['-m', 'pip', 'install', '--no-index', '--no-deps', path.join(root, 'dist/ecg_waveparse-0.1.0-py3-none-any.whl')]);
  const jsRuntime = path.join(scratch, 'node_modules/ecg-waveparse/runtime');
  const pyRuntime = String(execFileSync(consumerPython, ['-c', 'from ecg_waveparse import _resources; print(_resources())'], { cwd: scratch })).trim();
  for (const directory of [path.dirname(jsRuntime), path.dirname(pyRuntime)]) {
    assert((await fs.readFile(path.join(directory, 'THIRD_PARTY_NOTICES.md'), 'utf8')).includes('CC BY-SA 4.0'));
    await fs.access(path.join(directory, 'docs/API.md'));
    assert.equal(createHash('sha256').update(await fs.readFile(path.join(directory, 'LICENSES/Open-ECG-Digitizer-CC-BY-SA-4.0.txt'))).digest('hex'), 'a1fdc6cdcf6bf2e6c0059da8f87108c7003fd7d1a2093179e7cb5767c7c6876b');
  }
  const manifest = JSON.parse(await fs.readFile(path.join(jsRuntime, 'payload-manifest.json'), 'utf8'));
  assert.deepEqual(manifest, JSON.parse(await fs.readFile(path.join(pyRuntime, 'payload-manifest.json'), 'utf8')));
  for (const [relative, expected] of Object.entries(manifest.files)) {
    for (const directory of [jsRuntime, pyRuntime]) {
      assert.equal(createHash('sha256').update(await fs.readFile(path.join(directory, relative))).digest('hex'), expected, relative);
    }
  }
  await fs.writeFile(path.join(scratch, 'consumer.mjs'), `import { Digitizer } from 'ecg-waveparse'; import assert from 'node:assert/strict'; assert.equal(await new Digitizer({workspaceDir:'./esm'}).getRun('run_absent'),null);`);
  await fs.writeFile(path.join(scratch, 'consumer.cjs'), `const {Digitizer}=require('ecg-waveparse'); new Digitizer({workspaceDir:'./cjs'}).getRun('run_absent').then(x=>{require('node:assert/strict').equal(x,null)});`);
  run(process.execPath, ['consumer.mjs']); run(process.execPath, ['consumer.cjs']);
  for (const suffix of ['mts', 'cts']) {
    await fs.writeFile(path.join(scratch, `consumer.${suffix}`), `import {Digitizer, type RunResult} from 'ecg-waveparse'; const digitizer = new Digitizer({workspaceDir:'./typed'}); const run: Promise<RunResult|null> = digitizer.getRun('run_absent'); void run;`);
  }
  run(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--target', 'ES2022', '--typeRoots', path.join(root, 'node_modules/@types'), 'consumer.mts', 'consumer.cts']);
  run(consumerPython, ['-c', `import asyncio\nfrom ecg_waveparse import Digitizer, WaveParseError\nd=Digitizer(workspace_dir='python')\nassert d.get_run('run_absent') is None\nassert asyncio.run(d.get_run_async('run_absent')) is None\ntry: d.get_run('../escape')\nexcept WaveParseError as e: assert e.code == 'invalid_request'\nelse: raise AssertionError('unsafe ID accepted')`]);
  const members = String(execFileSync('tar', ['-tzf', path.join(root, 'dist/ecg-waveparse-0.1.0-source.tar.gz')]));
  assert(!members.split('\n').some(name => /(^|\/)(input|storage|\.git|\.external)(\/|$)|\.(png|jpe?g|onnx|pth|pt)$/.test(name)), 'Private data or models in source distribution');
  console.log('Installed npm and wheel consumers passed: identical payloads, ESM/CJS, NodeNext types, Python sync/async, source allowlist.');
} finally { await fs.rm(scratch, { recursive: true, force: true }); }
