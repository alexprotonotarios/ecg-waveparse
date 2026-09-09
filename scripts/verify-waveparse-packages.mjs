import { promises as fs } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: { dist: { type: 'string', default: 'dist' } } });
const dist = path.resolve(root, values.dist);
const distribution = JSON.parse(await fs.readFile(path.join(root, 'config/waveparse-distribution.json'), 'utf8'));
const scratch = await fs.mkdtemp(path.join(tmpdir(), 'waveparse-consumer-'));
const python = process.env.WAVEPARSE_PYTHON || 'python3.12';
const run = (command, args, options = {}) => execFileSync(command, args, { cwd: scratch, stdio: 'inherit', ...options });
try {
  run(python, ['-c', String.raw`
import re, sys, tarfile, zipfile
from pathlib import Path
root = Path(sys.argv[1])
names = ['ecg-waveparse-0.1.0.tgz', 'ecg_waveparse-0.1.0-py3-none-any.whl', 'ecg_waveparse-0.1.0.tar.gz', 'ecg-waveparse-0.1.0-source.tar.gz']
for name in names:
    if name.endswith('.whl'):
        with zipfile.ZipFile(root / name) as archive:
            files = {p.filename: archive.read(p) for p in archive.infolist() if not p.is_dir()}
    else:
        with tarfile.open(root / name) as archive:
            members = archive.getmembers()
            assert all(p.isfile() or p.isdir() for p in members), 'Archive link or special file'
            files = {p.name: archive.extractfile(p).read() for p in members if p.isfile()}
    for path, data in files.items():
        assert not Path(path).is_absolute() and '..' not in Path(path).parts, path
        assert not re.search(r'(^|/)(benchmark|ecg_benchmark|input|storage|\.external|\.git|\.github/workflows)/|/(decoder_backends|local_grid_warp)\.py$|\.(png|jpe?g|pt|onnx)$', path), path
        if path.endswith(('.md', '.gitignore', '/PKG-INFO', '/METADATA')):
            assert not re.search(rb'benchmark|\bARVC\b|SEM-16|FINAL-EVALUATION|multisource_truth', data, re.I), path
    print('Inspected download:', name, len(files), 'files')
`, dist]);
  await fs.writeFile(path.join(scratch, 'package.json'), '{"private":true,"type":"module"}');
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--cache', path.join(scratch, 'cache'), path.join(dist, 'ecg-waveparse-0.1.0.tgz')]);
  run(python, ['-m', 'venv', path.join(scratch, 'venv')]);
  const consumerPython = path.join(scratch, 'venv/bin/python');
  run(consumerPython, ['-m', 'pip', 'install', '--no-index', '--no-deps', path.join(dist, 'ecg_waveparse-0.1.0-py3-none-any.whl')]);
  const jsRuntime = path.join(scratch, 'node_modules/ecg-waveparse/runtime');
  const pyRuntime = String(execFileSync(consumerPython, ['-c', 'from ecg_waveparse import _resources; print(_resources())'], { cwd: scratch })).trim();
  for (const directory of [path.dirname(jsRuntime), path.dirname(pyRuntime)]) {
    assert((await fs.readFile(path.join(directory, 'THIRD_PARTY_NOTICES.md'), 'utf8')).includes('CC BY-SA 4.0'));
    assert.deepEqual((await fs.readdir(path.join(directory, 'docs'))).sort(), ['API.md']);
    const readme = await fs.readFile(path.join(directory, 'README.md'), 'utf8');
    assert(!/benchmark|\bARVC\b|SEM-16|FINAL-EVALUATION/i.test(readme), 'Development material in installed README');
    assert.equal(createHash('sha256').update(await fs.readFile(path.join(directory, 'LICENSES/Open-ECG-Digitizer-CC-BY-SA-4.0.txt'))).digest('hex'), 'a1fdc6cdcf6bf2e6c0059da8f87108c7003fd7d1a2093179e7cb5767c7c6876b');
  }
  const manifest = JSON.parse(await fs.readFile(path.join(jsRuntime, 'payload-manifest.json'), 'utf8'));
  assert.deepEqual(Object.keys(manifest.files).sort(), [...distribution.runtimeFiles, 'cli.cjs', 'runner.cjs', 'runtime_setup.py'].sort());
  assert.deepEqual(manifest, JSON.parse(await fs.readFile(path.join(pyRuntime, 'payload-manifest.json'), 'utf8')));
  for (const [relative, expected] of Object.entries(manifest.files)) {
    for (const directory of [jsRuntime, pyRuntime]) {
      assert.equal(createHash('sha256').update(await fs.readFile(path.join(directory, relative))).digest('hex'), expected, relative);
    }
  }
  const inventory = async (dir, prefix = '') => {
    const files = [];
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (entry.name === '__pycache__') continue;
      const relative = prefix + entry.name;
      if (entry.isDirectory()) files.push(...await inventory(path.join(dir, entry.name), relative + '/'));
      else { assert(entry.isFile(), `Special installed resource: ${relative}`); files.push(relative); }
    }
    return files.sort();
  };
  for (const directory of [jsRuntime, pyRuntime]) {
    assert.deepEqual(await inventory(directory), [...Object.keys(manifest.files), 'payload-manifest.json'].sort());
    const profile = JSON.parse(await fs.readFile(path.join(directory, 'config/candidate-selector-calibration.v2.json'), 'utf8'));
    assert(!('benchmarkRunId' in profile.training) && !('suiteId' in profile.training));
  }
  run(process.execPath, [path.join(scratch, 'node_modules/ecg-waveparse/dist/cli.cjs'), '--help']);
  run(consumerPython, ['-m', 'ecg_waveparse', '--help']);
  await fs.writeFile(path.join(scratch, 'consumer.mjs'), `import { Digitizer } from 'ecg-waveparse'; import assert from 'node:assert/strict'; assert.equal(await new Digitizer({workspaceDir:'./esm'}).getRun('run_absent'),null);`);
  await fs.writeFile(path.join(scratch, 'consumer.cjs'), `const {Digitizer}=require('ecg-waveparse'); new Digitizer({workspaceDir:'./cjs'}).getRun('run_absent').then(x=>{require('node:assert/strict').equal(x,null)});`);
  run(process.execPath, ['consumer.mjs']); run(process.execPath, ['consumer.cjs']);
  await fs.writeFile(path.join(scratch, 'evidence.mjs'), `import { Digitizer } from 'ecg-waveparse'; import assert from 'node:assert/strict'; const d = new Digitizer({workspaceDir:'./esm'}); assert.equal(await d.getEvidence('run_absent'),null); await assert.rejects(d.getEvidence('../escape'),{code:'invalid_request'});`);
  run(process.execPath, ['evidence.mjs']);
  for (const suffix of ['mts', 'cts']) {
    await fs.writeFile(path.join(scratch, `consumer.${suffix}`), `import {Digitizer, type RunResult} from 'ecg-waveparse'; const digitizer = new Digitizer({workspaceDir:'./typed'}); const run: Promise<RunResult|null> = digitizer.getRun('run_absent'); void run;`);
  }
  run(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--target', 'ES2022', '--typeRoots', path.join(root, 'node_modules/@types'), 'consumer.mts', 'consumer.cts']);
  run(consumerPython, ['-c', `import asyncio\nfrom ecg_waveparse import Digitizer, WaveParseError\nd=Digitizer(workspace_dir='python')\nassert d.get_run('run_absent') is None\nassert asyncio.run(d.get_run_async('run_absent')) is None\ntry: d.get_run('../escape')\nexcept WaveParseError as e: assert e.code == 'invalid_request'\nelse: raise AssertionError('unsafe ID accepted')`]);
  const extracted = path.join(scratch, 'source');
  await fs.mkdir(extracted);
  run('tar', ['-xzf', path.join(dist, 'ecg-waveparse-0.1.0-source.tar.gz'), '-C', extracted]);
  run(process.execPath, [path.join(extracted, 'scripts/audit-waveparse-source.mjs'), extracted, path.join(scratch, 'source-audit.json')]);
  const members = String(execFileSync('tar', ['-tzf', path.join(dist, 'ecg-waveparse-0.1.0-source.tar.gz')]));
  assert(!members.split('\n').some(name => /(^|\/)(input|storage|\.git|\.external)(\/|$)|\.(png|jpe?g|onnx|pth|pt)$/.test(name)), 'Private data or models in source distribution');
  console.log('Installed npm and wheel consumers passed: identical payloads, ESM/CJS, NodeNext types, Python sync/async, source allowlist.');
} finally { await fs.rm(scratch, { recursive: true, force: true }); }
