import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import ts from 'typescript';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = async (file) => JSON.parse(await fs.readFile(path.join(root, file), 'utf8'));
const release = await readJson('config/waveparse-release.json');
const js = path.join(root, 'packages/javascript');
const py = path.join(root, 'packages/python');
const stage = path.join(root, 'build/waveparse/runtime');
const digest = (data) => createHash('sha256').update(data).digest('hex');
async function walk(directory) {
  const result = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    if (entry.name === '__pycache__' || entry.name.startsWith('.')) continue;
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await walk(file));
    else if (entry.isFile()) result.push(file);
    else throw new Error(`Symlink/special file excluded: ${file}`);
  }
  return result.sort();
}
async function copy(relative, destination = stage) {
  const target = path.join(destination, relative);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(path.join(root, relative), target);
}
const referenceStub = 'export const LOCAL_REFERENCE_DIGITIZATIONS = [];';
const plugins = [{ name: 'exclude-local-reference', setup(builder) {
  builder.onLoad({ filter: /\/local-reference\.ts$/ }, () => ({ contents: referenceStub, loader: 'ts' }));
}}];
await fs.rm(path.join(root, 'build/waveparse'), { recursive: true, force: true });
await fs.mkdir(stage, { recursive: true });
await fs.rm(path.join(js, 'dist'), { recursive: true, force: true });
await fs.mkdir(path.join(js, 'dist'), { recursive: true });
const base = { absWorkingDir: root, platform: 'node', target: 'node22', bundle: true, plugins, logLevel: 'warning', sourcemap: false, metafile: true };
const bundleInputs = new Set();
function recordBundle(result) {
  for (const input of Object.keys(result.metafile.inputs)) {
    // If a future change bundles an external JS dependency, its licence must
    // be deliberately included instead of silently disappearing in the bundle.
    if (input.split('/').includes('node_modules')) throw new Error(`Bundled dependency needs attribution: ${input}`);
    bundleInputs.add(input);
  }
}
recordBundle(await build({ ...base, entryPoints: ['packages/runtime/runner.ts'], outfile: path.join(stage, 'runner.cjs'), format: 'cjs' }));
for (const format of ['esm', 'cjs']) {
  recordBundle(await build({ ...base, entryPoints: ['packages/javascript/src/index.ts'], outfile: path.join(js, 'dist', format === 'esm' ? 'index.js' : 'index.cjs'), format,
    define: { __WAVEPARSE_VERSION__: JSON.stringify(release.version), __WAVEPARSE_FORMAT__: JSON.stringify(format), ...(format === 'cjs' ? { 'import.meta.url': 'undefined' } : {}) },
  }));
}
recordBundle(await build({ ...base, entryPoints: ['packages/javascript/src/cli.ts'], outfile: path.join(stage, 'cli.cjs'), format: 'cjs',
  define: { __WAVEPARSE_VERSION__: JSON.stringify(release.version), __WAVEPARSE_FORMAT__: '"cjs"', 'import.meta.url': 'undefined' },
}));
await fs.copyFile(path.join(stage, 'cli.cjs'), path.join(js, 'dist/cli.cjs'));
await fs.chmod(path.join(js, 'dist/cli.cjs'), 0o755);
await fs.chmod(path.join(stage, 'cli.cjs'), 0o755);

const declarations = ts.createProgram([path.join(root, 'packages/javascript/src/index.ts')], {
  declaration: true, emitDeclarationOnly: true, outDir: path.join(js, 'dist/types'), rootDir: root,
  module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
  target: ts.ScriptTarget.ES2022, esModuleInterop: true, strict: true, skipLibCheck: true,
  resolveJsonModule: true, baseUrl: root, paths: { '@/*': ['src/*'] },
});
const emitted = declarations.emit();
const diagnostics = [...ts.getPreEmitDiagnostics(declarations), ...emitted.diagnostics];
if (diagnostics.length) throw new Error(ts.formatDiagnosticsWithColorAndContext(diagnostics, { getCanonicalFileName: (f) => f, getCurrentDirectory: () => root, getNewLine: () => '\n' }));
// TypeScript's declaration emitter retains extensionless relative imports.
// Add .js suffixes so declarations work with NodeNext as well as bundlers.
for (const file of await walk(path.join(js, 'dist/types'))) {
  if (file.endsWith('.d.ts')) {
    let source = await fs.readFile(file, 'utf8');
    source = source.replace(/(["'])@\/([^"']+)\1/g, (_, quote, name) => {
      let relative = path.relative(path.dirname(file), path.join(js, 'dist/types/src', name)).split(path.sep).join('/');
      if (!relative.startsWith('.')) relative = './' + relative;
      return quote + relative + '.js' + quote;
    });
    source = source.replace(/(from\s+["'])(\.[^"']+)(["'])/g, (_, prefix, name, suffix) => prefix + (/\.(js|json)$/.test(name) ? name : name + '.js') + suffix);
    await fs.writeFile(file, source);
  }
}
await fs.writeFile(path.join(js, 'dist/index.d.ts'), 'export * from "./types/packages/javascript/src/index.js";\n');

const resources = [
  'config/ecg-domain.v1.json', 'config/capture-quality.v1.json', 'config/digitizer-policy.v2.json',
  'config/candidate-selector-calibration.v2.json', 'config/waveparse-release.json', 'config/waveparse-runtime.json',
  'requirements-runtime-bootstrap.txt', 'requirements-runtime-macos-arm64.txt', 'requirements-runtime-linux-x64.txt',
  'scripts/prepare_ecg_input.py', 'scripts/detect_ecg_layout.py', 'output/render_digitized_paper.py',
];
for (const file of await walk(path.join(root, 'ecg_pipeline'))) {
  if (/\.(py|js|yml)$/.test(file)) resources.push(path.relative(root, file));
}
for (const file of resources) await copy(file);
await fs.copyFile(path.join(root, 'packages/runtime/runtime_setup.py'), path.join(stage, 'runtime_setup.py'));
const manifest = {};
for (const file of await walk(stage)) manifest[path.relative(stage, file).split(path.sep).join('/')] = digest(await fs.readFile(file));
await fs.writeFile(path.join(stage, 'payload-manifest.json'), JSON.stringify({ version: release.version, files: manifest }, null, 2) + '\n');
for (const destination of [path.join(js, 'runtime'), path.join(py, 'src/ecg_waveparse/runtime')]) {
  await fs.rm(destination, { recursive: true, force: true });
  await fs.cp(stage, destination, { recursive: true });
}
for (const directory of [js, py]) {
  await fs.rm(path.join(directory, 'docs'), { recursive: true, force: true });
  await fs.cp(path.join(root, 'docs/waveparse'), path.join(directory, 'docs'), { recursive: true });
  for (const file of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) await fs.copyFile(path.join(root, file), path.join(directory, file));
  await fs.rm(path.join(directory, 'LICENSES'), { recursive: true, force: true });
  await fs.cp(path.join(root, 'LICENSES'), path.join(directory, 'LICENSES'), { recursive: true });
  await fs.copyFile(path.join(root, 'docs/waveparse/README.md'), path.join(directory, 'README.md'));
}
const jsManifest = await readJson('packages/javascript/package.json');
if (jsManifest.version !== release.version || !String(await fs.readFile(path.join(py, 'pyproject.toml'))).includes(`version = "${release.version}"`)) throw new Error('Package versions disagree.');
await fs.mkdir(path.join(root, 'dist'), { recursive: true });
execFileSync('npm', ['pack', js, '--pack-destination', path.join(root, 'dist'), '--ignore-scripts', '--cache', path.join(root, 'build/npm-cache')], { cwd: root, stdio: 'inherit' });

// Corresponding source, without existing repository history or clinical data.
const source = path.join(root, 'build/waveparse/source');
await fs.mkdir(source, { recursive: true });
const sourceFiles = new Set([...resources, 'package.json', 'pnpm-lock.yaml', 'tsconfig.json', 'requirements-runtime.in', 'requirements-waveparse-build.txt', 'config/release-runtime.json', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'scripts/build-waveparse.mjs', 'scripts/freeze-waveparse-runtime.py']);
for (const prefix of ['src/lib', 'packages/runtime', 'packages/javascript/src', 'packages/python/src/ecg_waveparse', 'docs/waveparse', 'LICENSES']) {
  for (const file of await walk(path.join(root, prefix))) {
    if (file.includes('/runtime/') && prefix.endsWith('ecg_waveparse')) continue;
    if (file.endsWith('.pyc')) continue;
    sourceFiles.add(path.relative(root, file));
  }
}
sourceFiles.add('packages/javascript/package.json'); sourceFiles.add('packages/python/pyproject.toml');
for (const file of ['scripts/waveparse.test.ts', 'scripts/test_waveparse_runtime.py', 'scripts/verify-waveparse-packages.mjs', 'scripts/waveparse-linux-smoke.sh', 'scripts/waveparse-inference-smoke.mjs', 'scripts/generate_benchmark.py', 'ecg_benchmark/__init__.py', 'ecg_benchmark/generate.py', 'ecg_benchmark/io.py', 'benchmark/suites/smoke.json', 'benchmark/schemas/case-manifest.schema.json', 'benchmark/schemas/annotations.schema.json', '.github/workflows/waveparse-packages.yml']) {
  try { await fs.access(path.join(root, file)); sourceFiles.add(file); } catch { /* Optional before first test build. */ }
}
for (const file of ['scripts/waveparse-runtime-inventory.py', 'scripts/waveparse-regression.mjs', 'scripts/audit-waveparse-source.mjs', 'scripts/score_digitization.py', 'scripts/check_waveparse_regression.py', 'ecg_benchmark/scoring.py', 'ecg_benchmark/gates.py', 'benchmark/suites/smoke-gates.json', 'benchmark/suites/waveparse-3x4.json', 'benchmark/suites/waveparse-12x1.json']) sourceFiles.add(file);
sourceFiles.add('scripts/digitizer-reliability.test.ts');
sourceFiles.add('scripts/test_native_grid_digitizer.py');
for (const file of sourceFiles) {
  if (file === 'src/lib/local-reference.ts') continue;
  await copy(file, source);
}
await fs.writeFile(path.join(source, 'src/lib/local-reference.ts'), 'import type { KnownDigitization } from "./runs";\nexport const LOCAL_REFERENCE_DIGITIZATIONS: KnownDigitization[] = [];\n');
await fs.copyFile(path.join(root, 'docs/waveparse/README.md'), path.join(source, 'README.md'));
const sourcePackage = await readJson('package.json');
sourcePackage.name = 'ecg-waveparse-source';
sourcePackage.scripts = Object.fromEntries(Object.entries(sourcePackage.scripts).filter(([name]) => name.startsWith('waveparse:')));
await fs.writeFile(path.join(source, 'package.json'), JSON.stringify(sourcePackage, null, 2) + '\n');
await fs.writeFile(path.join(source, 'RELEASE-PREPARATION.txt'), 'Registry publishing is disabled. See THIRD_PARTY_NOTICES.md before public distribution.\n');
await fs.writeFile(path.join(source, '.gitignore'), 'node_modules/\nbuild/\ndist/\n.venv/\n__pycache__/\n*.pyc\n*.tsbuildinfo\n.env*\npackages/javascript/dist/\npackages/javascript/runtime/\npackages/python/src/ecg_waveparse/runtime/\npackages/javascript/docs/\npackages/python/docs/\npackages/javascript/README.md\npackages/python/README.md\npackages/javascript/LICENSE\npackages/python/LICENSE\npackages/javascript/LICENSES/\npackages/python/LICENSES/\npackages/javascript/THIRD_PARTY_NOTICES.md\npackages/python/THIRD_PARTY_NOTICES.md\n');
await fs.writeFile(path.join(root, 'dist/bundle-inputs.json'), JSON.stringify({ externalJavaScriptDependencies: [], inputs: [...bundleInputs].sort() }, null, 2) + '\n');
execFileSync('tar', ['-czf', path.join(root, 'dist', `${release.name}-${release.version}-source.tar.gz`), '-C', source, '.'], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
console.log(`Prepared ${release.name} ${release.version}; ${Object.keys(manifest).length} shared runtime files. Build Python distributions with python -m build --no-isolation packages/python --outdir dist.`);
