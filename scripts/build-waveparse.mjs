import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import ts from 'typescript';
import { parseArgs } from 'node:util';
import { load as loadYaml, dump as dumpYaml } from 'js-yaml';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: { outdir: { type: 'string', default: 'dist' } } });
const dist = path.resolve(root, values.outdir);
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
    source = source.replace(/((?:from\s+|import\(\s*)["'])(\.[^"']+)(["'])/g, (_, prefix, name, suffix) => prefix + (/\.(js|json)$/.test(name) ? name : name + '.js') + suffix);
    await fs.writeFile(file, source);
  }
}
await fs.writeFile(path.join(js, 'dist/index.d.ts'), 'export * from "./types/packages/javascript/src/index.js";\n');

const resources = [
  'config/ecg-domain.v1.json', 'config/capture-quality.v1.json', 'config/digitizer-policy.v2.json', 'config/digitizer-policy.v3.json',
  'config/candidate-selector-calibration.v2.json', 'config/waveparse-release.json', 'config/waveparse-runtime.json',
  'config/selector-parameter-contract.v1.json',
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
await fs.mkdir(dist, { recursive: true });
execFileSync('npm', ['pack', js, '--pack-destination', dist, '--ignore-scripts', '--cache', path.join(root, 'build/npm-cache')], { cwd: root, stdio: 'inherit' });

// Corresponding source, without existing repository history or clinical data.
const source = path.join(root, 'build/waveparse/source');
await fs.mkdir(source, { recursive: true });
const sourceFiles = new Set([...resources, 'package.json', 'pnpm-lock.yaml', 'tsconfig.json', 'requirements-runtime.in', 'requirements-waveparse-build.txt', 'config/release-runtime.json', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'scripts/build-waveparse.mjs', 'scripts/freeze-waveparse-runtime.py']);
for (const prefix of ['src/lib', 'packages/runtime', 'packages/javascript/src', 'packages/python/src/ecg_waveparse', 'docs/waveparse', 'LICENSES']) {
  for (const file of await walk(path.join(root, prefix))) {
    if (file.includes('/runtime/') && prefix.endsWith('ecg_waveparse')) continue;
    if (file.endsWith('.pyc')) continue;
    // The public library source has no Next.js UI or UI-only utility imports.
    if (['src/lib/local-ui-access.ts', 'src/lib/utils.ts'].includes(path.relative(root, file))) continue;
    sourceFiles.add(path.relative(root, file));
  }
}
sourceFiles.add('packages/javascript/package.json'); sourceFiles.add('packages/python/pyproject.toml');
for (const file of ['scripts/waveparse.test.ts', 'scripts/test_waveparse_runtime.py', 'scripts/verify-waveparse-packages.mjs', 'scripts/waveparse-linux-smoke.sh', 'scripts/waveparse-inference-smoke.mjs', 'scripts/generate_benchmark.py', 'ecg_benchmark/__init__.py', 'ecg_benchmark/generate.py', 'ecg_benchmark/io.py', 'benchmark/suites/smoke.json', 'benchmark/schemas/case-manifest.schema.json', 'benchmark/schemas/annotations.schema.json', '.github/workflows/waveparse-packages.yml']) {
  try { await fs.access(path.join(root, file)); sourceFiles.add(file); } catch { /* Optional before first test build. */ }
}
for (const file of ['scripts/waveparse-runtime-inventory.py', 'scripts/waveparse-regression.mjs', 'scripts/audit-waveparse-source.mjs', 'scripts/score_digitization.py', 'scripts/check_waveparse_regression.py', 'ecg_benchmark/scoring.py', 'ecg_benchmark/gates.py', 'benchmark/suites/smoke-gates.json', 'benchmark/suites/waveparse-3x4.json', 'benchmark/suites/waveparse-12x1.json']) sourceFiles.add(file);
sourceFiles.add('scripts/digitizer-reliability.test.ts');
for (const file of ['scripts/waveparse-ci.test.ts', 'scripts/physical-contracts.test.ts', 'scripts/evidence-retention.test.ts', 'scripts/segment-evidence.test.ts', 'scripts/outcome-contracts.test.ts', 'scripts/test_score_digitization.py', 'scripts/test_scoring_coordinates.py', 'ecg_benchmark/coordinates.py']) sourceFiles.add(file);
for (const file of ['scripts/geometry-evidence.test.ts', 'scripts/selector-feature-identity.test.ts', 'scripts/lineage.test.ts', 'scripts/interpreter-evidence.test.ts', 'scripts/test_evaluation_contract.py', 'scripts/test_engineering_experiments.py', 'scripts/test_calibrate_candidate_selector.py', 'scripts/calibrate_candidate_selector.py', 'scripts/run_engineering_experiments.py', 'scripts/profile_waveparse.py', 'scripts/profile_model_loading.py', 'scripts/summarize_public_engineering_pilot.py', 'ecg_benchmark/evaluation.py', 'benchmark/requirements.txt']) sourceFiles.add(file);
for (const file of await walk(path.join(root, 'benchmark/protocols'))) sourceFiles.add(path.relative(root, file));
for (const file of ['scripts/durable-write-faults.test.ts', 'scripts/input-error-taxonomy.test.ts', 'scripts/test_input_boundaries.py', 'scripts/selection_ablation.py', 'scripts/test_selection_ablation.py']) sourceFiles.add(file);
for (const file of ['scripts/build_public_engineering_pilot.py', 'scripts/test_public_engineering_pilot.py', 'ecg_benchmark/paired_render.py', 'ecg_benchmark/wfdb_source.py']) sourceFiles.add(file);
for (const file of ['scripts/test_printed_calibration.py', 'scripts/test_rendered_calibration.py', 'scripts/evaluate_printed_calibration.py', 'scripts/verify_printed_calibration_installed.mjs', 'scripts/evaluate_detected_grid.py']) sourceFiles.add(file);
for (const file of ['scripts/experimental_pretrained_heatmap.py', 'scripts/evaluate_pretrained_crops.py', 'scripts/evaluate_full_image_decoders.py']) sourceFiles.add(file);
for (const file of ['scripts/build_pmcardio_truth.py', 'scripts/test_pmcardio_identity.py', 'ecg_benchmark/manifest.py']) sourceFiles.add(file);
for (const file of ['ecg_benchmark/measurements.py', 'scripts/test_labelled_measurements.py', 'scripts/build_labelled_measurement_fixtures.py']) sourceFiles.add(file);
sourceFiles.add('requirements-waveparse-tests.txt');
for (const file of ['scripts/profile_warm_inference.py', 'scripts/build_capture_development_subset.py', 'benchmark/schemas/paired-catalog.schema.json']) sourceFiles.add(file);
for (const file of ['scripts/evaluation-protocol.mjs', 'scripts/evaluation-protocol.test.ts']) sourceFiles.add(file);
for (const file of ['scripts/run-selection-case.mjs', 'scripts/run_selection_experiment.py', 'scripts/summarize_selection_experiment.py', 'scripts/test_current_selection_experiment.py']) sourceFiles.add(file);
for (const file of ['scripts/platform-inference.mjs', 'scripts/platform_inference.py', 'scripts/run_platform_matrix.py', 'scripts/test_platform_matrix.py']) sourceFiles.add(file);
for (const file of ['scripts/cancel_neural_sitecustomize.py', 'scripts/cancel-neural-inference.mjs', 'scripts/cancel_neural_inference.py']) sourceFiles.add(file);
for (const file of ['scripts/freeze_engineering_snapshot.py', 'scripts/freeze_final_engineering_evaluation.py', 'scripts/test_final_engineering_membership.py']) sourceFiles.add(file);
for (const file of ['scripts/build_physical_settings_fixtures.py', 'scripts/run_physical_settings_experiment.py']) sourceFiles.add(file);
sourceFiles.add('scripts/test_physical_settings.py');
sourceFiles.add('scripts/test_fidelity_inference_wrapper.py');
for (const file of ['scripts/platform_evidence.py', 'scripts/test_platform_evidence.py']) sourceFiles.add(file);
for (const file of ['examples/waveparse.mjs', 'examples/waveparse.py', 'docker/waveparse.Dockerfile', '.dockerignore']) sourceFiles.add(file);
sourceFiles.add('scripts/test_native_grid_digitizer.py');
sourceFiles.add('scripts/test_waveparse_regression_gates.py');
sourceFiles.add('benchmark/suites/waveparse-smoke-gates.json');
for (const file of ['.github/workflows/publish-waveparse.yml', 'scripts/verify_waveparse_release.py', 'scripts/test_waveparse_release.py']) sourceFiles.add(file);
for (const file of sourceFiles) {
  if (file === 'src/lib/local-reference.ts') continue;
  await copy(file, source);
}
await fs.writeFile(path.join(source, 'src/lib/local-reference.ts'), 'import type { KnownDigitization } from "./runs";\nexport const LOCAL_REFERENCE_DIGITIZATIONS: KnownDigitization[] = [];\n');
await fs.copyFile(path.join(root, 'docs/waveparse/README.md'), path.join(source, 'README.md'));
const sourcePackage = await readJson('package.json');
sourcePackage.name = 'ecg-waveparse-source';
sourcePackage.scripts = Object.fromEntries(Object.entries(sourcePackage.scripts).filter(([name]) => name.startsWith('waveparse:')));
const availableBuildDependencies = { ...sourcePackage.dependencies, ...sourcePackage.devDependencies };
sourcePackage.devDependencies = Object.fromEntries(['@types/node', 'esbuild', 'js-yaml', 'typescript'].map(name => {
  if (!availableBuildDependencies[name]) throw new Error(`Missing source build dependency: ${name}`);
  return [name, availableBuildDependencies[name]];
}));
// Preserve tsx's original lockfile section so pruning needs no registry metadata.
if (!availableBuildDependencies.tsx) throw new Error('Missing source build dependency: tsx');
sourcePackage.dependencies = { tsx: availableBuildDependencies.tsx };
await fs.writeFile(path.join(source, 'package.json'), JSON.stringify(sourcePackage, null, 2) + '\n');
// Keep the exact reachable lock records, including every platform's optional binary.
// No dependency resolution, registry access, or application lock rewrite occurs here.
const sourceLock = loadYaml(await fs.readFile(path.join(root, 'pnpm-lock.yaml'), 'utf8'));
if (String(sourceLock.lockfileVersion) !== '9.0' || !sourceLock.importers?.['.'] || !sourceLock.snapshots) throw new Error('Unsupported source lock format');
const importer = {};
for (const kind of ['dependencies', 'devDependencies']) {
  importer[kind] = {};
  for (const [name, specifier] of Object.entries(sourcePackage[kind])) {
    const locked = sourceLock.importers['.'][kind]?.[name];
    if (!locked || locked.specifier !== specifier) throw new Error(`Source dependency is not locked: ${name}`);
    importer[kind][name] = locked;
  }
}
const pending = Object.values(importer).flatMap(dependencies => Object.entries(dependencies).map(([name, entry]) => `${name}@${entry.version}`));
const snapshots = {}, packages = {};
while (pending.length) {
  const key = pending.pop();
  if (Object.hasOwn(snapshots, key)) continue;
  const snapshot = sourceLock.snapshots[key], packageKey = key.split('(')[0];
  if (!snapshot || !sourceLock.packages[packageKey]) throw new Error(`Missing source dependency lock record: ${key}`);
  snapshots[key] = snapshot;
  packages[packageKey] = sourceLock.packages[packageKey];
  for (const [name, version] of Object.entries({ ...snapshot.dependencies, ...snapshot.optionalDependencies })) pending.push(`${name}@${version}`);
}
const sortedEntries = object => Object.fromEntries(Object.entries(object).sort(([a], [b]) => a.localeCompare(b)));
await fs.writeFile(path.join(source, 'pnpm-lock.yaml'), dumpYaml({ ...sourceLock, importers: { '.': importer }, packages: sortedEntries(packages), snapshots: sortedEntries(snapshots) }, { lineWidth: -1, noRefs: true }));
await fs.writeFile(path.join(source, 'RELEASE-PREPARATION.txt'), 'Source preview. Registry publishing remains disabled. See THIRD_PARTY_NOTICES.md for code licences and the separately downloaded runtime.\n');
await fs.writeFile(path.join(source, '.gitignore'), 'node_modules/\nbuild/\ndist/\n.venv/\n__pycache__/\n*.pyc\n*.tsbuildinfo\n.env*\npackages/javascript/dist/\npackages/javascript/runtime/\npackages/python/src/ecg_waveparse/runtime/\npackages/javascript/docs/\npackages/python/docs/\npackages/javascript/README.md\npackages/python/README.md\npackages/javascript/LICENSE\npackages/python/LICENSE\npackages/javascript/LICENSES/\npackages/python/LICENSES/\npackages/javascript/THIRD_PARTY_NOTICES.md\npackages/python/THIRD_PARTY_NOTICES.md\n');
await fs.writeFile(path.join(dist, 'bundle-inputs.json'), JSON.stringify({ externalJavaScriptDependencies: [], inputs: [...bundleInputs].sort() }, null, 2) + '\n');
execFileSync('tar', ['-czf', path.join(dist, `${release.name}-${release.version}-source.tar.gz`), '-C', source, '.'], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
console.log(`Prepared ${release.name} ${release.version}; ${Object.keys(manifest).length} shared runtime files. Archives: ${dist}. Build Python distributions with python -m build --no-isolation packages/python --outdir PATH.`);
