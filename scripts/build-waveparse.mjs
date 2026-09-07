import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import ts from 'typescript';
import { parseArgs } from 'node:util';
import { load as loadYaml, dump as dumpYaml } from 'js-yaml';
import { distributionResource, sourceScripts } from './waveparse-distribution.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: { outdir: { type: 'string', default: 'dist' } } });
const dist = path.resolve(root, values.outdir);
const readJson = async (file) => JSON.parse(await fs.readFile(path.join(root, file), 'utf8'));
const release = await readJson('config/waveparse-release.json');
if (process.versions.node !== release.node) throw new Error(`Node.js ${release.node} is required to build this release.`);
const distribution = await readJson('config/waveparse-distribution.json');
const userDocs = path.join(root, 'packages/distribution');
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
  await fs.writeFile(target, distributionResource(relative, await fs.readFile(path.join(root, relative))));
}
const referenceStub = 'export const LOCAL_REFERENCE_DIGITIZATIONS = [];';
const plugins = [{ name: 'exclude-local-reference', setup(builder) {
  builder.onLoad({ filter: /\/local-reference\.ts$/ }, () => ({ contents: referenceStub, loader: 'ts' }));
  builder.onLoad({ filter: /\/config\/(candidate-selector-calibration\.v2|digitizer-policy\.v[23])\.json$/ }, async ({ path: file }) => ({
    contents: String(distributionResource(path.relative(root, file), await fs.readFile(file))), loader: 'json',
  }));
}}];
await fs.rm(path.join(root, 'build/waveparse'), { recursive: true, force: true });
await fs.mkdir(stage, { recursive: true });
await fs.rm(path.join(js, 'dist'), { recursive: true, force: true });
await fs.mkdir(path.join(js, 'dist'), { recursive: true });
const base = { absWorkingDir: root, platform: 'node', target: 'node22', bundle: true, plugins, logLevel: 'warning', sourcemap: false, metafile: true,
  // Keep class semantics identical in the application and standalone checkout.
  tsconfigRaw: { compilerOptions: { useDefineForClassFields: false, baseUrl: '.', paths: { '@/*': ['src/*'] } } },
};
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

const declarationOptions = {
  declaration: true, emitDeclarationOnly: true, outDir: path.join(js, 'dist/types'), rootDir: root,
  module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
  target: ts.ScriptTarget.ES2022, esModuleInterop: true, strict: true, skipLibCheck: true,
  resolveJsonModule: true, baseUrl: root, paths: { '@/*': ['src/*'] },
};
const declarationHost = ts.createCompilerHost(declarationOptions);
const readDeclarationFile = declarationHost.readFile;
declarationHost.readFile = file => {
  const bytes = readDeclarationFile(file);
  return bytes === undefined ? undefined : String(distributionResource(path.relative(root, file), bytes));
};
const declarations = ts.createProgram(['packages/javascript/src/index.ts', 'packages/javascript/src/cli.ts', 'packages/runtime/runner.ts'].map(file => path.join(root, file)), declarationOptions, declarationHost);
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

const resources = distribution.runtimeFiles;
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
  await fs.mkdir(path.join(directory, 'docs'), { recursive: true });
  for (const file of distribution.documentation) await fs.copyFile(path.join(userDocs, file), path.join(directory, 'docs', file));
  await fs.copyFile(path.join(root, 'LICENSE'), path.join(directory, 'LICENSE'));
  await fs.copyFile(path.join(userDocs, 'THIRD_PARTY_NOTICES.md'), path.join(directory, 'THIRD_PARTY_NOTICES.md'));
  await fs.rm(path.join(directory, 'LICENSES'), { recursive: true, force: true });
  for (const file of distribution.licences) await copy(file, directory);
  await fs.copyFile(path.join(userDocs, 'README.md'), path.join(directory, 'README.md'));
}
const jsManifest = await readJson('packages/javascript/package.json');
if (jsManifest.version !== release.version || !String(await fs.readFile(path.join(py, 'pyproject.toml'))).includes(`version = "${release.version}"`)) throw new Error('Package versions disagree.');
await fs.mkdir(dist, { recursive: true });
execFileSync('npm', ['pack', js, '--pack-destination', dist, '--ignore-scripts', '--cache', path.join(root, 'build/npm-cache')], { cwd: root, stdio: 'inherit' });

// Corresponding source, without existing repository history or clinical data.
const source = path.join(root, 'build/waveparse/source');
await fs.mkdir(source, { recursive: true });
const sourceFiles = new Set([...resources, ...distribution.sourceFiles, ...distribution.licences, ...bundleInputs]);
// Type-only imports are needed to rebuild declarations, even when absent from JS.
for (const file of declarations.getSourceFiles()) {
  if (file.fileName.includes('/node_modules/')) continue;
  const relative = path.relative(root, file.fileName);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error(`Source outside checkout: ${relative}`);
  sourceFiles.add(relative);
}
for (const file of await walk(path.join(py, 'src/ecg_waveparse'))) {
  const relative = path.relative(root, file);
  if (!relative.includes('/runtime/') && !file.endsWith('.pyc')) sourceFiles.add(relative);
}
for (const file of sourceFiles) {
  if (file === 'src/lib/local-reference.ts') continue;
  await copy(file, source);
}
await fs.writeFile(path.join(source, 'src/lib/local-reference.ts'), 'import type { KnownDigitization } from "./runs";\nexport const LOCAL_REFERENCE_DIGITIZATIONS: KnownDigitization[] = [];\n');
await fs.copyFile(path.join(userDocs, 'README.md'), path.join(source, 'README.md'));
await fs.copyFile(path.join(userDocs, 'THIRD_PARTY_NOTICES.md'), path.join(source, 'THIRD_PARTY_NOTICES.md'));
await fs.mkdir(path.join(source, 'docs'), { recursive: true });
for (const file of distribution.documentation) await fs.copyFile(path.join(userDocs, file), path.join(source, 'docs', file));
await fs.mkdir(path.join(source, '.github/workflows'), { recursive: true });
await fs.copyFile(path.join(userDocs, 'workflow.yml'), path.join(source, '.github/workflows/waveparse-packages.yml'));
await fs.writeFile(path.join(source, 'tsconfig.json'), JSON.stringify({ compilerOptions: {
  target: 'ES2022', lib: ['esnext', 'dom'], strict: true, noEmit: true, esModuleInterop: true, useDefineForClassFields: false,
  module: 'esnext', moduleResolution: 'bundler', resolveJsonModule: true, skipLibCheck: true,
  paths: { '@/*': ['./src/*'] },
}, include: ['src/**/*.ts', 'packages/*/src/**/*.ts', 'packages/runtime/*.ts'], exclude: ['node_modules', 'build', 'dist', 'packages/javascript/dist', 'packages/javascript/runtime', 'packages/python/src/ecg_waveparse/runtime'] }, null, 2) + '\n');
const sourcePackage = await readJson('package.json');
sourcePackage.name = 'ecg-waveparse-source';
sourcePackage.scripts = sourceScripts;
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
await fs.writeFile(path.join(source, '.gitignore'), 'node_modules/\nbuild/\ndist/\n.venv/\n.build-venv/\n__pycache__/\n*.pyc\n*.tsbuildinfo\n.env*\npackages/javascript/dist/\npackages/javascript/runtime/\npackages/python/src/ecg_waveparse/runtime/\npackages/javascript/docs/\npackages/python/docs/\npackages/javascript/README.md\npackages/python/README.md\npackages/javascript/LICENSE\npackages/python/LICENSE\npackages/javascript/LICENSES/\npackages/python/LICENSES/\npackages/javascript/THIRD_PARTY_NOTICES.md\npackages/python/THIRD_PARTY_NOTICES.md\n');
await fs.writeFile(path.join(dist, 'bundle-inputs.json'), JSON.stringify({ externalJavaScriptDependencies: [], inputs: [...bundleInputs].sort() }, null, 2) + '\n');
await fs.writeFile(path.join(source, 'source-files.json'), JSON.stringify((await walk(source)).map(file => path.relative(source, file)).concat(['.github/workflows/waveparse-packages.yml', '.gitignore', '.dockerignore']).sort(), null, 2) + '\n');
execFileSync(process.execPath, [path.join(root, 'scripts/audit-waveparse-source.mjs'), source, path.join(dist, 'source-audit.json')], { stdio: 'inherit' });
execFileSync('tar', ['-czf', path.join(dist, `${release.name}-${release.version}-source.tar.gz`), '-C', source, '.'], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
console.log(`Prepared ${release.name} ${release.version}; ${Object.keys(manifest).length} shared runtime files. Archives: ${dist}. Build Python distributions with python -m build --no-isolation packages/python --outdir PATH.`);
