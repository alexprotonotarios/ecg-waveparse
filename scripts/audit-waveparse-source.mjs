// Inspect only the fresh allowlisted source export, before it gets a git history.
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const [directory, output] = process.argv.slice(2);
assert(directory && output, 'Usage: node audit-waveparse-source.mjs SOURCE_DIR REPORT_JSON');
const root = path.resolve(directory);
const files = [];
const forbiddenDirectory = /^(input|storage|\.git|\.external|node_modules|generated|results|dist|build)$/;
const forbiddenExtension = /\.(png|jpe?g|webp|pdf|dcm|csv|onnx|pth|pt|pkl|pickle|safetensors|pem|key|sqlite|db)$/i;
const credentialPatterns = [
  /(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}/,
  /AKIA[0-9A-Z]{16}/,
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /(?:sk_live_|sk_test_|wos_sk_)[A-Za-z0-9]{20,}/,
];
async function walk(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const absolute = path.join(dir, entry.name);
    const relative = path.relative(root, absolute).split(path.sep).join('/');
    assert(!entry.isSymbolicLink(), `Symlink excluded: ${relative}`);
    if (entry.isDirectory()) {
      assert(!forbiddenDirectory.test(entry.name), `Private/generated directory: ${relative}`);
      await walk(absolute);
    } else {
      assert(entry.isFile(), `Special file: ${relative}`);
      assert(!entry.name.startsWith('.env') && !forbiddenExtension.test(entry.name), `Data/credential file: ${relative}`);
      const data = await fs.readFile(absolute);
      assert(data.length < 2 * 1024 * 1024, `Unexpected large source file: ${relative}`);
      assert(!data.includes(0), `Binary file: ${relative}`);
      assert(!credentialPatterns.some(pattern => pattern.test(data.toString('utf8'))), `Credential-shaped content in ${relative}`);
      files.push({ path: relative, bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') });
    }
  }
}
await walk(root);
files.sort((a, b) => a.path.localeCompare(b.path));
for (const required of ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'LICENSES/Open-ECG-Digitizer-CC-BY-SA-4.0.txt', '.github/workflows/waveparse-packages.yml']) {
  assert(files.some(file => file.path === required), `Missing release source: ${required}`);
}
const stub = await fs.readFile(path.join(root, 'src/lib/local-reference.ts'), 'utf8');
assert(/LOCAL_REFERENCE_DIGITIZATIONS: KnownDigitization\[\] = \[\]/.test(stub), 'Private reference source was not stubbed');
const report = { schemaVersion: 1, fileCount: files.length, bytes: files.reduce((sum, file) => sum + file.bytes, 0),
  checks: { dataFilesAbsent: true, originalGitHistoryAbsent: true, symlinksAbsent: true, knownCredentialPatternsAbsent: true, referenceStubVerified: true, thirdPartyLicenceIncluded: true },
  limitation: 'Allowlist and targeted credential checks; not proof that arbitrary unrecognised secrets cannot exist.', files };
await fs.mkdir(path.dirname(path.resolve(output)), { recursive: true });
await fs.writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(`Audited ${report.fileCount} source files (${report.bytes} bytes); no images, models, history or recognised credentials.`);
