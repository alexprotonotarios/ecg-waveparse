import assert from 'node:assert/strict';

export const EVALUATION_SOURCE_FILES = [
  'scripts/waveparse-regression.mjs', 'scripts/evaluation-protocol.mjs',
  'scripts/score_digitization.py', 'ecg_benchmark/scoring.py',
  'ecg_benchmark/coordinates.py', 'ecg_benchmark/measurements.py',
  'ecg_benchmark/io.py', 'ecg_pipeline/domain.py', 'config/ecg-domain.v1.json',
];

export function validateEvaluationProtocol(protocol) {
  assert.equal(protocol.version, 1);
  assert.equal(protocol.clinicalValidationUse, false);
  assert(['engineering_development', 'final_engineering_evaluation'].includes(protocol.collection));
  assert(Array.isArray(protocol.membership) && protocol.membership.length > 0);
  assert.equal(new Set(protocol.membership.map(m => m.caseId)).size, protocol.membership.length);
  const lock = protocol.sourceLock;
  assert(lock && /^[a-f0-9]{64}$/.test(lock.payloadManifestSha256));
  for (const file of EVALUATION_SOURCE_FILES) {
    assert(/^[a-f0-9]{64}$/.test(lock.scorerFiles?.[file] ?? ''), `Missing source lock: ${file}`);
  }
  if (protocol.collection === 'final_engineering_evaluation') {
    assert.equal(protocol.previouslyExposed, false);
    assert.equal(protocol.sourceType, 'ptb_waveform_render');
    assert.equal(new Set(protocol.membership.map(m => m.patientId)).size, protocol.membership.length,
      'The final protocol requires one recording per independent patient group');
    for (const member of protocol.membership) {
      assert(/^patient\d{3}$/.test(member.patientId));
      assert(member.record.startsWith(member.patientId + '/') && member.record.split('/').length === 2);
    }
    assert(typeof protocol.accessControlReference === 'string' && protocol.accessControlReference.length > 0);
    assert(typeof protocol.accessLogFile === 'string' && /^[a-zA-Z0-9_-]+\.jsonl$/.test(protocol.accessLogFile));
    assert(/^[a-f0-9]{64}$/.test(lock.sourceSnapshotSha256));
    assert(typeof lock.sourceSnapshotManifest === 'string' && lock.sourceSnapshotManifest.length > 0);
  }
}
