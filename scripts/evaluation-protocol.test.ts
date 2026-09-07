import assert from 'node:assert/strict'
import test from 'node:test'
import { EVALUATION_SOURCE_FILES, validateEvaluationProtocol } from './evaluation-protocol.mjs'

const fixture = () => ({
  version: 1, clinicalValidationUse: false, collection: 'final_engineering_evaluation',
  sourceType: 'ptb_waveform_render', previouslyExposed: false,
  membership: [{ caseId: 'final_01', patientId: 'patient123', record: 'patient123/s001' }],
  accessControlReference: 'owner-only local final collection', accessLogFile: 'access.jsonl',
  sourceLock: { payloadManifestSha256: 'a'.repeat(64), sourceSnapshotSha256: 'b'.repeat(64),
    sourceSnapshotManifest: 'benchmark/results/frozen-source.json',
    scorerFiles: Object.fromEntries(EVALUATION_SOURCE_FILES.map((p: string) => [p, 'c'.repeat(64)])) },
})

test('final evaluation refuses exposed groups, repeated patients and missing source locks', () => {
  validateEvaluationProtocol(fixture())
  const exposed = fixture(); exposed.previouslyExposed = true
  assert.throws(() => validateEvaluationProtocol(exposed))
  const repeated = fixture(); repeated.membership.push({ ...repeated.membership[0], caseId: 'final_02' })
  assert.throws(() => validateEvaluationProtocol(repeated), /independent patient/)
  const unlocked = fixture(); delete unlocked.sourceLock.scorerFiles['ecg_benchmark/coordinates.py']
  assert.throws(() => validateEvaluationProtocol(unlocked), /Missing source lock/)
  const traversal = fixture(); traversal.accessLogFile = '../access.jsonl'
  assert.throws(() => validateEvaluationProtocol(traversal))
})
