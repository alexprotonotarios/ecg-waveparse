/** Compact observations of predicates actually evaluated by the selector.
 * Nothing in this module supplies a selection decision or computes alignment. */
type Measurements = Record<string, number | string | boolean | null>
type Gate = { gate: string; passed: boolean; measurements?: Measurements }
type Peer = { candidateId: string; gates: Gate[]; passed: boolean }
export type SelectionPredicate = {
  context: string
  predicate: "lead-trust" | "full-rhythm" | "source-grid-trust"
  candidateId: string
  lead: string
  peerOrder: string[]
  gates: Gate[]
  peers: Peer[]
  passed: boolean
  witnessCandidateId?: string
  unevaluatedPeerCount: number
}
export type SelectionDiagnostics = {
  version: 1
  method: "evaluated-selection-predicates-v1"
  scope: "before-stability-and-publication"
  shortCircuit: "first-failed-gate-or-first-successful-peer"
  candidates: Array<{ candidateId: string; status: string; eligible: boolean; reason: string }>
  predicates: SelectionPredicate[]
  publication?: { outcome: string; reasonCode: string; candidateId?: string }
  fusion?: {
    referenceCandidateId: string
    diagnostic: { kind: "copied-reference-candidate"; candidateId: string; representsFinalFusion: false }
    leads: Array<{ lead: string; sourceCandidateId: string; fullWidthRhythm: boolean;
      sourceStartSample: number; sourceSampleCount: number }>
  }
}

const pools = new WeakMap<object, { recorder: SelectionEvidenceRecorder; context: string }>()

class GateObservation {
  constructor(readonly gates: Gate[]) {}
  check(gate: string, passed: boolean, measurements?: Measurements) {
    this.gates.push({ gate, passed, ...(measurements ? { measurements } : {}) })
    return passed
  }
}

class PredicateObservation extends GateObservation {
  constructor(private recorder: SelectionEvidenceRecorder, private result: SelectionPredicate) {
    super(result.gates)
  }
  peer(candidateId: string) {
    const result: Peer = { candidateId, gates: [], passed: false }
    this.result.peers.push(result)
    const observation = new GateObservation(result.gates)
    return {
      check: observation.check.bind(observation),
      finish: (passed: boolean) => { result.passed = passed; return passed },
    }
  }
  finish(passed: boolean) {
    this.result.passed = passed
    this.result.unevaluatedPeerCount = this.result.peerOrder.length - this.result.peers.length
    const witness = this.result.peers.find(peer => peer.passed)
    if (witness) this.result.witnessCandidateId = witness.candidateId
    this.recorder.record(this.result)
    return passed
  }
}

export class SelectionEvidenceRecorder {
  private predicates = new Map<string, SelectionPredicate>()
  readonly candidates: SelectionDiagnostics["candidates"] = []
  fusion?: SelectionDiagnostics["fusion"]
  observe(pool: Array<{ id: string }>, context: string) {
    pools.set(pool, { recorder: this, context })
  }
  record(value: SelectionPredicate) {
    // Keep the latest actual evaluation for each context/predicate/lead, not
    // an unbounded history of repeated calls during uncertainty calculations.
    this.predicates.set(JSON.stringify([value.context, value.predicate, value.candidateId, value.lead]), value)
  }
  snapshot(publication: NonNullable<SelectionDiagnostics["publication"]>): SelectionDiagnostics {
    return structuredClone({ version: 1, method: "evaluated-selection-predicates-v1",
      scope: "before-stability-and-publication", shortCircuit: "first-failed-gate-or-first-successful-peer",
      candidates: this.candidates, predicates: [...this.predicates.values()], publication,
      ...(this.fusion ? { fusion: this.fusion } : {}) })
  }
}

export function beginSelectionPredicate(pool: Array<{ id: string }>, candidateId: string,
  lead: string, predicate: SelectionPredicate["predicate"]) {
  const observed = pools.get(pool)
  if (!observed) return undefined
  return new PredicateObservation(observed.recorder, { context: observed.context,
    predicate, candidateId, lead, peerOrder: pool.map(peer => peer.id), gates: [], peers: [],
    passed: false, unevaluatedPeerCount: pool.length })
}
