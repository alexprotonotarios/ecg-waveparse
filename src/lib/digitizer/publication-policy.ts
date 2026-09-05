export type PublicationReasonCode =
  | "source_verified"
  | "adaptive_preprocessed"
  | "lead_fusion"
  | "peer_trusted"
  | "reviewable_constrained"
  | "supported_partial_layout"
  | "input_quality_review_only"
  | "input_quality_insufficient"
  | "unstable_neural_confirmation"
  | "semantic_identity_unconfirmed"
  | "no_publishable_candidate"

export type PublicationDecision<T> =
  | {
      outcome: "needs_review" | "partial"
      reasonCode: PublicationReasonCode
      candidate: T
      partialLeadSelection: boolean
    }
  | {
      outcome: "failed"
      reasonCode:
        | "no_publishable_candidate"
        | "input_quality_insufficient"
        | "semantic_identity_unconfirmed"
      partialLeadSelection: false
    }

export function resolvePublicationDecision<T>({
  sourceVerified,
  adaptivePreprocessed,
  fused,
  safest,
  reviewableConstrained,
  supportedPartial,
  rankCandidate,
}: {
  sourceVerified?: T
  adaptivePreprocessed?: T
  fused?: T
  safest?: T
  reviewableConstrained?: T
  supportedPartial?: T
  rankCandidate?: (candidate: T) => number
}): PublicationDecision<T> {
  const completeCandidates: Array<{
    candidate: T
    reasonCode:
      | "source_verified"
      | "adaptive_preprocessed"
      | "lead_fusion"
      | "peer_trusted"
  }> = []
  if (sourceVerified !== undefined) {
    completeCandidates.push({
      candidate: sourceVerified,
      reasonCode: "source_verified",
    })
  }
  if (adaptivePreprocessed !== undefined) {
    completeCandidates.push({
      candidate: adaptivePreprocessed,
      reasonCode: "adaptive_preprocessed",
    })
  }
  if (fused !== undefined) {
    completeCandidates.push({ candidate: fused, reasonCode: "lead_fusion" })
  }
  if (safest !== undefined) {
    completeCandidates.push({ candidate: safest, reasonCode: "peer_trusted" })
  }
  if (rankCandidate) {
    completeCandidates.sort(
      (first, second) =>
        rankCandidate(first.candidate) - rankCandidate(second.candidate)
    )
  }
  const complete = completeCandidates[0]
  if (complete) {
    return selected("needs_review", complete.reasonCode, complete.candidate)
  }
  if (reviewableConstrained) {
    return selected(
      "needs_review",
      "reviewable_constrained",
      reviewableConstrained
    )
  }
  if (supportedPartial) {
    return {
      ...selected("partial", "supported_partial_layout", supportedPartial),
      partialLeadSelection: true,
    }
  }
  return {
    outcome: "failed",
    reasonCode: "no_publishable_candidate",
    partialLeadSelection: false,
  }
}

function selected<T>(
  outcome: "needs_review" | "partial",
  reasonCode: Exclude<
    PublicationReasonCode,
    | "no_publishable_candidate"
    | "unstable_neural_confirmation"
    | "semantic_identity_unconfirmed"
    | "input_quality_insufficient"
    | "input_quality_review_only"
  >,
  candidate: T
) {
  return { outcome, reasonCode, candidate, partialLeadSelection: false } as const
}

export function applyInputQualityPublicationGate<T>(
  decision: PublicationDecision<T>,
  _inputQualityOutcome: "acceptable" | "review" | "insufficient" | undefined
): PublicationDecision<T> {
  void _inputQualityOutcome
  // Resolution, blur, glare, and screen-artifact classifiers estimate likely
  // fidelity. They are not direct evidence that an ECG trace is absent. A
  // candidate that has already passed the trace, lead-identity, coverage, and
  // corroboration requirements remains publishable with review metadata.
  return decision
}
