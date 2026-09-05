export const REVIEW_CONFIRMATION_KEYS = [
  "sourceCompared",
  "leadIdentityVerified",
  "scaleAndGapsReviewed",
] as const

export type DigitizerReviewConfirmations = Record<
  (typeof REVIEW_CONFIRMATION_KEYS)[number],
  boolean
>

export type DigitizerReviewDraft = {
  runId: string
  reviewer: string
  notes: string
  confirmations: DigitizerReviewConfirmations
}

type ReviewDraftRun = {
  id: string
  review?: {
    reviewer?: string
    notes?: string
    confirmations?: DigitizerReviewConfirmations
  }
}

export function reviewDraftForRun(run: ReviewDraftRun): DigitizerReviewDraft {
  return {
    runId: run.id,
    reviewer: run.review?.reviewer ?? "",
    notes: run.review?.notes ?? "",
    confirmations: run.review?.confirmations ?? {
      sourceCompared: false,
      leadIdentityVerified: false,
      scaleAndGapsReviewed: false,
    },
  }
}

export function synchronizeReviewDraft(
  draft: DigitizerReviewDraft,
  run: ReviewDraftRun
): DigitizerReviewDraft {
  return draft.runId === run.id ? draft : reviewDraftForRun(run)
}

export function hasAllReviewConfirmations(
  confirmations: Partial<DigitizerReviewConfirmations> | undefined
): confirmations is DigitizerReviewConfirmations {
  return REVIEW_CONFIRMATION_KEYS.every(
    (key) => confirmations?.[key] === true
  )
}

export function canAcceptQuantitativeReview(run: {
  publicationDecision?: { outcome: "needs_review" | "partial" | "failed" }
  assets: {
    canonicalCsv?: unknown
    segmentsCsv?: unknown
  }
}) {
  return Boolean(
    run.publicationDecision?.outcome === "needs_review" &&
      run.assets.canonicalCsv &&
      run.assets.segmentsCsv
  )
}

export function statusAfterReviewDecision(
  decision: "accepted" | "rejected",
  publicationOutcome: "needs_review" | "partial" | "failed" | undefined
) {
  if (decision === "accepted") return "completed" as const
  return publicationOutcome ?? ("needs_review" as const)
}
