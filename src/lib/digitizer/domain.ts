import domainDocument from "../../../config/ecg-domain.v1.json"

export const ECG_LEADS = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
] as const

export type EcgLead = (typeof ECG_LEADS)[number]
export type CanonicalEcgLayout = "standard_3x4" | "standard_6x2" | "standard_12x1"
export type PartialEcgLayout =
  | "standard_6x1_limb"
  | "cabrera_6x1_limb"
  | "precordial_6x1"
  | "precordial_3x2"
export type EcgLayout = CanonicalEcgLayout | PartialEcgLayout

export type DigitizerCandidateKind =
  | "source-model"
  | "adaptive-preprocessed"
  | "geometry-constrained"
  | "native-grid"
  | "crop-proposal"
  | "panel-component"
  | "panel-composite"
  | "peer-gap-repair"
  | "lead-fusion"

export type DigitizerCandidateCapabilities = {
  derived: boolean
  nativeGrid: boolean
  preprocessed: boolean
  geometryConstrained: boolean
  sourcePixelEvidence: boolean
  selectionEligible: boolean
}

export type EcgLayoutSpec = {
  pageRows: number
  rows: readonly (readonly string[])[]
}

function validateDomainDocument() {
  if (domainDocument.version !== 1) {
    throw new Error(`Unsupported ECG domain version: ${domainDocument.version}`)
  }
  if (JSON.stringify(domainDocument.leads) !== JSON.stringify(ECG_LEADS)) {
    throw new Error("ECG domain lead order does not match the canonical 12-lead order.")
  }
  for (const [layout, rawSpec] of Object.entries(domainDocument.layouts)) {
    if (!Number.isInteger(rawSpec.pageRows) || rawSpec.pageRows <= 0) {
      throw new Error(`ECG domain layout ${layout} has an invalid page row count.`)
    }
    const unknownLead = rawSpec.rows.flat().find(
      (lead) => !ECG_LEADS.includes(lead as EcgLead)
    )
    if (unknownLead) {
      throw new Error(`ECG domain layout ${layout} contains unknown lead ${unknownLead}.`)
    }
  }
}

validateDomainDocument()

export const SAMPLE_RATE_HZ = domainDocument.sampleRateHz
export const PAGE_DURATION_SECONDS = domainDocument.pageDurationSeconds
export const ECG_LAYOUTS: Readonly<Record<string, EcgLayoutSpec>> =
  Object.freeze(domainDocument.layouts)

export const PARTIAL_LAYOUT_LEADS: Readonly<Record<string, readonly string[]>> =
  Object.freeze({
    standard_6x1_limb: [
      ...new Set(ECG_LAYOUTS.standard_6x1_limb.rows.flat()),
    ],
    cabrera_6x1_limb: [
      ...new Set(ECG_LAYOUTS.cabrera_6x1_limb.rows.flat()),
    ],
    precordial_6x1: [...new Set(ECG_LAYOUTS.precordial_6x1.rows.flat())],
    precordial_3x2: [...new Set(ECG_LAYOUTS.precordial_3x2.rows.flat())],
  })

export const LAYOUT_COLUMNS: Readonly<
  Record<string, Readonly<Record<string, number>>>
> = Object.freeze(
  Object.fromEntries(
    Object.entries(ECG_LAYOUTS).map(([layout, spec]) => [
      layout,
      Object.fromEntries(
        spec.rows.flatMap((row) => row.map((lead, column) => [lead, column]))
      ),
    ])
  ) as Record<string, Record<string, number>>
)
