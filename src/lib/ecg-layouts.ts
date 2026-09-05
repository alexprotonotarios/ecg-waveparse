export const SUPPORTED_ECG_LAYOUTS = [
  "standard_3x4",
  "standard_6x2",
  "standard_12x1",
] as const

export type SupportedEcgLayout = (typeof SUPPORTED_ECG_LAYOUTS)[number]

export function normalizeSupportedEcgLayout(
  layout?: string
): SupportedEcgLayout | undefined {
  if (layout === "cabrera_12x1" || layout?.startsWith("cabrera_12x1_")) {
    return "standard_12x1"
  }
  return SUPPORTED_ECG_LAYOUTS.find(
    (candidate) => layout === candidate || layout?.startsWith(`${candidate}_`)
  )
}

export function isSupportedEcgLayout(layout?: string): boolean {
  return normalizeSupportedEcgLayout(layout) !== undefined
}
