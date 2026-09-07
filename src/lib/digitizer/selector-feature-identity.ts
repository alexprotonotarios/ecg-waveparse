import parameterContract from "../../../config/selector-parameter-contract.v1.json"

export type ProfileParameterIdentity = {
  parameterContractVersion: number
  parameterKeys: string[]
  parameterSignatures: Record<string, string[]>
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical)
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => [key, canonical(item)]))
  return value
}

export function selectorParameterSignature(parameters: object) {
  const fields = parameters as Record<string, unknown>
  return JSON.stringify(canonical(Object.fromEntries(parameterContract.keys.map(key => [key, fields[key] ?? null]))))
}

export function profileParameterCompatibility<T extends { id: string }>(profile: ProfileParameterIdentity | undefined, layout: string | undefined, candidate: T) {
  // Historical parameter contract v1 describes the upstream 10 mm/mV conversion.
  // A corrected nondefault gain is outside its fitted domain, even when the old
  // profile has no reconstructible parameter identity. It cannot inherit a prior.
  const gain = (candidate as T & { gainMmPerMv?: number }).gainMmPerMv
  if (gain !== undefined && gain !== 10) return "incompatible" as const
  if (!profile) return "historical_unverified" as const
  if (profile.parameterContractVersion !== parameterContract.version || JSON.stringify(profile.parameterKeys) !== JSON.stringify(parameterContract.keys)) return "incompatible" as const
  const expected = selectorParameterSignature(candidate)
  const compatible = profile.parameterSignatures[`${layout}/${candidate.id}`]?.some(signature => {
    // JSON numeric spellings differ between Python and JavaScript (1.0, 1e-07).
    // Compare canonical parsed values; never reinterpret a changed numeric value.
    try { return JSON.stringify(canonical(JSON.parse(signature))) === expected } catch { return false }
  })
  return compatible ? "compatible" as const : "incompatible" as const
}
