import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { promises as fs } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import { validateSegmentEvidence, type SegmentEvidenceMap } from "./segment-evidence"
import { EvidenceContractError } from "./physical-contracts"

const execute = promisify(execFile)
const digest = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex")
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
const fail = (): never => { throw new EvidenceContractError("source_time_publication_mismatch", "Source timing evidence does not match the selected publication.") }

export type PublicationUncertaintyRow = {
  lead: string; leadSample: number; timeSeconds: number; canonicalSample: number; valueUv: number; reviewEstimateUv: number;
  status: "observed" | "missing" | "uncertain_annotation" | "uncertain_candidate_disagreement" | "uncertain_annotation_and_disagreement" | "uncertainty_unavailable";
  annotationOverlap: boolean; candidateCount: number; candidateSpreadUv: number;
}
type Identity = { sourceSha256: string; coordinateSha256: string; inputCanonicalSha256: string; inputUncertaintySha256: string; inputSegmentMapSha256: string }
export type SourceTimeManifest = Identity & {
  version: 1; method: "source_grid_time_publication_v1"; state: "converted" | "refused"; reason?: string;
  sourceLabelIdentityPromoted: false; sourceInkVerified: false; requiresReview: true;
  waveformConverted: boolean; originRangesAreGeometricNotStatistical: true;
  outputCanonicalSha256?: string; expectedSamples?: number; returnedSamples?: number; sourcePublicationExclusions?: number;
  files: Record<string, { sha256: string; bytes: number }>
}
export type SourceTimePublication = { manifest: SourceTimeManifest; bundlePath: string; bundleSha256: string;
  canonicalPath?: string; uncertaintyPath?: string; segmentPath?: string; diagnosticPath?: string; segmentMap?: SegmentEvidenceMap; uncertaintyRows?: PublicationUncertaintyRow[] }

export function sourceTimePublicationEligible(candidate: { layout?: string; parameters: { vectorizer?: string; cropBox?: unknown; paperSpeedMmPerSecond?: number; gainMmPerMv?: number }; sourceCoordinatePath?: string }, derived: boolean, imageSize: [number, number]) {
  return !derived && candidate.layout === "standard_3x4_with_r1" && candidate.parameters.vectorizer === "dynamic-path" && !candidate.parameters.cropBox &&
    Boolean(candidate.sourceCoordinatePath) && (candidate.parameters.paperSpeedMmPerSecond ?? 25) === 25 && (candidate.parameters.gainMmPerMv ?? 10) === 10 &&
    imageSize.every(n => Number.isSafeInteger(n) && n >= 200) && imageSize[0] * imageSize[1] <= 12_000_000
}

export function parseSourceTimeManifest(value: unknown, expected: Identity): SourceTimeManifest {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  const m = value as SourceTimeManifest
  if (m.version !== 1 || m.method !== "source_grid_time_publication_v1" || !["converted", "refused"].includes(m.state) ||
      m.sourceLabelIdentityPromoted !== false || m.sourceInkVerified !== false || m.requiresReview !== true ||
      m.originRangesAreGeometricNotStatistical !== true || m.waveformConverted !== (m.state === "converted") ||
      Object.entries(expected).some(([key, value]) => !hash(value) || m[key as keyof Identity] !== value) ||
      !m.files || typeof m.files !== "object" || Array.isArray(m.files)) return fail()
  const inputs = ["input-canonical.csv", "input-uncertainty.csv", "input-segments.json", "source-observation.json"]
  const outputs = ["canonical.csv", "uncertainty.csv", "uncertainty.json", "segments.json", "lineage.csv", "mapping.npz", "conversion.npz", "source-overlay.png"]
  const names = [...inputs, ...(m.state === "converted" ? outputs : [])].sort()
  if (JSON.stringify(Object.keys(m.files).sort()) !== JSON.stringify(names) || Object.values(m.files).some(f => !f || !hash(f.sha256) || !Number.isSafeInteger(f.bytes) || f.bytes < 1 || f.bytes > 128 * 1024 * 1024) ||
      m.files["input-canonical.csv"].sha256 !== expected.inputCanonicalSha256 || m.files["input-uncertainty.csv"].sha256 !== expected.inputUncertaintySha256 ||
      m.files["input-segments.json"].sha256 !== expected.inputSegmentMapSha256) return fail()
  if (m.state === "converted" ? (!hash(m.outputCanonicalSha256) || m.outputCanonicalSha256 !== m.files["canonical.csv"].sha256 || m.expectedSamples !== 18750 ||
      !Number.isSafeInteger(m.returnedSamples) || m.returnedSamples! < 1 || m.returnedSamples! > 18750 || !Number.isSafeInteger(m.sourcePublicationExclusions) || m.sourcePublicationExclusions! < 0)
    : typeof m.reason !== "string" || !m.reason || m.outputCanonicalSha256 !== undefined) return fail()
  return m
}

export async function verifySourceTimePublication(directory: string, identities: Identity): Promise<SourceTimePublication> {
  const manifestBytes = await fs.readFile(path.join(directory, "manifest.json"))
  const manifest = parseSourceTimeManifest(JSON.parse(manifestBytes.toString()), identities)
  for (const [name, identity] of Object.entries(manifest.files)) {
    const file = path.join(directory, name); const stat = await fs.lstat(file)
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== identity.bytes || digest(await fs.readFile(file)) !== identity.sha256) return fail()
  }
  const bundlePath = path.join(directory, "evidence.zip")
  const result: SourceTimePublication = { manifest, bundlePath, bundleSha256: digest(await fs.readFile(bundlePath)) }
  if (manifest.state === "refused") return result
  const map = validateSegmentEvidence(JSON.parse(await fs.readFile(path.join(directory, "segments.json"), "utf8")), {
    sourceSha256: identities.sourceSha256, canonicalSha256: manifest.outputCanonicalSha256!,
  })
  const original = validateSegmentEvidence(JSON.parse(await fs.readFile(path.join(directory, "input-segments.json"), "utf8")), {
    sourceSha256: identities.sourceSha256, canonicalSha256: identities.inputCanonicalSha256,
  })
  if (map.runId !== original.runId || map.segments.length !== 12) return fail()
  for (const segment of map.segments) {
    const before = original.segments.find(s => s.lead === segment.lead); const proof = segment.sourceTimeConversion
    if (!before || segment.lineageState !== "source_time_conversion_recorded" || !proof || proof.lineageSha256 !== manifest.files["lineage.csv"].sha256 ||
        proof.coordinateSha256 !== identities.coordinateSha256 || proof.inputCanonicalSha256 !== identities.inputCanonicalSha256 ||
        proof.inputUncertaintySha256 !== identities.inputUncertaintySha256 || proof.inputSegmentMapSha256 !== identities.inputSegmentMapSha256 ||
        (["identityState", "identityMethod", "candidateId", "sourceLabel", "polarity", "segmentId", "rowIndex", "panelIndex", "role", "canonicalStartSample", "sampleCount"] as const).some(k => segment[k] !== before[k])) return fail()
  }
  const raw = JSON.parse(await fs.readFile(path.join(directory, "uncertainty.json"), "utf8")) as Array<Record<string, unknown>>
  if (!Array.isArray(raw) || raw.length !== 18750) return fail()
  const seen = new Set<string>()
  const num = (value: unknown) => value === "" || value === null ? NaN : typeof value === "number" && Number.isFinite(value) ? value : fail()
  const uncertaintyRows = raw.map(row => {
    if (!row || typeof row !== "object") return fail()
    const segment = map.segments.find(s => s.lead === row.lead)
    const sample = num(row.canonical_sample), local = num(row.lead_sample), time = num(row.t_s_at_500hz), value = num(row.value_uv)
    const key = `${row.lead}:${sample}`
    if (!segment || !Number.isSafeInteger(local) || local < 0 || local >= segment.sampleCount || sample !== segment.canonicalStartSample + local || time !== local / 500 || seen.has(key) ||
        ![0, 1].includes(Number(row.annotation_overlap)) || !["observed", "missing", "uncertain_annotation", "uncertain_candidate_disagreement", "uncertainty_unavailable"].includes(String(row.status)) ||
        Number.isFinite(value) !== (segment.support.find(span => local >= span.startSample && local < span.endSample)?.status === "returned")) return fail()
    seen.add(key)
    return { lead: String(row.lead), leadSample: local, timeSeconds: time, canonicalSample: sample, valueUv: value,
      reviewEstimateUv: num(row.review_estimate_uv), status: row.status as PublicationUncertaintyRow["status"],
      annotationOverlap: row.annotation_overlap === 1, candidateCount: num(row.candidate_count), candidateSpreadUv: num(row.candidate_spread_uv) }
  })
  if (uncertaintyRows.filter(r => Number.isFinite(r.valueUv)).length !== manifest.returnedSamples) return fail()
  return { ...result, canonicalPath: path.join(directory, "canonical.csv"), uncertaintyPath: path.join(directory, "uncertainty.csv"),
    segmentPath: path.join(directory, "segments.json"),
    diagnosticPath: path.join(directory, "source-overlay.png"), segmentMap: map, uncertaintyRows }
}

export async function applySourceTimePublication(context: {
  sourcePath: string; sourceSha256: string; coordinatePath: string; coordinateSha256: string; canonicalPath: string;
  uncertaintyPath: string; segmentPath: string; directory: string; python: string; resourceRoot: string;
  environment: NodeJS.ProcessEnv; signal?: AbortSignal; acquire: () => Promise<() => void>;
}): Promise<SourceTimePublication> {
  const identities = { sourceSha256: context.sourceSha256, coordinateSha256: context.coordinateSha256,
    inputCanonicalSha256: digest(await fs.readFile(context.canonicalPath)), inputUncertaintySha256: digest(await fs.readFile(context.uncertaintyPath)),
    inputSegmentMapSha256: digest(await fs.readFile(context.segmentPath)) }
  await fs.rm(context.directory, { recursive: true, force: true })
  const release = await context.acquire()
  try {
    context.signal?.throwIfAborted()
    await execute(context.python, ["-m", "ecg_pipeline.source_time", "--source", context.sourcePath,
      "--coordinates", context.coordinatePath, "--canonical", context.canonicalPath, "--uncertainty", context.uncertaintyPath,
      "--segments", context.segmentPath, "--output-dir", context.directory,
      "--expected-source-sha256", identities.sourceSha256, "--expected-coordinate-sha256", identities.coordinateSha256],
    { cwd: context.resourceRoot, env: context.environment, signal: context.signal, timeout: 180_000, maxBuffer: 1024 * 1024 })
  } finally { release() }
  return verifySourceTimePublication(context.directory, identities)
}
