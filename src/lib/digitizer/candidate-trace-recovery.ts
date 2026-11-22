import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { promises as fs } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import type { DigitizerCandidateSummary, RunAsset, RunRecord } from "@/lib/runs"
import { ECG_LAYOUTS, ECG_LEADS } from "./domain"
import { EvidenceContractError } from "./physical-contracts"

const execute = promisify(execFile)
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex")
const isHash = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
const REASONS = ["no_publishable_candidate", "semantic_identity_unconfirmed", "calibration_conflict", "unsupported_calibration"]
const REVIEW_REASONS = ["source_verified", "adaptive_preprocessed", "lead_fusion", "peer_trusted", "reviewable_constrained"]
const PARTIAL_REASONS = ["supported_partial_layout", "input_quality_review_only", "unstable_neural_confirmation"]
const MAX_BYTES = 32 * 1024 ** 2

export type CandidateTraceRecovery = {
  version: 1
  method: "candidate-trace-recovery-v1"
  state: "diagnostic_available"
  candidateId: string
  sourceSha256: string
  manifestSha256: string
  originalPublicationReason: string
  finiteSamples: number
  tracedColumns: number
  leadIdentityConfirmed: false
  calibrationVerified: false
  quantitativeUseApproved: false
  clinicalValidationUse: false
  gapsPreserved: true
  waveformChanged: false
}

type Candidate = {
  id: string
  status: string
  canonicalPath?: string
  diagnosticPath?: string
  decoderCoordinatePath?: string
  panelObservationPath?: string
  additionalTracesPath?: string
  panelTraceRecovery?: Record<string, unknown>
  rank: number
  summary: DigitizerCandidateSummary
  preparation?: unknown
}

type Context = {
  candidates: Candidate[]
  originalSourcePath: string
  sourceSha256: string
  runDirectory: string
  python: string
  environment: NodeJS.ProcessEnv
  resourceRoot?: string
  signal?: AbortSignal
  asset: (label: string, absolutePath: string, sourceSha256: string) => Promise<RunAsset>
}

/** Count the actual exported samples; inferred column names do not verify leads. */
export function summarizeRecoveredCsv(bytes: Buffer) {
  if (bytes.length === 0 || bytes.length > MAX_BYTES) return undefined
  const lines = bytes.toString("utf8").trimEnd().split(/\r?\n/)
  const columns = lines.shift()?.split(",").map(value => value.trim()) ?? []
  if (columns.length !== ECG_LEADS.length || columns.some((value, index) => value !== ECG_LEADS[index]) || !lines.length) return undefined
  const finiteByColumn = Object.fromEntries(columns.map(column => [column, 0]))
  let finiteSamples = 0
  for (const line of lines) {
    const values = line.split(",")
    if (values.length !== columns.length) return undefined
    for (let index = 0; index < values.length; index++) {
      const cell = values[index].trim()
      if (!cell || /^nan$/i.test(cell)) continue
      if (!Number.isFinite(Number(cell))) return undefined
      finiteByColumn[columns[index]]++
      finiteSamples++
    }
  }
  if (!finiteSamples) return undefined
  return {
    rows: lines.length, finiteSamples, finiteByColumn,
    missingByColumn: Object.fromEntries(columns.map(column => [column, lines.length - finiteByColumn[column]])),
    tracedColumns: Object.values(finiteByColumn).filter(count => count > 0).length,
  }
}

/** Diagnostic coverage is independent of the thresholds for measurements. */
export function hasPotentialRhythmOverwrite(summary: DigitizerCandidateSummary, support: NonNullable<ReturnType<typeof summarizeRecoveredCsv>>) {
  return /^standard_3x4_with_r[123]$/.test(summary.layout ?? "") &&
    ECG_LEADS.some(lead => lead !== "II" && support.finiteByColumn[lead] > support.rows / 4 + 250)
}

export function diagnosticCandidatePriority(summary: DigitizerCandidateSummary, support: NonNullable<ReturnType<typeof summarizeRecoveredCsv>>) {
  if (summary.sourceFidelity?.passed === true) return 0
  // A populated wrong geometry can be tracing labels, redactions or the grid.
  // Keep it as a fallback, but prefer an observed layout with supported traces.
  if (summary.sourceFidelity?.passed === false) return 3
  if (hasPotentialRhythmOverwrite(summary, support)) return 3
  const layout = summary.layout?.replace(/_constrained$/, "").replace(/_with_r[123]$/, "")
  const expected = layout ? ECG_LAYOUTS[layout]?.rows.flat() : undefined
  const supported = expected?.length && expected.every(lead => support.finiteByColumn[lead] >= 250)
  if (supported && Number.isFinite(summary.layoutCost) && summary.layoutCost! <= 0.75) {
    return summary.parameters.geometryConfirmedLayout === true ? 2 : 1
  }
  return 4
}

export function validateCandidateTraceRecovery(value: CandidateTraceRecovery, sourceSha256?: string, mode: "failed" | "needs_review" = "failed") {
  if (!value || value.version !== 1 || value.method !== "candidate-trace-recovery-v1" ||
      value.state !== "diagnostic_available" || typeof value.candidateId !== "string" || !value.candidateId ||
      !isHash(value.sourceSha256) || !isHash(value.manifestSha256) ||
      (sourceSha256 && value.sourceSha256 !== sourceSha256) || !(mode === "needs_review" ? REVIEW_REASONS : [...REASONS, ...PARTIAL_REASONS]).includes(value.originalPublicationReason) ||
      !Number.isSafeInteger(value.finiteSamples) || value.finiteSamples <= 0 ||
      !Number.isSafeInteger(value.tracedColumns) || value.tracedColumns < 1 || value.tracedColumns > ECG_LEADS.length ||
      value.leadIdentityConfirmed !== false || value.calibrationVerified !== false ||
      value.quantitativeUseApproved !== false || value.clinicalValidationUse !== false ||
      value.gapsPreserved !== true || value.waveformChanged !== false) {
    throw new EvidenceContractError("invalid_candidate_trace_recovery", "Recovered traces require a source-bound, provisional diagnostic record with unchanged gaps.")
  }
}

/** Preserve a recovered candidate independently of the quantitative decision. */
export async function retainCandidateTraceDiagnostic(failure: Partial<RunRecord>, context: Context): Promise<Partial<RunRecord> | undefined> {
  const besideQuantitative = failure.status === "needs_review" && failure.publicationDecision?.outcome === "needs_review" &&
    REVIEW_REASONS.includes(failure.publicationDecision.reasonCode) && Boolean(failure.assets?.canonicalCsv)
  const partialDiagnostic = failure.status === "partial" && failure.publicationDecision?.outcome === "partial" &&
    PARTIAL_REASONS.includes(failure.publicationDecision.reasonCode)
  if (!besideQuantitative && !partialDiagnostic && (failure.status !== "failed" || !REASONS.includes(failure.publicationDecision?.reasonCode ?? ""))) return undefined
  context.signal?.throwIfAborted()
  if (hash(await fs.readFile(context.originalSourcePath)) !== context.sourceSha256) {
    throw new EvidenceContractError("source_integrity_mismatch", "The original source changed before trace recovery.")
  }
  const candidates: Array<{ candidate: Candidate; signal: Buffer;
    support: NonNullable<ReturnType<typeof summarizeRecoveredCsv>>; priority: number }> = []
  const panelRecoveryAttempts: Record<string, unknown>[] = []
  for (const candidate of context.candidates.filter(candidate => candidate.status === "completed" && candidate.canonicalPath && candidate.diagnosticPath)) {
    context.signal?.throwIfAborted()
    const signal = await fs.readFile(candidate.canonicalPath!)
    const support = summarizeRecoveredCsv(signal)
    if (!support) continue
    candidates.push({ candidate, signal, support, priority: diagnosticCandidatePriority(candidate.summary, support) })
    const panelCapture = candidate.panelObservationPath ?? candidate.decoderCoordinatePath
    if (candidate.summary.layout === "standard_3x4_with_r1" &&
        (hasPotentialRhythmOverwrite(candidate.summary, support) || candidate.summary.parameters.provisionalLayoutHint === true) &&
        panelCapture && context.resourceRoot) {
      const captureSha256 = hash(await fs.readFile(panelCapture))
      try {
        if (!/^[a-zA-Z0-9_-]+$/.test(candidate.id)) throw new Error("Unsafe panel recovery identifier")
        const output = path.join(context.runDirectory, "candidates", "trace-panel-recovery", candidate.id)
        const { stdout } = await execute(context.python, ["-m", "ecg_pipeline.panel_trace_recovery",
          "--capture", panelCapture, "--capture-sha256", captureSha256,
          "--source", context.originalSourcePath, "--source-sha256", context.sourceSha256, "--output", output],
        { cwd: context.resourceRoot, env: context.environment, signal: context.signal, timeout: 30_000, maxBuffer: 1024 * 1024 })
        const receipt = JSON.parse(stdout) as Record<string, unknown>
        if (receipt.method !== "observed-decoder-panel-preservation-v1" || receipt.state !== "recovered" ||
            receipt.sourceSha256 !== context.sourceSha256 || receipt.captureSha256 !== captureSha256 ||
            receipt.canonicalExportChanged !== false || receipt.waveformChanged !== false || receipt.gapsPreserved !== true ||
            receipt.leadIdentityConfirmed !== false || receipt.calibrationVerified !== false ||
            receipt.quantitativeUseApproved !== false || receipt.clinicalValidationUse !== false ||
            receipt.additionalTracesContainAllObservedDecoderRows !== true ||
            receipt.csvPath !== path.join(output, "candidate-signal.csv") ||
            receipt.previewPath !== path.join(output, "candidate-preview.png") ||
            receipt.additionalTracesPath !== path.join(output, "candidate-additional-traces.csv")) {
          throw new Error("Invalid panel preservation receipt")
        }
        if (candidate.summary.parameters.provisionalLayoutHint === true &&
            (receipt.provisionalLayoutHint !== true || !Number.isSafeInteger(receipt.observedDecoderRows) ||
              (receipt.observedDecoderRows as number) < 4 || (receipt.observedDecoderRows as number) > 6)) {
          throw new Error("Invalid provisional observed-row receipt")
        }
        const preservedSignal = await fs.readFile(receipt.csvPath as string)
        const preservedSupport = summarizeRecoveredCsv(preservedSignal)
        if (!preservedSupport || hash(preservedSignal) !== receipt.signalSha256 ||
            hash(await fs.readFile(receipt.previewPath as string)) !== receipt.previewSha256 ||
            hash(await fs.readFile(receipt.additionalTracesPath as string)) !== receipt.additionalTracesSha256 ||
            hash(await fs.readFile(panelCapture)) !== captureSha256 ||
            hash(await fs.readFile(candidate.canonicalPath!)) !== hash(signal)) {
          throw new Error("Panel preservation bytes changed")
        }
        panelRecoveryAttempts.push({ candidateId: candidate.id, ...receipt })
        const preserved: Candidate = { ...candidate, id: candidate.id + "-panels",
          canonicalPath: receipt.csvPath as string, diagnosticPath: receipt.previewPath as string,
          additionalTracesPath: receipt.additionalTracesPath as string, panelTraceRecovery: receipt,
          summary: { ...candidate.summary, id: candidate.id + "-panels", label: candidate.summary.label + " preserved panels" } }
        candidates.push({ candidate: preserved, signal: preservedSignal, support: preservedSupport,
          priority: diagnosticCandidatePriority(preserved.summary, preservedSupport) })
      } catch (error) {
        context.signal?.throwIfAborted()
        panelRecoveryAttempts.push({ candidateId: candidate.id, method: "observed-decoder-panel-preservation-v1",
          state: "refused", captureSha256, sourceSha256: context.sourceSha256,
          reason: error instanceof Error ? error.message.slice(0, 1000) : String(error) })
      }
    }
  }
  const completeObservedPanels = (item: (typeof candidates)[number]) => Boolean(item.candidate.panelTraceRecovery &&
    item.candidate.summary.layout === "standard_3x4_with_r1" &&
    ((Number.isFinite(item.candidate.summary.layoutCost) && item.candidate.summary.layoutCost! <= 0.75) ||
      (item.candidate.summary.parameters.provisionalLayoutHint === true &&
        item.candidate.summary.parameters.selectionEligible === false &&
        item.candidate.panelTraceRecovery.provisionalLayoutHint === true)) &&
    ECG_LEADS.every(lead => item.support.finiteByColumn[lead] >= 250))
  candidates.sort((first, second) => Number(completeObservedPanels(second)) - Number(completeObservedPanels(first)) ||
    first.priority - second.priority ||
    (Number.isFinite(first.candidate.summary.layoutCost) ? first.candidate.summary.layoutCost! : Number.POSITIVE_INFINITY) -
      (Number.isFinite(second.candidate.summary.layoutCost) ? second.candidate.summary.layoutCost! : Number.POSITIVE_INFINITY) ||
    first.candidate.rank - second.candidate.rank || first.candidate.id.localeCompare(second.candidate.id))
  for (const { candidate, signal, support, priority } of candidates) {
    context.signal?.throwIfAborted()
    const preview = await fs.readFile(candidate.diagnosticPath!)
    if (!preview.length || preview.length > MAX_BYTES) continue
    const directory = path.join(context.runDirectory, "candidates", "trace-recovery")
    await fs.mkdir(directory, { recursive: true })
    const summary = Buffer.from(JSON.stringify({ candidate: candidate.summary, preparation: candidate.preparation ?? null,
      panelTraceRecovery: candidate.panelTraceRecovery ?? null, panelRecoveryAttempts }, null, 2) + "\n")
    const files: Record<string, Buffer> = { "candidate-signal.csv": signal, "candidate-preview.png": preview, "candidate-summary.json": summary }
    if (candidate.additionalTracesPath) {
      const additional = await fs.readFile(candidate.additionalTracesPath)
      if (!additional.length || additional.length > MAX_BYTES || hash(additional) !== candidate.panelTraceRecovery?.additionalTracesSha256) {
        throw new EvidenceContractError("artifact_integrity_mismatch", "The additional rhythm trace changed before retention.")
      }
      files["candidate-additional-traces.csv"] = additional
    }
    const manifest = Buffer.from(JSON.stringify({
      schemaVersion: 1, method: "candidate-trace-recovery-v1", state: "diagnostic_available",
      selectionMethod: "diagnostic-complete-observed-panels-v5", diagnosticPriority: priority,
      candidateId: candidate.id, originalSourceSha256: context.sourceSha256,
      originalPublicationDecision: failure.publicationDecision, ...support,
      leadIdentityConfirmed: false, calibrationVerified: false, quantitativeUseApproved: false,
      clinicalValidationUse: false, gapsPreserved: true, waveformChanged: false,
      quantitativeUncertainty: "unavailable", columnNamesAreInferred: true,
      timingAndAmplitudeAreProvisional: true, previewCoordinateSpace: "candidate_input",
      sourceCorrespondenceVerified: false,
      panelTraceRecovery: candidate.panelTraceRecovery ?? null,
      additionalTracesAreUnverified: Boolean(candidate.additionalTracesPath),
      instructions: "Compare the candidate preview with the untouched source. Lead labels, timing, amplitudes and morphology may be wrong. Missing samples remain gaps. This bundle is for tracing review; measurements require separately verified mapping and calibration.",
      files: Object.fromEntries(Object.entries(files).map(([name, bytes]) => [name, { sha256: hash(bytes), bytes: bytes.length }])),
    }, null, 2) + "\n")
    for (const [name, bytes] of Object.entries({ ...files, "manifest.json": manifest })) await fs.writeFile(path.join(directory, name), bytes)
    const archive = path.join(directory, "review.zip")
    await execute(context.python, ["-m", "zipfile", "-c", archive, ...Object.keys(files), "manifest.json"],
      { cwd: directory, env: context.environment, signal: context.signal, timeout: 30_000, maxBuffer: 1024 * 1024 })
    await execute(context.python, ["-c", [
      "import hashlib,json,sys,zipfile", "from pathlib import Path", "folder=Path(sys.argv[1])",
      "manifest=json.loads((folder/'manifest.json').read_text())", "with zipfile.ZipFile(sys.argv[2]) as archive:",
      " assert sorted(archive.namelist())==sorted([*manifest['files'],'manifest.json'])",
      " for name in archive.namelist():", "  data=archive.read(name); assert data==(folder/name).read_bytes()",
      "  if name!='manifest.json': assert hashlib.sha256(data).hexdigest()==manifest['files'][name]['sha256']",
    ].join("\n"), directory, archive], { cwd: directory, env: context.environment, signal: context.signal, timeout: 30_000, maxBuffer: 1024 * 1024 })
    if (hash(await fs.readFile(context.originalSourcePath)) !== context.sourceSha256 ||
        hash(await fs.readFile(candidate.canonicalPath!)) !== hash(signal) || hash(await fs.readFile(candidate.diagnosticPath!)) !== hash(preview)) {
      throw new EvidenceContractError("artifact_integrity_mismatch", "Source or candidate changed during trace recovery.")
    }
    const exports = path.join(context.runDirectory, "exports")
    await fs.mkdir(exports, { recursive: true })
    const manifestExport = path.join(exports, "candidate-trace-recovery.json")
    const archiveExport = path.join(exports, "candidate-trace-recovery.zip")
    const previewExport = path.join(exports, "candidate-trace-preview.png")
    const signalExport = path.join(exports, "candidate-trace-provisional.csv")
    const archiveHash = hash(await fs.readFile(archive))
    await fs.copyFile(archive, archiveExport)
    await fs.writeFile(manifestExport, manifest)
    await fs.writeFile(previewExport, preview)
    const retainedPreview = await context.asset("Provisional recovered trace preview", previewExport, context.sourceSha256)
    const retainedManifest = await context.asset("Provisional tracing and missing-sample evidence", manifestExport, context.sourceSha256)
    const retainedBundle = await context.asset("Provisional trace CSV and review bundle", archiveExport, context.sourceSha256)
    if (retainedPreview.identity?.sha256 !== hash(preview) || retainedManifest.identity?.sha256 !== hash(manifest) || retainedBundle.identity?.sha256 !== archiveHash) {
      throw new EvidenceContractError("artifact_integrity_mismatch", "Recovered trace assets changed while being retained.")
    }
    const traceRecovery: CandidateTraceRecovery = {
      version: 1, method: "candidate-trace-recovery-v1", state: "diagnostic_available",
      candidateId: candidate.id, sourceSha256: context.sourceSha256, manifestSha256: hash(manifest),
      originalPublicationReason: failure.publicationDecision!.reasonCode,
      finiteSamples: support.finiteSamples, tracedColumns: support.tracedColumns,
      leadIdentityConfirmed: false, calibrationVerified: false, quantitativeUseApproved: false,
      clinicalValidationUse: false, gapsPreserved: true, waveformChanged: false,
    }
    validateCandidateTraceRecovery(traceRecovery, context.sourceSha256, besideQuantitative ? "needs_review" : "failed")
    if (besideQuantitative) {
      await fs.writeFile(signalExport, signal)
      const retainedSignal = await context.asset("Provisional trace CSV with original gaps", signalExport, context.sourceSha256)
      if (retainedSignal.identity?.sha256 !== hash(signal)) {
        throw new EvidenceContractError("artifact_integrity_mismatch", "The provisional CSV changed while being retained.")
      }
      return { ...failure, provisionalTrace: traceRecovery, assets: { ...failure.assets,
        provisionalTraceCsv: retainedSignal, provisionalTracePreview: retainedPreview,
        provisionalTraceManifest: retainedManifest, provisionalTraceBundle: retainedBundle,
      } }
    }
    const assets = { ...failure.assets, diagnostic: retainedPreview, diagnosticManifest: retainedManifest, diagnosticBundle: retainedBundle }
    for (const key of ["canonicalCsv", "segmentsCsv", "uncertaintyCsv", "segmentMapJson", "coordinateEvidence", "sourceTimeEvidence", "paperRender"] as const) delete assets[key]
    return { ...failure, status: "partial", assets, traceRecovery,
      publicationDecision: { ...failure.publicationDecision!, outcome: "partial", reasonCode: "diagnostic_candidate_trace", candidateId: candidate.id },
      message: "A provisional trace was recovered. Inspect the preview and trace CSV beside the original. Lead labels, timing, amplitudes and morphology require verification; missing samples remain gaps.",
      layout: undefined, leadCount: undefined, layoutCost: undefined, selectedCandidateId: undefined,
      sampleRateHz: undefined, effectiveSampleRateHz: undefined, paperSpeedMmPerSecond: undefined,
      gainMmPerMv: undefined, calibrationConfidence: undefined, reliability: undefined,
      updatedAt: new Date().toISOString(),
    }
  }
  return undefined
}
