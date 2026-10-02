import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { promises as fs } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import type { RunAsset, RunRecord } from "@/lib/runs"
import { verifySourcePhotoBundle, type SourcePhotoAttempt } from "./source-photo-contract"

const execute = promisify(execFile)
type Context = {
  workingSourcePath: string
  originalSourcePath: string
  imageSize: [number, number]
  sourceSha256: string
  runDirectory: string
  python: string
  resourceRoot: string
  environment: NodeJS.ProcessEnv
  signal?: AbortSignal
  acquire: () => Promise<() => void>
  asset: (label: string, absolutePath: string, sourceSha256: string) => Promise<RunAsset>
}

export function sourcePhotoFallbackEligible(failure: Partial<RunRecord>): boolean {
  return failure.status === "failed" && ["no_publishable_candidate", "semantic_identity_unconfirmed"].includes(failure.publicationDecision?.reasonCode ?? "")
}

/** Separate diagnostic publication after ordinary quantitative selection declines. */
export async function retainSourcePhotoDiagnostic(failure: Partial<RunRecord>, context: Context): Promise<Partial<RunRecord>> {
  if (!sourcePhotoFallbackEligible(failure)) return failure
  context.signal?.throwIfAborted()
  const started = performance.now()
  // An unreadable working raster is an input/integrity failure. Do not fabricate
  // a source identity or turn it into an ordinary diagnostic abstention.
  const workingSourceSha256 = createHash("sha256").update(await fs.readFile(context.workingSourcePath)).digest("hex")
  const attempt: SourcePhotoAttempt = {
    version: 2, method: "source-photo-waveform-v1", state: "failed",
    originalSourceSha256: context.sourceSha256, workingSourceSha256, runtimeMs: 0,
    originalPublicationReason: failure.publicationDecision!.reasonCode,
    productionEligible: false, semanticIdentityConfirmed: false, quantitativeUseApproved: false,
  }
  const directory = path.join(context.runDirectory, "candidates", "source-photo-diagnostic")
  const bundle = path.join(directory, "bundle")
  try {
    if (createHash("sha256").update(await fs.readFile(context.originalSourcePath)).digest("hex") !== context.sourceSha256) {
      throw new Error("The original source changed before diagnostic extraction.")
    }
    await fs.rm(directory, { recursive: true, force: true })
    await fs.mkdir(directory, { recursive: true })
    const release = await context.acquire()
    try {
      context.signal?.throwIfAborted()
      await execute(context.python, ["-m", "ecg_pipeline.source_photo", "--input", context.workingSourcePath,
        "--output-dir", bundle, "--expected-source-sha256", workingSourceSha256, "--original-source", context.originalSourcePath],
      { cwd: context.resourceRoot, env: context.environment, signal: context.signal, timeout: 120_000, maxBuffer: 1024 * 1024 })
    } finally { release() }
    const { candidate, manifestSha256, names } = await verifySourcePhotoBundle(bundle, {
      originalSha256: context.sourceSha256, workingSha256: workingSourceSha256, imageSize: context.imageSize,
    })
    const zip = path.join(directory, "diagnostic-bundle.zip")
    await execute(context.python, ["-m", "zipfile", "-c", zip, ...names],
      { cwd: bundle, env: context.environment, signal: context.signal, timeout: 30_000, maxBuffer: 1024 * 1024 })
    // Check again before retaining the archive and copies; no unverified member is published.
    const confirmed = await verifySourcePhotoBundle(bundle, {
      originalSha256: context.sourceSha256, workingSha256: workingSourceSha256, imageSize: context.imageSize,
    })
    if (confirmed.manifestSha256 !== manifestSha256) throw new Error("The diagnostic manifest changed during export.")
    await execute(context.python, ["-c", [
      "import hashlib,json,sys,zipfile",
      "from pathlib import Path",
      "folder=Path(sys.argv[1]); manifest=json.loads((folder/'candidate.json').read_text())",
      "with zipfile.ZipFile(sys.argv[2]) as archive:",
      " assert sorted(archive.namelist())==sorted([*manifest['files'],'candidate.json'])",
      " for name in archive.namelist():",
      "  data=archive.read(name); assert data==(folder/name).read_bytes()",
      "  if name!='candidate.json': assert hashlib.sha256(data).hexdigest()==manifest['files'][name]['sha256']",
    ].join("\n"), bundle, zip],
    { cwd: context.resourceRoot, env: context.environment, signal: context.signal, timeout: 30_000, maxBuffer: 1024 * 1024 })
    const zipSha256 = createHash("sha256").update(await fs.readFile(zip)).digest("hex")
    const exports = path.join(context.runDirectory, "exports")
    await fs.mkdir(exports, { recursive: true })
    const manifestExport = path.join(exports, "source-photo-candidate.json")
    const bundleExport = path.join(exports, "source-photo-diagnostic.zip")
    await fs.copyFile(path.join(bundle, "candidate.json"), manifestExport)
    await fs.copyFile(zip, bundleExport)
    attempt.state = candidate.state
    attempt.manifestSha256 = manifestSha256
    attempt.runtimeMs = performance.now() - started
    attempt.message = candidate.branch
    const assets = {
      ...failure.assets,
      diagnosticManifest: await context.asset("Photo diagnostic manifest", manifestExport, context.sourceSha256),
      diagnosticBundle: await context.asset("Photo diagnostic review bundle", bundleExport, context.sourceSha256),
    }
    if (assets.diagnosticManifest.identity?.sha256 !== manifestSha256 || assets.diagnosticBundle.identity?.sha256 !== zipSha256) {
      throw new Error("The diagnostic bundle changed while being retained.")
    }
    if (candidate.state !== "diagnostic_available") return { ...failure, assets, sourcePhotoAttempt: attempt }
    const overlay = path.join(exports, "source-photo-overlay.png"), paper = path.join(exports, "source-photo-paper.png")
    await fs.copyFile(path.join(bundle, "source-overlay.png"), overlay)
    await fs.copyFile(path.join(bundle, "ecg-paper.png"), paper)
    const diagnosticAssets: RunRecord["assets"] = { ...assets,
      diagnostic: await context.asset("Photo diagnostic source overlay", overlay, context.sourceSha256),
      paperRender: await context.asset("Photo diagnostic paper render", paper, context.sourceSha256),
    }
    if (diagnosticAssets.diagnostic?.identity?.sha256 !== candidate.files["source-overlay.png"].sha256 ||
        diagnosticAssets.paperRender?.identity?.sha256 !== candidate.files["ecg-paper.png"].sha256) {
      throw new Error("The diagnostic preview changed while being retained.")
    }
    for (const key of ["canonicalCsv", "segmentsCsv", "uncertaintyCsv", "segmentMapJson"] as const) delete diagnosticAssets[key]
    return {
      ...failure, status: "partial", assets: diagnosticAssets, sourcePhotoAttempt: attempt,
      publicationDecision: { ...failure.publicationDecision!, outcome: "partial", reasonCode: "diagnostic_source_photo", candidateId: candidate.candidateId },
      message: "A diagnostic trace was recovered from the photo. Compare the overlay and paper view with the original; gaps and tracing errors remain. The review bundle contains the signal and its evidence. Quantitative use is not approved.",
      layout: "standard_3x4_with_r1", leadCount: 12, layoutCost: undefined,
      sampleRateHz: undefined, effectiveSampleRateHz: undefined, paperSpeedMmPerSecond: undefined,
      gainMmPerMv: undefined, calibrationConfidence: undefined, selectedCandidateId: undefined, reliability: undefined,
      updatedAt: new Date().toISOString(),
    }
  } catch (error) {
    context.signal?.throwIfAborted()
    attempt.state = "failed"
    delete attempt.manifestSha256
    attempt.runtimeMs = performance.now() - started
    attempt.message = error instanceof Error ? error.message.slice(0, 2000) : "Diagnostic photo extraction failed."
    const exports = path.join(context.runDirectory, "exports")
    await fs.mkdir(exports, { recursive: true })
    const receipt = path.join(exports, "source-photo-attempt.json")
    await fs.writeFile(receipt, JSON.stringify(attempt, null, 2) + "\n")
    return { ...failure, sourcePhotoAttempt: attempt, assets: { ...failure.assets,
      diagnosticManifest: await context.asset("Photo diagnostic attempt", receipt, context.sourceSha256),
    } }
  }
}
