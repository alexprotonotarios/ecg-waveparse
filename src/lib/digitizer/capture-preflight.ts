import { execFile } from "node:child_process"
import { promises as fs } from "node:fs"
import os from "node:os"
import path from "node:path"
import { promisify } from "node:util"

import {
  evaluatePreprocessingCaptureQuality,
  type CaptureMethod,
  type CaptureQualitySummary,
} from "@/lib/digitizer/capture-quality"
import { parsePreprocessingReport } from "@/lib/digitizer/contracts"

const execFileAsync = promisify(execFile)
import { RESOURCE_ROOT as ROOT_DIR } from "@/lib/runtime-paths"
const INPUT_PREPARER_PATH = path.join(
  ROOT_DIR,
  "scripts",
  "prepare_ecg_input.py"
)
const PREFLIGHT_TIMEOUT_MS = 90_000

export class CapturePreflightUnavailableError extends Error {
  readonly code = "capture_preflight_unavailable"

  constructor(message: string) {
    super(message)
    this.name = "CapturePreflightUnavailableError"
  }
}

export async function preflightCaptureBytes({
  fileName,
  bytes,
  captureMethod,
}: {
  fileName: string
  bytes: Buffer
  captureMethod: CaptureMethod
}): Promise<CaptureQualitySummary> {
  const temporaryDirectory = await fs.mkdtemp(
    path.join(os.tmpdir(), "ecg-capture-preflight-")
  )
  const extension = safeImageExtension(fileName)
  const inputPath = path.join(
    /* turbopackIgnore: true */ temporaryDirectory,
    `source${extension}`
  )
  const reportPath = path.join(
    /* turbopackIgnore: true */ temporaryDirectory,
    "capture-quality.json"
  )
  const output = (name: string) =>
    path.join(
      /* turbopackIgnore: true */ temporaryDirectory,
      `${name}.png`
    )

  try {
    await fs.writeFile(inputPath, bytes, { mode: 0o600 })
    try {
      await execFileAsync(
        openEcgPythonPath(),
        [
          INPUT_PREPARER_PATH,
          "--input",
          inputPath,
          "--prepared",
          output("prepared"),
          "--working-source",
          output("working-source"),
          "--mask",
          output("mask"),
          "--enhanced",
          output("enhanced"),
          "--background",
          output("background"),
          "--trace-probability",
          output("trace-probability"),
          "--grid-probability",
          output("grid-probability"),
          "--exclusion-mask",
          output("exclusion-mask"),
          "--geometry-corrected",
          output("geometry-corrected"),
          "--artifact-preprocessed",
          output("artifact-preprocessed"),
          "--report",
          reportPath,
        ],
        {
          cwd: ROOT_DIR,
          env: {
            ...process.env,
            PYTHONUNBUFFERED: "1",
          },
          maxBuffer: 4 * 1024 * 1024,
          timeout: PREFLIGHT_TIMEOUT_MS,
        }
      )
    } catch (error) {
      throw new CapturePreflightUnavailableError(
        `Capture quality verification could not run: ${failureMessage(error)}`
      )
    }
    const report = parsePreprocessingReport(
      await fs.readFile(reportPath, "utf8")
    )
    return evaluatePreprocessingCaptureQuality(report, captureMethod)
  } finally {
    await fs.rm(temporaryDirectory, { recursive: true, force: true })
  }
}

function openEcgPythonPath() {
  const openEcgDirectory = path.resolve(
    /* turbopackIgnore: true */ ROOT_DIR,
    process.env.OPEN_ECG_DIGITIZER_DIR ??
      [".external", "open-ecg-digitizer"].join(path.sep)
  )
  return path.join(openEcgDirectory, ".venv", "bin", "python")
}

function safeImageExtension(fileName: string) {
  const extension = path.extname(fileName).toLowerCase()
  return [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"].includes(
    extension
  )
    ? extension
    : ".png"
}

function failureMessage(error: unknown) {
  if (error instanceof Error) return error.message
  return "the deterministic image analyser was unavailable"
}
