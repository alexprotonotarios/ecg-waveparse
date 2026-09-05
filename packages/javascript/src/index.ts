import { spawn } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"
import type { RunRecord, RunAssetKey, RunAsset } from "../../../src/lib/runs"

declare const __WAVEPARSE_VERSION__: string
declare const __WAVEPARSE_FORMAT__: string
export const version = __WAVEPARSE_VERSION__

export type RunResult = Omit<RunRecord, "assets"> & {
  schemaVersion: 1
  waveparse: { version: string; runtimeManifestSha256: string; device: string }
  assets: Partial<Record<RunAssetKey, RunAsset & { absolutePath: string }>>
}
export type ReviewOptions = {
  decision: "accepted" | "rejected"
  reviewer: string
  notes: string
  confirmations?: { sourceCompared: boolean; leadIdentityVerified: boolean; scaleAndGapsReviewed: boolean }
}
export type DigitizerOptions = { workspaceDir: string; runtimeDir?: string }
export type DigitizeOptions = {
  device?: "cpu" | "mps"
  timeoutMs?: number
  signal?: AbortSignal
  onProgress?: (event: { type: "started"; runId: string }) => void
}
export class WaveParseError extends Error {
  constructor(public code: string, message: string, public runId?: string) {
    super(message)
    this.name = "WaveParseError"
  }
}

function resourceRoot() {
  const directory = __WAVEPARSE_FORMAT__ === "cjs" ? __dirname : path.dirname(fileURLToPath(import.meta.url))
  return path.resolve(directory, "../runtime")
}

/** Runs the installed engine locally. No downloads or network calls occur here. */
export class Digitizer {
  private readonly options: DigitizerOptions
  constructor(options: DigitizerOptions) {
    if (!options?.workspaceDir) throw new WaveParseError("invalid_request", "workspaceDir is required.")
    this.options = {
      workspaceDir: path.resolve(options.workspaceDir),
      ...(options.runtimeDir ? { runtimeDir: path.resolve(options.runtimeDir) } : {}),
    }
  }

  digitize(inputPath: string, options: DigitizeOptions = {}): Promise<RunResult> {
    return this.request({ operation: "digitize", inputPath: path.resolve(inputPath), device: options.device ?? "cpu", timeoutMs: options.timeoutMs ?? 1_800_000 }, options)
  }
  getRun(runId: string): Promise<RunResult | null> {
    return this.request({ operation: "get", runId })
  }
  review(runId: string, options: ReviewOptions): Promise<RunResult> {
    return this.request({ operation: "review", runId, ...options })
  }

  private request<T>(request: Record<string, unknown>, options: DigitizeOptions = {}): Promise<T> {
    if (options.signal?.aborted) return Promise.reject(new WaveParseError("cancelled", "Digitization cancelled."))
    return new Promise((resolve, reject) => {
      const root = resourceRoot()
      const child = spawn(process.execPath, [path.join(root, "runner.cjs")], {
        stdio: ["pipe", "pipe", "pipe"], detached: process.platform !== "win32",
        env: { ...process.env, WAVEPARSE_RESOURCE_ROOT: root, WAVEPARSE_WORKSPACE_ROOT: this.options.workspaceDir },
      })
      let buffer = "", errors = "", runId: string | undefined
      let final: { result?: T; error?: { code: string; message: string; runId?: string } } | undefined
      let forceTimer: ReturnType<typeof setTimeout> | undefined
      let callbackError: Error | undefined
      const terminate = () => {
        child.kill("SIGTERM")
        forceTimer ??= setTimeout(() => {
          try { if (child.pid) process.kill(process.platform === "win32" ? child.pid : -child.pid, "SIGKILL") } catch { /* Already exited. */ }
        }, 30_000)
        forceTimer.unref()
      }
      options.signal?.addEventListener("abort", terminate, { once: true })
      if (options.signal?.aborted) terminate()
      child.stdout.setEncoding("utf8").on("data", (chunk: string) => {
        buffer += chunk
        if (buffer.length > 16 * 1024 * 1024) { callbackError = new WaveParseError("protocol_error", "Runner output exceeded the protocol limit."); terminate(); return }
        let end: number
        while ((end = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, end); buffer = buffer.slice(end + 1)
          try {
            const event = JSON.parse(line)
            if (event.type === "started") { runId = event.runId; options.onProgress?.(event) }
            else if (event.type === "result" || event.type === "error") final = event
            else throw new Error("Unknown runner event.")
          } catch (error) { callbackError = error instanceof Error ? error : new Error(String(error)); terminate() }
        }
      })
      child.stderr.setEncoding("utf8").on("data", (chunk: string) => { errors = (errors + chunk).slice(-8192) })
      child.stdin.on("error", () => { /* Exit/error event reports the failure. */ })
      child.on("error", (error) => reject(new WaveParseError("runtime_unavailable", error.message)))
      child.on("close", (code) => {
        if (forceTimer) clearTimeout(forceTimer)
        options.signal?.removeEventListener("abort", terminate)
        if (callbackError) return reject(callbackError)
        if (final?.error) return reject(new WaveParseError(final.error.code, final.error.message, final.error.runId))
        if (options.signal?.aborted) return reject(new WaveParseError("cancelled", "Digitization cancelled; the admitted source is preserved.", runId))
        if (code !== 0 || !final || !("result" in final)) return reject(new WaveParseError("runner_failed", errors || `Runner exited with code ${code}.`, runId))
        resolve(final.result as T)
      })
      child.stdin.end(JSON.stringify({ protocolVersion: 1, ...request, ...this.options }) + "\n")
    })
  }
}
