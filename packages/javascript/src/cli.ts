#!/usr/bin/env node
import { spawn } from "node:child_process"
import path from "node:path"
import { Digitizer, version } from "./index"

function option(name: string) { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1] }
const command = process.argv[2]
async function main() {
  if (command === "--version") { console.log(version); return }
  if (command === "setup" || command === "doctor") {
    const resources = path.resolve(__dirname, "../runtime")
    const args = [path.join(resources, "runtime_setup.py"), command, "--resource-root", resources]
    for (const key of ["--runtime-dir", "--device", "--workspace-dir"]) { const value = option(key); if (value) args.push(key, value) }
    const python = option("--python") ?? process.env.WAVEPARSE_PYTHON ?? "python3.12"
    await new Promise<void>((resolve, reject) => {
      const child = spawn(python, args, { stdio: "inherit", env: { ...process.env, WAVEPARSE_NODE: process.execPath } })
      const stop = () => child.kill("SIGTERM")
      process.once("SIGINT", stop); process.once("SIGTERM", stop)
      child.on("error", reject)
      child.on("close", (code) => { process.removeListener("SIGINT", stop); process.removeListener("SIGTERM", stop); process.exitCode = code ?? 1; resolve() })
    })
    return
  }
  if (command !== "digitize" || !option("--input") || !option("--workspace-dir")) {
    console.log("ECG WaveParse\n  ecg-waveparse setup [--python /path/to/python3.12] [--runtime-dir DIR]\n  ecg-waveparse doctor [--runtime-dir DIR] [--workspace-dir DIR]\n  ecg-waveparse digitize --input ECG.png --workspace-dir DIR [--runtime-dir DIR] [--device cpu|mps] [--timeout-ms MS]")
    if (command && command !== "--help") process.exitCode = 1
    return
  }
  const controller = new AbortController()
  const stop = () => controller.abort()
  process.once("SIGINT", stop); process.once("SIGTERM", stop)
  try {
    const digitizer = new Digitizer({ workspaceDir: option("--workspace-dir")!, runtimeDir: option("--runtime-dir") })
    const result = await digitizer.digitize(option("--input")!, { device: option("--device") as "cpu" | "mps" | undefined, timeoutMs: option("--timeout-ms") ? Number(option("--timeout-ms")) : undefined, signal: controller.signal })
    console.log(JSON.stringify(result, null, 2))
    if (result.status === "failed" || result.status === "timed_out") process.exitCode = 2
  } finally { process.removeListener("SIGINT", stop); process.removeListener("SIGTERM", stop) }
}
main().catch((error) => { console.error(JSON.stringify({ error: { code: error.code ?? "execution_error", message: error.message, runId: error.runId } })); process.exitCode = 1 })
