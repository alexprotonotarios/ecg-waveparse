import { readFileSync } from "node:fs"
import os from "node:os"
import path from "node:path"

export function runtimeDirectory(resourceRoot: string) {
  const manifest = JSON.parse(readFileSync(path.join(resourceRoot, "config/waveparse-runtime.json"), "utf8"))
  const base = process.platform === "darwin"
    ? path.join(os.homedir(), "Library/Caches")
    : (process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"))
  return path.join(base, "ecg-waveparse", manifest.runtimeId, `${process.platform}-${process.arch}`)
}
