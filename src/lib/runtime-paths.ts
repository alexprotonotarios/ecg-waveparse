import path from "node:path"

// Set only in isolated WaveParse runner processes. Existing app/CLI defaults
// remain the checkout root; installed resources are never writable run storage.
export const RESOURCE_ROOT = path.resolve(
  /* turbopackIgnore: true */
  process.env.WAVEPARSE_RESOURCE_ROOT ?? process.cwd()
)
export const WORKSPACE_ROOT = path.resolve(
  /* turbopackIgnore: true */
  process.env.WAVEPARSE_WORKSPACE_ROOT ?? process.cwd()
)
