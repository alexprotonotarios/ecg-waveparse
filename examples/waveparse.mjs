// Execute from a consumer with ecg-waveparse installed. Synthetic/licensed input only.
import { Digitizer } from 'ecg-waveparse';
const [input, workspaceDir, runtimeDir, existingRunId] = process.argv.slice(2);
if (!input || !workspaceDir || !runtimeDir) throw new Error('Usage: node waveparse.mjs INPUT WORKSPACE RUNTIME [EXISTING_RUN_ID]');
const digitizer = new Digitizer({ workspaceDir, runtimeDir });
const run = existingRunId ? await digitizer.getRun(existingRunId) : await digitizer.digitize(input);
if (!run) throw new Error('Run not found.');
const evidence = await digitizer.getEvidence(run.id);
console.log(JSON.stringify({runId:run.id,status:run.status,sourceSha256:evidence.sourceSha256,
  reconstruction:evidence.reconstruction.state,segmentCount:evidence.segments.length,
  originalAvailable:evidence.sourceInspectionAvailable,limitations:evidence.limitations},null,2));
// Use evidence.original locally to compare the source with overlays and CSVs.
// CSV units are uV, export rate is 500 Hz; missing intervals remain missing.
// Panel position does not establish acquisition time. Do not auto-accept a run.
