"""Run against an installed ecg-waveparse wheel; no source imports or remote APIs."""
import json
import sys
from ecg_waveparse import Digitizer

if len(sys.argv) not in (4,5):
    raise SystemExit("Usage: python waveparse.py INPUT WORKSPACE RUNTIME [EXISTING_RUN_ID]")
input_path,workspace,runtime=sys.argv[1:4]
digitizer=Digitizer(workspace_dir=workspace,runtime_dir=runtime)
run=digitizer.get_run(sys.argv[4]) if len(sys.argv)==5 else digitizer.digitize(input_path)
if run is None:raise SystemExit("Run not found.")
evidence=digitizer.get_evidence(run["id"])
print(json.dumps({"runId":run["id"],"status":run["status"],"sourceSha256":evidence["sourceSha256"],
                  "reconstruction":evidence["reconstruction"]["state"],"segmentCount":len(evidence["segments"]),
                  "originalAvailable":evidence["sourceInspectionAvailable"],"limitations":evidence["limitations"]},indent=2))
# Values are uV at a 500 Hz export grid, with explicit gaps. Source sampling and
# acquisition simultaneity are separate evidence. A paper render is a visualization.
# An abstention still exposes the original for local image/caliper inspection.
