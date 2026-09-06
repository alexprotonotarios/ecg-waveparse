"""One real extraction through an installed Python wheel, with durable reload."""
import hashlib
import json
import sys
import time
from pathlib import Path
from importlib.resources import files
from ecg_waveparse import Digitizer

source,runtime,workspace,output,device=sys.argv[1:]
payload=files('ecg_waveparse').joinpath('runtime/payload-manifest.json').read_bytes()
result={'interface':'installed_python','device':device,'payloadManifestSha256':hashlib.sha256(payload).hexdigest()}
digitizer=Digitizer(workspace_dir=workspace,runtime_dir=runtime)
start=time.monotonic()
try:
    result['run']=digitizer.digitize(source,device=device,timeout_ms=600000)
    result['evidence']=digitizer.get_evidence(result['run']['id'])
    result['reload']=digitizer.get_run(result['run']['id'])
except Exception as error:
    result['error']={'code':getattr(error,'code',type(error).__name__),'message':str(error),'runId':getattr(error,'run_id',None)}
result['elapsedMs']=(time.monotonic()-start)*1000
with Path(output).open('x') as handle:json.dump(result,handle,indent=2)
print(json.dumps({'interface':result['interface'],'device':device,'status':result.get('run',{}).get('status','failure'),'elapsedMs':result['elapsedMs']}))
