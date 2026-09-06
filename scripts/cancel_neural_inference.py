"""Cancel the installed async Python client after observed neural computation."""
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from ecg_waveparse import Digitizer


async def main():
    source,runtime,workspace,marker,output=sys.argv[1:]
    digitizer=Digitizer(workspace_dir=workspace,runtime_dir=runtime)
    source_hash=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    task=asyncio.create_task(digitizer.digitize_async(source,device='cpu',timeout_ms=600000))
    observed=None
    while not task.done():
        try:observed=json.loads(Path(marker).read_text());break
        except (FileNotFoundError,json.JSONDecodeError):await asyncio.sleep(.05)
    assert observed,'The experiment never observed actual convolution work.'
    start=time.monotonic();task.cancel()
    try:await task;raise AssertionError('Cancellation was not propagated.')
    except asyncio.CancelledError:pass
    latency=(time.monotonic()-start)*1000
    assert latency<30000
    records=list((Path(workspace)/'storage/runs').glob('run_*/metadata.json'))
    assert len(records)==1
    run_id=json.loads(records[0].read_text())['id']
    run=await digitizer.get_run_async(run_id)
    assert run['processing']['state']==run['status']=='failed'
    assert run['sourceIdentity']['sha256']==source_hash and 'canonicalCsv' not in run['assets']
    evidence=await digitizer.get_evidence_async(run_id);assert evidence['sourceInspectionAvailable'] is True
    alive=True
    try:os.kill(observed['pid'],0)
    except ProcessLookupError:alive=False
    assert not alive,'The cancelled neural process remains alive.'
    result={'version':1,'interface':'python','clinicalValidationUse':False,'sourceSha256':source_hash,
            'boundary':observed['boundary'],'cancellationLatencyMs':latency,'runId':run_id,'status':run['status'],
            'processingState':run['processing']['state'],'sourceInspectionAvailable':True,'quantitativeOutputPresent':False,
            'candidateProcessAlive':alive,'instrumentation':'One-shot profile hook; immutable package and model files unchanged.'}
    with Path(output).open('x') as handle:json.dump(result,handle,indent=2)


if __name__=='__main__':asyncio.run(main())
