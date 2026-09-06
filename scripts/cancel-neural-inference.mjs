import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const [consumer,source,runtime,workspace,marker,output]=process.argv.slice(2);
const require=createRequire(path.resolve(consumer,'package.json'));
const {Digitizer}=require('ecg-waveparse');
const digitizer=new Digitizer({workspaceDir:workspace,runtimeDir:runtime});
const controller=new AbortController();
let observed,requestedAt,runId;
const sourceHash=createHash('sha256').update(await fs.readFile(source)).digest('hex');
const poll=setInterval(async()=>{
  if (observed) return;
  try { observed=JSON.parse(await fs.readFile(marker,'utf8')); }
  catch(error) { if(error.code==='ENOENT'||error instanceof SyntaxError) return; throw error; }
  requestedAt=performance.now();controller.abort();
},50);
let error;
try { await digitizer.digitize(source,{device:'cpu',timeoutMs:600000,signal:controller.signal,onProgress:event=>{runId=event.runId;}}); }
catch(caught) { error=caught; }
finally {clearInterval(poll);}
const latency=requestedAt===undefined?null:performance.now()-requestedAt;
assert(observed,'The experiment never observed actual convolution work.');
assert.equal(error?.code,'cancelled');assert(latency<30000);
const run=await digitizer.getRun(runId);
assert.equal(run.processing.state,'failed');assert.equal(run.status,'failed');
assert.equal(run.sourceIdentity.sha256,sourceHash);assert(!run.assets.canonicalCsv);
const evidence=await digitizer.getEvidence(runId);assert.equal(evidence.sourceInspectionAvailable,true);
let candidateProcessAlive=true;
try {process.kill(observed.pid,0);}catch(caught){if(caught.code==='ESRCH')candidateProcessAlive=false;else throw caught;}
assert.equal(candidateProcessAlive,false,'The cancelled neural process remains alive.');
await fs.writeFile(output,JSON.stringify({version:1,interface:'javascript',clinicalValidationUse:false,sourceSha256:sourceHash,
  boundary:observed.boundary,cancellationLatencyMs:latency,runId,status:run.status,processingState:run.processing.state,
  sourceInspectionAvailable:true,quantitativeOutputPresent:false,candidateProcessAlive,instrumentation:'One-shot profile hook; immutable package and model files unchanged.'},null,2)+'\n',{flag:'wx'});
