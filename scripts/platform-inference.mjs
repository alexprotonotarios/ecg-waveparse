import { createRequire } from 'node:module';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
const [consumer,input,runtime,workspace,output,device]=process.argv.slice(2);
const require=createRequire(path.resolve(consumer,'package.json'));
const {Digitizer}=require('ecg-waveparse');
const payload=path.resolve(path.dirname(require.resolve('ecg-waveparse')),'../runtime/payload-manifest.json');
const result={interface:'installed_javascript',device,payloadManifestSha256:createHash('sha256').update(await fs.readFile(payload)).digest('hex')};
const d=new Digitizer({workspaceDir:workspace,runtimeDir:runtime});
const start=performance.now();
try {
  result.run=await d.digitize(input,{device,timeoutMs:600000});
  result.evidence=await d.getEvidence(result.run.id);
  result.reload=await d.getRun(result.run.id);
} catch(error) { result.error={code:error.code,message:error.message,runId:error.runId}; }
result.elapsedMs=performance.now()-start;
await fs.writeFile(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({interface:result.interface,device,status:result.run?.status??'failure',elapsedMs:result.elapsedMs}));
