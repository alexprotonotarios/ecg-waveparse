// Exercise real preparation, OCR, shared policy, persistence and source reload
// through a separately installed package. Inputs are synthetic, never patients.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { promises as fs } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';

const [consumer,runtime,fixtures,output]=process.argv.slice(2);
assert(output,'Usage: CONSUMER RUNTIME FIXTURES NEW_OUTPUT');
await fs.mkdir(output); // Never overwrite a previous test workspace.
const require=createRequire(path.resolve(consumer,'package.json'));
const {Digitizer}=require('ecg-waveparse');
const digitizer=new Digitizer({workspaceDir:path.resolve(output,'workspace'),runtimeDir:path.resolve(runtime)});
const cases=[['speed_conflict','calibration_conflict'],['gain_conflict','calibration_conflict'],['unsupported_speed','unsupported_calibration']];
const report={version:1,clinicalValidationUse:false,attempted:cases.length,cases:[]};
for(const [id,expected] of cases){
  const input=path.resolve(fixtures,`${id}.png`),started=performance.now();
  const row={caseId:id,expectedReason:expected};
  try{
    const run=await digitizer.digitize(input,{device:'cpu',timeoutMs:120000});
    const evidence=await digitizer.getEvidence(run.id);
    const sourceHash=createHash('sha256').update(await fs.readFile(input)).digest('hex');
    row.status=run.status;row.reasonCode=run.publicationDecision?.reasonCode;
    row.hasCanonicalCsv=Boolean(run.assets.canonicalCsv);row.candidateCount=run.digitizer?.candidates?.length;
    row.runId=run.id;row.sourceSha256=sourceHash;
    assert.equal(run.status,'failed');assert.equal(row.reasonCode,expected);
    assert.equal(row.hasCanonicalCsv,false);assert.equal(row.candidateCount,0);
    assert.equal(evidence.sourceInspectionAvailable,true);assert.equal(evidence.sourceSha256,sourceHash);
    assert.equal((await digitizer.getRun(run.id)).publicationDecision.reasonCode,expected);
    row.passed=true;
  }catch(error){row.passed=false;row.errorCode=error.code??error.name;row.error=error.message;}
  row.runtimeMs=performance.now()-started;report.cases.push(row);
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(row));
}
process.exitCode=report.cases.every(c=>c.passed)?0:1;
