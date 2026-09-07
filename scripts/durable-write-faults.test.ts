import assert from "node:assert/strict"
import test from "node:test"
import {promises as fs} from "node:fs"
import os from "node:os"
import path from "node:path"
import {runStorageTestUtils} from "../src/lib/runs"

test("durable writes remain old or complete under failures at every publication boundary",async()=>{
  const directory=await fs.mkdtemp(path.join(os.tmpdir(),"waveparse-write-fault-"))
  try {
    const source=path.join(directory,"untouched-source"),target=path.join(directory,"metadata.json")
    await fs.writeFile(source,"untouched")
    for (const stage of ["opened","written","synced","renamed","directory_synced"] as const) {
      await fs.writeFile(target,'{"state":"old"}')
      await assert.rejects(runStorageTestUtils.atomicWritePrivateFile(target,'{"state":"complete-new"}',false,point=>{
        if (point===stage) throw Object.assign(new Error("Injected disk exhaustion"),{code:"ENOSPC"})
      }),{code:"ENOSPC"})
      const value=JSON.parse(await fs.readFile(target,"utf8"))
      assert.equal(value.state,["renamed","directory_synced"].includes(stage) ? "complete-new" : "old")
      assert.equal(await fs.readFile(source,"utf8"),"untouched")
      assert.deepEqual((await fs.readdir(directory)).sort(),["metadata.json","untouched-source"])
    }
  } finally {await fs.rm(directory,{recursive:true,force:true})}
})
