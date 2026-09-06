"""Sample a local command's process-tree RSS and workspace size without logging inputs."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def process_tree_rss(pid: int) -> tuple[int, int]:
    output=subprocess.check_output(["ps","-axo","pid=,ppid=,rss="],text=True,timeout=5)
    rows=[tuple(map(int,line.split())) for line in output.splitlines() if len(line.split())==3]
    owned={pid}
    while True:
        children={process for process,parent,_ in rows if parent in owned}
        if children<=owned:break
        owned|=children
    return sum(rss*1024 for process,_,rss in rows if process in owned),sum(process in owned for process,_,_ in rows)


def directory_bytes(directory: Path) -> int:
    size=0
    for parent,dirs,files in os.walk(directory,followlinks=False):
        dirs[:]=[name for name in dirs if not (Path(parent)/name).is_symlink()]
        for name in files:
            path=Path(parent)/name
            try:
                if not path.is_symlink():size+=path.stat().st_size
            except FileNotFoundError:pass
    return size


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--interval',type=float,default=1);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or not .25<=args.interval<=10:parser.error('Command and a 0.25–10 s interval are required.')
    if args.output.exists():parser.error('Refusing to overwrite profiling evidence.')
    started=time.monotonic();process=subprocess.Popen(command)
    samples=[];errors=[]
    try:
        while process.poll() is None:
            try:
                rss,count=process_tree_rss(process.pid)
                samples.append({'elapsedSeconds':time.monotonic()-started,'processTreeRssBytes':rss,'processCount':count,'workspaceBytes':directory_bytes(args.workspace)})
            except (OSError,subprocess.SubprocessError,ValueError) as error:
                errors.append(type(error).__name__)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        process.terminate();process.wait(timeout=30)
        raise
    finally:
        result={'version':1,'commandName':Path(command[0]).name,'platform':platform.platform(),'architecture':platform.machine(),'exitCode':process.poll(),'elapsedSeconds':time.monotonic()-started,'samplingIntervalSeconds':args.interval,'sampleCount':len(samples),'peakSampledProcessTreeRssBytes':max((s['processTreeRssBytes'] for s in samples),default=None),'peakSampledWorkspaceBytes':max((s['workspaceBytes'] for s in samples),default=None),'measurementErrors':errors,'samples':samples,'limitations':['RSS includes shared pages in each process; values are sampled, not an operating-system hard maximum.','Workspace size excludes the separately installed runtime and package.'],'clinicalValidationUse':False}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
