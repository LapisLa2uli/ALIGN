"""Run single-sample UI exports in isolation and record Windows process memory."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak_ws','ws','peak_pool_paged','pool_paged',
            'peak_pool_nonpaged','pool_nonpaged','pagefile','peak_pagefile','private')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sample=out/'sample';sample.mkdir()
    for name in ('performance_audio.wav','verified_score.musicxml','labels.json'):
        path=args.sample/name
        if path.exists():shutil.copy2(path,sample/name)
    protected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sample.iterdir()}
    env=os.environ.copy();env['NUMBA_CACHE_DIR']=str(ROOT/'align-model/runs/stack-v9/numba-cache')
    command=[sys.executable,str(ROOT/'align-model/scripts/publish_datacreate_v9.py'),
        '--sample',str(sample),'--output',str(out/'run'),'--device','cuda']
    kernel=ctypes.WinDLL('kernel32'); psapi=ctypes.WinDLL('psapi')
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    started=time.monotonic();samples=[]
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        handle=kernel.OpenProcess(0x410,False,proc.pid)
        try:
            while proc.poll() is None:
                mem=Counters();mem.cb=ctypes.sizeof(mem)
                if psapi.GetProcessMemoryInfo(handle,ctypes.byref(mem),mem.cb):
                    try:stage=json.loads((sample/'alignment_progress.json').read_text())['step']
                    except (OSError,ValueError):stage='loading'
                    samples.append({'seconds':round(time.monotonic()-started,2),'stage':stage,
                        'rss':mem.ws,'peak_rss':mem.peak_ws,'private':mem.private})
                if time.monotonic()-started>240:
                    proc.terminate();proc.wait();raise TimeoutError('Profiling exceeded 240 seconds')
                time.sleep(.25)
        finally:
            kernel.CloseHandle(handle)
            (out/'memory.json').write_text(json.dumps(samples,indent=2))
    if proc.returncode:raise RuntimeError((out/'stderr.log').read_text())
    assert all(hashlib.sha256((sample/n).read_bytes()).hexdigest()==digest for n,digest in protected.items())
    doc=json.loads((sample/'feedback_v9.json').read_text())
    summary={'source':str(args.sample),'seconds':round(time.monotonic()-started,2),
        'peak_rss_gib':max(s['peak_rss'] for s in samples)/2**30,
        'peak_private_gib':max(s['private'] for s in samples)/2**30,
        'status':doc['status'],'labels':len(doc['labels']),
        'location':doc['diagnostics']['passage_location'],'protected_unchanged':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
