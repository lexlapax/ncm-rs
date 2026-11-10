"""Run trusted local lifecycle commands in fresh, counterbalanced processes.

Parent wall includes process creation/import/linker/context startup and process exit.
Child load/execute/drop intervals are separate; this is NOT the steady-state matrix.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

PHASES = ["context", "load", "execute", "drop"]


def validate(rows):
    if [r.get("phase") for r in rows] != PHASES:
        raise ValueError("incomplete/reordered lifecycle")
    for row in rows:
        if row.get("schema_version") != 1 or not 0 <= row["internal_ms"] < float("inf"):
            raise ValueError("invalid lifecycle measurement")
    digest = rows[2]["output_ids_sha256"]
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("missing output identity")
    return digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands",type=Path); parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("output exists")
    jobs=json.loads(args.commands.read_bytes())
    if [j['name'] for j in jobs] != ['previous','candidate','python']: raise ValueError("expected three-way commands")
    result={"schema_version":1,"scope":__doc__,"started_utc":datetime.now(timezone.utc).isoformat(),
            "host":platform.platform(),"commands":jobs,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "binary_sha256":{j['name']:hashlib.sha256(Path(j['argv'][0]).read_bytes()).hexdigest() for j in jobs},"runs":[]}
    for repeat in range(3):
        for job in jobs[repeat:]+jobs[:repeat]:
            start=time.perf_counter()
            with tempfile.TemporaryFile(mode='w+') as errors:
                with subprocess.Popen(job['argv'],cwd=job['cwd'],env=job.get('env',os.environ),
                                      stdout=subprocess.PIPE,stderr=errors,text=True) as process:
                    rows=[]
                    for line in process.stdout:
                        row=json.loads(line);row['parent_elapsed_ms']=(time.perf_counter()-start)*1000;rows.append(row)
                    code=process.wait();elapsed=(time.perf_counter()-start)*1000
                errors.seek(0);stderr=errors.read()
            if code: raise RuntimeError(f"{job['name']} failed ({code}): {stderr}")
            digest=validate(rows)
            result['runs'].append({"runtime":job['name'],"repeat":repeat+1,"phases":rows,
                "startup_wall_ms":rows[0]['parent_elapsed_ms'],"post_drop_exit_ms":elapsed-rows[-1]['parent_elapsed_ms'],
                "total_process_ms":elapsed,"output_ids_sha256":digest,"stderr":stderr})
    if len({r['output_ids_sha256'] for r in result['runs']})!=1: raise ValueError("output mismatch")
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    with args.output.open('x') as file: json.dump(result,file,indent=2,allow_nan=False);file.write('\n')


if __name__=='__main__': main()
