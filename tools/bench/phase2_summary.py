"""Fail-closed codec/tokenizer benchmark qualification and descriptive statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("encode","decode","reconstruct","resident_decoder_block")
TEXT_SCOPES = ("encode","decode","decode_skip_special")


def require(condition, message):
    if not condition: raise ValueError(message)


def stats(values):
    require(bool(values) and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values),"invalid timing")
    ordered=sorted(values)
    return {"n":len(values),"median_ms":statistics.median(values),"p95_ms":ordered[math.ceil(.95*len(values))-1],
            "min_ms":min(values),"max_ms":max(values),"stddev_ms":statistics.pstdev(values),
            "iqr_ms":ordered[math.ceil(.75*len(values))-1]-ordered[math.ceil(.25*len(values))-1]}


def aggregate(rows, key):
    require(len(rows)==3 and all(len(row[key])==30 for row in rows),"expected three processes of thirty trials")
    result=stats([v for row in rows for v in row[key]])
    result["process_medians_ms"]=[statistics.median(row[key]) for row in rows]
    return result


def floats(reference, candidate):
    require(isinstance(reference,list) and isinstance(candidate,list) and len(reference)==len(candidate)>0,"floating shape/empty mismatch")
    maximum=0.; fraction=0.
    for x,y in zip(reference,candidate):
        require(math.isfinite(x) and math.isfinite(y),"nonfinite output")
        delta=abs(x-y); budget=5e-4+2e-4*abs(x)
        require(delta<=budget,"logit tolerance failure")
        maximum=max(maximum,delta);fraction=max(fraction,delta/budget)
    return {"elements":len(reference),"max_abs_error":maximum,"max_budget_fraction":fraction}


def reports(root,prefix):
    result={runtime:[json.loads((root/f"{prefix}-{runtime}-{i}.json").read_bytes()) for i in range(1,4)] for runtime in ("rust","python")}
    for runs in result.values():
        for run in runs:
            require(run["schema_version"]==1 and run["warmups"]==5 and run["trials"]==30,"unsupported protocol")
            require(run.get("diagnostic_profiling", False) is False,"profiled diagnostic is not an acceptance measurement")
            require(run["source_sha256"]==runs[0]["source_sha256"],"sources changed between processes")
    return result


def codec(root, device="cuda"):
    require(device in ("cuda","metal"),"unsupported measurement device")
    runs=reports(root,"codec")
    workload=hashlib.sha256((ROOT/"tests/fixtures/codec-bench-v1.json").read_bytes()).hexdigest()
    anchor=hashlib.sha256((ROOT/"docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes()).hexdigest()
    manifests={run["manifest_sha256"] for group in runs.values() for run in group}
    require(manifests=={anchor},"artifact mismatch")
    indexed={}
    for runtime,group in runs.items():
        for i,run in enumerate(group):
            require(run["workload_sha256"]==workload and run["precision"]=="fp32" and run["device"]==device,"workload/precision mismatch")
            rows={(r["batch"],r["scope"]):r for r in run["measurements"]}
            require(len(run["measurements"])==len(rows)==12 and set(rows)=={(b,s) for b in (1,2,4) for s in SCOPES},"incomplete measurements")
            indexed[runtime,i]=rows
    output=[]
    for batch in (1,2,4):
        for scope in SCOPES:
            groups={runtime:[indexed[runtime,i][batch,scope] for i in range(3)] for runtime in runs}
            baseline=groups["python"][0];errors=[]
            count=batch*({"encode":322,"decode":89,"reconstruct":411,"resident_decoder_block":0}[scope])
            retained=batch*89*(256 if scope=="resident_decoder_block" else 12*256 if scope=="encode" else 12*256+12 if scope=="decode" else 24*256+12)
            for rows in groups.values():
                for row in rows:
                    for key in ("input_ids_sha256","supplied_codes_sha256","output_ids_sha256","output_ids_count","output_float_elements"):
                        require(row[key]==baseline[key],f"identity/scope mismatch: {key}")
                    require(row["output_ids_count"]==count and row["output_float_elements"]==retained,"incorrect output scope")
                    if scope in ("decode","reconstruct"):
                        require(len(row["first_logits"])==batch*89*12,"logit shape mismatch")
                        errors.append(floats(baseline["first_logits"],row["first_logits"]))
                    else: require(row["first_logits"] is None,"unexpected logits")
            row={"batch":batch,"scope":scope,"quality":errors,"output_ids_sha256":baseline["output_ids_sha256"],"runtimes":{}}
            for runtime,rows in groups.items():
                wall=aggregate(rows,"trials_ms")
                span=aggregate(rows,"cuda_stream_spans_ms") if device=="cuda" else None
                wall["ms_per_sample"]=wall["median_ms"]/batch;wall["samples_per_second"]=batch*1000/wall["median_ms"]
                # Codec positions include Sudoku separators; these are not generated text tokens.
                wall["input_or_output_positions_per_second"]=batch*89*1000/wall["median_ms"]
                row["runtimes"][runtime]={"wall":wall,"stream_span":span,"first_ms":[r["first_ms"] for r in rows],
                    "host_hwm_bytes":[r["host"]["VmHWM"] for r in rows],"allocator_bytes":[r["pool_bytes" if runtime=="rust" and device=="cuda" else "allocator_bytes"] for r in rows]}
            row["python_over_rust_median_ratio"]=row["runtimes"]["python"]["wall"]["median_ms"]/row["runtimes"]["rust"]["wall"]["median_ms"]
            output.append(row)
    stages=[]
    expected={"encoder_pre_quant","decoder_logits","output_selection" if device=="cuda" else "output_selection_readback"}|{f"encode.level_{i:02}" for i in range(12)}
    stage_key="cuda_stream_spans_ms" if device=="cuda" else "synchronized_stage_ms"
    for runtime,group in runs.items():
        for run in group:
            require([r["batch"] for r in run["instrumented_stages"]]==[1,2,4],"incomplete stage batches")
            for row in run["instrumented_stages"]:require(set(row[stage_key])==expected,"incomplete stage names")
        for index,batch in enumerate((1,2,4)):
            for name in sorted(expected):
                samples=[{"values":r["instrumented_stages"][index][stage_key][name]} for r in group]
                stages.append({"runtime":runtime,"batch":batch,"stage":name,**aggregate(samples,"values")})
    loads={runtime:{"warm":aggregate(group,"load_trials_ms"),"first_ms":[r["first_load_ms"] for r in group],
        "first_execution_ms":[r["first_execution_ms"] for r in group],"load_memory":[r["load_memory"] for r in group],
        "host_final":[r["host_final"] for r in group]} for runtime,group in runs.items()}
    return {"measurements":output,"instrumented_stages":stages,"loads":loads}


def tokenizers(root):
    output=[];fixture_bytes=(ROOT/"tests/fixtures/tokenizers-v1.json").read_bytes()
    fixture=hashlib.sha256(fixture_bytes).hexdigest();oracle=json.loads(fixture_bytes)
    workload=json.loads((ROOT/"tests/fixtures/tokenizer-bench-v1.json").read_bytes())
    require(workload["schema_version"]==1 and workload["fixture_sha256"]==fixture,"invalid tokenizer workload")
    for profile in ("bert","gpt2","qwen3","tinystories"):
        runs=reports(root,f"tokenizer-{profile}");base=runs["python"][0]
        pinned=oracle["profiles"][profile]
        require(base["manifest_sha256"]==pinned["manifest_sha256"] and base["strings_per_trial"]==89,"unpinned tokenizer/count")
        positions=sum(len(c["ids"]) for c in pinned["cases"])
        non_special=sum(x not in workload["special_ids"][profile] for c in pinned["cases"] for x in c["ids"])
        require(base["token_positions"]==positions,"incorrect position count")
        for group in runs.values():
            for run in group:
                require(run["fixture_sha256"]==fixture and run["profile"]==profile,"tokenizer identity mismatch")
                for key in ("manifest_sha256","strings_per_trial","input_utf8_bytes","token_positions"):
                    require(run[key]==base[key],"tokenizer scope mismatch")
                require([r["scope"] for r in run["measurements"]]==list(TEXT_SCOPES),"incomplete tokenizer measurements")
        measurements=[]
        for index,scope in enumerate(TEXT_SCOPES):
            row={"scope":scope,"runtimes":{}}
            hashes=set()
            for runtime,group in runs.items():
                rows=[r["measurements"][index] for r in group];hashes.update(r["output_sha256"] for r in rows)
                timing=aggregate(rows,"trials_ms");timing["strings_per_second"]=base["strings_per_trial"]*1000/timing["median_ms"]
                timing["token_positions_per_second"]=base["token_positions"]*1000/timing["median_ms"]
                timing["non_special_token_positions_per_second"]=non_special*1000/timing["median_ms"]
                row["runtimes"][runtime]={"timing":timing,"first_ms":[r["first_ms"] for r in rows]}
            require(len(hashes)==1,"tokenizer output mismatch");row["output_sha256"]=hashes.pop();measurements.append(row)
        output.append({"profile":profile,"strings_per_trial":base["strings_per_trial"],"input_utf8_bytes":base["input_utf8_bytes"],"token_positions":base["token_positions"],"non_special_token_positions":non_special,"measurements":measurements,
            "loads":{runtime:{"warm":aggregate(group,"load_trials_ms"),"first_ms":[r["first_load_ms"] for r in group],"host_final":[r["host_final"] for r in group]} for runtime,group in runs.items()}})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    result={"schema_version":1,"codec":codec(a.directory),"tokenizers":tokenizers(a.directory),"all_benchmark_output_checks_passed":True,
        "summary_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_workload_sha256":hashlib.sha256((ROOT/"tests/fixtures/tokenizer-bench-v1.json").read_bytes()).hexdigest(),
        "limitations":["Benchmark logits/IDs supplement, not replace, full internal trace qualification.","Descriptive distributions over three processes; no statistical speedup guarantee.","Instrumented stage spans perturb dispatch and do not sum to headline API timings.","Tokenizer total and non-special position throughputs are separately labeled; codec is not generation."]}
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")


if __name__=="__main__":main()
