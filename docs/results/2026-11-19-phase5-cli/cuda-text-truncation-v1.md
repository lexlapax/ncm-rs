# CUDA overlength text-input screen

2026-11-19. For XSum and WMT14, a synthetic 1,101-token `hello ` repeated input was passed through
`--text`; the equivalent 512-token and 96-token ID rows came from the independently pinned Python
reference described in the [boundary screen](cuda-text-input-v1.md#synthetic-truncation-and-padding-boundaries).
For each profile, six fresh CUDA processes ran the pre-change alias with IDs, candidate alias with
IDs, unified command with IDs, unified command with text, and the latter two with `--decode`.
The first four stdout byte streams matched; both decoded streams matched and retained the same
codes and IDs as the undecoded baseline. A one-row output shape check passed. This is only two
synthetic requests, not a corpus generation, quality, performance, or Metal gate.

| Profile | Device | Text input SHA-256 | ID input SHA-256 | Common undecoded SHA-256 | Common decoded SHA-256 |
| --- | --- | --- | --- | --- | --- |
| XSum | cuda-cudnn | `6e71a3cba06649311640000a2445c54b51b2a5966bfcf9068aeedd487597830e` | `095c5743c45d9b12a272b77f08b173aa3fbb1c072efb3a8d8111aa57f8b6080e` | `110300dee351509a6053191ce1f454304a5e213985e95300d13ad80dbc3ff209` | `a86041d600b370e4b9532cec6d3f7686a3df32ccda65061c7d695f1919c5dbbe` |
| WMT14 | cuda | `6e71a3cba06649311640000a2445c54b51b2a5966bfcf9068aeedd487597830e` | `9e6195f06bd191620d2d7a81139100c9c7221299e8248cf34a1d60fcb415c38f` | `3c104474d2a2908d7c77222703e3e677f3c30cfbbe66111382c935edf361ae15` | `9ba41ffb3e526e71520091c6bf1e5465f6891aed22055da68c8290b92216b6f9` |

Baseline binary SHA-256 was `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`,
candidate `cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`. Qualified XSum
and WMT14 manifest SHA-256s were `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4`
and `01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81`. Raw inputs/outputs
remain ignored under `reports/phase5-text-truncation-cuda-v1/`. An active desktop shared the GPU;
no reservation or timing conclusion is implied. The reproduction block below was separately run
with fresh reference files and output directory; its four output hashes matched the table. The
ignored extracted shell script SHA-256 is `6dd59fe07fdea572086dd96ef135787db9a5d162b4620769265864c52666eae2`.

## Reproduce

Generate fresh boundary references as in the linked screen, build or retain the identified binaries
and artifacts, then use a new output directory and the exact reference hashes. The script below
checks fixed-request bytes and decoded output shapes, not evaluator decoding parity.

```sh
setpriv --inh-caps=-all env LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" OMP_NUM_THREADS=1 bash -c '
set -eC
out=reports/NEW-text-truncation
mkdir "$out"
python3 - "$out" <<'"'"'PY'"'"'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1])
for p,sha in [("xsum","2f7c8ed36cfd669a59ecf2fc5e13fe2fb584941c61c93e069a0825a7c5ff46f6"),("wmt14","03886e0864b60f9fc7992311f83b3e20511b9ce73274fef5f51624d41d960fb6")]:
    raw=Path(f"reports/NEW-{p}-boundaries.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError(p)
    ref=json.loads(raw)
    if ref["profile"]!=p or ref["original_lengths"][2]<=len(ref["ids"][2]): raise ValueError(p)
    for suffix,value in [("text",[ref["texts"][2]]),("ids",[ref["ids"][2]])]:
        with (out/f"{p}-{suffix}.json").open("x") as f:
            json.dump(value,f,ensure_ascii=True,separators=(",",":")); f.write("\n")
PY
old=reports/phase5-cli-equivalence/baseline-ncm
new=target/phase5-candidate/release/ncm
printf "%s  %s\n" f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 "$old" cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 "$new" | sha256sum -c -
for profile in xsum wmt14; do
  if [ "$profile" = xsum ]; then
    artifact=artifacts/canonical/xsum-inference-rope-v2-final; device=cuda-cudnn; sha=8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4
  else
    artifact=artifacts/canonical/wmt14-inference-v1-final; device=cuda; sha=01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81
  fi
  printf "%s  %s\n" "$sha" "$artifact/manifest.json" | sha256sum -c -
  "$old" "sample-$profile" "$artifact" "$device" "$out/$profile-ids.json" > "$out/$profile-baseline.json"
  "$new" "sample-$profile" "$artifact" "$device" "$out/$profile-ids.json" > "$out/$profile-alias.json"
  "$new" sample "$artifact" "$device" "$out/$profile-ids.json" > "$out/$profile-unified.json"
  "$new" sample "$artifact" "$device" "$out/$profile-text.json" --text > "$out/$profile-text-output.json"
  "$new" sample "$artifact" "$device" "$out/$profile-ids.json" --decode > "$out/$profile-id-decoded.json"
  "$new" sample "$artifact" "$device" "$out/$profile-text.json" --text --decode > "$out/$profile-text-decoded.json"
  cmp "$out/$profile-baseline.json" "$out/$profile-alias.json"
  cmp "$out/$profile-baseline.json" "$out/$profile-unified.json"
  cmp "$out/$profile-baseline.json" "$out/$profile-text-output.json"
  cmp "$out/$profile-id-decoded.json" "$out/$profile-text-decoded.json"
done
python3 - "$out" <<'"'"'PY'"'"'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name,codes,ids in [("xsum",200,48),("wmt14",313,64)]:
    a=json.loads((root/f"{name}-baseline.json").read_bytes())
    b=json.loads((root/f"{name}-id-decoded.json").read_bytes())
    if set(a)!={"codes","ids"} or len(a["codes"])!=1 or len(a["ids"])!=1 or len(a["codes"][0])!=codes or len(a["ids"][0])!=ids: raise ValueError(name)
    if set(b)!={"codes","ids","texts"} or b["codes"]!=a["codes"] or b["ids"]!=a["ids"] or len(b["texts"])!=1 or not isinstance(b["texts"][0],str): raise ValueError(name)
PY
'
```
