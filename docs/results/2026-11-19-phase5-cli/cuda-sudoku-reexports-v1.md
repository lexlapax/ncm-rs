# Sudoku provenance-only re-exports: CUDA screen

2026-11-20. The user approved proceeding with the provenance-only medium/hard re-exports; see
[export investigation](linux-export-repro-v1.md) for the original and alternate manifest hashes,
byte-level comparisons and their limitations. This screen does not itself qualify the re-exports.

On a shared desktop GPU, the `cuda-cudnn` feature debug CLI, SHA-256
`7509d9dd32f6e27b3a5f8231d500aef8e3da2691a1a0ce7c30c997739cfeca3b`, ran the `cuda` device
with each original and alternate artifact on the existing two-row fixture. Outputs were
byte-identical within each pair and matched the earlier [routing screen](cuda-routing-screen-v2.md):

| Difficulty | Input SHA-256 | Both outputs SHA-256 |
| --- | --- | --- |
| Medium | `8ce8ac13af0fddc753e55ee06ab2898647c5a652b931754e701a715ad60a57b7` | `b5a22993f2df60ece95b6b188e009a55cf10ad59a2591e69e478a025125d104d` |
| Hard | `355ed0c09c83ca5cb3c67fcb5bd77057e2ccdcc7a05f054322ebcba283925ece` | `655dbfdb50d1210923246a730a08332156df64418365328d55a2f20af8b7caf0` |

The original medium/hard manifest SHA-256 values were `facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219` and `9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b`; the alternate values were `9bc8e30673e426d6b10ef14204e2b0a0162fdc6a38c92e453fe5c156fab375c4` and `25ec837745601d69435a41f7ef5131b70d5fe17e1ca908e7b660d15531031158`.

Each invocation used `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1`, and the reference cuDNN
`LD_LIBRARY_PATH`:

```sh
# Use a fresh trusted output directory; binary and inputs must match the identities above.
mkdir reports/phase5-sudoku-reexport-cuda-v1
for difficulty in medium hard; do
  for kind in original alternate; do
    if [ "$kind" = original ]; then artifact="artifacts/canonical/sudoku-$difficulty-phase3"; else artifact="artifacts/canonical/phase5-repro-probe-sudoku-$difficulty-pinned"; fi
    setpriv --inh-caps=-all env LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" OMP_NUM_THREADS=1 \
      target/debug/ncm sample "$artifact" cuda "artifacts/canonical/sudoku-data-phase3/$difficulty-two-inputs.json" \
      > "reports/phase5-sudoku-reexport-cuda-v1/$difficulty-$kind.json" || exit 1
  done
  cmp "reports/phase5-sudoku-reexport-cuda-v1/$difficulty-original.json" "reports/phase5-sudoku-reexport-cuda-v1/$difficulty-alternate.json" || exit 1
done
```

The four outputs remain ignored locally. This is two fixed inputs per difficulty, not corpus quality,
performance, clean-clone reproducibility, comprehensive CLI equivalence or Metal validation.
