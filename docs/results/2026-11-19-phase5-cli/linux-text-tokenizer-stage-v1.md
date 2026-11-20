# Host tokenizer-stage diagnostic for CUDA CLI text paths

2026-11-20. The [CUDA cold-request screen](cuda-text-cold-performance-v1.md) observed median paired text-minus-ID increases of 14.766 ms for XSum and 74.469 ms for WMT14 on one row. To narrow—not prove—the cause, an ignored release-mode CLI test exercised the actual artifact-bound `Decoder` on the same first reference text and retained decoded IDs. The artifact was fully loaded before timing. Each of three fresh processes per profile ran five warmups then 30 rounds; profile order alternated between processes. Every round checked encoded IDs against the pinned reference and decoded text against the pinned retained output. No CUDA inference was run in this diagnostic.

| Profile | Stage | Pooled median ms | Pooled p95 ms | Process medians ms |
| --- | --- | ---: | ---: | --- |
| XSum | Construct decoder/tokenizer | 15.3171 | 15.8450 | 15.3336 / 15.4276 / 15.2563 |
| XSum | Encode and pad first row | 0.4768 | 0.5148 | 0.4755 / 0.4792 / 0.4780 |
| XSum | Decode retained generated IDs | 0.0053 | 0.0061 | 0.0054 / 0.0054 / 0.0051 |
| WMT14 | Construct decoder/tokenizer | 61.8562 | 63.2698 | 62.0366 / 61.7288 / 61.7401 |
| WMT14 | Encode and pad first row | 0.0419 | 0.0478 | 0.0423 / 0.0427 / 0.0411 |
| WMT14 | Decode retained generated IDs | 0.0055 | 0.0067 | 0.0054 / 0.0056 / 0.0053 |

Construction includes authenticating the embedded manifest, payload and license, parsing the tokenizer, and profile metadata checks. It excludes artifact loading and tokenizer destruction; each timed round constructs a new decoder from already-owned artifact bytes. Encoding/decoding follow that construction within the same process; reference comparisons occur outside their timed spans. Median process peak RSS was 881.7 MiB XSum and 1096.6 MiB WMT14, including the preloaded full artifacts, not isolated tokenizer memory.

Construction costs are of the same order as the separate cold-request text-path differences, while encoding and decoding of these particular rows are much smaller. This supports tokenizer setup as an investigation target, but the runs have different timing boundaries and workloads; it does not attribute the full cold delta, establish a general bottleneck or justify bypassing integrity checks. CPU activity was not isolated or continuously monitored. XSum text was reconstructed from converted IDs. This diagnostic is not CUDA/Metal performance qualification, a full stage profile or an optimization result.

The test binary SHA-256 was `3abfad4b149c2a42de9a90e93ee1f1887131fc83021cce9cda72b681fcb245dd`, and `crates/ncm-cli/src/decode.rs` SHA-256 was `2ff018008e48bb499a1c53c2291e76e5677ddeb50d741d0256b47716c437c905`; both matched before/after collection. The [sanitized timing series](text-tokenizer-stage-summary-v1.json) SHA-256 is `c3cdc03400dda2a59c8ef9e22748b55f17351b05b58b6a3158d9c816a4383036`; its exporter replayed byte-for-byte and includes input-log hashes, 90 samples per stage and RSS, but no text, IDs or paths. Raw logs remain ignored under `reports/phase5-tokenizer-stage-cpu-v2/`. Parser negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. An initial v1 diagnostic completed, but the first full check then rejected an unnecessary clone in the test-only assertion; the assertion was corrected and all six v2 measurements were recollected before the passing check. No production inference code changed.

Reproduce after preparing the qualified artifacts and the corpus/decoded references identified in the test. Build with `setpriv --inh-caps=-all cargo test --release --locked -p ncm-cli --bin ncm --no-run`, then use the emitted test executable as `BIN`:

```sh
mkdir reports/NEW-tokenizer-stage
for rep in 0 1 2; do
  if [ "$rep" = 1 ]; then order='wmt14 xsum'; else order='xsum wmt14'; fi
  for name in $order; do
    if [ "$name" = xsum ]; then art=xsum-inference-rope-v2-final; else art=wmt14-inference-v1-final; fi
    setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      NCM_TEXT_TIMING_PROFILE="$name" NCM_TEXT_TIMING_ARTIFACT="$PWD/artifacts/canonical/$art" \
      NCM_TEXT_TIMING_REFERENCE="$PWD/reports/phase5-text-corpus-v1/$name.json" \
      NCM_TEXT_TIMING_OUTPUT="$PWD/reports/phase5-cli-equivalence/$name-decoded.json" \
      /usr/bin/time -v -o "reports/NEW-tokenizer-stage/$name-r$rep.time" "$BIN" \
      decode::tests::text_tokenizer_stage_diagnostic --ignored --exact --nocapture \
      > "reports/NEW-tokenizer-stage/$name-r$rep.log" 2>&1
  done
done
setpriv --inh-caps=-all python3 tools/bench/summarize_phase5_text_stages.py reports/NEW-tokenizer-stage
```
