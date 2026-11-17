# WMT14 CUDA CFG development scoring

The separate CFG=1 diagnostic uses the upstream EOS/PAD null branch and the
same rows 0–127 designated in the [base development record](cuda-development-quality-v1.md).
It is not the released CFG=0 setting. The opt-in untraced Rust CFG API matches
its traced codes, logits and IDs exactly on the two-row fixture. The 35-row
smoke, including a three-row final batch, matches upstream exactly. The full
128-row development captures match all 40,064 codes, 8,192 raw IDs and decoded
hypotheses. The upstream capture SHA-256 is
`93095430f9870d7a326e6cca5bb6de4ca7a71b31b3da4a3686de4b0400bfd747`, Rust
`7eb4ec50025f4b4d96103aac17ff58c18386427dbc388549bf89278963f9111a`, and
the exact comparator
`d9ecd7b15e0f4332d6a4b02ed17a73d71e0a270a1edfd3905dcd85a2ddf315f6`.
The capture binary SHA-256 is
`0437378dcf1d5937ea55e2065fcf187f8f30856b0593b8fa31e4a38f5d126736` and
the upstream capture script SHA-256 is
`c0f4380e84e20b5732260eead9b035b1b38257bcc4419cab927bd6e167d5aae8`.

The same pinned tokenizer, sacreBLEU 2.5.1 environment and 1,000-resample
seed-42 bootstrap as the base development run produce identical Python/Rust
metrics and bootstrap intervals:

| Metric | Development value | Baseline 95% bootstrap interval |
| --- | ---: | ---: |
| BLEU | 18.1520 | 15.8795–20.4379 |
| BLEU lowercase | 19.1999 | 16.8054–21.5923 |
| chrF | 47.4474 | 45.0556–49.3364 |
| Length ratio | 0.9346 | 0.9083–0.9632 |

The Python score and independent offline replay both have SHA-256
`66b9396c31cc0fa95f7c2f22368beceef565a5044004ba81050b631c8e1d4129`;
the Rust score report, which has different input provenance, is
`2e4ed760d860f7838da4594fe388c1feca679a077f5bcbf459c70f272001af31`.
The 128-row upstream and Rust captures used `tools/reference/wmt14_cfg_capture.py`
and `target/release/examples/wmt14_cfg_capture` with start 0 and count 128;
the checked comparator is `tools/reference/compare_wmt14_cfg_capture.py`.
The Rust capture checks artifact and prefix identities but does not perform
source postflight, and neither capture is resumable. These are development
results, not held-out quality, performance or Metal evidence. Desktop GPU
processes were present. Focused model and comparator tests and feature Clippy
pass. Full `tools/check.sh --cuda` passed 18 real-device cases (log SHA-256
`e79c968d6840b6cdc4e3193d1bc477007a12ef33b32ed871fc2d641008d5f33f`).
