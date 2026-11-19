# Phase 5 Linux export reproducibility investigation

2026-11-19. This is preparation for `docs/USAGE.md`, **not** the required fresh-clone setup verification. Exports used the existing pinned checkpoints and tokenizer assets under `artifacts/`, and the existing exporters; no established artifact or manifest was changed. Scratch outputs are under ignored `artifacts/canonical/phase5-repro-probe-*` and `reports/phase5-*`. Production inference was not run against the new outputs.

## Results

| Profile | Qualified manifest SHA-256 | Re-export result |
| --- | --- | --- |
| Sudoku easy | `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832` | Exact match using the historical exporter, base commit and Python 3.14.9. |
| Sudoku medium | `facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219` | No exact match yet; see below. |
| Sudoku hard | `9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b` | No exact match yet; see below. |
| TinyStories | `0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1` | Exact match using the historical exporter, base commit and Python 3.14.9. |
| XSum v2 | `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4` | Exact match using the historical exporter, base commit and Python 3.14.9. |
| WMT14 | `01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81` | Exact match with the current exporter and Python 3.14.10. |
| LM1B small | `58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76` | Exact match using the exporter at `9d791f88c75614bf531d897603d1896394145a79` and Python 3.14.10. |
| LM1B medium | `2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c` | Exact match with the current exporter and Python 3.14.10. |
| LM1B large | `86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a` | Exact match with the current exporter and Python 3.14.10. |

The XSum reference rotary preparation regenerated the pinned payload SHA-256 `5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a`. All export reports stated bit-exact tensor roundtrips where reported. Current-HEAD exports of Sudoku easy, TinyStories, XSum and LM1B small did **not** match their qualified manifest hashes. Comparing parsed manifests against the retained qualified artifacts showed that all payload records except `provenance.json`, all tensor inventories, and all other manifest fields match. The provenance differences are respectively converter commit/hash/Python version, converter commit/Python version, converter commit/Python version, and the LM1B-small watched exporter hash. These are not newly approved artifact identities.

## Historical provenance and unresolved medium/hard

For easy, TinyStories and XSum, provenance records the base Git commit at export time and the hash of an exporter that was then uncommitted. Exact reconstruction used separate worktrees at the recorded base, with the exporter restored from the subsequent commit at the exact recorded hash:

| Profile | Base commit | Exporter source commit | Exporter SHA-256 |
| --- | --- | --- | --- |
| Sudoku easy | `f7aad49c3b904f76c86a3d5e36c3b552d2995000` | `bbdf4222b734c4ebcb6cfa3ed1f23dbdf0c5f58c` | `cfcae7d716fa7a9692c57019454bd28da5f8098c6d3bec32698795076dc0034e` |
| TinyStories | `fdc7db4b6d0316bfef74085994b26a6619cd6847` | `93ae5eb6d51d43a02ae764b2dc928c57d9bd7335` | `f757c3cb694922c2f8cf96a3f6cb40c47ea6bfa8dae262de3b957ee5e204754d` |
| XSum | `3bdb96b2981145ea71a387b10e662318af85cdc3` | `a92a27516957ae9f3e2e8e52e8d6bf189a600dfc` | `eb90493398ca55087b775a9207200e11d6f312a8e0cd154284569f64cc244319` |

The worktrees were removed after the test; the explicitly substituted exporters were known copies, not user work. A project-local CPython 3.14.9 was built from the official source tarball SHA-256 `e02d9ac21cebda6e46435369744a92c65cc7cd8669697694122cf97ffacd1e53`, using the existing reference venv's Python 3.14 packages (`torch 2.15.0+cu132`) through `PYTHONPATH`. The build-tree interpreter performed the exports. `make install` failed because this host's `_ssl` module could not import (`undefined symbol: TLSv1_method`); therefore the incomplete install is **not** a qualified general-purpose Python environment. These were offline exports and did not use SSL. No global tooling was changed.

Medium and hard provenance names base `e1ffb121a8094505173c03d27669b6486f9b9abd`, Python 3.14.9 and exporter SHA-256 `810a02e856b13dc6fcca30771512fe0a82f0a717fc7b9db9ad72700988556b67`. The recorded exporter is not either version in the path's Git history (`cfcae7d…` or `e639675…`). A search of 1,623 Git blobs sized 5,000–20,000 bytes and all 214 unreachable blobs did not find the recorded hash; this is not proof it exists nowhere else. Using the current exporter at the recorded base and Python version produced medium manifest `9bc8e30673e426d6b10ef14204e2b0a0162fdc6a38c92e453fe5c156fab375c4` and hard `25ec837745601d69435a41f7ef5131b70d5fe17e1ca908e7b660d15531031158`. In each comparison against the retained qualified artifact, `provenance.json` was the only different file record, `converter_sha256` its only different provenance field, and tensor inventories and other manifest fields matched. These alternate hashes are **not** accepted replacements for the qualified hashes.

A subsequent read-only search of the Mac checkout (`702168c`, clean) found only the tracked current Sudoku exporter (`e639675…`). A complete hash scan found no matching blob among all 4,919 Linux Git blobs (331,616,501 bytes) or all 4,694 Mac Git blobs (319,427,740 bytes). A separate Linux scan of 2,171 regular files (100–200,000 bytes) in `reports/`, `.reference/`, `docs/`, `tools/` and `tests/`, excluding environments, build trees and symlinks, also found no match. No Mac files were changed and no MLX test was run. These searches do not prove the original exporter is unavailable elsewhere.

## TinyStories guide command replay

The [draft guide](../../USAGE.md#linux-tinystories-development-replay) now gives the historical
TinyStories export and qualified manifest check. On Linux, the downloader's `--verify` checked
all five retained files in `artifacts/upstream/tinystories-phase4` (578,228,368 bytes). The guide's
worktree/export sequence was run with that retained path and fresh replay-specific output/report
paths, rather than its intended fresh-download paths. It verified the historical exporter hash,
reported bit-exact roundtrips, and produced manifest SHA-256
`0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1`. The ignored report
`reports/phase5-guide-tiny-replay.json` has SHA-256
`493c6f6f71cf2e481a76d6d0b178d80ca320ea0309b2c4ecbcf2056ec5218490`. The temporary worktree
contained only the expected restored exporter; after inspecting its status and hash, it was
removed. The output remains in ignored `artifacts/canonical/phase5-guide-tiny-replay/`.
This did not exercise a fresh clone, fresh download, Mac, inference or model quality.

## XSum guide command replay

The [draft guide](../../USAGE.md#linux-xsum-development-replay) now gives the rotary reference,
historical XSum exporter and qualified manifest checks. On Linux, the downloader's `--verify`
checked all three retained files in `artifacts/upstream/xsum-phase4` (653,983,635 bytes). The
guide's generation and export sequence was run using that retained source and fresh
replay-specific output/report paths. The generated rotary payload matched SHA-256
`5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a`; its manifest hash was
`2fcdd99756cdf5a77e7b93ef48e03097366b87603d3297b2b6a1a70d03f90c5e`. The worktree initially
contained the base commit's older tracked exporter; the first attempt's absent-file guard
correctly stopped before replacing it. After checking its pristine hash and status, the replay
substituted the recorded historical exporter. It reported bit-exact roundtrips and produced
qualified manifest SHA-256 `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4`.
The ignored report `reports/phase5-guide-xsum-replay.json` hashes to
`e9544b614c9a7b4a98f6b96d56cfce467c25182fab3285fe011043dc6d7df9fe`. The worktree contained
only that expected tracked-file change and was removed after inspection. The output and rotary
reference remain under ignored `artifacts/canonical/phase5-guide-xsum-replay/` and
`reports/phase5-guide-xsum-rope/`. No fresh clone, fresh download, Mac, inference or quality
validation is implied.

## WMT14 and LM1B guide command replays

The [draft guide](../../USAGE.md#linux-wmt14-and-lm1b-development-replays) documents retained-input
Linux export replays for WMT14 and all three LM1B sizes. The downloader verified each profile's
three retained files. Commands used `artifacts/upstream/wmt14-phase4` or the shared retained
`artifacts/upstream/lm1b-phase4`, and replay-specific output paths, rather than the guide's separate
fresh-download paths. Python was 3.14.10 and Torch 2.15.0+cu132. WMT14 and medium/large used the
current exporter; small used the tracked exporter at `9d791f88c75614bf531d897603d1896394145a79`
in a new worktree. All four exports produced their qualified manifest hashes in the results table
above. WMT14 reported bit-exact roundtrip; LM1B uses the same checked tensor-saving helper.
Outputs remain under ignored `artifacts/canonical/phase5-guide-{wmt14,lm1b-small,lm1b-medium,lm1b-large}-replay/`.
The small output was moved out of its worktree after checking its hash and the destination; the
worktree's only remaining ignored files were generated Python bytecode and it was then removed.

The pinned upstream LM1B rotary generator was also run to the new ignored path
`reports/phase5-guide-lm1b-rope.safetensors`, producing SHA-256
`9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc` as required. These replays
are not fresh-clone or download verification and do not run inference, qualify a Mac build, or
establish model quality or performance.

## Byte-level comparison of alternate exports

`tools/reference/compare_artifact_provenance.py` independently verifies both requested manifest hashes and each declared file's size and SHA-256, compares every non-provenance file byte-for-byte, compares all other manifest fields including tensor inventories, and allows exactly the explicitly listed provenance keys to differ. It revalidates inputs after comparison. It assumes stable trusted directories and is not a substitute for `ncm-io` validation, GPU qualification or user approval. The tool SHA-256 is `4856913542f7c652974c28c11325cf5c5ea4fd4ce34e97d36d41a1652a11c729`; six tests including tampered and rehashed payload, tensor, JSON and symlink negative controls pass through `tools/check.sh`.

For medium and hard, the tool passed using `--allow-provenance-key converter_sha256` and the qualified/alternate manifest hashes above. Raw result SHA-256s: medium `f42e02081005f5e0e072187009fd6e43736a494b2c271f461cafc7e4985a6e18`, hard `08ed1797ddcb6c6eb655b387feafb03f8174595f34a9efd78abaefd2c3546e5b`. The results establish those local files' equality outside provenance; they do not approve either alternate manifest. To reproduce, pass the two paths, their manifest hashes and the allowed key to the tool with a fresh `--output` path.

Next: find the original medium/hard exporter bytes or obtain an explicit decision on whether to accept provenance-only re-exports after appropriate qualification. Then write and actually follow complete fresh-clone setup instructions on Linux before the Metal setup run. Exact matching here used retained local inputs and is not evidence that downloads, fresh clones, or Mac exports work.
