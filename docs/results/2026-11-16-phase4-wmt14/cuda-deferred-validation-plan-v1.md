# WMT14 CUDA deferred selection-validation candidate

The [batch-32 comparison](cuda-performance-v1.md) found the experimental
reference backend slower than Python. The generator currently reads selected
codes to the host at every level to reject tied or nonfinite maxima. This
candidate uses the already supported sentinel-safe embedding on generated
codes, retains the original indices unchanged, then reads and rejects any
sentinel in the concatenated codes before decoding or returning. Forced-code
embedding is unchanged. Invalid rows are temporarily mapped to a valid
placeholder solely for safe intermediate device execution; the sentinel is
never accepted as output. Backends without the safe embedding fail explicitly.

A real-CUDA regression constructs unique, tied and NaN rows, checks the safe
embedding and confirms that the original invalid sentinels survive
concatenation. It passed with the experimental/cuDNN feature set. The full
CUDA check, full base and CFG exact-output replay, frozen quality scoring and
repeated performance/memory comparison remain pending. The first bounded
batch-32 timing/output smoke is diagnostic only. Prior failures and retest
limitations remain documented; no oracle or tolerance changes are authorized.

The batch-32 CFG=0 smoke reproduced the previous Rust codes and raw IDs exactly;
its single-process median was 1.276 s, which does not establish a speedup.
The local full CUDA check passed 18 device cases (log SHA-256
`483287251448c6e952ea510bb5393a86cad1159d1924ae6abafe913b6c69bdc5`).
The targeted sentinel test log SHA-256 is
`25cc6516ae17ce7f59c727cbabd25123fc5f15cb565388b847da5296ff55b1d4`.
The additional experimental-feature regression and full candidate replays
remain pending.
The release regression with `wmt14-experimental,cuda-cudnn` also passed; its
log SHA-256 is
`f48a337da159a50186f6727a5caa77a8dd6231970517be1fbd5e833a1b43d2a5`.
