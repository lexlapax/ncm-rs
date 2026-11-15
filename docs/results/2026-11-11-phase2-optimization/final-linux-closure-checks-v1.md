# Final Linux tokenizer and model-lifetime checks

On the current `f155794` runtime, a freshly built `asset_probe` ran with `env -i PATH=/usr/bin:/bin` and passed all four pinned tokenizer profiles, including their goldens and rejection cases. Its report SHA-256 is `6124d2dbd456dce9375e5d6bb17472875ae75674114eb51cd4f279046b08bee9`, identical to the prior qualified probe. No Python was on the probe's PATH.

The freshly built `codec_memory` diagnostic ran 144 solved-board reconstruction requests over ten model lifetimes, rotating batches 1, 2 and 4. The default CUDA pool's live used count returned to its zero-byte initial baseline after each model drop; warmed live usage was stable within each cycle. Reserved pool memory remained 100,663,296 bytes after each drop, and process RSS/context/library allocations have different coverage. This is a finite lifetime observation, not proof of an infinite bound or a timing/physical-memory claim.

For both commands, executable hashes matched immediately before and after. [Raw reports, command metadata and hashes](raw/final-linux-closure-checks-v1.tar.xz) have SHA-256 `0775e2ea9bffcd262dd74dbfc74c26c5d2860b9689327d35371f629275576c93`. The capture did not modify GPU clocks, power policy, pins, tolerances or input fixtures.
