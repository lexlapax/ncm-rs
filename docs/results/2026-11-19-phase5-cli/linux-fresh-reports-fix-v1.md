# Fresh-clone check directory regression

2026-11-20. An attempted full Linux guide replay began in a fresh remote clone at
`740eb3b679b77251eddbc562003aa05732fe164d`. The extracted guide wrapper
`reports/phase5-linux-full-v2-wrapper.sh` (SHA-256
`727f4bfdfee94eb9efcf7c5b6938573f1cea54e33c3f0820c42d546c663a492f`) ran under
`setpriv --inh-caps=-all` and `PIP_NO_CACHE_DIR=1`. Its CUDA CLI build succeeded, but guide block 2
(`tools/check.sh --cuda`) failed in four report-tool tests because a fresh clone did not contain
`reports/`. The captured log `reports/phase5-linux-full-v2.log` SHA-256 is
`9164e6dde3ee3ea05ca56a5370417def3558887d2663f059f6d03394ac9a2c46`; later guide blocks were not run.

`tools/check.sh` now creates the required directory before running tests. The exact patch was applied
to the otherwise unmodified clone; the patched file SHA-256 was
`b89942e890ed57a3237109377dfd639b62720ff132ea55363b15b7fffc7d7bb0` in both checkouts. Retrying
`setpriv --inh-caps=-all` with the guide's CUDA environment in that clone passed the full check,
including 18 real CUDA operator cases. The ignored retry log is
`reports/phase5-linux-full-v2-check-retry.log`. The original checkout's default `tools/check.sh`
also passed. This tests the previously missing-directory scenario, but the fresh clone's build and
retry used a patched worktree and the remainder of the guide has not yet been replayed. This is not
a full fresh-clone guide, quality, performance or Metal gate.
