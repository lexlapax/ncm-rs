# Linux fresh-clone guide replay, resumed after check fix

2026-11-20. A fresh remote clone initially at `740eb3b679b77251eddbc562003aa05732fe164d`
ran the draft Linux guide. The build passed, but the check failed because `reports/` was absent;
[the failure, patch and successful CUDA check retry](linux-fresh-reports-fix-v1.md) are retained.
After inspecting the clone's sole tracked change, it was restored and the clone fast-forwarded to
`cc24da0`, which includes that same patch. The remaining guide blocks 4–16 were then extracted
without edits and run under `setpriv --inh-caps=-all` and `PIP_NO_CACHE_DIR=1`. Thus this is a
**resumed replay across two commits**, not a clean uninterrupted full-guide pass at one commit.

The resume wrapper `reports/phase5-linux-full-v2-resume.sh` SHA-256 is
`b3098443bca4658e1f341bc2a60b20941dbe7ba0fc1eb2293489d5dd221fbf32`; its captured log
`reports/phase5-linux-full-v2-resume.log` SHA-256 is
`e8843b7cd981ac28eec4b00f39bf63411ab3e2784d5f5a54d61f83666b76dde6`. The patched-clone check retry
log SHA-256 is `71d07166091055c71072095130e19e017979122ac044189812f137f4ddaab333`; its retained
CUDA operator report `reports/phase5-linux-full-v2-cuda.json` SHA-256 is
`4d1437acf092917cf4a3e259eafaa43da3b2b61a371e79bdb4d3bb97344d8900` (18 operator cases).
The release CLI SHA-256 was `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`.

Fresh downloads and exports for all nine profiles completed, including the approved medium/hard
candidate manifests (`9bc8e30673e426d6b10ef14204e2b0a0162fdc6a38c92e453fe5c156fab375c4`,
`25ec837745601d69435a41f7ef5131b70d5fe17e1ca908e7b660d15531031158`). All guide manifest and
rotary hash checks passed. Four tokenizer identities matched their lock. The synthetic Sudoku alias
and unified output matched; TinyStories, XSum, WMT14 and LM1B-small examples each yielded one
decoded row, matching the hashes from the earlier [seven-profile replay](linux-guide-integrated-replay-v1.md).
The Python 3.14.9 build again logged the documented `_ssl` `TLSv1_method` import failure; that
interpreter was used only for offline export.

The five nested worktrees were inspected: easy and TinyStories contained only the expected restored
untracked exporter; medium/hard and XSum contained only their expected tracked exporter change
(XSum also had ignored Python bytecode); LM1B-small contained only ignored bytecode. Exporter hashes
matched the guide. The worktrees and the inspected temporary clone were removed; logs and the CUDA
operator report remain ignored locally. The GPU was shared with a desktop. This replay is not a
quality or performance gate, uninterrupted single-commit guide verification, or any Metal evidence.
