# WMT14 CFG row 2181 investigation

The [frozen full comparison](cuda-heldout-quality-v1.md) fails for row 2181:
seven codes and seven raw IDs differ, all differing raw IDs after the common
first EOS at position 23. The exact raw-ID gate remains failed.

The original four-row batch starts at 2180. A pinned upstream CUDA trace
replayed all four rows' full-capture codes and IDs exactly; the native trace did
the same for its retained capture. In row 2181, the first differing code is
global position 53: level 5, position 6. All four rows' code histories through
the first 47 positions are identical. The checked comparison's SHA-256 is
`814e806ff4eeb4901790524b1bc4578f5a0061c5f0909befb17b788545357b9b`.

Python selects code 32 with logit 6.352477550506592 over code 1071 at
6.352476119995117, a gap of 0.000001430511474609375. Rust selects 1071 at
6.352487564086914 over code 32 at 6.352478504180908. For all four rows across
levels 0–5, every head logit lies inside the unchanged accumulated band. The
first five levels have no argmax differences; level 5 has one. This is an
in-band near-tie crossing, not a waiver of the exact-output gate.

Two diagnostic candidates did not fix it. The existing strict CUDA/cuDNN
arithmetic path still has seven code and seven raw-ID differences in the batch,
with level-5 logits for codes 32/1071 of 6.35247802734375/6.352482318878174.
Adding independently verified upstream rotary tables for positions 0–95 to
that strict path also leaves seven code and seven ID differences, with logits
6.352475643157959/6.352485179901123. These are bounded exploratory tests, not
replacement quality runs. The upstream table exactly matches the first 96
positions of the previously pinned XSum rotary reference. An isolated Rust
`powf`/trigonometric probe found 362 cosine and 635 sine values different from
that reference, maximum absolute errors 3.769993782043457e-6 and
7.241964340209961e-6 respectively; the table difference alone did not resolve
this choice.

A further same-batch trace retained the conditional and null head logits before
guidance. At level 5 position 6, Python conditional scores for codes 32/1071
are 5.684321880340576/3.2823526859283447; Rust's are
5.684320449829102/3.2823596000671387. Python null scores are
4.348010063171387/-2.857893705368042; Rust's are
4.348004341125488/-2.857896566390991. Both branch vectors are within the
unchanged band, with maximum absolute errors 5.91278076171875e-5 and
2.86102294921875e-5 respectively. Recomputing `1.5 * conditional - 0.5 * null`
in each runtime's retained logits exactly reproduces its guided scores: the
observed difference is already present before the guidance operation. The
Python and Rust branch trace payload hashes are
`b582e4ae075452e4ff92b8e1242894ceeeb0248509cc2e9a34a9eea9fccdb1b6` and
`82824b3e17f49bb39788d88eded377af63187410f890327822bdad87f4fcb2a4`.
The first Python branch attempt mistakenly shadowed its revision variable and
failed source postflight; the corrected second capture passed, and the failed
log is retained. The temporary Rust instrumentation was reverted; its example
and model patch remain in ignored `reports/`, SHA-256
`969049fd60b03f1e1892bfa1e4f95a3cb7abb95be7c86563b0d024f60d1e74e6` and
`163c72bc347fb1712f08bf77c8b9ece810f07e414ce68824408468a01c40f6d9`.

The next trace retained inputs to the untied level-5 head in both branches.
Across the `[8, 26, 768]` head-input tensor, the largest Python/Rust absolute
difference is 0.0000247955322265625 and no values exceed the existing component
band. Feeding the exact Python head inputs and checked head weights to the Rust
CUDA projection produces all 851,968 branch logits bit-for-bit equal to Python;
the recomposed guided logits select Python's code 32. This localizes the
crossing to accumulated differences before that projection, rather than the
head matrix multiplication or guidance arithmetic. It does not identify a
single earlier operator or resolve the gate. The Python and Rust head-trace
payload hashes are `31d8695fe164d86fc4e2d9340ffeb4ef138b078cb129a55b07161716d7b4624f`
and `ed4a7a8ba3dcec3a9608828728d4bc8fc41f5e8b6bee75a4022555f925ea5beb`.
The temporary Rust example and model instrumentation are retained in ignored
`reports/` with SHA-256 `881e3e66159eadc818cc53cb52b8a2e4190209a48603802944b390b8d8e2f88c`
and `677eb24c7403de4408c58abd706df325e53e307bb5f29d813ae8848ac5601f38`.
An initial diagnostic build missed a trait import and failed; its log is retained.

Tracing the same four-row batch at level 5 through all twelve transformer
blocks shows that the stack input already differs: the prefix portion matches
bit-for-bit, while the target portion has a maximum absolute error of
0.000003814697265625, with no values outside the component band. At the
divergent target position its maximum input error is 0.0000019073486328125.
The target-side maximum error grows to 0.00012969970703125 after block 11;
all traced block values remain inside the component band. Codec reconstruction
level 0 already differs by at most 0.0000035762786865234375 in the normal
path, or 0.000001430511474609375 in the strict diagnostic. These observations
locate drift before the stack and its subsequent growth, but do not prove a
single causal operator or resolve the exact gate. Both Python and Rust stack
traces reproduce their respective full-batch codes and IDs. Their payload hashes
are `5d2feb3a3a0cb48b20be0c9aae3ef7a3b28a95c92c3e9915d0a477bf362bffea` and
`910f5dce9cd222b4ca0b6db1127819ed7d164c51bc3355ea437f66a2fad9ae71`.
The temporary Rust example and instrumentation patch remain in ignored
`reports/` with hashes `ab24f9b9ad6aec8754e23f1ea800fd4291500b8629eafa5c3713086e95891a56`
and `e01762d4e310c0f9f6fa2f5d21587d26f4651a89541ab1fcbae803e4fca95cf2`.

Stage traces of the initial learned upsampling cascade show that its input
embedding is bit-identical. In the normal CUDA path the first transposed
convolution is the first difference, maximum absolute error 2.86102294921875e-6.
The strict cuDNN path matches that first convolution bit-for-bit, then differs
at the following group normalization by at most 7.152557373046875e-7. All five
traced cascade stages stay within the component band in both paths. This
explains where discrepancies first appear but does not establish that correcting
one stage alone will pass the frozen full-set gate. The normal and strict
cascade trace hashes are
`4df1af9e7db6039848b90b517b46a1bfbfe36ae6fd5c0d7311ef18edbe800097` and
`84c0f3cd2426c03ed057fd9cc3d8d2c0d6300355ec31e2adae683cfa23e6fae4`;
temporary examples and instrumentation patch remain in ignored `reports/`.

A separate diagnostic generalized the existing XSum reference-order CUDA group
normalization kernel to the WMT14 cascade's lengths 4 and 16 while using the
strict cuDNN convolution path. It did not reproduce Python exactly: the first
normalization still differs by up to 7.152557373046875e-7, and the batch still
has seven code and seven ID differences. At the original crossing the candidate
scores for codes 32/1071 are 6.352480888366699/6.352488040924072, retaining
the Rust choice. This candidate is reverted; no tolerance or supported runtime
was changed. Its retained trace SHA-256 is
`f06706ecdc8e9c1ed39b84a25fe603e2b0869d185dbb6259fdbf9f13141334a9`;
the temporary source patch SHA-256 is
`c1cb0992f30ccaa20e01ffb9aec60ed5b594c5ec148ef2028532e1cb5cdfc4d6`.

Inspection of the [group-normalization source](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cuda/group_norm_kernel.cu)
at the exact installed PyTorch revision `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e` showed that group normalization
uses 32 threads when the reduction width is below 512, and 512 otherwise. The
earlier diagnostic incorrectly used 512 for both WMT14 widths. Correcting that
dispatch makes all five initial cascade stages and the first cumulative
reconstruction bit-identical in the strict cuDNN path, but that path still
produces the seven differences. Restricting cuDNN convolution and reference-
order normalization to the first reconstruction while preserving the ordinary
subsequent generator arithmetic makes all codes and IDs match on this four-row
batch, without injecting upstream activations. This is a diagnostic candidate,
not yet a qualified implementation or a full-set gate pass. Its trace SHA-256
is `1ccbe1fcd38c6bd02f7c552e4560f87d12f3ebc11e9259ca2b6abd0224536e88`;
the complete temporary candidate patch is retained under ignored `reports/`,
SHA-256 `e3037cb42fa9584eceb355126d5e40e0e590f0659914428d92ca046d41250b9f`.
The retrieved pinned PyTorch source is retained in ignored `reports/` with
SHA-256 `55d01b174b911be6767e6f218cae0e5a3f40fce057c1d614b3b574880c980609`.

The temporary candidate was replaced by an explicitly selected experimental
WMT14 CUDA reference backend. It uses strict cuDNN only for the three initial
transposed convolutions and first residual convolution, and a separate bounded
reference-order group-normalization kernel only for the two initial-cascade
shapes. Other operations retain the ordinary backend behavior. The generic
backend's default refinement remains unchanged. Eight real-CUDA normalization
cases across lengths 4/16 and batches 1/4/8/32 match captured upstream inputs
bit-for-bit; invalid initial-refinement shapes are rejected. The original
four-row CFG batch now matches all codes and IDs, and the full trace repeats
byte-for-byte with SHA-256
`3cab3d2b13e3f85a478db6b15770181bdcaf6b2c8faf9b80f02f6e01a21d814e`,
identical to the earlier oracle-injection diagnostic trace.
Both 128-row base and CFG development outputs remain byte-identical to the
previous Rust captures and therefore to upstream. This is a candidate screen;
full base/CFG output replay, quality and performance remain
required before accepting the change. The former failed full CFG result remains
preserved; no tolerance or gate was changed. Full `tools/check.sh --cuda`
passed 18 real-device cases (log SHA-256
`4b348ff75621bcad3e7b150246459422fe327ae459cf98864ad34d0c14236180`),
and the separate CUDA/cuDNN-feature release regression passed (log SHA-256
`079ee0a5be4bc499b0504652a53e97f831ca6e2b5fbdfcb35434597122862112`).
The upcoming full-set candidate comparison is a retest on data already
inspected during this investigation, not a fresh held-out experiment.

An isolation experiment substituted the captured upstream level-0 cumulative
state into the Rust CFG run for this same batch. Python's conditional and null
branches had identical first cumulative states, and the separately reconstructed
Python state matched them exactly. With that one captured state injected, all
four rows' generated codes and raw IDs matched Python; at the former crossing,
Rust selected code 32 with scores 6.352484703063965/6.352484226226807 for
codes 32/1071. This shows sensitivity to the first reconstruction in this batch,
not that any particular cascade operation is solely responsible. It is an
oracle-fed diagnostic, not a candidate implementation or a gate pass. The
upstream cumulative trace SHA-256 is
`5c1b70f3e740fba582a80727237f5434d59af441bce94c29af485753531031cc`;
the injected Rust trace is
`3cab3d2b13e3f85a478db6b15770181bdcaf6b2c8faf9b80f02f6e01a21d814e`.
The temporary Rust example and model patch remain under ignored `reports/`,
SHA-256 `e47d8063f8e913c2797b6a4f54869a81b39b305f7893b02976edacbf2b8da293`
and `6875cceb61d140db07a24177baec8f54e4eeed31d12506f174c93f9726e27ac1`.

The strict trace SHA-256 is
`02867dc64785543b41a3cbd1ce6a91ae7b8246b42293428ec36ad9be34bc0168`,
and the reference-table trace SHA-256 is
`2308581078b81463b8ad0a12c4d1c1f62e94aec7047c5748fd3bd9db718ee`.
The latter candidate's source and model patch are retained under ignored
`reports/phase4-wmt14-cfg-row-rope-source-v1.rs` and
`reports/phase4-wmt14-cfg-row-rope-model-v1.patch`, SHA-256
`0ca47f5fc182b045c3093f3cbfd631d0d0d62d85113cfd85fcad3da4042f6490` and
`9033b728177d8fd5113f306c2f52a454c2d04db21205d99ea57a3d1eaac2c6fb`.
The first two strict builds failed to find the unversioned cuDNN linker name;
the third succeeded using the existing repository-local link directory. All
logs and raw traces remain in `reports/`. No artifact, upstream pin, oracle,
numerical band or gate was changed.
