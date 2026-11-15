# Unmasked attention: same-process comparison

The [planned diagnostic](mlx-attention-toggle-plan-v1.md) compared the existing
zero-mask attention with the previously qualified no-mask alternative inside
the same loaded trained model. Production sources and qualified Linux executable
identities were verified first. The isolated Metal target cloned prior build
artifacts and rebuilt local workspace packages. Three fresh processes completed;
finite initial outputs met existing component tolerances, and output IDs matched
and remained stable. There were five warmups and 30 timings per mode. Table
entries are unmasked/zero-mask median / nearest-rank p95 ratios; lower is faster.

| Batch | Scope | Process 1 | Process 2 | Process 3 |
|---:|---|---:|---:|---:|
| 1 | decode | 0.9432 / 1.0009 | 1.0037 / 1.0118 | 1.0007 / 0.9942 |
| 1 | reconstruct | 0.9894 / 0.9706 | 0.9934 / 1.0351 | 0.9975 / 0.9948 |
| 1 | resident_decoder_block | 1.0089 / 1.0366 | 0.9986 / 0.9745 | 0.9888 / 1.0088 |
| 2 | decode | 1.0110 / 0.9888 | 1.0147 / 1.0294 | 0.9930 / 1.0237 |
| 2 | reconstruct | 0.9966 / 1.0042 | 1.0005 / 1.0539 | 1.0016 / 1.0007 |
| 2 | resident_decoder_block | 1.0026 / 0.9788 | 0.9902 / 0.9517 | 0.9781 / 0.8680 |
| 4 | decode | 0.9964 / 1.0229 | 0.9760 / 0.9803 | 1.0095 / 0.9850 |
| 4 | reconstruct | 0.9973 / 1.0170 | 1.0290 / 1.0160 | 0.9954 / 1.0082 |
| 4 | resident_decoder_block | 0.9852 / 1.0075 | 0.9771 / 0.8517 | 1.0066 / 1.0136 |

Batch-4 block medians were 0.9852/0.9771/1.0066; decode and reconstruction
were mixed. This same-process observation does not establish a useful or
consistent full-codec improvement, and does not explain the previously failed
Apple gate. The shared process and interleaved caches also limit independence.
The [earlier separate-process result](mlx-attention-nomask-metal-v1.md) remains
retained. Do not integrate the candidate or change any acceptance result on
this evidence. Eighteen activity samples saw WindowServer up to 37.8% CPU;
they cannot establish GPU exclusivity. No thermal warning was recorded.

The isolated patched codec and ops sources have SHA-256
`8685cb3537ef38901519edbc6982e13d0c622fdfa7456e819d98ca2d7e06d78b` and
`4ec33db005a724a404f96a3fb40b7404e4e80d86ff5f78c4c8c91e5126ef01e4`;
the example is `2238061bd65e85028797f8a1b1d059caf9c529568368636d85627ba57ffb84e6` and binary is
`5b85b6448a0ccd03e31b7609d2b39cc89ab253f48059ee2e74b348c859ec686d`.
The [reports, commands, activity and runner](raw/mlx-attention-toggle-v1.tar.xz)
have SHA-256 `afe13f11b476e91927c339ca41f5dd19f0f9d2663726216fdd727b918a39ae02`.
