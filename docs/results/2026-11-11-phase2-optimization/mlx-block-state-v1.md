# Apple resident-block regression — isolated state diagnostics

The combined candidate's [initial Apple audit](final-mlx-initial-v1.md) remains
**69/78**, including batch-4 resident-block candidate/accepted median ratio
1.35310. These diagnostics investigate that failure; none changes the fixed
acceptance workload or closes a gate.

- **Fresh synthetic latent:** nine rotating fresh processes compared accepted,
  current and Apple-unpacked Q/K/V blocks with the same synthetic resident
  tensor and no earlier codec execution. Candidate/accepted block median
  ratios at batches 1/2/4 were 0.888/1.002/0.992; all output float hashes
  matched. The unpacked variant was similarly close. Process medians varied.
- **Fresh actual latent:** six alternating fresh processes uploaded the exact
  accepted fixture's retained latents, whose hashes match the prior accepted
  and candidate encoder snapshots, without earlier codec execution. Ratios
  were 0.855/1.001/1.007, and input and output hashes matched across all runs.
- **Materialized latent after the workload:** a separate six-process diagnostic
  read and reuploaded each exact latent outside timing, then retained the
  normal encode/decode/reconstruction sequence. Block ratios were
  **1.073/1.085/1.346**. Thus changing the retained tensor allocation alone did
  not remove the observed batch-4 gap.
- **Process-local cache clear:** another six-process diagnostic cleared each
  process's unused MLX cache immediately before block timing. Cache counters
  fell to zero, but block ratios were **1.061/1.098/1.237**. This modified
  preparation is not acceptance evidence or a proposed global cache policy.

Together these results indicate that the regression depends on execution
history or runtime state rather than an observable change in final latent
values/strides alone. They do not identify the cause, establish GPU clock
behavior, or justify altering acceptance warmups. A separate single-gather
interpolation candidate did not consistently improve the workload; see
[alternatives](mlx-block-alternatives-v1.md). Production and the failed audit
remain unchanged.

[Raw reports, source, source identities and build logs](raw/mlx-block-state-v1.tar.xz)
have SHA-256 `533a4bfdcce17aa01fe2acaee05b31e3cbce55242d03e7a0fc539fdbd979ab9b`.
The raw fixture-derived latent arrays remain outside Git; their hashes are
retained in the reports.
