# Streamed artifact hashing — rejected

The candidate hashed each bounded artifact file while `read_to_end` filled the
owned buffer, rather than making a subsequent full-file hash pass. It preserved
the existing file inventory, byte limits, full-file and per-tensor hashes, and
backend API. `cargo test -p ncm-io --locked` passed the nine unit and artifact
tests, including corrupt, truncated and malformed inputs. The exact rejected
[patch](raw/streamed-hash-v1.patch.gz) has compressed SHA-256
`1cb8c4ab18675210159f62873f3b4bfedfe3591cd7df499d860edb383f9a81e1`
and decompressed SHA-256
`c3f15671f1510b73857c66ceb335bcc1806fe7fb9bb8e7b2634a6330f023624f`.

Release CUDA benchmarks compared the saved geometry candidate with the modified
binary in three fresh processes each, ordered previous/candidate,
candidate/previous, previous/candidate. CPU 2 and the existing codec workload
were used, with five warmups and thirty checked-load trials per process.
The per-process load medians were 192.742, 190.237, 191.250 ms for the previous
binary and 200.539, 202.486, 201.976 ms for the candidate. The median of process
medians increased from **191.25 to 201.98 ms** (ratio **1.0561**). All twelve
codec batch/scope output-ID hashes matched. Host high-water readings at the load
checkpoint ranged 605–608 MB previous and 604–606 MB candidate; these three
processes do not establish a memory improvement.

[Raw reports and binary hashes](raw/streamed-hash-v1.tar.xz) have SHA-256
`59b1eee7a6a6237ba0a91bbe948881e1bffaf0fabedb0714c619e7bc5e5488e6`.
This targeted experiment is not a Python comparison, full numerical trace,
certified quiet-window run, lifecycle analysis or 78-cell acceptance audit.
The candidate was reverted; the release `codec_bench` binary still reflects the
rejected patch and must be rebuilt before reuse. Linux remains 72/78.
