# LM1B-small CUDA codec reference preparation

The pinned upstream codec was strictly loaded from the existing LM1B-small
config and shared tokenizer checkpoint. Two raw-ID rows from the previously
inspected seed-11 development capture were used as 128-token codec inputs.
The reference runs on the real CUDA device with deterministic cuDNN, TF32
disabled and math SDPA, using the existing `codec-fp32-v1` trace policy without
changing tolerances. It instruments boundaries without modifying upstream
model math and checks traced and ordinary encode decisions agree.

Two fresh processes captured the same 362 tensors, including all twelve
quantizer levels, learned first-level up/down operations, the positional
four-block decoder and both encoded-code and forced-code decoding. Their
reports are byte-identical, SHA-256
`4b24d8a08f20facd81bdd21fc7546414c570c146f93bff775c99007c0b328cc8`;
their safetensors payloads are byte-identical, SHA-256
`944b64e5ccaf24913a29a08a1825c68458e9dba6902df1b31872d93983c29f82`.
The reference source SHA-256 is
`c2918aae52f62bc27d2b311714dd944a9718843efe6d2ab6b20b2b0f40c7d714`.
Input, checkpoint, config, pinned upstream and watched source identities were
checked before and after. Raw evidence is in ignored
`reports/phase4-lm1b-small-codec-cuda-reference-v{1,2}/`.

Under the existing diagnostic margin policy, 27 of 1,282 encode choices and
one of 256 forced-decode choices are near boundaries; encoded-code decoding
has zero near-boundary final choices. These are diagnostic flags, not failures
or tolerance changes. No Rust comparison has yet run.

The command was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/lm1b_small_codec_trace.py --upstream .reference/upstream-ncm
--source artifacts/upstream/lm1b-phase4 --reference
reports/phase4-lm1b-small-cuda-development-v1.json --output NEW_REPORT_DIR`.
A focused test checks input count, width, types and vocabulary bounds. These
two inspected inputs are preparation evidence only; they do not establish
LM1B Rust parity, task quality, performance, Metal or Phase 4 completion.

The repository standard `tools/check.sh` passed; log SHA-256
`630805501f4585651458df760790e4f84cc25f7aaa33a7fe86403444636cae5a`.
