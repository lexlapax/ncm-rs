# Phase 5 step 1: Apple Silicon / Metal CLI verification

2026-11-20. Tested on `macstudio` (Apple M1 Ultra, macOS 26.7.3, Rust 1.99.1, native MLX 0.32.2 via `.reference/mlx-native`). `dotsetup.rmlx` was idle in memory and untouched.

## Build, feature folding, and workspace checks

- Baseline pre-Phase-5 binary built at `702168c` with `--features mlx,wmt14-experimental,lm1b-mlx-experimental`: SHA-256 `d8224daaf8172aacc1f1e2cf6f2fa1fc481e6c81a0b0a6185f80d30c885460e7`.
- Candidate release binary built at `fc52a57` (identical Rust source to `a18a624`) with `--features mlx` only (verifying `wmt14-experimental` and `lm1b-mlx-experimental` folding): SHA-256 `51f281f0a58e92bc2b23cfab647dc9760f0a0cd00892216cd9b290c3f88cb6de`.
- `cargo clippy -p ncm-cli --all-targets --features mlx --locked -- -D warnings` and `cargo test -p ncm-cli --features mlx --locked` passed (12 unit tests + `unified_rejections` integration test).
- At `a18a624` (after fixing two macOS portability issues in Python report tests), `./tools/check.sh --mlx` passed: formatting, workspace check/clippy/test/doc, 493 macOS report tests, reference tests, MLX release clippy/test/doc, and `mlx_probe` (`Metal: 18 cases passed`, `reports/local-mlx-20261121T001701Z-13925/mlx.json`).

## Nine-profile baseline / alias / unified equivalence and invalid-device checks

`tools/bench/phase5_cli_equivalence.py --platform mlx` compared baseline alias, candidate alias, and candidate unified `ncm sample` in separate processes across all nine qualified artifacts. Every profile produced byte-identical stdout across all three routes (`reports/phase5-mlx-equivalence-v1/result.json`, SHA-256 `15c740f67345f1824f40387d157fe4e701ee0dcc4a3b10ac7182acb369604aff`):

| Profile | Device | Common stdout SHA-256 |
| --- | --- | --- |
| `sudoku-easy` | `mlx` | `bd2a86aa8d3c626178bfd1d07bd2bbbdfee2a98643298c02c189e59bb2a06212` |
| `sudoku-medium` | `mlx` | `b5a22993f2df60ece95b6b188e009a55cf10ad59a2591e69e478a025125d104d` |
| `sudoku-hard` | `mlx` | `655dbfdb50d1210923246a730a08332156df64418365328d55a2f20af8b7caf0` |
| `tiny` | `mlx` | `7c828e7bcf538bd65a60762429e4df318c6d71a52e3d8cb20210bfaf14eb6837` |
| `xsum` | `mlx` | `87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca` |
| `wmt14` | `mlx` | `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` |
| `lm1b-small` | `mlx` | `103f054a4f8a201867c24397e7c14aecb1d6e0fb91fec51568f177b00220152d` |
| `lm1b-medium` | `mlx` | `21cd2625d4caa3559b28e5ab490d363314afc8b0fb0af61d840efc1a938cac18` |
| `lm1b-large` | `mlx` | `12af7a2fe810deca255c46f01f5ede7fd33e43f1ac61aeac53372c7231c6803b` |

`tools/bench/phase5_cli_invalid_devices.py` rejected `phase5-invalid-device` with exit 1, empty stdout, and the expected profile-specific diagnostic across all nine profiles (`reports/phase5-mlx-invalid-devices-v1.json`, SHA-256 `41759dfd12fec1f2eb1fc5cdb600d944477b1e140e888d1a4ec6406f0b5eafd5`).

## Experimental flags, LM1B default sidecar, and Metal-specific options

Run on `macstudio` (`reports/phase5-mlx-step1-screens-v2/`):

- **XSum experimental flags:** positive CFG (`1`, `--experimental-positive-cfg-xsum`) matched baseline == alias == unified (`75aeac4fe53bb7416fbd2d8a75814671ca84d5721eb30b8699d09b7ecd2b57ed`); stochastic (`0 42`, `--experimental-stochastic-xsum`) matched baseline == alias == unified (`801defdf5ed80c6a0bf17c75c2a426789293b169b827d722aee0a11cc713a215`). Positive CFG + seeded failed explicitly on all three routes with `Seeded XSum Metal generation requires CFG 0`. Unflagged positive CFG and unflagged seed failed explicitly on both alias and unified commands.
- **WMT14 CFG boundary and `mlx-retry-padded`:** CFG 0 (`17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a`), unflagged CFG 1 (`374b657b197af78011c8f532aed3c5b1e868631e263562d2ac906839f0159fcb`), flagged CFG 2 (`--experimental-cfg-wmt14`, `4d22adc794c2c8418345f94588af8764eb0570a10f9047845465804b526e0a0e`), and `mlx-retry-padded` (`17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a`) all matched baseline == alias == unified. Unflagged CFG 2 failed explicitly on both alias and unified commands.
- **LM1B default sibling rotary sidecar and `--top-p-fp32`:** `artifacts/canonical/lm1b-reference-rope-v1.safetensors` (SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`) produced byte-identical stdout without `--reference-rope` compared to explicit `--reference-rope` across small, medium, and large. `--top-p-fp32` on small matched baseline == alias == unified (`11a8c49a289afb7b60f715fb39ca5235c32c753649c80c7fa7863ca7384c2245`).

## `--text` and `--decode` parity on macOS / Metal

1. **Corpus & boundary tokenizer tests on macOS (`ncm-cli --features mlx`):** all six artifact-bound tests passed against the pinned SHA-256 references:
   - `corpus_text_encoding_matches_every_reference_row` (11,334 XSum rows + 3,003 WMT14 rows);
   - `synthetic_text_boundaries_match_pinned_python` (XSum + WMT14 overlength/Unicode/special-token boundaries);
   - `synthetic_decode_boundaries_match_pinned_python` (135 rows each for XSum + WMT14);
   - `tinystories_text_matches_upstream_vocabulary_reference` (8,199 rows / 8,064 accepted);
   - `tinystories_single_token_decode_matches_upstream_vocabulary` (8,192 vocab IDs);
   - `tinystories_multi_token_decode_matches_upstream` (137 multi-token sequences).
2. **End-to-end Metal `--text` + `--decode` generation equivalence** (`reports/phase5-mlx-step1-screens-v2/`, `reports/phase5-mlx-step1-screens-v3/result.json` SHA-256 `2ef8181e6ef79a1d34e27ec8e5399be8a87aa1d2b04046557758557c4b3ad4e6`):
   - **LM1B small/medium/large:** `[""] --text --seed 42 --decode` matched `[[101]] --seed 42 --decode` byte-for-byte (`3f28b8157b25387ce9c191d3dbd40ce7f8fc2bb0fe9d7789bd2d88b30624c7e0`, `db81ce43834a6953717f5a0d647fdeab30b3672db28842e9e3b8b75426bcf260`, `861f13d04d1ce43b916805e09a3942a8e76703d6453ce11186e0fd9853772d0e`); `["hello"]` rejected.
   - **TinyStories:** `["hello"] --text --seed 42 --decode` (token 5663) matched `[[5663]] --seed 42 --decode` byte-for-byte (`c150908c02da9b00f998c55c00155dcf165c7cfcbf69ed3336789d00268fe078`), matching baseline codes/IDs (`d39feef86a300fb22bc5677e0bc1e6908ccf58adfbdbe71abd60c349c91dd7d7`); `["hello world"]` rejected.
   - **XSum:** row 0 (`6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328`), 4-row batch 1234..1238 (`c02f73fd7b449a2f2e20a5364c3f62c1e6d1ab8ec6a1b64af14f3ed097605d5f`), and >512-token overlength truncation (`a86041d600b370e4b9532cec6d3f7686a3df32ccda65061c7d695f1919c5dbbe`) all matched `--text --decode` == ID `--decode` byte-for-byte and matched baseline codes/IDs.
   - **WMT14 CFG=0 and CFG=1:** CFG=0 row 0 (`fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32`), batch 4 (`1a8a15ede3c8ce3247f5a342b2738777ca40ae086d83f85aab4f5c1ca82f886b`), >96-token overlength (`9ba41ffb3e526e71520091c6bf1e5465f6891aed22055da68c8290b92216b6f9`), plus CFG=1 row 0 (`4449a45d316ab9f50346d8fcd1ff721fc46c144075405f9bc0ecb57c84ef9d92`) and batch 4 (`370ac5b8bdc5844d3c5e907b5bb710769e944405fda3ebb5901a8738d4540ff8`) all matched `--text --decode` == ID `--decode` byte-for-byte and matched baseline codes/IDs.

## Scope

This verifies Step 1 CLI routing, feature folding, flag boundaries, invalid-device rejection, LM1B sibling rotary sidecar, and `--text`/`--decode` parity on real Metal hardware. Phase 4 quality/timing qualification and accepted Metal exceptions remain unchanged; Mac fresh-clone `docs/USAGE.md` verification follows separately.
