# Guarded and width-portable Q/K rotary follow-up

This isolated follow-up layers the existing compiler-initialization error guard and `phase4_generator_qk_rope_portable.patch` on the screened fused candidate. The Metal kernel checks `threads_per_simdgroup` in the shader: width 32 keeps the screened SIMD reduction; other widths calculate the same 64-element squared sum independently per lane. A forced scalar test is included to exercise that branch on the available hardware. This is an explicit implementation of the same equation, not a change to model weights, tolerances or the pinned Python reference.

Formatting, model/core check, exact five-patch replay and the focused release CUDA pinned generator golden and strict-cuDNN retained TinyStories batch passed. CUDA log `reports/phase4-qk-rope-portable-cuda-v1.log` SHA-256 `55b3e1b3057ae9bc8d685e148984ee0508dfb7bafa33d16b3789b7f67bb604ce`. Full CUDA has passed for the preceding candidate, not yet for this source.

Next: on real Metal, validate the normal operator and forced scalar branch, compiled-initializer error handling, seed identity and speed against the screened candidate. The scalar branch on actual non-32-wide GPU hardware is not yet qualified. Full platform checks, quality/performance evidence for any changed outputs, and production integration remain pending.
