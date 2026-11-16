"""Host-only preflight for the pinned WMT14 checkpoint structure.

This is preparation for a future exporter, not a runtime artifact or model support.
"""
from tinystories_artifact import TRAINING, tensor_map


def checkpoint_parts(generator, tokenizer):
    """Separate the exact named EMA/hit state, preserving all other tensors.

    The classification is provisional until the WMT14 inference call graph and
    exporter are qualified. No checkpoint data are discarded by this function.
    """
    if not isinstance(generator, dict) or set(generator) != {'model'}:
        raise ValueError('unexpected WMT14 generator checkpoint metadata')
    if not isinstance(tokenizer, dict) or set(tokenizer) != {'model'}:
        raise ValueError('unexpected WMT14 tokenizer checkpoint metadata')
    model = tensor_map(generator['model'], 186)
    codec = tensor_map(tokenizer['model'], 96)
    expected = {f'vq.{kind}_{index}' for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit')
                for index in range(11)}
    observed = {name for name in codec if TRAINING.fullmatch(name)}
    if observed != expected:
        raise ValueError('unexpected WMT14 EMA/hit inventory')
    return {'generator': model,
            'codec_other': {name: value for name, value in codec.items() if name not in expected},
            'ema_and_hits': {name: value for name, value in codec.items() if name in expected}}
