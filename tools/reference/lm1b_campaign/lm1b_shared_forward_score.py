"""Development scorer that cross-checks FLM perplexity on shared GPT-2 logits."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from lm1b_heldout_score import upstream_functions, valid_targets


def corpus_ppl(texts, tokenizer, model, evaluator):
    """Run the pinned upstream evaluator with an independent per-batch reduction."""
    if not texts:
        raise ValueError('no texts to score')

    enc = tokenizer(texts, return_tensors='pt', return_token_type_ids=False,
                    truncation=True, padding=True, max_length=128)
    ids, attention = enc['input_ids'].cuda(), enc['attention_mask'].cuda()
    consumed = 0
    pending = None
    nll_sum = tok_sum = 0.0

    def reduce_previous_batch():
        nonlocal pending, nll_sum, tok_sum
        if pending is None:
            return
        previous_ids, logits = pending
        pending = None
        nll = F.cross_entropy(logits[..., :-1], previous_ids[..., 1:], reduction='none')
        valid = valid_targets(previous_ids, tokenizer.eos_token_id).to(nll.dtype)
        nll_sum += float((nll * valid).sum())
        tok_sum += float(valid.sum())

    def checked_forward(input_ids, *, attention_mask):
        nonlocal consumed, pending
        end = min(consumed + 16, len(ids))
        if (end == consumed or not torch.equal(input_ids, ids[consumed:end])
                or not torch.equal(attention_mask, attention[consumed:end])):
            raise ValueError('upstream scorer changed the token or attention batch')

        # Upstream still holds the prior logits when it asks for the next
        # forward. Verify that batch before allocating the next one.
        reduce_previous_batch()
        output = model(input_ids, attention_mask=attention_mask)
        pending = input_ids, output[0].transpose(-1, -2)
        consumed = end
        return output

    reference_ppl, _ = upstream_functions(evaluator, tokenizer, checked_forward)
    expected = reference_ppl(texts)
    reduce_previous_batch()
    if consumed != len(ids) or tok_sum <= 0:
        raise ValueError('upstream scorer left an incomplete or empty corpus')
    actual = float(np.exp(nll_sum / tok_sum))
    if not math.isfinite(actual) or actual != expected:
        raise ValueError('shared-forward score differs from pinned upstream')
    return actual, nll_sum, tok_sum
