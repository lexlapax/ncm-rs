"""Compare WMT14 head captures and project each input against the same weights in FP64."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tinystories_artifact import digest_file


def from_bits(bits, shape):
    if len(bits) != shape.numel() or any(type(x) is not int or not 0 <= x < 2**32 for x in bits):
        raise ValueError('invalid WMT14 FP32 bit patterns')
    values = torch.tensor(bits, dtype=torch.uint32).view(torch.float32).reshape(shape)
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite WMT14 capture')
    return values


def scores(values):
    top = values.topk(3)
    return [{'code': code, 'score': score} for code, score in zip(top.indices.tolist(), top.values.tolist())]


def compare(reference, candidate):
    report_path = reference / 'report.json'
    report = json.loads(report_path.read_bytes())
    trace = reference / 'trace.safetensors'
    data = json.loads(candidate.read_bytes())
    level, start, position = (data[key] for key in ('level', 'batch_start', 'position'))
    if (any(type(x) is not int for x in (level, start, position, data['row'])) or data['cfg'] != 0
            or report['cfg_scale'] != 0.0 or report['observed_head'] != level or report['batch_start'] != start
            or digest_file(trace) != report['trace_sha256']):
        raise ValueError('WMT14 captures do not identify the same verified base head')

    head_name = f'diagnostic.head_{level:02d}'
    with safe_open(trace, framework='pt', device='cpu') as capture:
        reference_input = capture.get_tensor(f'{head_name}.input')
        shape = reference_input.shape
        row = data['row'] - start
        if len(shape) != 3 or shape[2] != 768 or not 0 <= row < shape[0] or not 0 <= position < shape[1]:
            raise ValueError('WMT14 head site is out of range')
        if data['head_input_shape'] != list(shape):
            raise ValueError('WMT14 head input shapes differ')
        candidate_input = from_bits(data['head_input_bits'], shape)
        candidate_logits = from_bits(data['row_site_logits_bits'], torch.Size([4096]))
        reference_logits = capture.get_tensor(f'generator.level_{level:02d}.logits')[row, position]
        weights = capture.get_tensor(f'{head_name}.weight').double()
        native_codes = torch.tensor(data['codes'], dtype=torch.int64)
        native_ids = torch.tensor(data['raw_ids'], dtype=torch.int64)
        reference_codes = capture.get_tensor('generator.codes')
        reference_ids = capture.get_tensor('generator.greedy_ids')
        if (weights.shape != (4096, 768) or reference_logits.shape != (4096,)
                or native_codes.shape != reference_codes.shape or native_ids.shape != reference_ids.shape):
            raise ValueError('unexpected WMT14 comparison shapes')
        earlier_codes = sum(capture.get_slice(f'generator.level_{i:02d}.logits').get_shape()[1]
                            for i in range(level))

    site_reference = reference_input[row, position]
    site_candidate = candidate_input[row, position]
    if any(logits[codes[row, earlier_codes + position]] != logits.max()
           for codes, logits in ((native_codes, candidate_logits), (reference_codes, reference_logits))):
        raise ValueError('captured WMT14 logits and selected code disagree')
    return {
        'reference_report_sha256': digest_file(report_path),
        'candidate_sha256': digest_file(candidate),
        'row': data['row'],
        'level': level,
        'position': position,
        'preceding_level_code_differences': int((native_codes[:, :earlier_codes] != reference_codes[:, :earlier_codes]).sum()),
        'batch_code_differences': int((native_codes != reference_codes).sum()),
        'batch_raw_id_differences': int((native_ids != reference_ids).sum()),
        'head_input_max_abs_difference': float((reference_input - candidate_input).abs().max()),
        'site_input_max_abs_difference': float((site_reference - site_candidate).abs().max()),
        'site_logit_max_abs_difference': float((reference_logits - candidate_logits).abs().max()),
        'reference_fp32_top': scores(reference_logits),
        'candidate_fp32_top': scores(candidate_logits),
        'reference_input_fp64_top': scores(weights @ site_reference.double()),
        'candidate_input_fp64_top': scores(weights @ site_candidate.double()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
