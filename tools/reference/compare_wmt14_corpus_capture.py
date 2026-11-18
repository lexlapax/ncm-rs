"""Compare batchwise WMT14 JSONL against a frozen base or guided CUDA reference."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_wmt14_cuda_capture import compare

REFERENCE_SHA256 = {
    0: 'e287a9c784d927cc319572fed6679bbf4cd9ffc1489c0f15028f71e13c47bce0',
    1: '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256',
}
ROWS = 3003
BATCH = 4


def compare_batches(reference, batches, start, count):
    end = start + count
    if not 0 <= start < end <= ROWS or start % BATCH or (end != ROWS and end % BATCH):
        raise ValueError('expected complete aligned WMT14 batches')
    if len(reference) != ROWS:
        raise ValueError('unexpected reference length')
    result = {
        'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': [],
        'rejected_batches': [], 'retried_batches': [], 'rows_compared': 0,
    }
    common = {'start', 'count'}
    outputs = common | {'codes', 'raw_ids'}
    for offset, record in zip(range(start, end, BATCH), batches, strict=True):
        size = min(BATCH, end - offset)
        if (type(record) is not dict or type(record.get('start')) is not int
                or type(record.get('count')) is not int
                or (record['start'], record['count']) != (offset, size)):
            raise ValueError('capture batch range differs')
        if 'error' in record:
            if record.keys() != common | {'error'} or not isinstance(record['error'], str):
                raise ValueError('invalid WMT14 rejection record')
            result['rejected_batches'].append(record)
            continue
        if 'internal_retry_batch' in record:
            if (record.keys() != outputs | {'internal_retry_batch'}
                    or type(record['internal_retry_batch']) is not int
                    or record['internal_retry_batch'] != 8):
                raise ValueError('invalid WMT14 retry record')
            result['retried_batches'].append(offset)
        elif record.keys() != outputs:
            raise ValueError('invalid WMT14 output record')
        if any(type(record[key]) is not list or len(record[key]) != size
               for key in ('codes', 'raw_ids')):
            raise ValueError('capture row count differs')
        actual = [
            {'row': row, 'codes': codes, 'raw_ids': ids}
            for row, (codes, ids) in enumerate(zip(record['codes'], record['raw_ids'], strict=True), offset)
        ]
        differences = compare(reference[offset:offset + size], actual, offset, size)
        for key, value in differences.items():
            result[key] += value
        result['rows_compared'] += size
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--cfg', type=int, required=True, choices=REFERENCE_SHA256)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=ROWS)
    args = parser.parse_args()

    reference_bytes = args.reference.read_bytes()
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    if reference_sha != REFERENCE_SHA256[args.cfg]:
        raise ValueError('unexpected frozen reference')
    reference = json.loads(reference_bytes)['outputs']

    capture_bytes = args.capture.read_bytes()
    batches = (json.loads(line) for line in capture_bytes.splitlines())
    result = {
        'cfg': args.cfg, 'start': args.start, 'count': args.count,
        'reference_sha256': reference_sha, 'capture_sha256': hashlib.sha256(capture_bytes).hexdigest(),
        **compare_batches(reference, batches, args.start, args.count),
    }
    with args.receipt.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')
    if result['rejected_batches'] or result['code_differences'] or result['id_differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
