"""Compare the TinyStories CLI's first seeded batch with a retained quality capture."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda-cudnn', 'mlx'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, artifact, reference, output = (p.resolve() for p in
                                           (args.binary, args.artifact, args.reference, args.output))
    ref_bytes = reference.read_bytes()
    ref = json.loads(ref_bytes)
    if ref['seed'] != 42 or ref['batch'] != 128 or len(ref['codes']) < 128 or len(ref['raw_ids']) < 128:
        raise ValueError('expected a seed-42 batch-128 retained quality capture')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'prefixes.json'
    inputs.write_text(json.dumps([[0]] * 128) + '\n')
    command = [str(binary), 'sample-tinystories', str(artifact), args.device, str(inputs), '42']
    completed = subprocess.run(command, check=False, capture_output=True)
    (output / 'stdout.json').write_bytes(completed.stdout)
    (output / 'stderr.txt').write_bytes(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'CLI exited {completed.returncode}; see {output / "stderr.txt"}')
    actual = json.loads(completed.stdout)
    if actual['seed'] != 42 or actual['codes'] != ref['codes'][:128] or actual['ids'] != ref['raw_ids'][:128]:
        raise AssertionError('CLI codes or IDs differ from the retained first batch')
    report = {'command': command, 'reference_sha256': digest(ref_bytes),
              'binary_sha256': digest(binary.read_bytes()), 'output_sha256': digest(completed.stdout),
              'rows': 128, 'codes_and_ids_match': True}
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
