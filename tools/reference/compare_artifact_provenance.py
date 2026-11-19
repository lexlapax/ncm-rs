"""Verify identical artifact payloads while reporting explicitly allowed provenance changes.

This compares two trusted, stable artifact directories; it does not approve new
manifest identities, run inference or replace ncm-io's full artifact validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

MAX_MANIFEST = 4 * 1024 * 1024
MAX_PROVENANCE = 4 * 1024 * 1024
MAX_PAYLOAD = 4 * 1024 * 1024 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'non-finite JSON value: {value}')


def same_json(first, second):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return canonical(first) == canonical(second)


def regular(path, limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'not a bounded regular file: {path}')


def read_json(path, limit):
    regular(path, limit)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'file grew beyond bound: {path}')
    return raw, json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def digest(path, limit):
    regular(path, limit)
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > limit:
                raise ValueError(f'file grew beyond bound: {path}')
            h.update(chunk)
    return count, h.hexdigest()


def validate_artifact(root, expected_manifest):
    if root.is_symlink() or not root.is_dir():
        raise ValueError('artifact must be a real directory')
    raw, manifest = read_json(root / 'manifest.json', MAX_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError('manifest hash differs from requested identity')
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise ValueError('invalid artifact manifest')
    files = manifest['files']
    if 'provenance.json' not in files or not files:
        raise ValueError('missing provenance')
    for name, record in files.items():
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..')
                or not isinstance(record, dict) or set(record) != {'bytes', 'sha256'}
                or type(record['bytes']) is not int or not 0 <= record['bytes'] <= MAX_PAYLOAD
                or not isinstance(record['sha256'], str)):
            raise ValueError('invalid file record')
        count, sha = digest(root / name, MAX_PAYLOAD)
        if count != record['bytes'] or sha != record['sha256']:
            raise ValueError(f'payload verification failed: {name}')
    _, provenance = read_json(root / 'provenance.json', MAX_PROVENANCE)
    if not isinstance(provenance, dict):
        raise ValueError('invalid provenance')
    return manifest, provenance


def equal_bytes(first, second):
    with first.open('rb') as a, second.open('rb') as b:
        while True:
            left, right = a.read(1024 * 1024), b.read(1024 * 1024)
            if left != right:
                return False
            if not left:
                return True


def compare(old_root, new_root, old_sha, new_sha, allowed):
    old, old_provenance = validate_artifact(old_root, old_sha)
    new, new_provenance = validate_artifact(new_root, new_sha)
    if set(old) != set(new) or any(not same_json(old[k], new[k]) for k in old if k != 'files'):
        raise ValueError('manifest metadata or tensor inventory differs')
    if set(old['files']) != set(new['files']):
        raise ValueError('file inventories differ')
    if any(not same_json(old['files'][k], new['files'][k]) for k in old['files'] if k != 'provenance.json'):
        raise ValueError('non-provenance payload differs')
    for name in old['files']:
        if name != 'provenance.json' and not equal_bytes(old_root / name, new_root / name):
            raise ValueError(f'non-provenance payload bytes differ: {name}')
    if set(old_provenance) != set(new_provenance):
        raise ValueError('provenance keys differ')
    changed = {key: {'old': old_provenance[key], 'new': new_provenance[key]}
               for key in old_provenance if not same_json(old_provenance[key], new_provenance[key])}
    if set(changed) != set(allowed):
        raise ValueError(f'provenance changes differ from allowed keys: {sorted(changed)}')
    # Verify inputs a second time, including all payloads, to detect mutation during comparison.
    if not same_json(validate_artifact(old_root, old_sha), (old, old_provenance)):
        raise ValueError('old artifact changed during comparison')
    if not same_json(validate_artifact(new_root, new_sha), (new, new_provenance)):
        raise ValueError('new artifact changed during comparison')
    return {'scope': 'verified byte-identical non-provenance files and equal manifest metadata; no qualification',
            'old_manifest_sha256': old_sha, 'new_manifest_sha256': new_sha,
            'file_count': len(old['files']), 'provenance_changes': changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--old-sha256', required=True)
    parser.add_argument('--new-sha256', required=True)
    parser.add_argument('--allow-provenance-key', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    if len(set(args.allow_provenance_key)) != len(args.allow_provenance_key):
        raise ValueError('duplicate allowed provenance key')
    result = compare(args.old, args.new, args.old_sha256, args.new_sha256, args.allow_provenance_key)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
