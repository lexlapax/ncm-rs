"""Deterministic source-level index of explicit Rust struct storage signals.

This is a review aid, not a Rust parser or a runtime allocation profiler. It
indexes declared source types including the vendored tokenizer, not types
synthesized by macros or external crates outside this checkout.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECL = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)')
SIGNALS = ('Vec', 'String', 'HashMap', 'BTreeMap', 'HashSet', 'BTreeSet', 'Box', 'Arc', 'Rc', 'Cow',
           'Mutex', 'RwLock', 'OnceLock', 'Tensor', 'Array', 'Atomic', 'Cache', 'Regex',
           'AHashMap', 'AHashSet', 'Vocab', 'VocabR', 'MergeMap', 'MatchingSet', 'VecDeque', 'SmallVec', 'CudaSlice')

def body(lines, index, start_column):
    text = lines[index][start_column:] + '\n' + '\n'.join(lines[index + 1:index + 250])
    # Strip line comments for brace counting. Literal braces and generated macro
    # syntax can still defeat this intentionally conservative lexical scan.
    clean = '\n'.join(line.split('//', 1)[0] for line in text.splitlines())
    positions = [(clean.find(c), c) for c in ('{', '(', ';') if clean.find(c) >= 0]
    if not positions: return '', False
    start, delimiter = min(positions)
    if delimiter == ';': return clean[:start + 1], True
    end_char = '}' if delimiter == '{' else ')'
    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == delimiter: depth += 1
        elif clean[i] == end_char:
            depth -= 1
            if depth == 0: return clean[:i + 1], True
    return clean, False

def main():
    records = []
    for base in ('crates', 'vendor/tokenizers-0.22.2'):
        for path in sorted((ROOT/base).rglob('*.rs')):
            raw = path.read_bytes(); lines = raw.decode('utf-8').splitlines()
            for i, line in enumerate(lines):
                match = DECL.match(line)
                if not match: continue
                excerpt, complete = body(lines, i, match.end())
                signals = [name for name in SIGNALS if re.search(r'\b' + re.escape(name) + r'\b', excerpt)]
                records.append({'path':str(path.relative_to(ROOT)), 'line':i+1, 'name':match.group(1),
                                'storage_signals':signals, 'lexical_body_complete':complete,
                                'file_sha256':hashlib.sha256(raw).hexdigest()})
    print(json.dumps({'schema_version':1, 'scope':'explicit declarations; lexical heuristics, not measured memory or generated types',
                      'records':records}, indent=2, sort_keys=True))
if __name__ == '__main__': main()
