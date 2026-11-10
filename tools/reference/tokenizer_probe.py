"""Time the unmodified pinned upstream Sudoku character tokenizer."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("upstream", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    fixture = json.loads(args.fixture.read_bytes())
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == fixture["upstream_commit"]
    assert not subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from tokenization.sudoku_tokenizer import SudokuTokenizer
    tokenizer = SudokuTokenizer()
    inputs = [(c["text"], c["ids"]) for c in fixture["cases"]]
    for case in fixture["cases"]:
        assert tokenizer.encode(case["text"]) == case["ids"]
        assert tokenizer.decode(case["ids"]) == case["decoded"]
    def run():
        for _ in range(1024):
            for text, ids in inputs:
                tokenizer.encode(text)
                tokenizer.decode(ids)
    for _ in range(5):
        run()
    trials = []
    for _ in range(30):
        start = time.perf_counter()
        run()
        trials.append((time.perf_counter() - start) * 1000)
    with args.output.open("x") as f:
        json.dump({"runtime":"python-upstream-sudoku", "warmups":5, "trials_ms":trials,
                   "repetitions":1024, "cases":len(inputs), "parity_passed":True,
                   "scope":"1024 repetitions of all five golden encode+decode pairs; not model tokenization"}, f, indent=2)


if __name__ == "__main__":
    main()
