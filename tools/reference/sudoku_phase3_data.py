"""Convert the newly pinned Phase 3 Sudoku boards into bounded non-executable JSON."""
import argparse
import hashlib
import io
import json
import pickle
import pickletools
from pathlib import Path

REVISION = "6580ba41a4b0a976dde876fbe84d25327617b0e6"
HASHES = {
    "easy": "403353dd5d11e49d42627aee84c7b1ceb399fda03d01f6210ce78d0e61f2b29d",
    "medium": "d49cd2b380bdce748b10a7f0847637b5e160bd107b67f91dca93670b15190871",
    "hard": "dd7b6763e5594094a1366ac8423f87fc9431056721c61df52ce3490023f17d5e",
    "solutions": "b52de75f5e4642adca58b18967bcd4bbcafa699c61b21ac33ee0fd7389e0d576",
}
ALLOWED = {"PROTO", "FRAME", "EMPTY_LIST", "MEMOIZE", "MARK", "BININT1", "APPENDS", "STOP"}

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("global references are forbidden")
    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent references are forbidden")

def read(path, digest, size, length):
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("input size or SHA-256 mismatch")
    if any(op.name not in ALLOWED for op, _, _ in pickletools.genops(raw)):
        raise ValueError("unexpected pickle opcode")
    stream = io.BytesIO(raw)
    boards = Restricted(stream).load()
    if stream.tell() != len(raw) or type(boards) is not list or len(boards) != 2000:
        raise ValueError("unexpected board list")
    for board in boards:
        if type(board) is not list or len(board) != length or any(type(n) is not int or not 0 <= n <= 11 for n in board):
            raise ValueError("invalid board tokens")
        grid = board[1:-1] if length == 91 else board
        if length == 91 and (board[0] != 11 or board[-1] != 11):
            raise ValueError("invalid prefix BOS tokens")
        for i, n in enumerate(grid):
            if (i + 1) % 10 == 0:
                if n != 10: raise ValueError("invalid row separator")
            elif n > 9 or (length == 89 and n == 0):
                raise ValueError("invalid cell token")
    return boards

def convert(root, output, difficulty):
    puzzles = read(root / difficulty / "puzzles.pickle", HASHES[difficulty], 372063, 91)
    solutions = read(root / difficulty / "solutions.pickle", HASHES["solutions"], 364063, 89)
    for puzzle, solution in zip(puzzles, solutions, strict=True):
        if any(a not in (0, b) for a, b in zip(puzzle[1:-1], solution, strict=True)):
            raise ValueError("puzzle clue differs from reference")
    with output.open("x") as file:
        json.dump({"schema_version": 1, "dataset_revision": REVISION, "difficulty": difficulty,
                   "inputs": puzzles, "references": solutions}, file, separators=(",", ":"))
        file.write("\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("difficulty", choices=("easy", "medium", "hard"))
    a = p.parse_args()
    convert(a.source, a.output, a.difficulty)
