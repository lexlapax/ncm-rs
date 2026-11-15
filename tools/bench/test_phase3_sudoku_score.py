"""Check Sudoku scoring semantics against small known boards."""
import unittest
from phase3_sudoku_score import score, valid

class ScoreTests(unittest.TestCase):
    def test_solved_and_clues_and_malformed(self):
        cells = [((r * 3 + r // 3 + c) % 9) + 1 for r in range(9) for c in range(9)]
        board = []
        for row in range(9):
            if row: board.append(10)
            board.extend(cells[row*9:(row+1)*9])
        prefix = [11] + [v if i % 3 else 0 for i, v in enumerate(board)] + [11]
        result = score([prefix], [board], [board])
        self.assertEqual((result['solved'], result['valid_grids'], result['malformed'], result['clues_preserved']), (1, 1, 0, 1))
        broken = board[:]; broken[0] = 0
        result = score([prefix], [board], [broken])
        self.assertEqual((result['solved'], result['valid_grids'], result['malformed'], result['clues_preserved']), (0, 0, 1, 0))
        self.assertTrue(valid(''.join('|' if x == 10 else str(x) for x in board)))
        with self.assertRaises(ValueError): score([prefix], [board], [board[:-1]])

if __name__ == '__main__': unittest.main()
