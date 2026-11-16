import unittest

from run_phase4_tinystories_mlx_bench import runtime_order


class RuntimeOrderTests(unittest.TestCase):
    def test_batch_128_preserves_full_matrix_order_when_selected_alone(self):
        self.assertEqual([runtime_order(128, rep) for rep in range(3)],
                         [('python', 'rust'), ('rust', 'python'), ('python', 'rust')])


if __name__ == '__main__':
    unittest.main()
