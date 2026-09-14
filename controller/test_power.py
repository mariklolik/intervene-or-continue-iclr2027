import importlib.util
import unittest


class PowerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('power_sensitivity'),
                             'Exact paired binary power implementation must exist')
        import power_sensitivity
        return power_sensitivity

    def test_no_rejection_with_five_discordances_at_five_percent(self):
        self.assertEqual(self.module().exact_power(5, .2, 1, .05), 0)

    def test_six_discordances_reject_only_unanimous_directions(self):
        self.assertAlmostEqual(self.module().exact_power(6, .2, 1, .05), .6**6 + .4**6)

    def test_random_discordance_exact_summation(self):
        self.assertAlmostEqual(self.module().exact_power(6, .1, .5, .05), .5**6 * (.6**6 + .4**6))

    def test_null_power_respects_alpha(self):
        self.assertLessEqual(self.module().exact_power(100, 0, .2, .05), .05 + 1e-12)

    def test_invalid_pair_probability_rejected(self):
        with self.assertRaises(ValueError):
            self.module().exact_power(100, .2, .1, .05)


if __name__ == '__main__':
    unittest.main()
