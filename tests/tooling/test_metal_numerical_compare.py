"""Pure-stdlib contracts for the tolerant numerical row comparator."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "benchmarks" / "tools" / "metal_numerical_compare.py"
spec = importlib.util.spec_from_file_location("metal_numerical_compare", TOOL)
comparator = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(comparator)


class NumericalComparisonTests(unittest.TestCase):
    def compare(self, expected, actual, *, rtol=0.0, atol=1e-9, budget=1_000_000):
        return comparator.compare_rows(expected, actual, ["key", "value"], ["value"], rtol, atol, budget)

    def test_adversarial_nontransitive_greedy_trap_finds_perfect_matching(self):
        # First expected value can match either actual; the second can match only
        # the first actual. Augmenting paths must move the first match to -0.5.
        result = self.compare([(1, 0.0), (1, 1.0)], [(1, 0.5), (1, -0.5)], atol=0.6)
        self.assertEqual(result["status"], "equal")

    def test_duplicate_multiplicity_cannot_reuse_approximate_candidate(self):
        result = self.compare([(1, 0.0), (1, 0.0)], [(1, 0.0), (1, 0.4)], atol=0.1)
        self.assertEqual(result["status"], "different")
        self.assertEqual(result["reason"], "no_tolerant_perfect_matching")

    def test_row_count_and_exact_key_differences_fail(self):
        self.assertEqual(self.compare([(1, 1.0)], [], atol=1.0)["reason"], "row_count_mismatch")
        self.assertEqual(self.compare([(1, 1.0)], [(2, 1.0)], atol=1.0)["reason"], "exact_key_mismatch")

    def test_large_integer_exact_key_is_not_rounded_through_float(self):
        large = 2**53 + 1
        result = comparator.compare_rows([(large, 1.0)], [(float(large), 1.0)], ["key", "value"], ["value"])
        self.assertEqual(result["status"], "different")

    def test_tolerant_null_nan_infinity_and_signed_zero_rules(self):
        expected = [(1, None), (2, float("nan")), (3, float("inf")), (4, -0.0)]
        actual = [(4, 0.0), (3, float("inf")), (2, float("nan")), (1, None)]
        self.assertEqual(self.compare(expected, actual)["status"], "equal")
        self.assertEqual(self.compare([(1, float("inf"))], [(1, -float("inf"))])["status"], "different")
        self.assertEqual(self.compare([(1, None)], [(1, float("nan"))])["status"], "different")

    def test_absolute_tolerance_near_zero(self):
        self.assertEqual(self.compare([(1, 1e-13)], [(1, 8e-13)], atol=1e-12)["status"], "equal")

    def test_relative_tolerance_at_large_magnitude(self):
        self.assertEqual(self.compare([(1, 1e12)], [(1, 1e12 + 500)], rtol=1e-9, atol=0)["status"], "equal")

    def test_finite_difference_outside_budget_fails(self):
        self.assertEqual(self.compare([(1, 1.0)], [(1, 1.01)], rtol=1e-6, atol=1e-6)["status"], "different")

    def test_candidate_pair_budget_exhaustion_is_unverified(self):
        result = self.compare([(1, 1.0)], [(1, 1.0)], budget=0)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["reason"], "candidate_pair_budget_exceeded")

    def test_tolerant_columns_reject_boolean_or_integer_coercion(self):
        with self.assertRaises(ValueError):
            self.compare([(1, True)], [(1, 1.0)])
        with self.assertRaises(ValueError):
            self.compare([(1, 1)], [(1, 1.0)])

    def test_exact_columns_preserve_numeric_type(self):
        result = comparator.compare_rows([(True, 1.0)], [(1, 1.0)], ["key", "value"], ["value"])
        self.assertEqual(result["status"], "different")


if __name__ == "__main__":
    unittest.main()
