"""Calibration math and leakage checks; no GPU or compiler invocation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("cost_driver", Path(__file__).resolve().parents[1] / "scripts/bench_accelerator_cost.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class CalibrationTests(unittest.TestCase):
    def test_generated_and_heldout_queries_are_disjoint(self):
        cases = driver.matrix(64)
        train = {driver.case_id(c) for c in cases if c["role"] == "train"}
        heldout = {driver.case_id(c) for c in cases if c["role"] == "heldout"}
        self.assertFalse(train & heldout)
        changed = dict(cases[0], rows=999999, null_percent=50)
        self.assertEqual(driver.case_id(changed), driver.case_id(cases[0]))
        self.assertIn("chain32", {c["workload"] for c in cases})
        self.assertIn("count", {c["workload"] for c in cases})

    def test_coordinate_descent_fits_nonnegative_terms(self):
        x = [[0.0] * len(driver.FEATURES) for _ in range(4)]
        for i, row in enumerate(x):
            row[0], row[1] = 1, i + 1
        coefficients = driver.nnls(x, [12, 14, 16, 18])
        for row, target in zip(x, [12, 14, 16, 18]):
            actual = sum(coefficients[k] * row[j] for j, k in enumerate(driver.FEATURES))
            self.assertAlmostEqual(actual, target, places=5)
        self.assertTrue(all(v >= 0 for v in coefficients.values()))

    def test_coverage_rejection_is_explicit(self):
        features = dict.fromkeys(driver.FEATURES, 1.0)
        model = dict(coefficients=features, minimum=dict.fromkeys(driver.FEATURES, 0), maximum=dict.fromkeys(driver.FEATURES, 2), residual_factor=1)
        self.assertEqual(driver.predict(model, features), len(driver.FEATURES))
        features["rows"] = 3
        self.assertIsNone(driver.predict(model, features))

    def test_incomplete_or_unvalidated_output_is_rejected(self):
        with self.assertRaises(ValueError):
            driver.parse_run("VALIDATED False\n")


if __name__ == "__main__":
    unittest.main()
