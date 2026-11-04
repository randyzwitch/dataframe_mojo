"""Placement models stay conservative and reject incomparable measurements."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from metal_calibration import fit, publish


class MetalCalibrationTests(unittest.TestCase):
    def captures(self):
        return [
            {
                "schema_version": 1,
                "identity": {
                    "runtime_build_id": "dataframe-mojo-metal-v2;cpu="
                    + "a" * 64
                    + ";metal="
                    + "b" * 64,
                    "device": "M1",
                    "host": "host;thermal=0",
                    "workers": 8,
                },
                "status": "measured",
                "verified": True,
                "direct_gpu": True,
                "key": "sum:chunks=1:nulls=0",
                "rows": n,
                "cpu_ns": [2 * n + 50, 2 * n + 100, 2 * n + 150],
                "accel_ns": [n + 900, n + 1000, n + 1100],
            }
            for n in [100, 1000, 10000]
        ]

    def test_bounds_every_observed_sample_and_does_not_extrapolate(self):
        records = self.captures()
        model = fit(records)["operators"][records[0]["key"]]
        self.assertEqual((model["min_rows"], model["max_rows"]), (100, 10000))
        for entry in records:
            cpu = model["cpu"]
            accel = model["accel"]
            self.assertLessEqual(
                cpu["fixed_ns"] + cpu["row_ns"] * entry["rows"],
                min(entry["cpu_ns"]),
            )
            self.assertGreaterEqual(
                accel["fixed_ns"] + accel["row_ns"] * entry["rows"],
                max(entry["accel_ns"]),
            )
        self.assertGreater(model["accel"]["fixed_ns"], 0)

    def test_cannot_mix_builds_devices_or_thread_budgets(self):
        for field, value in [
            ("runtime_build_id", "changed"),
            ("device", "M2"),
            ("host", "host;thermal=1"),
            ("workers", 4),
        ]:
            records = self.captures()
            records[1]["identity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fit(records)

    def test_requires_checked_gpu_samples_and_three_scales(self):
        for change in [
            {"verified": False},
            {"direct_gpu": False},
            {"rows": 0},
            {"cpu_ns": [1, float("nan"), 2]},
            {"accel_ns": [1, 2]},
            {"schema_version": 2},
        ]:
            records = self.captures()
            records[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                fit(records)
        with self.assertRaises(ValueError):
            fit(self.captures()[:2])
        records = self.captures()
        records[0]["identity"][
            "runtime_build_id"
        ] = "development-source-no-calibration"
        with self.assertRaises(ValueError):
            fit(records)

    def test_unsupported_cases_are_preserved_and_publish_is_immutable(self):
        records = self.captures()
        skipped = copy.deepcopy(records[0])
        skipped.update(status="skipped", reason="CPU accumulation order")
        records.append(skipped)
        profile = fit(records)
        self.assertEqual(profile["skipped"], [skipped])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = publish(profile, root)
            self.assertEqual(first, publish(profile, root))
            changed = copy.deepcopy(profile)
            changed["identity"]["workers"] = 4
            second = publish(changed, root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())


if __name__ == "__main__":
    unittest.main()
