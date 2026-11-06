"""Placement models stay conservative and reject incomparable measurements."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from metal_calibration import fit, publish, _typical


class MetalCalibrationTests(unittest.TestCase):
    def captures(self):
        return [
            {
                "schema_version": 2,
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
                "pipeline_key": "pipeline:select:filter:int32",
                "rows": n,
                "cpu_ns": [2 * n + 50, 2 * n + 100, 2 * n + 150],
                "accel_ns": [n + 900, n + 1000, n + 1100],
            }
            for n in [100, 1000, 10000]
        ]

    def test_fits_typical_samples_with_one_switching_margin(self):
        records = self.captures()
        model = fit(records)["regions"][records[0]["key"]]
        self.assertEqual((model["min_rows"], model["max_rows"]), (100, 10000))
        for entry in records:
            n = entry["rows"]
            cpu = model["cpu"]
            typical = model["accel_typical"]
            self.assertAlmostEqual(
                cpu["fixed_ns"] + cpu["row_ns"] * n, 2 * n + 100
            )
            self.assertAlmostEqual(
                typical["fixed_ns"] + typical["row_ns"] * n, n + 1000
            )
        self.assertAlmostEqual(model["switch_margin"], 0.05)
        self.assertAlmostEqual(model["accel"]["fixed_ns"], 1050)

    def test_boundary_refit_does_not_keep_an_oversteep_slope(self):
        model = _typical([(1, 1), (2, 3), (3, 5)])
        self.assertEqual(model, {"fixed_ns": 0.0, "row_ns": 1.5})
        self.assertEqual(
            _typical([(1, 10), (2, 9), (3, 8)]), {"fixed_ns": 9, "row_ns": 0}
        )

    def test_a_single_slow_gpu_sample_does_not_hide_clear_wins(self):
        records = self.captures()
        for entry in records:
            entry["accel_ns"][2] = 1e9
        model = fit(records)["regions"][records[0]["key"]]
        self.assertLess(
            model["accel"]["fixed_ns"] + 5000 * model["accel"]["row_ns"],
            model["cpu"]["fixed_ns"] + 5000 * model["cpu"]["row_ns"],
        )

    def test_near_ties_stay_on_cpu_but_clear_wins_do_not(self):
        for ratio, choose_gpu in [(1.03, False), (1.5, True)]:
            records = self.captures()
            for entry in records:
                n = entry["rows"]
                entry["cpu_ns"] = [2 * n + 100] * 3
                entry["accel_ns"] = [(2 * n + 100) / ratio] * 3
            model = fit(records)["regions"][records[0]["key"]]
            cpu = model["cpu"]
            gpu = model["accel"]
            self.assertEqual(
                gpu["fixed_ns"] + 5000 * gpu["row_ns"]
                < cpu["fixed_ns"] + 5000 * cpu["row_ns"],
                choose_gpu,
            )

    def test_a_consistently_losing_scale_is_not_hidden_by_the_fit(self):
        records = self.captures()
        for entry in records:
            cpu = 2 * entry["rows"] + 100
            entry["cpu_ns"] = [cpu] * 3
            entry["accel_ns"] = [
                cpu * (1.2 if entry["rows"] == 1000 else 1 / 1.5)
            ] * 3
        model = fit(records)["regions"][records[0]["key"]]
        cpu = model["cpu"]
        gpu = model["accel"]
        self.assertGreaterEqual(
            gpu["fixed_ns"] + 1000 * gpu["row_ns"],
            cpu["fixed_ns"] + 1000 * cpu["row_ns"],
        )

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
            {"schema_version": 99},
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

    def test_cpu_pipeline_envelope_is_conservative_across_layouts(self):
        records = self.captures()
        other = copy.deepcopy(records)
        for entry in other:
            entry["key"] = "sum:chunks=8:nulls=1"
            entry["cpu_ns"] = [t * 0.5 for t in entry["cpu_ns"]]
        profile = fit(records + other)
        self.assertIn(records[0]["key"], profile["operators"])
        model = profile["operators"][records[0]["pipeline_key"]]["cpu"]
        for entry in records + other:
            self.assertLessEqual(
                model["fixed_ns"] + model["row_ns"] * entry["rows"],
                min(entry["cpu_ns"]),
            )
            region = profile["regions"][entry["key"]]
            self.assertEqual(
                region["cpu"], profile["operators"][region["cpu_key"]]["cpu"]
            )
        self.assertEqual(len(profile["regions"]), 2)
        slow = profile["regions"][records[0]["key"]]
        fast = profile["regions"][other[0]["key"]]
        evaluate = lambda m: m["fixed_ns"] + 10000 * m["row_ns"]
        self.assertLess(evaluate(slow["accel"]), evaluate(slow["cpu"]))
        self.assertGreater(evaluate(fast["accel"]), evaluate(fast["cpu"]))


if __name__ == "__main__":
    unittest.main()
