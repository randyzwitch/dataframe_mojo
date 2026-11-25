"""Placement models stay conservative and reject incomparable measurements."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from metal_calibration import fit, install, publish, _typical


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

    def test_cpu_executor_crossover_is_layout_specific(self):
        records = self.captures()
        for entry in records:
            entry["cpu_materialized_ns"] = [entry["rows"] + 500] * 3
        profile = fit(records)
        region = profile["regions"][records[0]["key"]]
        self.assertEqual(region["cpu_materialized"], {"row_ns": 1, "fixed_ns": 500})
        self.assertEqual(region["cpu"], {"row_ns": 2, "fixed_ns": 100})
        self.assertNotIn("cpu_materialized", profile["operators"][region["cpu_key"]])
        del records[1]["cpu_materialized_ns"]
        records[1]["cpu_materialized_error"] = "answer differs from reference"
        fallback = fit(records)["regions"][records[0]["key"]]
        self.assertNotIn("cpu_materialized", fallback)
        self.assertEqual(fallback["cpu"], region["cpu"])

    def test_materialized_cpu_samples_must_be_paired_and_finite(self):
        for invalid in ([1, 2], [1, 2, 3, 4], [1, -1, 2], [1, float("nan"), 2], None):
            records = self.captures()
            records[0]["cpu_materialized_ns"] = invalid
            with self.subTest(samples=invalid), self.assertRaisesRegex(ValueError, "paired"):
                fit(records)

    def test_gpu_margin_uses_residuals_against_the_cheaper_cpu(self):
        records = self.captures()
        for record in records:
            n = record["rows"]
            record["cpu_ns"] = [5 * n] * 3
            record["cpu_materialized_ns"] = [n, 2 * n, 3 * n]
            record["accel_ns"] = [n] * 3
        region = fit(records)["regions"][records[0]["key"]]
        self.assertAlmostEqual(region["switch_margin"], 1)

    def test_startup_uses_only_paired_cold_and_warm_measurements(self):
        records = self.captures()
        key = records[0]["key"]
        self.assertNotIn("startup_ns", fit(records)["regions"][key])
        records[0]["first_accel_ns"] = 6100
        records[2]["first_accel_ns"] = 16000
        self.assertEqual(fit(records)["regions"][key]["startup_ns"], 5000)
        records[1]["first_accel_ns"] = -1
        with self.assertRaisesRegex(ValueError, "first Metal execution"):
            fit(records)

    def test_boundary_refit_does_not_keep_an_oversteep_slope(self):
        model = _typical([(1, 1), (2, 3), (3, 5)])
        self.assertEqual(model, {"fixed_ns": 0.0, "row_ns": 1.5})
        self.assertEqual(
            _typical([(1, 10), (2, 9), (3, 8)]), {"fixed_ns": 9, "row_ns": 0}
        )

    def test_install_selects_an_immutable_profile_for_one_build_and_worker_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self.captures()
            original = install(fit(records), Path(tmp))
            records[0]["first_accel_ns"] = 9000
            updated = install(fit(records), Path(tmp))
            self.assertNotEqual(original, updated)
            self.assertTrue(original.is_file())
            self.assertEqual(updated.parent.parts[-2:], ("a" * 64, "8"))
            self.assertEqual((updated.parent / "active").read_text().strip(), updated.name)

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

    def test_affine_guard_covers_a_bend_without_pricing_it_as_per_row_growth(self):
        # CPU grows linearly. GPU latency rises quickly at small sizes,
        # then flattens. Relative headroom at the bend must protect that
        # measured CPU win without hiding the later GPU crossover.
        template = self.captures()[0]
        records = []
        for rows, gpu in [(1, 1), (3, 5), (7, 9), (31, 10), (127, 18)]:
            records.append(dict(template, rows=rows, cpu_ns=[0.6 * rows] * 3, accel_ns=[gpu] * 3))
        region = fit(records)["regions"][template["key"]]
        def cost(model, n):
            return model["fixed_ns"] + model["row_ns"] * n
        for record in records:
            self.assertGreaterEqual(
                cost(region["accel"], record["rows"]) + 1e-12,
                1.05 * record["accel_ns"][0],
            )
        self.assertGreater(cost(region["accel"], 7), cost(region["cpu"], 7))
        self.assertLess(cost(region["accel"], 19), cost(region["cpu"], 19))

    def test_decreasing_paired_bounds_use_nonnegative_coefficients(self):
        records = self.captures()
        for entry, gpu in zip(records, (90, 80, 70)):
            entry["cpu_ns"] = [entry["rows"]] * 3
            entry["accel_ns"] = [gpu] * 3
        region = fit(records)["regions"][records[0]["key"]]
        self.assertEqual(region["accel"], {"fixed_ns": 94.5, "row_ns": 0.0})

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

    def test_cpu_executor_consensus_requires_distinct_layouts_and_overlapping_scales(self):
        records = self.captures()
        for e in records:
            e["cpu_layout_family"] = "sum:int32:plain"
            e["cpu_materialized_ns"] = [v / 2 for v in e["cpu_ns"]]
        other = copy.deepcopy(records)
        for e in other:
            e["key"] = "eight chunks"
            e["rows"] *= 2
        name = records[0]["pipeline_key"] + "|sum:int32:plain"
        profile = fit(records + other)
        self.assertEqual(profile["materialized_cpu_ranges"], {name: {"min_rows": 200, "max_rows": 10000}})
        self.assertNotIn(name, profile["operators"])
        self.assertNotIn(name, profile["regions"])
        self.assertFalse(fit(records + records)["materialized_cpu_ranges"])
        for field, value in [("pipeline_key", "pipeline:other"), ("cpu_layout_family", "sum:int32:nullable")]:
            different = copy.deepcopy(other)
            for e in different:
                e[field] = value
            self.assertFalse(fit(records + different)["materialized_cpu_ranges"])
        for e in other:
            e["rows"] += 10000
        self.assertFalse(fit(records + other)["materialized_cpu_ranges"])

    def test_one_tie_loss_or_missing_cpu_sample_vetoes_the_preference(self):
        records = self.captures()
        for e in records:
            e["cpu_layout_family"] = "sum:int32:plain"
            e["cpu_materialized_ns"] = [v / 2 for v in e["cpu_ns"]]
        other = copy.deepcopy(records)
        for e in other:
            e["key"] = "four chunks"
        for factor in [1, 2]:
            mixed = copy.deepcopy(other)
            mixed[1]["cpu_materialized_ns"][0] = factor * mixed[1]["cpu_ns"][0]
            self.assertFalse(fit(records + mixed)["materialized_cpu_ranges"])
        for field in ["cpu_materialized_ns", "cpu_layout_family"]:
            missing = copy.deepcopy(other)
            del missing[0][field]
            self.assertFalse(fit(records + missing)["materialized_cpu_ranges"])

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
