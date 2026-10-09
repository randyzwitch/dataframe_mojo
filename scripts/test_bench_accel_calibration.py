"""Calibration integrity checks; no GPU required."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import bench_accel_calibration as bench


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.case = bench.cases(True)[0]
        self.stdout = "device,name,0,test GPU\nanswer,a0,0,12\n"
        for mode in (*bench.MODES, "context_init", "first_query"):
            for rep in range(3 if mode in bench.MODES else 1):
                self.stdout += f"timing,{mode},{rep},100\n"
        for name in (
            "upload_bytes",
            "download_bytes",
            "workspace_bytes",
            "peak_requested_device_bytes",
            "synchronizations",
            "device_id",
        ):
            self.stdout += f"metric,{name},0,0\n"
        self.stdout += "metric,kernel_launches,0,2\n"

    def test_all_dimensions_in_both_tiers(self):
        for quick, count in ((True, 64), (False, 432)):
            cases = bench.cases(quick)
            self.assertEqual(len(cases), count)
            for key in self.case:
                self.assertGreater(len({c[key] for c in cases}), 1)

    def test_incomplete_duplicate_and_nonfinite_samples_rejected(self):
        self.assertEqual(
            len(
                bench.parse_output(self.stdout, self.case, 3)["samples_ns"][
                    "cpu"
                ]
            ),
            3,
        )
        for broken in (
            self.stdout.replace("timing,cpu,2,100\n", ""),
            self.stdout + "timing,cpu,2,100\n",
            self.stdout.replace("timing,cpu,2,100", "timing,cpu,2,nan"),
        ):
            with self.assertRaises(ValueError):
                bench.parse_output(broken, self.case, 3)

    def test_failures_and_slow_large_cases_block_crossover(self):
        record = {
            "case": self.case,
            "status": "ok",
            **bench.parse_output(self.stdout, self.case, 3),
        }
        record["samples_ns"]["cpu"] = [1000, 1000, 1000]
        larger = copy.deepcopy(record)
        larger["case"]["rows"] *= 10
        larger["status"] = "failed"
        larger["error"] = "wrong answer"
        result = bench.summarize([record, larger])
        self.assertEqual(result["failed"], 1)
        self.assertTrue(
            all(
                e["first_measured_rows_with_margin_through_larger_sizes"]
                is None
                for e in result["placement_evidence"]
            )
        )
        larger["status"] = "ok"
        larger["samples_ns"]["gpu_fresh_handle"] = [100, 100, 900]
        result = bench.summarize([record, larger])
        self.assertIsNone(
            result["placement_evidence"][0][
                "first_measured_rows_with_margin_through_larger_sizes"
            ]
        )
        self.assertEqual(
            result["placement_evidence"][1][
                "first_measured_rows_with_margin_through_larger_sizes"
            ],
            1000,
        )

    def test_partial_sweep_cannot_recommend_crossover(self):
        record = {
            "case": self.case,
            "status": "ok",
            **bench.parse_output(self.stdout, self.case, 3),
        }
        record["samples_ns"]["cpu"] = [1000, 1000, 1000]
        result = bench.summarize([record], bench.cases(True))
        self.assertFalse(result["complete"])
        self.assertTrue(
            all(
                e["first_measured_rows_with_margin_through_larger_sizes"]
                is None
                for e in result["placement_evidence"]
            )
        )

    def test_kernel_speed_cannot_imply_end_to_end_win(self):
        record = {
            "case": self.case,
            "status": "ok",
            **bench.parse_output(self.stdout, self.case, 3),
        }
        record["samples_ns"]["kernel_interval"] = [1, 1, 1]
        self.assertTrue(
            all(
                e["first_measured_rows_with_margin_through_larger_sizes"]
                is None
                for e in bench.summarize([record])["placement_evidence"]
            )
        )

    def test_failure_preserved_in_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            bench.write_results(
                output,
                {},
                [
                    {
                        "case": self.case,
                        "status": "failed",
                        "error": "wrong answer",
                    }
                ],
            )
            self.assertIn("wrong answer", output.read_text())
            self.assertIn(
                "wrong answer", output.with_suffix(".csv").read_text()
            )

    def test_changed_binary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "consumer"
            binary.write_text("binary")
            (root / "provider.json").write_text('{"packages": {}}')
            binary.with_suffix(".manifest.json").write_text(
                '{"benchmark_sha256": "source", "binary_sha256": "old", "provider": {"packages": {}}}'
            )
            with patch.object(bench, "digest", side_effect=["source", "new"]):
                with self.assertRaisesRegex(ValueError, "binary or provider"):
                    bench.verify_provenance(binary, root)


if __name__ == "__main__":
    unittest.main()
