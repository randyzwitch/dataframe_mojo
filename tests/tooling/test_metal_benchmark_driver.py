"""Measured Mojo benchmark profiles accept operation-dependent synchronization."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "metal_benchmark_driver", ROOT / "benchmarks/tools/bench_metal.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def output(implementation="mojo", waits=7):
    return (
        "FIRST 10 20\nSAMPLE 0 11 21\nVALIDATED True\n"
        "BUILD_ID development-source-no-calibration\n"
        f"PROFILE_S runtime_implementation {implementation}\n"
        "PROFILE_I kernel_launches 13\n"
        f"PROFILE_I synchronizations {waits}\n"
        "PROFILE_I upload_bytes 64\nPROFILE_I download_bytes 32\n"
        "PROFILE_I peak_requested_bytes 128\n"
    )


class MetalBenchmarkDriver(unittest.TestCase):
    def test_measured_sync_counts_preserved_without_native_assumptions(self):
        parsed = driver.parse_run(output(), 1)
        self.assertEqual(parsed["profile"]["synchronizations"], 7)
        self.assertEqual(parsed["profile"]["kernel_launches"], 13)

    def test_whole_query_totals_override_region_observations(self):
        text = output() + (
            "PROFILE_I profile_regions 2\n"
            "PROFILE_I region_0_kernel_launches 13\n"
            "PROFILE_I region_1_kernel_launches 8\n"
            "PROFILE_I query_kernel_launches 21\n"
            "PROFILE_I query_synchronizations 10\n"
            "PROFILE_I query_upload_bytes 96\n"
            "PROFILE_I query_download_bytes 48\n"
            "PROFILE_I query_peak_requested_bytes 256\n"
        )
        parsed = driver.parse_run(text, 1)["profile"]
        self.assertEqual(parsed["kernel_launches"], 21)
        self.assertEqual(parsed["region_0_kernel_launches"], 13)
        self.assertEqual(parsed["synchronizations"], 10)
        self.assertEqual(parsed["peak_requested_bytes"], 256)
        with self.assertRaises(ValueError):
            driver.parse_run(
                text.replace("PROFILE_I query_peak_requested_bytes 256\n", ""), 1
            )

    def test_native_or_missing_profile_cannot_claim_mojo_measurements(self):
        for text in (
            output("native"),
            output(waits=-1),
            output().replace("PROFILE_I upload_bytes 64\n", ""),
            output().replace("BUILD_ID development-source-no-calibration\n", ""),
            output().replace("VALIDATED True", "VALIDATED False"),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                driver.parse_run(text, 1)

    def test_profile_timing_fields_are_retained(self):
        parsed = driver.parse_run(
            output()
            + "PROFILE_F region_0_gpu_ms 2.5\n"
            + "PROFILE_F region_0_staging_copy_ms 1.25\n"
            + "PROFILE_F query_wall_ms 4.75\n",
            1,
        )
        self.assertEqual(parsed["profile"]["region_0_gpu_ms"], 2.5)
        self.assertEqual(parsed["profile"]["region_0_staging_copy_ms"], 1.25)
        self.assertEqual(parsed["profile"]["query_wall_ms"], 4.75)

    def test_optional_pipeline_and_gpu_timing_fields_are_retained(self):
        parsed = driver.parse_run(
            output()
            + "PROFILE_F region_0_pipeline_compile_ms 0.75\n"
            + "PROFILE_F region_0_gpu_ms 1.5\n",
            1,
        )
        self.assertEqual(parsed["profile"]["region_0_pipeline_compile_ms"], 0.75)
        self.assertEqual(parsed["profile"]["region_0_gpu_ms"], 1.5)

    def test_source_fingerprint_includes_untracked_mojo_content(self):
        with tempfile.TemporaryDirectory() as folder:
            source_root = Path(folder) / "dataframe"
            source_root.mkdir()
            source = source_root / "new_untracked_module.mojo"
            source.write_text("fn first(): pass\n")
            before = driver.source_fingerprint(source_root)
            source.write_text("fn second(): pass\n")
            after = driver.source_fingerprint(source_root)
        self.assertNotEqual(before, after)

    def test_timeout_keeps_partial_output_in_log_and_result(self):
        error = subprocess.TimeoutExpired(["bench"], 3, output=b"partial\xff")
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "worker.txt"
            result = driver.retain_timeout(error, 3, log)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["timeout_seconds"], 3)
            self.assertIn("Timed out", result["error"])
            self.assertEqual(result["captured_output"], "partial\ufffd")
            self.assertEqual(log.read_text(), result["captured_output"])

    def test_report_lists_cases_where_metal_is_slower(self):
        data = {
            "metadata": {"scope": "Development mechanisms only."},
            "runs": [
                {
                    "runtime": "mojo",
                    "rows": 64,
                    "workload": "projection",
                    "variant": "base",
                    "status": "ok",
                    "samples": [{"cpu_ns": 100, "metal_ns": 200}],
                    "profile": {"kernel_launches": 1, "peak_requested_bytes": 4},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.md"
            driver.render(data, target)
            report = target.read_text()
        self.assertIn("## Measured Metal slowdowns", report)
        self.assertIn("| mojo | 64 | projection | base | 0.50 | 2.00x |", report)


if __name__ == "__main__":
    unittest.main()
