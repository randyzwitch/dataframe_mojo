"""Measured Mojo benchmark profiles accept operation-dependent synchronization."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "metal_benchmark_driver", ROOT / "scripts/bench_metal.py"
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


if __name__ == "__main__":
    unittest.main()
