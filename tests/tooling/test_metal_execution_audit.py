"""Execution evidence must describe submitted work and the result actually used."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from metal_execution_fixture import pipeline, evidence

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "observed_metal_driver", ROOT / "benchmarks/tools/bench_metal_suites.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
audit = driver.metal_execution_audit


class MetalExecutionAudit(unittest.TestCase):
    def test_completion_requires_actual_work_and_consistent_counters(self):
        self.assertEqual(
            audit.validate_pipeline(pipeline())["gpu_completed_launches"], 2
        )
        for change in (
            dict(gpu_launches=0, gpu_completed_launches=0),
            dict(gpu_completed_launches=3),
            dict(gpu_completed_launches=1),
            dict(gpu_launches=True),
            dict(result_device="cpu"),
            dict(device_status="cpu"),
            dict(result_used="True"),
            dict(phase_timing_kind="device_elapsed"),
            dict(context_wall_ms=-1),
            dict(synchronize_wall_ms=float("nan")),
            dict(node=2),
            dict(executions=0),
            dict(plan_id=-1),
            dict(plan_id=2),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_pipeline(pipeline(**change))

    def test_cpu_retry_keeps_completed_gpu_attempt_separate(self):
        failed = pipeline(
            device_status="failed",
            result_device="none",
            result_used=False,
            fallback_reason="data_fault",
        )
        cpu = pipeline(
            execution_attempt=1,
            result_device="cpu",
            device_status="cpu_fallback",
            fallback_reason="data_fault_adjudication",
            gpu_launches=0,
            gpu_completed_launches=0,
        )
        result = audit.summarize(
            [dict(status="cpu", pipelines=[failed, cpu])], 1
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["gpu_completed_calls"], 1)
        self.assertEqual(result["gpu_result_calls"], 0)
        self.assertEqual(result["cpu_fallback_calls"], 1)
        self.assertFalse(result["whole_query_gpu_certificate"])

    def test_empty_and_hybrid_results_are_not_gpu_certificates(self):
        host = pipeline(
            device_status="no_device_work",
            result_device="host",
            gpu_launches=0,
            gpu_completed_launches=0,
        )
        result = audit.summarize(
            [dict(status="no_device_work", pipelines=[host])], 1
        )
        self.assertEqual(result["gpu_result_calls"], 0)
        self.assertEqual(result["no_device_work_calls"], 1)
        producer = pipeline(
            node=0,
            pipeline=0,
            operator="FILTER",
            result_device="cpu",
            device_status="cpu",
            gpu_launches=0,
            gpu_completed_launches=0,
        )
        hybrid = audit.summarize(
            [dict(status="gpu_observed", pipelines=[producer, pipeline()])], 1
        )
        self.assertEqual(hybrid["gpu_result_calls"], 1)
        self.assertEqual(hybrid["cpu_compute_calls"], 1)

    def test_admission_recomputes_evidence_and_rejects_missing_or_forged_records(
        self,
    ):
        valid = evidence(audit)
        self.assertTrue(audit.verified_summary(valid, 2))
        self.assertFalse(audit.verified_summary(valid, 3))
        for change in (
            dict(gpu_result_calls=3),
            dict(calls=[]),
            dict(whole_query_gpu_certificate=True),
        ):
            self.assertFalse(audit.verified_summary(dict(valid, **change), 2))
        duplicate = copy.deepcopy(valid)
        duplicate["calls"][0]["pipelines"].append(pipeline(query_id=0))
        self.assertFalse(audit.verified_summary(duplicate, 2))
        legacy = audit.summarize([dict(status="no_marker", pipelines=[])], 1)
        self.assertEqual(legacy["status"], "incomplete")
        with self.assertRaises(ValueError):
            audit.summarize(
                [dict(status="cpu_only", pipelines=[pipeline()])], 1
            )

    def test_protocol_binds_pipeline_records_to_each_collection(self):
        output = (
            "runtime_pipeline\t" + json.dumps(pipeline()) + "\n"
            "runtime_audit\tgpu_observed\ncollections\t1\nordered_equal\tTrue\n"
            "match\texact\ncoverage\tq1\tok\t\n"
        )
        result = driver.parse(output, "q1")
        self.assertTrue(audit.verified_summary(result["runtime_execution"], 1))
        with self.assertRaises(ValueError):
            driver.parse(
                output.replace("runtime_audit\tgpu_observed\n", ""), "q1"
            )


if __name__ == "__main__":
    unittest.main()
