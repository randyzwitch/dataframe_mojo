"""Policy and optional Parquet integration tests for the numerical audit CLI."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "benchmarks" / "tools" / "metal_numerical_audit.py"
spec = importlib.util.spec_from_file_location("metal_numerical_audit", TOOL)
audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(audit)

SOURCE_PINS = {
    "benchmarks/suites/h2o.mojo": "bba2b834539145cbd45d1751bf72ed172e94672f5328e64352c32015fbbae05c",
    "benchmarks/suites/pdsh.mojo": "f27444f0de2e128f05ed457b65fa1d1fc93ebae1f3f7eabe1dd9e306819b990e",
    "benchmarks/suites/tpcds.mojo": "14d4f9e0288dcadbb01f4c5aae341951ebb60dd011af478a35010688a8c74e68",
}


def minimal_policy(columns):
    return {
        "format": audit.POLICY_FORMAT,
        "rtol": 1e-9,
        "atol": 1e-12,
        "comparison": "abs(a-b) <= max(atol, rtol * max(abs(a), abs(b)))",
        "source_sha256": dict(SOURCE_PINS),
        "cases": {"h2o_groupby/q3": columns},
    }


class PolicyValidationTests(unittest.TestCase):
    def test_valid_policy(self):
        self.assertEqual(audit.validate_policy(minimal_policy(["value"]))["rtol"], 1e-9)

    def test_rejects_changed_tolerance_or_source_pin_set(self):
        policy = minimal_policy(["value"])
        policy["rtol"] = 1e-8
        with self.assertRaises(audit.AuditInputError):
            audit.validate_policy(policy)
        policy = minimal_policy(["value"])
        del policy["source_sha256"]["benchmarks/suites/pdsh.mojo"]
        with self.assertRaises(audit.AuditInputError):
            audit.validate_policy(policy)

    def test_rejects_bad_or_duplicate_column_registry(self):
        with self.assertRaises(audit.AuditInputError):
            audit.validate_policy(minimal_policy(["value", "value"]))
        with self.assertRaises(audit.AuditInputError):
            audit.validate_policy(minimal_policy([3]))

    def test_duplicate_or_missing_required_exports_fail(self):
        cpu = dict(engine="cpu", repetition=0, status="written")
        metal = dict(engine="metal", repetition=0, status="written")
        for exports in ([cpu], [cpu, metal, metal], [cpu, dict(metal, repetition=True)]):
            with self.subTest(exports=exports), self.assertRaises(audit.AuditInputError):
                audit._validate_export_identities(exports, "q3")


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


@unittest.skipIf(pa is None, "PyArrow is required for end-to-end audit tests")
class NumericalAuditParquetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.answer_root = self.root / "answers"
        self.answer_root.mkdir()
        self.coverage_path = self.root / "coverage.json"
        self.policy_path = self.root / "policy.json"

    def _export(self, engine, table, dtype_name, encoded):
        stem = f"answer-{engine}-0"
        answer = self.answer_root / f"{stem}.parquet"
        sidecar = self.answer_root / f"{stem}.schema.json"
        pq.write_table(table, answer)
        descriptor = {
            "format": "dataframe-answer-schema-v1", "engine": engine, "repetition": 0,
            "rows": table.num_rows,
            "columns": [
                {"index": 0, "name": "key", "dtype": {"name": "int64", "encoded": "P5:int64"}},
                {"index": 1, "name": "value", "dtype": {"name": dtype_name, "encoded": encoded}},
            ],
        }
        sidecar.write_text(json.dumps(descriptor))
        return {
            "engine": engine, "repetition": 0, "path": str(answer), "schema_path": str(sidecar),
            "status": "written", "sha256": audit.sha256(answer),
            "schema_sha256": audit.sha256(sidecar), "original_schema": descriptor,
        }

    def _run(self, cpu_table, metal_table, dtype_name="float64", encoded="P7:float64", declared=("value",)):
        exports = [self._export("cpu", cpu_table, dtype_name, encoded), self._export("metal", metal_table, dtype_name, encoded)]
        coverage = {
            "provenance": {"evaluation": {"source_sha256": dict(SOURCE_PINS)}},
            "cases": [{"suite": "h2o_groupby", "query": "q3", "variant": "v3", "status": "wrong_answer", "answer_exports": exports}],
        }
        policy = minimal_policy(list(declared))
        self.coverage_path.write_text(json.dumps(coverage))
        self.policy_path.write_text(json.dumps(policy))
        return audit.audit_coverage(self.coverage_path, self.answer_root, self.policy_path)

    def _float_table(self, values, keys=None):
        return pa.table({"key": pa.array(keys or [1] * len(values), type=pa.int64()), "value": pa.array(values, type=pa.float64())})

    def test_float64_cpu_metal_within_policy_matches(self):
        report = self._run(self._float_table([10.0]), self._float_table([10.0 + 5e-9]))
        self.assertEqual(report["cases"][0]["comparison_status"], "tolerance_match")

    def test_decimal_registry_field_remains_exact(self):
        types = [pa.decimal128(7, 2)]
        if hasattr(pa, "decimal32"):
            types.append(pa.decimal32(7, 2))
        for typ in types:
            with self.subTest(storage=str(typ)):
                cpu = pa.table({"key": pa.array([1], type=pa.int64()), "value": pa.array([Decimal("1.00")], type=typ)})
                metal = pa.table({"key": pa.array([1], type=pa.int64()), "value": pa.array([Decimal("1.01")], type=typ)})
                report = self._run(cpu, metal, "decimal32[7,2]", "P14:decimal32[7,2]")
                self.assertEqual(report["cases"][0]["tolerant_columns"], [])
                self.assertEqual(report["cases"][0]["comparison_status"], "different")

    def test_decimal_precision_mismatch_is_rejected(self):
        table = pa.table({"key": pa.array([1], type=pa.int64()), "value": pa.array([Decimal("1.00")], type=pa.decimal128(8, 2))})
        with self.assertRaisesRegex(audit.AuditInputError, "physical dtype"):
            self._run(table, table, "decimal32[7,2]", "P14:decimal32[7,2]")

    def test_physical_schema_mismatch_fails_closed(self):
        cpu = self._float_table([1.0])
        metal = pa.table({"key": pa.array([1], type=pa.int64()), "value": pa.array([1.0], type=pa.float32())})
        with self.assertRaisesRegex(audit.AuditInputError, "physical dtype"):
            self._run(cpu, metal)

    def test_parquet_hash_tampering_fails_closed(self):
        self._run(self._float_table([1.0]), self._float_table([1.0]))
        export_path = self.answer_root / "answer-metal-0.parquet"
        with export_path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(audit.AuditInputError, "Parquet hash"):
            audit.audit_coverage(self.coverage_path, self.answer_root, self.policy_path)

    def test_duplicate_float_rows_preserve_multiplicity(self):
        cpu = self._float_table([1.0, 1.0], [1, 1])
        metal = self._float_table([1.0, 1.0 + 1e-4], [1, 1])
        report = self._run(cpu, metal)
        self.assertEqual(report["cases"][0]["comparisons"][0]["comparison_status"], "different")

    def test_changed_suite_source_fails_closed(self):
        self._run(self._float_table([1.0]), self._float_table([1.0]))
        receipt = json.loads(self.coverage_path.read_text())
        receipt["provenance"]["evaluation"]["source_sha256"]["benchmarks/suites/h2o.mojo"] = "0" * 64
        self.coverage_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(audit.AuditInputError, "source hash"):
            audit.audit_coverage(self.coverage_path, self.answer_root, self.policy_path)


if __name__ == "__main__":
    unittest.main()
