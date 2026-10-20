"""Pure-Python contracts for the independent ordering oracle."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "benchmarks" / "tools" / "metal_ordering_oracle.py"
spec = importlib.util.spec_from_file_location("metal_ordering_oracle", TOOL)
oracle = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(oracle)


class SelectionValidationTests(unittest.TestCase):
    def test_tied_ranks_allow_any_member_at_the_tie(self):
        full = [(9, "a"), (8, "b"), (8, "c"), (2, "d")]
        result = oracle.validate_selection(full, [(9, "a"), (8, "c")], sort_index=0, limit=2)
        self.assertTrue(result["valid"])
        self.assertTrue(result["rank_sequence_correct"])

    def test_unordered_group_head_accepts_any_valid_subset(self):
        full = [("a", 2), ("b", 4), ("c", 1), ("d", 3)]
        result = oracle.validate_selection(full, [("c", 1), ("a", 2)], sort_index=None, limit=2)
        self.assertTrue(result["valid"])
        self.assertIsNone(result["rank_sequence_correct"])

    def test_fabricated_aggregate_row_fails_membership(self):
        result = oracle.validate_selection([("a", 4), ("b", 7)], [("a", 4), ("b", 8)], limit=2)
        self.assertFalse(result["valid"])
        self.assertFalse(result["exact_membership"])
        self.assertEqual(result["invalid_row_multiplicity"], 1)

    def test_wrong_rank_sequence_fails(self):
        full = [(10, "top"), (8, "middle"), (1, "bottom")]
        result = oracle.validate_selection(full, [(8, "middle"), (10, "top")], sort_index=0, limit=2)
        self.assertTrue(result["exact_membership"])
        self.assertFalse(result["rank_sequence_correct"])
        self.assertFalse(result["valid"])

    def test_omitted_duplicate_rank_fails_even_when_replaced_by_valid_lower_row(self):
        full = [(9, "same"), (9, "same"), (7, "lower")]
        result = oracle.validate_selection(full, [(9, "same"), (7, "lower")], sort_index=0, limit=2)
        self.assertTrue(result["exact_membership"])
        self.assertFalse(result["rank_sequence_correct"])
        self.assertFalse(result["valid"])

    def test_bad_count_fails(self):
        result = oracle.validate_selection([(3, "a"), (2, "b"), (1, "c")], [(3, "a")], sort_index=0, limit=2)
        self.assertFalse(result["row_count_correct"])
        self.assertFalse(result["valid"])

    def test_negative_offset_and_limit_fail_closed(self):
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1,)], [(1,)], offset=-1)
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1,)], [(1,)], limit=-1)

    def test_null_and_nonfinite_values_fail_closed(self):
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1, "a")], [(1, None)])
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1.0,)], [(float("inf"),)])

    def test_empty_relation_accepts_nonnegative_rank_index(self):
        result = oracle.validate_selection([], [], sort_index=0, limit=10)
        self.assertTrue(result["valid"])
        self.assertEqual(result["expected_rank_sequence"], [])
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([], [], sort_index=-1)

    def test_signed_zero_is_exact_row_identity(self):
        result = oracle.validate_selection([(0.0,)], [(-0.0,)], limit=1)
        self.assertFalse(result["exact_membership"])
        self.assertFalse(result["valid"])

    def test_source_hash_must_match_reviewed_revision(self):
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_source_identity({oracle.SOURCE_PATH: "0" * 64})
        self.assertEqual(
            oracle.validate_source_identity({oracle.SOURCE_PATH: oracle.REVIEWED_SOURCE_SHA256}),
            oracle.REVIEWED_SOURCE_SHA256,
        )

    def test_exact_types_do_not_allow_python_numeric_coercion(self):
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1, "a")], [(True, "a")])
        with self.assertRaises(oracle.OracleInputError):
            oracle.validate_selection([(1, "a")], [(1.0, "a")])


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


@unittest.skipIf(pa is None, "PyArrow is required for file-protocol tests")
class ExportFileProtocolTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        import json
        import tempfile

        self._hashlib = hashlib
        self._json = json
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.answer = self.root / "answer.parquet"
        self.schema = self.root / "answer.schema.json"
        self.table = pa.table({"key": pa.array([1, 2], type=pa.int64()), "value": pa.array([1.5, 2.5], type=pa.float64())})
        pq.write_table(self.table, self.answer)
        self.descriptor = {
            "format": "dataframe-answer-schema-v1",
            "engine": "metal",
            "repetition": 0,
            "rows": 2,
            "columns": [
                {"index": 0, "name": "key", "dtype": {"name": "int64", "encoded": "P5:int64"}},
                {"index": 1, "name": "value", "dtype": {"name": "float64", "encoded": "P7:float64"}},
            ],
        }
        self.schema.write_text(self._json.dumps(self.descriptor))
        self.export = {"engine": "metal", "repetition": 0, "path": str(self.answer), "schema_path": str(self.schema)}
        self.receipt = {
            "sha256": oracle._sha256(self.answer),
            "schema_sha256": oracle._sha256(self.schema),
            "original_schema": self.descriptor,
        }
        self.pa = pa

    def _check(self):
        return oracle._check_schema_export(self.export, self.root, self.receipt, self.pa)

    def test_valid_typed_parquet_and_complete_descriptor_pass(self):
        answer, descriptor, table = self._check()
        self.assertEqual(answer, self.answer)
        self.assertEqual(descriptor, self.descriptor)
        self.assertEqual(table.schema, self.table.schema)

    def test_physical_parquet_dtype_mismatch_fails(self):
        wrong = pa.table({"key": pa.array([1.0, 2.0], type=pa.float64()), "value": pa.array([1.5, 2.5], type=pa.float64())})
        pq.write_table(wrong, self.answer)
        self.receipt["sha256"] = oracle._sha256(self.answer)
        with self.assertRaises(oracle.OracleInputError):
            self._check()

    def test_parquet_content_hash_tampering_fails_before_read(self):
        with self.answer.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(oracle.OracleInputError, "answer parquet hash"):
            self._check()

    def test_schema_descriptor_engine_or_repetition_tampering_fails(self):
        for field, value in (("engine", "cpu"), ("repetition", 1)):
            with self.subTest(field=field):
                descriptor = dict(self.descriptor)
                descriptor[field] = value
                self.schema.write_text(self._json.dumps(descriptor))
                self.receipt["schema_sha256"] = oracle._sha256(self.schema)
                with self.assertRaisesRegex(oracle.OracleInputError, "schema descriptor differs"):
                    self._check()
                self.schema.write_text(self._json.dumps(self.descriptor))
                self.receipt["schema_sha256"] = oracle._sha256(self.schema)

    def test_receipt_descriptor_identity_mismatch_fails(self):
        self.export["engine"] = "cpu"
        with self.assertRaisesRegex(oracle.OracleInputError, "schema descriptor differs"):
            self._check()


if __name__ == "__main__":
    unittest.main()
