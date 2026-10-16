"""Independent typed files and ownership/protocol checks for optional exports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
from metal_answer_exports import collect_exports, parse_export
import bench_metal_suites as driver


def fixture_schema(engine="cpu", repetition=0):
    return dict(
        format="dataframe-answer-schema-v1",
        engine=engine,
        repetition=repetition,
        rows=3,
        columns=[
            dict(
                index=i,
                name=name,
                dtype=dict(
                    name=dtype, encoded="P" + str(len(dtype)) + ":" + dtype
                ),
            )
            for i, (name, dtype) in enumerate(
                [("URL\tname", "string"), ("key", "int64"), ("f32", "float32")]
            )
        ],
    )


class AnswerExports(unittest.TestCase):
    def test_complete_typed_independent_parquet_fixture(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest(
                "Independent Parquet fixture requires the oracle environment"
            )
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "answer"
            source = pl.DataFrame(
                {
                    "URL\tname": pl.Series(
                        ["https://example/" + "λ" * 300, None, 'quote"\nline'],
                        dtype=pl.String,
                    ),
                    "key": pl.Series(
                        [9223372036854775807, -1, None], dtype=pl.Int64
                    ),
                    "f32": pl.Series([1.25, None, -0.5], dtype=pl.Float32),
                }
            )
            path = str(prefix) + "-cpu-0.parquet"
            source.write_parquet(path)
            schema_path = str(prefix) + "-cpu-0.schema.json"
            Path(schema_path).write_text(
                json.dumps(fixture_schema(), ensure_ascii=False)
            )
            payload = dict(
                engine="cpu",
                repetition=0,
                status="written",
                path=path,
                schema_path=schema_path,
            )
            observed = collect_exports(
                [parse_export(json.dumps(payload))], prefix, 1
            )
            self.assertEqual(observed[0]["status"], "written")
            self.assertEqual(
                pl.read_parquet(path).to_dicts(), source.to_dicts()
            )
            self.assertEqual(pl.read_parquet(path).schema, source.schema)
            self.assertEqual(
                observed[0]["original_schema"]["columns"][0]["name"],
                "URL\tname",
            )
            self.assertEqual(len(observed[0]["sha256"]), 64)
            self.assertEqual(observed[1]["status"], "not_observed")

    def test_wrong_answer_cannot_be_promoted_by_exports(self):
        payload = dict(
            engine="metal",
            repetition=0,
            status="unavailable",
            path="p",
            schema_path="s",
            error="no writer",
        )
        result = driver.parse(
            "answer_export\t"
            + json.dumps(payload)
            + "\ncoverage\tq38\twrong_answer\tfull result differs\n",
            "q38",
        )
        self.assertEqual(result["status"], "wrong_answer")
        self.assertEqual(result["answer_exports"][0]["status"], "unavailable")

    def test_unowned_duplicate_and_unexpected_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "answer"
            record = dict(
                engine="cpu",
                repetition=0,
                status="unavailable",
                path=str(prefix) + "-cpu-0.parquet",
                schema_path=str(prefix) + "-cpu-0.schema.json",
                error="writer missing",
            )
            self.assertEqual(
                collect_exports([record], prefix, 2)[0]["status"], "unavailable"
            )
            for records in (
                [record, record],
                [dict(record, path="/unowned.parquet")],
                [dict(record, engine="metal", repetition=99)],
            ):
                with self.assertRaises(ValueError):
                    collect_exports(records, prefix, 2)

    def test_missing_or_invalid_schema_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "answer"
            record = dict(
                engine="cpu",
                repetition=0,
                status="written",
                path=str(prefix) + "-cpu-0.parquet",
                schema_path=str(prefix) + "-cpu-0.schema.json",
            )
            self.assertEqual(
                collect_exports([record], prefix, 1)[0]["status"], "unavailable"
            )
            Path(record["schema_path"]).write_text(
                json.dumps(dict(fixture_schema(), rows=-1))
            )
            self.assertEqual(
                collect_exports([record], prefix, 1)[0]["status"], "unavailable"
            )

    def test_malformed_protocol_and_fractional_repetition_rejected(self):
        for value in (
            {},
            dict(
                engine=[],
                repetition=0,
                status="written",
                path="p",
                schema_path="s",
            ),
            [],
            dict(
                engine="cpu",
                repetition=0.5,
                status="written",
                path="p",
                schema_path="s",
            ),
        ):
            with self.assertRaises(ValueError):
                parse_export(json.dumps(value))

    def test_exports_follow_timing_and_precede_query_comparisons(self):
        text = (ROOT / "benchmarks/metal_suite_common.mojo").read_text()
        self.assertLess(
            text.index("elapsed +="),
            text.index('export_answer(actual, "metal", repetition)'),
        )
        self.assertLess(
            text.index('export_answer(actual, "metal", repetition)'),
            text.index("ordered_equal &= actual.equals(expected)"),
        )
        helper = (
            ROOT / "benchmarks/metal_suite_answer_exports.mojo"
        ).read_text()
        self.assertIn("write_parquet(frame, path)", helper)
        self.assertNotIn(".sort(", helper)
        self.assertNotIn(".head(", helper)
        self.assertNotIn(".cast(", helper)


if __name__ == "__main__":
    unittest.main()
