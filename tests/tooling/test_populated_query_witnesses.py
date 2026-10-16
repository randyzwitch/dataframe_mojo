"""Supplemental reports cannot promote empty or unrelated query outcomes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "populated_query_witnesses",
    ROOT / "benchmarks/tools/populated_query_witnesses.py",
)
witnesses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(witnesses)


class PopulatedQueryWitnesses(unittest.TestCase):
    def test_q41_offset_has_unique_nonempty_answer_with_bounded_input(self):
        rows = witnesses.fixture_rows("clickbench", "q41")["hits"]
        self.assertEqual(len(rows), 20001)
        counts = Counter(
            (r["WindowClientWidth"], r["WindowClientHeight"]) for r in rows
        )
        tail = sorted(counts.items(), key=lambda item: -item[1])[10000:10010]
        self.assertEqual(tail, [((20000, 2), 1)])
        self.assertEqual(Counter(counts.values()), {2: 10000, 1: 1})

    def test_backlog_keeps_all49_and_only8_selected_nonhybrid_cells(self):
        backlog = json.loads(witnesses.BACKLOG.read_text())
        keys = {
            (r["suite"], r["variant"], r["query"]) for r in backlog["cases"]
        }
        chosen = {
            ("tpcds", v, q)
            for v in ["base", "decimal"]
            for q in ["q3", "q19", "q82"]
        }
        chosen |= {("clickbench", "base", q) for q in ["q19", "q41"]}
        self.assertEqual(len(keys), 49)
        self.assertTrue(chosen <= keys)
        self.assertEqual(len(keys - chosen), 41)
        self.assertNotIn(("tpcds", "base", "q24"), keys)

    def test_audit_requires_source_inputs_population_and_separate_oracles(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case = {
                "suite": "clickbench",
                "variant": "base",
                "query": "q19",
                "witness_id": "witness19",
                "source_path": "canonical.mojo",
                "source_sha256": "sourcehash",
                "tables": {"hits": {"sha256": "inputhash"}},
                "expected_output_rows": 2,
                "before_output_rows": 0,
                "execution_boundary": "gpu_collects",
            }
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"cases": [case]}))
            receipt = folder / "receipt.json"
            measured = {
                "suite": "clickbench",
                "variant": "base",
                "query": "q19",
                "inputs": {"hits": {"sha256": "inputhash"}},
                "status": "ok",
                "output_rows": 2,
            }
            data = {
                "provenance": {
                    "original_source_sha256": {"canonical.mojo": "sourcehash"}
                },
                "cases": [measured],
            }
            receipt.write_text(json.dumps(data))
            result = witnesses.audit(manifest, [receipt])["cases"][0]
            self.assertFalse(result["literal_oracle_cpu_checked"])
            self.assertFalse(result["literal_oracle_metal_checked"])
            self.assertFalse(result["performance_evidence"])
            log = folder / "oracle.log"
            log.write_text("witness_oracle\twitness19\tcpu\t2\tTrue\n")
            result = witnesses.audit(manifest, [receipt], [log])["cases"][0]
            self.assertTrue(result["literal_oracle_cpu_checked"])
            self.assertFalse(result["literal_oracle_metal_checked"])
            for count in [0, 1, 3]:
                measured["output_rows"] = count
                receipt.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "row count"):
                    witnesses.audit(manifest, [receipt])
            measured["output_rows"] = 2
            measured["inputs"]["hits"]["sha256"] = "unrelated"
            receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "identity"):
                witnesses.audit(manifest, [receipt])
            measured["inputs"]["hits"]["sha256"] = "inputhash"
            data["provenance"]["original_source_sha256"][
                "canonical.mojo"
            ] = "otherquery"
            receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source identity"):
                witnesses.audit(manifest, [receipt])


if __name__ == "__main__":
    unittest.main()
