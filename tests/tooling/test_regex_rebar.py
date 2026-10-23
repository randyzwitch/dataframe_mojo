"""A performance report must not hide missing or incorrect results."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/tools"))
from bench_regex_rebar import report


class RegexRebarReportTests(unittest.TestCase):
    def result(self):
        cases = [dict(key="example/" + v, name="example", variant=v, expected=7)
                 for v in ("base", "reverse-lines", "double")]
        records = []
        for build, ns in (("parent", 200), ("candidate", 100)):
            for round_number in range(3):
                lines = [f"{case['key']}\tOK\t{rep}\t{ns}\t7"
                         for case in cases for rep in range(3)]
                records.append(dict(build=build, round=round_number,
                                    stdout="\n".join(lines), rejected=False))
        return dict(cases=cases, records=records)

    def test_all_variants_are_included(self):
        result = self.result()
        # Make the perturbation slower even though the base improved.
        for record in result["records"]:
            if record["build"] == "candidate":
                record["stdout"] = "\n".join(
                    line.replace("\t100\t7", "\t400\t7")
                    if "reverse-lines" in line else line
                    for line in record["stdout"].splitlines()
                )
        text, failures = report(result)
        self.assertFalse(failures)
        self.assertIn("2.000× | 0.500×", text)

    def test_compiler_overlapped_samples_cannot_complete_a_report(self):
        result = self.result()
        result["records"][0]["rejected"] = True
        text, failures = report(result)
        self.assertTrue(any(status == "MISSING" for _, _, status, _ in failures))
        self.assertIn("unsupported/failing", text)

    def test_wrong_results_are_not_promoted_as_fast(self):
        result = self.result()
        result["records"][3]["stdout"] = "example/base\tWRONG\t0\t1\t6"
        text, failures = report(result)
        self.assertTrue(any(status == "WRONG" for _, _, status, _ in failures))
        self.assertIn("unsupported/failing", text)


if __name__ == "__main__":
    unittest.main()
