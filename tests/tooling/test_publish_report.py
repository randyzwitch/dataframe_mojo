"""publish_report builds a newest-first index and republishes idempotently."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import publish_report


def fake(day, sha):
    return {
        "provenance": {
            "utc": f"{day}T02:00:00Z",
            "revision": sha * 40,
            "tier": "full",
            "mojo": "Mojo 1.2.0 (abc)",
            "polars": "1.44.2",
            "duckdb": "1.5.5",
        },
        "engines": ["mojo", "polars", "duckdb"],
        "runs": [],
    }


SUMMARY = {"pdsh": {"polars": {"geomean": 0.9, "queries": 22}}}


class PublishReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"

    def add(self, day, sha, summary=SUMMARY):
        results = self.root / f"{day}.json"
        results.write_text(json.dumps(fake(day, sha)))
        page = self.root / f"{day}.html"
        page.write_text(f"<p>report {day}</p>")
        return publish_report.publish(results, page, self.site, summary=summary)

    def test_index_lists_reports_newest_first(self):
        first = self.add("2026-10-18", "a")
        self.add("2026-10-25", "b")
        self.assertEqual(first.name, "2026-10-18-aaaaaaa")
        self.assertEqual(
            (first / "index.html").read_text(), "<p>report 2026-10-18</p>"
        )
        self.assertTrue((first / "results.json").exists())
        index = (self.site / "index.html").read_text()
        newer = index.index("cpu/2026-10-25-bbbbbbb/")
        older = index.index("cpu/2026-10-18-aaaaaaa/")
        self.assertLess(newer, older)
        for text in ("Mojo 1.2.0", "polars 1.44.2", "duckdb 1.5.5", "0.90"):
            self.assertIn(text, index)
        self.assertIn("queries-per-path table", index)

    def test_republishing_replaces_and_does_not_duplicate(self):
        self.add("2026-10-18", "a")
        self.add("2026-10-18", "a", summary={})
        self.assertEqual(len(list((self.site / "cpu").iterdir())), 1)
        index = (self.site / "index.html").read_text()
        self.assertEqual(index.count("cpu/2026-10-18-aaaaaaa/"), 1)

    def test_label_makes_a_separate_report(self):
        results = self.root / "r.json"
        results.write_text(json.dumps(fake("2026-10-18", "a")))
        page = self.root / "r.html"
        page.write_text("x")
        publish_report.publish(results, page, self.site, summary={})
        publish_report.publish(results, page, self.site, "quick run", {})
        names = sorted(p.name for p in (self.site / "cpu").iterdir())
        self.assertEqual(
            names, ["2026-10-18-aaaaaaa", "2026-10-18-aaaaaaa-quick-run"]
        )


if __name__ == "__main__":
    unittest.main()
