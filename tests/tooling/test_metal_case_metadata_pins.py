"""Pinned case-metadata hashes must match the current suite sources."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
import bench_metal_suites as suites  # noqa: E402

METADATA = ROOT / "benchmarks/tools/metal_case_metadata.json"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class CaseMetadataPins(unittest.TestCase):
    def test_pins_match_current_sources(self):
        cases = json.loads(METADATA.read_text())["cases"]
        stale = []
        for name, entry in sorted(cases.items()):
            path = ROOT / entry["source_path"]
            source = path.read_text()
            if digest(source) != entry["source_sha256"]:
                stale.append(name + ": canonical source_sha256")
            pin = entry.get("metal_derived_source_sha256")
            if pin is not None:
                # The coverage run derives the Metal worker with the audit
                # runtime; its hash is what the report compares against.
                derived = suites.derive(
                    source, path.stem, "metal", audit_runtime=True
                )
                if digest(derived) != pin:
                    stale.append(name + ": metal_derived_source_sha256")
        self.assertEqual(
            stale,
            [],
            "Stale pins in metal_case_metadata.json; re-verify each case's "
            "audit_reason against the current suite and update the hashes.",
        )


if __name__ == "__main__":
    unittest.main()
