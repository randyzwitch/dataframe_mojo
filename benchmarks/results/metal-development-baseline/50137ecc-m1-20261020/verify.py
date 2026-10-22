"""Verify archived baseline identities without executing queries or extracting tar files."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest.items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    performance_bytes = gzip.decompress((ROOT / "performance.json.gz").read_bytes())
    coverage_bytes = gzip.decompress((ROOT / "coverage.json.gz").read_bytes())
    data = json.loads(performance_bytes)
    audit = json.loads((ROOT / "audit.json").read_text())
    assert sha(performance_bytes) == audit["performance_sha256"]
    assert sha(coverage_bytes) == audit["receipt_sha256"]
    assert sha(coverage_bytes) == data["provenance"]["strict_coverage_receipts"][0]["sha256"]
    keys = [(r["suite"], r["variant"], r["query"], r["engine"], r["round"]) for r in data["runs"]]
    assert len(keys) == len(set(keys)) == 1812
    assert len(data["cases"]) == 302
    assert data["all_scheduled_workers_attempted"]
    assert data["engine_unchanged_after"] and data["inputs_unchanged_after"]
    with tarfile.open(ROOT / "worker-evidence.tar.gz") as archive:
        for run in data["runs"]:
            assert sha(archive.extractfile(run["log"]).read()) == run["log_sha256"], run["log"]
    with tarfile.open(ROOT / "measurement-harness.tar.gz") as archive:
        for name, expected in data["provenance"]["harness_sha256"].items():
            assert sha(archive.extractfile(name).read()) == expected, name
    print("PASS package, original receipts, 1812 unique workers, worker logs and harness hashes")


if __name__ == "__main__":
    main()
