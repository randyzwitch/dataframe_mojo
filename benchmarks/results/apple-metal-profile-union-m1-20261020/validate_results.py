"""Check archived profile regression evidence; does not execute Metal."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((ROOT / "receipt.json").read_text())
    assert receipt["revision"] == "d241b736141b764f0bb9f7ec85aafeca7b5e590f"
    assert receipt["engine_tree"] == "2d9b8558ddc55b46b5e866e82487177412ff87a1"
    assert receipt["build_exit"] == receipt["test_exit"] == 0
    assert sha(ROOT / "run_gate.py") == receipt["runner_sha256"]
    for name in ["build.log", "test.log"]:
        assert sha(ROOT / name) == receipt["files"][name], name
    assert (ROOT / "test.log").read_text().splitlines() == [
        "PASS profile region schema union and optional metrics",
        "PASS native join followed by scalar SUM profile",
        "PASS host phase attribution and mixed report schemas",
        "PASS resident results, immutable branches, exports and budgets",
    ]
    assert json.loads((ROOT / "status.json").read_text()) == {
        "build_exit": 0, "test_exit": 0,
    }
    plan = json.loads((ROOT / "plan.json").read_text())
    for field in ["revision", "bundle_sha256", "lock_sha256"]:
        assert plan[field] == receipt[field]
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(ROOT / name) == digest, name
    print("PASS archived profile schema regression evidence (no GPU run)")


if __name__ == "__main__":
    main()
