"""Qualify the pinned profile schema union change on the serialized Mac."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path("/tmp/metal-profile-union-gate-oct20")
TREE = Path("/Users/randyzwitch/GitHub/dataframe_mojo-metal-profile-union-oct20")
MANIFEST = Path("/Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml")
PIXI = "/Users/randyzwitch/.pixi/bin/pixi"


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    assert not TREE.exists(), "Refusing to overwrite a prior gate tree"
    assert not (ROOT / "identity.json").exists()
    patterns = [r"\bmojo\s+(?:build|test)\b", r"\bbench_metal\.py\b",
                r"\brun_pairs\.py\b", r"metal-[^\s]*(?:queue|gate)\.py"]
    for line in capture(["ps", "-Ao", "pid=,command="]).splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if match and int(match[1]) != os.getpid():
            assert not any(re.search(p, match[2]) for p in patterns), line
    assert sha(ROOT / "source.bundle") == plan["bundle_sha256"]
    assert sha(MANIFEST.with_name("pixi.lock")) == plan["lock_sha256"]
    prefix = [PIXI, "run", "--manifest-path", str(MANIFEST), "--locked", "-e", "gpu"]
    version = capture(prefix + ["mojo", "--version"])
    assert version == "Mojo 1.2.0.dev2026100905 (3b71af75)", version
    compiler = capture(prefix + ["which", "mojo"])
    assert sha(compiler) == "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b"
    subprocess.run(["git", "clone", "-q", str(ROOT / "source.bundle"), str(TREE)], check=True)
    assert capture(["git", "rev-parse", "HEAD"], TREE) == plan["revision"]
    assert not capture(["git", "status", "--porcelain"], TREE)
    identity = dict(plan, compiler_version=version, compiler_sha256=sha(compiler),
                    runner_sha256=sha(__file__), xcode=capture(["xcodebuild", "-version"]),
                    sdk=capture(["xcrun", "--sdk", "macosx", "--show-sdk-version"]))
    (ROOT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    binary = ROOT / "gate"
    commands = [
        ("build", prefix + ["mojo", "build", "-I", str(TREE),
                           str(TREE / "tests/tooling/mojo_gpu_profile_phases.mojo"), "-o", str(binary)]),
        ("test", [str(binary)]),
    ]
    status = {}
    for label, command in commands:
        print("START", label, datetime.now(timezone.utc).isoformat(), flush=True)
        with (ROOT / (label + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=TREE, stdout=log, stderr=subprocess.STDOUT)
        status[label + "_exit"] = result.returncode
        (ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print("STATUS", label, result.returncode, flush=True)
        assert result.returncode == 0, "Gate failed; preserve artifacts"
    assert "PASS native join followed by scalar SUM profile" in (ROOT / "test.log").read_text()
    assert "PASS profile region schema union and optional metrics" in (ROOT / "test.log").read_text()
    assert capture(["git", "rev-parse", "HEAD"], TREE) == plan["revision"]
    assert not capture(["git", "status", "--porcelain"], TREE)
    receipt = dict(identity, **status,
                   observed_after_tests_utc=datetime.now(timezone.utc).isoformat(),
                   engine_tree=capture(["git", "rev-parse", "HEAD:dataframe"], TREE),
                   files={name: sha(ROOT / name) for name in ["gate", "build.log", "test.log"]})
    (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
