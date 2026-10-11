#!/usr/bin/env python3
"""Stage a CPU-identified dataframe package without editing the source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from build_identity import cpu_build_id, identity_source

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path) -> None:
    compiler = subprocess.check_output(["mojo", "--version"], text=True).strip()
    identity = cpu_build_id(ROOT, compiler)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataframe-package-") as directory:
        stage = Path(directory)
        shutil.copytree(ROOT / "dataframe", stage / "dataframe")
        (stage / "dataframe/_build_identity.mojo").write_text(identity_source(identity))
        subprocess.run(["mojo", "precompile", "-I", str(stage), str(stage / "dataframe"), "-o", str(stage / "dataframe.mojoc")], check=True)
        shutil.copyfile(stage / "dataframe.mojoc", output / "dataframe.mojoc.tmp")
        (output / "dataframe.mojoc.tmp").replace(output / "dataframe.mojoc")
    (output / "cpu-build.json").write_text(json.dumps({"cpu_build_id": identity, "mojo": compiler}, indent=2) + "\n")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "dataframe", "scripts/build_package.py", "scripts/build_identity.py", "pixi.lock"], text=True).strip())
    metadata = {"provider": "core-with-optional-native-metal", "cpu_build_id": identity,
                "engine_revision": revision, "core_revision": revision,
                "engine_dirty": dirty, "core_dirty": dirty, "mojo": compiler,
                "packages": {"dataframe.mojoc": hashlib.sha256((output / "dataframe.mojoc").read_bytes()).hexdigest()}}
    (output / "provider.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    build(parser.parse_args().output.resolve())
