#!/usr/bin/env python3
"""Build an optional provider-enabled Mojo distribution without editing core.

Run in the pinned GPU environment. Both output packages are required; use the
output directory as the application's Mojo import path, without the source
checkout on that path. Applications continue importing only `dataframe`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataframe-accel-") as directory:
        stage = Path(directory)
        for package in ("dataframe", "dataframe_accel"):
            shutil.copytree(ROOT / package, stage / package)
        (stage / "dataframe" / "_accel_provider.mojo").write_text(
            '"""NVIDIA registration for the optional GPU distribution."""\n'
            "from dataframe_accel.provider import installed, execute, execute_profiled, describe\n"
        )
        artifacts = {}
        for package in ("dataframe", "dataframe_accel"):
            target = stage / f"{package}.mojoc"
            subprocess.run(
                [
                    "mojo",
                    "precompile",
                    "-I",
                    str(stage),
                    str(stage / package),
                    "-o",
                    str(target),
                ],
                check=True,
            )
            artifacts[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Publish only after both packages compile successfully.
        for name in artifacts:
            destination = output / name
            temporary = output / (name + ".tmp")
            shutil.copyfile(stage / name, temporary)
            os.replace(temporary, destination)
        metadata = {
            "provider": "nvidia",
            "packages": artifacts,
            "mojo": subprocess.check_output(["mojo", "--version"], text=True).strip(),
        }
        (output / "provider.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"NVIDIA provider registered in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "nvidia")
    build(parser.parse_args().output)
