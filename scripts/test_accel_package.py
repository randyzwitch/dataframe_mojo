#!/usr/bin/env python3
"""Check provider packaging with an application outside the source checkout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(package_dir: Path, provider: str) -> None:
    package_dir = package_dir.resolve()
    with tempfile.TemporaryDirectory(
        prefix="dataframe-provider-consumer-"
    ) as directory:
        work = Path(directory)
        source = work / "app.mojo"
        shutil.copyfile(ROOT / "tests" / "accel" / "registered_provider.mojo", source)
        binary = work / "app"
        subprocess.run(
            ["mojo", "build", "-I", str(package_dir), str(source), "-o", str(binary)],
            cwd=work,
            check=True,
        )

        def run(mode: str, **settings: str) -> None:
            env = os.environ.copy()
            env.pop("DATAFRAME_ACCEL_DEVICE", None)
            env.pop("DATAFRAME_ACCEL_MEMORY_LIMIT", None)
            env.update(settings)
            subprocess.run([str(binary), mode], cwd=work, env=env, check=True)

        if provider == "cpu":
            run("cpu", DATAFRAME_ACCEL_DEVICE="invalid")
        else:
            run("gpu")
            run("unavailable", CUDA_VISIBLE_DEVICES="")
            run("unavailable", DATAFRAME_ACCEL_DEVICE="2147483647")
            run("invalid-device", DATAFRAME_ACCEL_DEVICE="invalid")
            run("invalid-device", DATAFRAME_ACCEL_DEVICE="-1")
            run("unsupported", DATAFRAME_ACCEL_DEVICE="invalid")
            run("budget", DATAFRAME_ACCEL_MEMORY_LIMIT="0")
            run("invalid-memory", DATAFRAME_ACCEL_MEMORY_LIMIT="invalid")
            run("invalid-memory", DATAFRAME_ACCEL_MEMORY_LIMIT="-1")
            override_source = work / "override.mojo"
            shutil.copyfile(
                ROOT / "tests" / "accel" / "registered_override.mojo", override_source
            )
            override_binary = work / "override"
            subprocess.run(
                [
                    "mojo",
                    "build",
                    "-I",
                    str(package_dir),
                    str(override_source),
                    "-o",
                    str(override_binary),
                ],
                cwd=work,
                check=True,
            )
            env = os.environ.copy()
            env["DATAFRAME_ACCEL_DEVICE"] = "invalid"
            env["DATAFRAME_ACCEL_MEMORY_LIMIT"] = "invalid"
            subprocess.run([str(override_binary)], cwd=work, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "nvidia"), required=True)
    args = parser.parse_args()
    check(args.package_dir, args.provider)
