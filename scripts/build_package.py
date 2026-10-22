#!/usr/bin/env python3
"""Stage a CPU-identified dataframe package without editing the source tree.

The default flavor is CPU-only: `dataframe.mojoc` and `dataframe_provider.mojoc`
(the stub provider). `--provider metal` also stages `dataframe_metal.mojoc` and
a `dataframe_provider.mojoc` that re-exports it. Use the output directory as
the import path (`-I dist`).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from build_identity import cpu_build_id, identity_source

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path, provider: str = "") -> None:
    compiler = subprocess.check_output(["mojo", "--version"], text=True).strip()
    identity = cpu_build_id(ROOT, compiler)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataframe-package-") as directory:
        stage = Path(directory)
        shutil.copytree(ROOT / "dataframe", stage / "dataframe")
        # The CPU stub unless a backend's provider directory is chosen.
        shutil.copytree(
            ROOT / "providers" / provider / "dataframe_provider"
            if provider
            else ROOT / "dataframe_provider",
            stage / "dataframe_provider",
        )
        packages = ["dataframe", "dataframe_provider"]
        if provider:
            shutil.copytree(
                ROOT / f"dataframe_{provider}", stage / f"dataframe_{provider}"
            )
            packages.append(f"dataframe_{provider}")
        (stage / "dataframe/_build_identity.mojo").write_text(
            identity_source(identity)
        )
        for package in packages:
            subprocess.run(
                [
                    "mojo",
                    "precompile",
                    "-I",
                    str(stage),
                    str(stage / package),
                    "-o",
                    str(stage / f"{package}.mojoc"),
                ],
                check=True,
            )
        # Publish only after every package compiles.
        for package in packages:
            shutil.copyfile(
                stage / f"{package}.mojoc", output / f"{package}.mojoc.tmp"
            )
            (output / f"{package}.mojoc.tmp").replace(
                output / f"{package}.mojoc"
            )
    (output / "cpu-build.json").write_text(
        json.dumps({"cpu_build_id": identity, "mojo": compiler}, indent=2)
        + "\n"
    )
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "dataframe",
                "dataframe_provider",
                "dataframe_metal",
                "providers",
                "scripts/build_package.py",
                "scripts/build_identity.py",
                "pixi.lock",
            ],
            text=True,
        ).strip()
    )
    metadata = {
        "provider": "core-with-mojo-metal" if provider else "core-cpu",
        "implementation": "mojo",
        "runtime_build_id": "dataframe-mojo-metal-v1;source-compiler-sdk="
        + identity,
        "cpu_build_id": identity,
        "engine_revision": revision,
        "core_revision": revision,
        "engine_dirty": dirty,
        "core_dirty": dirty,
        "mojo": compiler,
        "packages": {
            f"{package}.mojoc": hashlib.sha256(
                (output / f"{package}.mojoc").read_bytes()
            ).hexdigest()
            for package in packages
        },
    }
    (output / "provider.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--provider",
        choices=["metal"],
        default="",
        help="also package this backend and its provider (default: CPU only)",
    )
    args = parser.parse_args()
    build(args.output.resolve(), args.provider)
