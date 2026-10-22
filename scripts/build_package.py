#!/usr/bin/env python3
"""Stage a CPU-identified dataframe package without editing the source tree.

One artifact per platform, chosen here and not by the user. On macOS arm64 it
holds `dataframe.mojoc`, `dataframe_provider.mojoc` (re-exporting the Metal
provider) and `dataframe_metal.mojoc`. Elsewhere Metal cannot run, so it holds
`dataframe.mojoc` and `dataframe_provider.mojoc` with the CPU-only provider.
`--provider cpu|metal` overrides the choice for testing. Use the output
directory as the import path (`-I dist`).
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
from build_identity import cpu_build_id, identity_source

ROOT = Path(__file__).resolve().parents[1]


def default_provider() -> str:
    mac = platform.system() == "Darwin" and platform.machine() == "arm64"
    return "metal" if mac else "cpu"


def build(output: Path, provider: str) -> None:
    compiler = subprocess.check_output(["mojo", "--version"], text=True).strip()
    identity = cpu_build_id(ROOT, compiler)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataframe-package-") as directory:
        stage = Path(directory)
        shutil.copytree(ROOT / "dataframe", stage / "dataframe")
        # The root provider is Metal; the CPU-only one lives in providers/cpu.
        shutil.copytree(
            ROOT / "providers/cpu/dataframe_provider"
            if provider == "cpu"
            else ROOT / "dataframe_provider",
            stage / "dataframe_provider",
        )
        packages = ["dataframe", "dataframe_provider"]
        if provider == "metal":
            shutil.copytree(ROOT / "dataframe_metal", stage / "dataframe_metal")
            packages.append("dataframe_metal")
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
        "provider": "core-with-mojo-metal" if provider == "metal" else "core-cpu",
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
        choices=["cpu", "metal"],
        default=default_provider(),
        help="override the platform's provider (testing only)",
    )
    args = parser.parse_args()
    build(args.output.resolve(), args.provider)
