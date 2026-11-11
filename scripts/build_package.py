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
from metal_build_identity import metal_build_id, metal_identity_source

ROOT = Path(__file__).resolve().parents[1]
JSON_PACKAGES = ("emberjson", "emberserde")


def dependency_directory() -> Path:
    """Installed packages beside the compiler selected on PATH."""
    compiler = shutil.which("mojo")
    if compiler is None:
        raise RuntimeError("mojo is not installed; run pixi install first")
    return Path(compiler).absolute().parent.parent / "lib/mojo"


def default_provider() -> str:
    mac = platform.system() == "Darwin" and platform.machine() == "arm64"
    return "metal" if mac else "cpu"


def build(output: Path, provider: str) -> None:
    compiler = subprocess.check_output(["mojo", "--version"], text=True).strip()
    identity = cpu_build_id(ROOT, compiler)
    metal_identity = (
        metal_build_id(ROOT, compiler) if provider == "metal" else None
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataframe-package-") as directory:
        stage = Path(directory)
        shutil.copytree(ROOT / "dataframe", stage / "dataframe")
        # The root provider is Metal; the CPU-only one lives in providers/cpu.
        shutil.copytree(
            ROOT / "providers/cpu/dataframe_provider" if provider
            == "cpu" else ROOT / "dataframe_provider",
            stage / "dataframe_provider",
        )
        packages = ["dataframe", "dataframe_provider"]
        dependencies = dependency_directory()
        for package in JSON_PACKAGES:
            shutil.copyfile(
                dependencies / f"{package}.mojoc",
                stage / f"{package}.mojoc",
            )
        if provider == "metal":
            shutil.copytree(ROOT / "dataframe_metal", stage / "dataframe_metal")
            (stage / "dataframe_metal/_build_identity.mojo").write_text(
                metal_identity_source(metal_identity)
            )
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
        packages.extend(JSON_PACKAGES)
        for package in packages:
            shutil.copyfile(
                stage / f"{package}.mojoc", output / f"{package}.mojoc.tmp"
            )
            (output / f"{package}.mojoc.tmp").replace(
                output / f"{package}.mojoc"
            )
        shutil.copyfile(
            ROOT / "scripts/vendor/ember-LICENSE-APACHE",
            output / "ember-LICENSE-APACHE",
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
                "scripts/metal_build_identity.py",
                "packages",
                "pixi.lock",
            ],
            text=True,
        ).strip()
    )
    metadata = {
        "provider": "core-with-mojo-metal" if provider
        == "metal" else "core-cpu",
        "implementation": "mojo",
        "runtime_build_id": (
            "dataframe-mojo-metal-v2;cpu="
            + identity
            + ";metal="
            + metal_identity if metal_identity else "dataframe-mojo-metal-v1;source-compiler-sdk="
            + identity
        ),
        "metal_build_id": metal_identity,
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
