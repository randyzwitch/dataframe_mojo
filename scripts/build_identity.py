"""Stable CPU source/compiler identity shared by optional package builders."""
import hashlib
from pathlib import Path


def cpu_build_id(root: Path, compiler: str, flags: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256(b"dataframe-cpu-build-v1\0")
    for source in sorted((root / "dataframe").rglob("*.mojo")):
        if source.name == "_build_identity.mojo":
            continue
        digest.update(str(source.relative_to(root)).encode() + b"\0")
        digest.update(source.read_bytes() + b"\0")
    digest.update((root / "pixi.lock").read_bytes())
    digest.update(compiler.encode() + b"\0")
    for flag in flags:
        digest.update(flag.encode() + b"\0")
    return digest.hexdigest()


def identity_source(identity: str) -> str:
    return ('"""Packaged CPU source/compiler/settings identity."""\n'
            f'comptime CPU_BUILD_ID = "{identity}"\n')
