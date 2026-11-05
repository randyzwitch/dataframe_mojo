"""Source/compiler identity for measured Metal handoff costs.

The CPU builder already stages CPU_BUILD_ID. Keep the Metal identity separate:
changing a kernel or the provider must invalidate Metal costs even when CPU
sources are unchanged. Generated identity files cannot hash themselves.
"""
import hashlib
from pathlib import Path

from build_identity import cpu_build_id


def metal_build_id(
    root: Path, compiler: str, flags: tuple[str, ...] = ()
) -> str:
    digest = hashlib.sha256(b"dataframe-metal-build-v1\0")
    digest.update(cpu_build_id(root, compiler, flags).encode() + b"\0")
    for directory in ("dataframe_metal", "dataframe_provider"):
        for source in sorted((root / directory).rglob("*.mojo")):
            if source.name == "_build_identity.mojo":
                continue
            digest.update(source.relative_to(root).as_posix().encode() + b"\0")
            digest.update(source.read_bytes() + b"\0")
    return digest.hexdigest()


def metal_identity_source(identity: str) -> str:
    if len(identity) != 64 or any(
        c not in "0123456789abcdef" for c in identity
    ):
        raise ValueError("Metal build identity must be a SHA256 hex digest")
    return (
        '"""Packaged Metal source/compiler/settings identity."""\n'
        f'comptime METAL_BUILD_ID = "{identity}"\n'
    )
