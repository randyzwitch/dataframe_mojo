#!/usr/bin/env python3
"""Keep the release version consistent across the files that state it.

A compiled Mojo package is version-locked and each tag targets one Mojo
version, so a tag that disagrees with pixi.toml ships a package labelled
with the wrong version and sends anyone following the README to a tag that
does not exist. This runs in CI, and with a tag argument before tagging.

Usage:
    python3 scripts/check_version.py            # internal consistency
    python3 scripts/check_version.py v0.1.2     # also match this tag
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fail(message: str) -> None:
    print(f"version check: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    pixi = (ROOT / "pixi.toml").read_text()
    readme = (ROOT / "README.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()

    # [workspace] and [package] each carry a version; they must agree.
    versions = re.findall(r"^version = \"([^\"]+)\"$", pixi, re.MULTILINE)
    if len(versions) != 2:
        fail(f"expected two version lines in pixi.toml, found {len(versions)}")
    if versions[0] != versions[1]:
        fail(f"pixi.toml versions disagree: {versions[0]} vs {versions[1]}")
    version = versions[0]

    # The changelog must have a dated section for it, not "Unreleased".
    if not re.search(rf"^## {re.escape(version)} - \d{{4}}-\d\d-\d\d$",
                     changelog, re.MULTILINE):
        fail(f"CHANGELOG.md has no dated '## {version} - YYYY-MM-DD' section")

    # Install snippets must point at this version's tag, or consumers follow
    # the README to a tag that does not exist.
    expected = f"v{version}"
    for name, text in (("README.md", readme), ("pixi.toml", pixi)):
        for found in re.findall(r"tag = \"(v[^\"]+)\"", text):
            if found != expected:
                fail(f"{name} install snippet says {found}, expected {expected}")

    if len(sys.argv) > 1:
        tag = sys.argv[1]
        if tag != expected:
            fail(f"tag {tag} does not match pixi.toml version {version}")
        print(f"version check: {version} consistent, matches tag {tag}")
        return
    print(f"version check: {version} consistent")


if __name__ == "__main__":
    main()
