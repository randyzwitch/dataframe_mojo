#!/usr/bin/env python3
"""Fail when a program under tests/ is not run or imported by anything.

Every tracked .mojo, .py and .sh file under tests/tooling, tests/support and
tests/*.mojo needs at least one reference: its file name or module path in the
CI workflows, pixi.toml, scripts/, the runner files, or an import line of
another file under tests/, or by a file that is itself reachable.
test_*.py files under tests/tooling are run by a
glob in CI, so they count as reachable. Files in the six area directories are
found by the contract runner and are not checked.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATHS = [
    ".github/workflows",
    "pixi.toml",
    "scripts",
    "tests/run_tests.sh",
    "tests/runner.mojo",
    "tests/tooling/contract_runner.mojo",
    "tests/tooling/suite_policy.mojo",
    "tests/tooling/package_smoke.sh",
]
AREA_DIRECTORIES = [
    "public_contracts",
    "storage_ownership",
    "kernel_equivalence",
    "planner_correctness",
    "execution_modes",
    "interchange",
]
IMPORT_LINE = re.compile(r"^\s*(import|from)\s")
CHECKED_SUFFIXES = {".mojo", ".py", ".sh"}
# A concrete program path as a task, script or workflow would spell it.
DANGLING = re.compile(r"tests/[A-Za-z0-9_./-]+\.(?:mojo|py|sh)")


def tracked(*prefixes):
    out = subprocess.run(
        ["git", "ls-files", *prefixes],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def read(path):
    try:
        return (ROOT / path).read_text(errors="replace")
    except OSError:
        return ""


def candidates():
    result = []
    for path in tracked("tests"):
        parts = path.split("/")
        if parts[-1] == "__init__.mojo":
            continue
        if Path(path).suffix not in CHECKED_SUFFIXES:
            continue
        in_dir = len(parts) > 2 and parts[1] in ("tooling", "support")
        top_mojo = len(parts) == 2 and path.endswith(".mojo")
        if in_dir or top_mojo:
            result.append(path)
    return result


def main():
    reference_text = ""
    for ref in REFERENCE_PATHS:
        for path in tracked(ref):
            reference_text += read(path) + "\n"
    # Files in the area directories are found by the contract runner, so
    # anything they mention by name or import counts as used.
    all_tests = [
        p
        for p in tracked("tests")
        if Path(p).suffix in {".mojo", ".py", ".sh"}
    ]
    areas = tuple("tests/" + a + "/" for a in AREA_DIRECTORIES)
    for path in all_tests:
        if path.startswith(areas):
            reference_text += read(path) + "\n"
    import_lines = {}
    for path in all_tests:
        if Path(path).suffix not in {".mojo", ".py"}:
            continue
        lines = [
            line for line in read(path).splitlines() if IMPORT_LINE.match(line)
        ]
        import_lines[path] = "\n".join(lines)

    def needles(path):
        suffix = Path(path).suffix
        return [Path(path).name, path[: -len(suffix)].replace("/", ".")]

    reachable = set()
    pending = candidates()
    changed = True
    while changed:
        changed = False
        for path in list(pending):
            name = Path(path).name
            if path.startswith("tests/tooling/") and re.fullmatch(
                r"test_.*\.py", name
            ):
                found = True
            else:
                found = any(n in reference_text for n in needles(path)) or any(
                    n in text
                    for other, text in import_lines.items()
                    if other != path
                    for n in needles(path)
                )
            if found:
                pending.remove(path)
                reachable.add(path)
                changed = True
                # A file that is run may name the files it runs.
                reference_text += read(path) + "\n"
    offenders = pending

    # The reverse failure: a pixi task, script or workflow naming a program
    # under tests/ that no longer exists (PR #920 deleted probes that #916
    # had just wired into the Mac script, and nothing caught it).
    dangling = []
    for ref in REFERENCE_PATHS:
        for path in tracked(ref):
            for match in sorted(set(DANGLING.findall(read(path)))):
                if not (ROOT / match).exists():
                    dangling.append((path, match))

    if offenders:
        print("Programs under tests/ that nothing runs or imports:")
        for path in offenders:
            print("  " + path)
        print("Delete them, or run them from CI, a pixi task or a script.")
    if dangling:
        print("References to programs under tests/ that do not exist:")
        for path, match in dangling:
            print("  " + path + " -> " + match)
        print("Restore the program or remove the reference.")
    if offenders or dangling:
        return 1
    print("Every program under tests/ is reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
