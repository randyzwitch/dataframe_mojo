#!/usr/bin/env python3
"""Map every dataframe module to the contract groups that import it.

    python3 scripts/check_module_coverage.py --write   # regenerate the map
    python3 scripts/check_module_coverage.py           # check it (CI)

A module is directly tested when a tests/<area>/test_*.mojo file, or a
tests/support module that such a test imports, imports it by name. The check
fails when tests/COVERAGE.md is stale or when tests/coverage_gaps.txt
disagrees with the modules no test imports.

Edit tests/coverage_gaps.txt by hand, then run --write: it rewrites that file
in the canonical sorted order (never adding or removing entries), so ordering
alone never fails the check.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "tests" / "COVERAGE.md"
BASELINE = ROOT / "tests" / "coverage_gaps.txt"
COMMAND = "python3 scripts/check_module_coverage.py --write"
IMPORT = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.M)
GROUP = re.compile(r"^# contract-group:\s*(\S+)", re.M)
SECTIONS = [
    "dataframe",
    "dataframe/io",
    "dataframe/regex",
    "dataframe/accelerator",
    "dataframe/accelerator/gpu",
]


def modules():
    found = {}
    for path in sorted((ROOT / "dataframe").rglob("*.mojo")):
        if path.name == "__init__.mojo":
            continue
        rel = path.relative_to(ROOT).as_posix()
        found[rel[: -len(".mojo")].replace("/", ".")] = rel
    return found


def imports(text):
    return set(IMPORT.findall(text))


def reach(known):
    """Return {module name: set of contract groups}."""
    tests = ROOT / "tests"
    support = {}
    for path in (tests / "support").glob("*.mojo"):
        support["tests.support." + path.stem] = imports(path.read_text())
    reached = {name: set() for name in known}
    for path in sorted(tests.glob("*/test_*.mojo")):
        text = path.read_text()
        groups = set(GROUP.findall(text)) or {"unmarked:" + path.parent.name}
        direct = imports(text)
        names = set(direct)
        for name in direct:
            names |= support.get(name, set())
        for name in names:
            if name in reached:
                reached[name] |= groups
    return reached


def section(rel):
    parent = rel.rsplit("/", 1)[0]
    return parent if parent in SECTIONS else None


def render(known, reached):
    tested = [n for n in known if reached[n]]
    gaps = [n for n in known if not reached[n]]
    out = [
        "# Module coverage map",
        "",
        "This file lists, for each module under `dataframe/`, the contract "
        "groups whose tests import it by name. Direct imports are a proxy: "
        "tests that go through the public DataFrame and LazyFrame API "
        "exercise far more code than this shows, and a module that is "
        "imported is not necessarily well tested. Execution-path coverage "
        "from `dataframe/trace.mojo` is a separate check. A test file counts "
        "when it is `tests/<area>/test_*.mojo` and its groups are its "
        "`# contract-group:` markers (`unmarked:<area>` when it has none). "
        "Support modules count one level deep: a module imported by a "
        "`tests/support` file that a test imports is credited to that test's "
        "groups. `__init__.mojo` files are skipped.",
        "",
        f"Generate with `{COMMAND}`. "
        "The check (no flag) fails when this file or "
        "`tests/coverage_gaps.txt` is out of date.",
        "",
        f"{len(known)} modules, {len(tested)} imported directly, "
        f"{len(gaps)} not.",
        "",
        "| module | contract groups |",
        "| --- | --- |",
    ]
    for name in tested:
        out.append(f"| `{known[name]}` | {', '.join(sorted(reached[name]))} |")
    out += ["", "## Modules no test imports directly", ""]
    for sec in SECTIONS:
        members = [known[n] for n in gaps if section(known[n]) == sec]
        out.append(f"### `{sec}/` ({len(members)})")
        out.append("")
        out += [f"- `{m}`" for m in members] or ["- none"]
        out.append("")
    return "\n".join(out).rstrip() + "\n", [known[n] for n in gaps]


def main():
    known = modules()
    text, gaps = render(known, reach(known))
    gaps = sorted(gaps)
    if "--write" in sys.argv[1:]:
        MAP.write_text(text)
        if BASELINE.exists():
            entries = {
                l.strip() for l in BASELINE.read_text().splitlines() if l.strip()
            }
            BASELINE.write_text("".join(f"{e}\n" for e in sorted(entries)))
        print(f"wrote {MAP.relative_to(ROOT)} ({len(gaps)} gaps)")
        return 0
    errors = []
    if not MAP.exists() or MAP.read_text() != text:
        errors.append(f"tests/COVERAGE.md is stale; run `{COMMAND}`.")
    base = (
        {l.strip() for l in BASELINE.read_text().splitlines() if l.strip()}
        if BASELINE.exists()
        else set()
    )
    if sorted(base) != sorted(set(base)) or BASELINE.exists() and (
        BASELINE.read_text().splitlines() != sorted(base)
    ):
        errors.append("tests/coverage_gaps.txt must be sorted, no blanks.")
    for rel in sorted(set(gaps) - base):
        errors.append(
            f"{rel}: new untested module; add a test or, with a reason, "
            "add it to tests/coverage_gaps.txt."
        )
    for rel in sorted(base - set(gaps)):
        errors.append(
            f"{rel}: now imported by a test (or gone); remove it from "
            "tests/coverage_gaps.txt."
        )
    for e in errors:
        print("error:", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
