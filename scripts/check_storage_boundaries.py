#!/usr/bin/env python3
"""Check storage ownership and synchronized metadata access."""
from pathlib import Path
import re
import sys
from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]
FACTS = "dataframe.storage.facts"


def check():
    failures = []
    for path in sorted((ROOT / "dataframe/storage").rglob("*.mojo")):
        name = module_name(path)
        if name == FACTS:
            for dependency in imports(path):
                if dependency.startswith("dataframe"):
                    failures.append(f"{name} imports an engine layer: {dependency}")
    methods = {
        "first", "last", "order_state", "publish_order", "progression_state",
        "publish_progression", "zones_ready", "zones", "publish_zones",
        "initialize_window",
    }
    for tree in ("dataframe", "dataframe_metal", "tests"):
        for path in sorted((ROOT / tree).rglob("*.mojo")):
            text = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", path.read_text())
            for match in re.finditer(r"\._facts\[\]\.(\w+)\s*(\()?", text):
                member, call = match.groups()
                if member not in methods or not call:
                    failures.append(f"{path.relative_to(ROOT)} accesses raw buffer facts: {member}")
                if member == "initialize_window" and path != ROOT / "dataframe/column.mojo":
                    failures.append(f"{path.relative_to(ROOT)} changes a published facts window")
            if path != ROOT / "dataframe/column.mojo" and re.search(
                r"unsafe_from_address\s*=\s*Int\([^\n]*\._facts\.ptr\(\)", text
            ):
                failures.append(f"{path.relative_to(ROOT)} casts the buffer facts address")
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print("Storage facts boundary holds.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
