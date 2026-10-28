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
    legacy = {
        "dataframe._buffer", "dataframe.column", "dataframe.bool_column",
        "dataframe.string_column", "dataframe.string_view", "dataframe.nested_column",
        "dataframe.nested_merge", "dataframe.field_metadata", "dataframe.huge_pages",
    }
    paths = sorted((ROOT / "dataframe/storage").rglob("*.mojo"))
    sources = {module_name(path): path for path in paths}
    graph = {}
    for name, path in sources.items():
        dependencies = set(imports(path))
        graph[name] = dependencies & sources.keys()
        allowed = {"dataframe.dtype"}
        # Nested values recursively hold Series; categorical child append uses
        # the existing dictionary unifier. These are value-representation seams,
        # not permission for storage to depend on query/expression execution.
        if name in {"dataframe.storage.nested", "dataframe.storage.nested_merge"}:
            allowed.add("dataframe.series")
        if name == "dataframe.storage.nested":
            allowed.add("dataframe.categorical")
        for dependency in dependencies:
            if not dependency.startswith("dataframe"):
                continue
            if dependency.startswith("dataframe.storage."):
                continue
            if name != FACTS and any(dependency == item or dependency.startswith(item + ".") for item in allowed):
                continue
            failures.append(f"{name} imports outside storage/value representation: {dependency}")
    visited, active = set(), []
    def visit(name):
        if name in active:
            failures.append("storage cycle: " + " -> ".join(active + [name]))
            return
        if name in visited:
            return
        active.append(name)
        for dependency in sorted(graph[name]):
            visit(dependency)
        active.pop()
        visited.add(name)
    for name in sources:
        visit(name)
    for tree in ("dataframe", "dataframe_metal", "dataframe_provider", "providers"):
        for path in sorted((ROOT / tree).rglob("*.mojo")):
            name = module_name(path)
            if name in legacy:
                source = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", path.read_text())
                if re.search(r"^(?:def|struct|trait|comptime)\s+", source, re.M):
                    failures.append(f"{name} defines implementation in a compatibility facade")
                continue
            for dependency in sorted(set(imports(path)) & legacy):
                failures.append(f"{name} imports legacy storage facade: {dependency}")
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
                if member == "initialize_window" and path != ROOT / "dataframe/storage/column.mojo":
                    failures.append(f"{path.relative_to(ROOT)} changes a published facts window")
            if path != ROOT / "dataframe/storage/column.mojo" and re.search(
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
