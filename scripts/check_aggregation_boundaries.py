#!/usr/bin/env python3
"""Check ownership and import direction for shared aggregation machinery."""

from pathlib import Path
import re
import sys

from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]


def check():
    paths = sorted((ROOT / "dataframe/aggregation").glob("*.mojo"))
    sources = {module_name(path): path for path in paths}
    graph = {}
    failures = []
    # These existing value types and frame operations retain their definitions
    # during extraction. Aggregation must own its implementation dependencies.
    frame_members = {"DataFrame", "concat"}
    for name, path in sources.items():
        dependencies = set(imports(path))
        graph[name] = dependencies & sources.keys()
        text = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", path.read_text())
        for match in re.finditer(
            r"from dataframe.frame import (\([^)]*\)|[^\n]+)", text
        ):
            members = {x.strip() for x in match[1].strip("()").split(",") if x.strip()}
            for member in sorted(members - frame_members):
                failures.append(f"{name} imports implementation from frame: {member}")
        for dependency in sorted(dependencies):
            if dependency in {"dataframe", "dataframe.streaming"} or dependency.startswith(
                ("dataframe.query_execution", "dataframe_provider", "dataframe._accel_provider")
            ):
                failures.append(f"{name} imports a higher layer: {dependency}")

    visited = set()
    active = []

    def visit(name):
        if name in active:
            failures.append("aggregation cycle: " + " -> ".join(active + [name]))
            return
        if name in visited:
            return
        active.append(name)
        for dependency in sorted(graph[name]):
            visit(dependency)
        active.pop()
        visited.add(name)

    for name in sorted(graph):
        visit(name)
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"Aggregation boundaries hold ({len(sources)} modules; no internal cycles).")
    return 0


if __name__ == "__main__":
    sys.exit(check())
