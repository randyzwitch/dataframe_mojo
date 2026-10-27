#!/usr/bin/env python3
"""Check eager join implementation ownership and import direction."""

from pathlib import Path
import sys

from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]


def check():
    sources = {
        module_name(path): path
        for path in sorted((ROOT / "dataframe/joins").glob("*.mojo"))
    }
    graph = {}
    failures = []
    for name, path in sources.items():
        dependencies = set(imports(path))
        graph[name] = dependencies & sources.keys()
        for dependency in sorted(dependencies):
            if dependency.startswith("dataframe.frame.") and dependency != "dataframe.frame.DataFrame":
                failures.append(f"{name} imports frame implementation: {dependency}")
            if dependency in {"dataframe", "dataframe.lazy", "dataframe.lazy_api", "dataframe.streaming"} or dependency.startswith(
                ("dataframe.query_execution", "dataframe_provider", "dataframe._accel_provider")
            ):
                failures.append(f"{name} imports a higher layer: {dependency}")

    visited = set()
    active = []

    def visit(name):
        if name in active:
            failures.append("join module cycle: " + " -> ".join(active + [name]))
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
    print(f"Join boundaries hold ({len(sources)} modules; no internal cycles).")
    return 0


if __name__ == "__main__":
    sys.exit(check())
