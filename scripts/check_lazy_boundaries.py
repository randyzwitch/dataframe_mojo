#!/usr/bin/env python3
"""Check dependency direction within the lazy query implementation."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("planning", "optimizer", "query_execution")


def module_name(path):
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def imports(path):
    text = path.read_text()
    # Docstrings and comments cannot introduce dependencies.
    text = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", text)

    def resolve(name):
        if name.startswith("."):
            dots = len(name) - len(name.lstrip("."))
            parent = module_name(path).split(".")[:-dots]
            name = ".".join(parent + [name.lstrip(".")]).rstrip(".")
        return name

    for match in re.finditer(
        r"^\s*from\s+([.\w]+)\s+import\s+(\([^)]*\)|[^\n]+)", text, re.M
    ):
        name = resolve(match[1])
        yield name
        # Include module imports such as `from . import graph`, as well as
        # imports directly from graph. Non-module members never enter the graph.
        for member in match[2].strip("()").split(","):
            member = member.strip().split(" as ")[0]
            if member:
                yield name + "." + member
    for match in re.finditer(r"^\s*import\s+([.\w]+)", text, re.M):
        yield resolve(match[1])


def check():
    paths = [ROOT / "dataframe/lazy_api.mojo"]
    for package in PACKAGES:
        paths.extend(sorted((ROOT / "dataframe" / package).rglob("*.mojo")))
    sources = {module_name(path): path for path in paths}
    graph = {}
    failures = []
    for name, path in sources.items():
        dependencies = set(imports(path))
        graph[name] = dependencies & sources.keys()
        if name == "dataframe.lazy_api":
            continue
        for dependency in sorted(dependencies):
            if dependency == "dataframe":
                failures.append(f"{name} imports the public package facade")
            if dependency in {"dataframe.lazy", "dataframe.lazy_api"}:
                failures.append(f"{name} imports the public query API: {dependency}")
            if dependency in {"dataframe._accel_provider", "dataframe_provider"}:
                failures.append(f"{name} imports provider dispatch: {dependency}")
            if name.startswith(("dataframe.planning.", "dataframe.optimizer.")):
                if dependency.startswith("dataframe.query_execution"):
                    failures.append(f"{name} imports query execution: {dependency}")
            if name.startswith("dataframe.planning.") and name != "dataframe.planning.physical":
                if dependency.startswith("dataframe.optimizer"):
                    failures.append(f"{name} imports optimizer passes: {dependency}")
            if name.startswith("dataframe.planning.cost."):
                if dependency == "dataframe.planning.physical":
                    failures.append(f"{name} imports physical orchestration: {dependency}")
                if name.endswith((".model", ".eligibility", ".sampling")) and dependency == "dataframe.planning.cost.aggregation":
                    failures.append(f"{name} imports higher strategy policy: {dependency}")
                if name.endswith((".model", ".eligibility")) and dependency in {
                    "dataframe.planning.properties", "dataframe.planning.cost.sampling",
                }:
                    failures.append(f"{name} mixes semantic eligibility with cost observations: {dependency}")
            if name == "dataframe.planning.model" and dependency in sources:
                failures.append(f"the node model imports a higher query layer: {dependency}")

    stream_source = (ROOT / "dataframe/query_execution/pipeline.mojo").read_text()
    stream_source = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", stream_source)
    if re.search(r"\bvar\s+(piped|insert|table|eager)\s*=", stream_source):
        failures.append("stream execution declares independent physical-strategy flags")

    visited = set()
    active = []

    def visit(name):
        if name in active:
            cycle = active[active.index(name) :] + [name]
            failures.append("query module cycle: " + " -> ".join(cycle))
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
    print(f"Lazy query boundaries hold ({len(sources)} modules; no internal cycles).")
    return 0


if __name__ == "__main__":
    sys.exit(check())
