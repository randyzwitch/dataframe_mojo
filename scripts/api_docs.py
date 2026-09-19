"""Render `mojo doc` JSON for the exported API as Markdown, or check coverage.

Usage:
    python3 scripts/api_docs.py API_JSON OUTPUT_MD
    python3 scripts/api_docs.py API_JSON --check

The public API is exactly the names imported in dataframe/__init__.mojo. The
check fails when an exported struct or function has no docstring summary, and
reports (without failing) the share of public methods that have one.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def exported_names() -> list[str]:
    text = (ROOT / "dataframe" / "__init__.mojo").read_text()
    names: list[str] = []
    for block in re.findall(r"from \.\w+ import (\([^)]*\)|[^\n]*)", text):
        for name in re.split(r"[(),\s]+", block):
            if name:
                names.append(name)
    return names


def index(decl: dict) -> dict[str, tuple[str, dict]]:
    found: dict[str, tuple[str, dict]] = {}
    for module in decl["modules"]:
        for struct in module["structs"]:
            found.setdefault(struct["name"], ("struct", struct))
        for function in module["functions"]:
            found.setdefault(function["name"], ("function", function))
    return found


def summary(item: dict) -> str:
    if item.get("summary"):
        return item["summary"]
    for overload in item.get("overloads", []):
        if overload.get("summary"):
            return overload["summary"]
    return ""


def public_methods(struct: dict) -> list[dict]:
    return [
        f
        for f in struct["functions"]
        if not f["name"].startswith("_") or f["name"] in ("__init__", "__getitem__")
        or (f["name"].startswith("__") and f["name"] not in ("__del__", "__copyinit__", "__moveinit__"))
    ]


def render(decl: dict, names: list[str], table: dict) -> str:
    lines = [
        "# API reference",
        "",
        "Generated from `mojo doc` by `scripts/api_docs.py`; covers every name",
        "exported by `dataframe/__init__.mojo`. Contracts live in the guides:",
        "[semantics](semantics.md), [expressions](expressions.md), [csv](csv.md).",
        "",
    ]
    for name in sorted(names, key=str.lower):
        if name not in table:
            continue
        kind, item = table[name]
        lines.append(f"## `{name}`")
        lines.append("")
        text = summary(item)
        if item.get("description"):
            text = (text + "\n\n" + item["description"]).strip()
        if text:
            lines += [text, ""]
        if kind == "function":
            for overload in item["overloads"]:
                lines += ["```mojo", overload["signature"], "```", ""]
            continue
        for method in public_methods(item):
            for overload in method["overloads"]:
                lines.append(f"- `{overload['signature']}`")
                if overload.get("summary"):
                    lines.append(f"  {overload['summary']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def check(names: list[str], table: dict) -> int:
    missing = [n for n in names if n in table and not summary(table[n][1])]
    unknown = [n for n in names if n not in table]
    total = documented = 0
    for name in names:
        if name in table and table[name][0] == "struct":
            for method in public_methods(table[name][1]):
                for overload in method["overloads"]:
                    total += 1
                    documented += bool(overload.get("summary"))
    print(f"exported symbols: {len(names)}; methods documented: {documented}/{total}")
    for name in unknown:
        print(f"not found in mojo doc output: {name}")
    for name in missing:
        print(f"missing docstring: {name}")
    return 1 if missing or unknown else 0


def main() -> int:
    decl = json.loads(Path(sys.argv[1]).read_text())["decl"]
    names = exported_names()
    table = index(decl)
    if sys.argv[2] == "--check":
        return check(names, table)
    Path(sys.argv[2]).write_text(render(decl, names, table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
