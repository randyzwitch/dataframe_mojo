"""Build a local Mermaid explorer from benchmark LazyFrame.explain() output.

Run with pixi run -e oracle python scripts/query_diagrams.py. Uses existing
smoke/base data only; never generates data or runs timed benchmarks. PDS-H
q11/q22 execute their scalar subqueries while constructing their final plans.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "suites"))
from datagen import data_root, PDSH_TABLES, TPCDS_TABLES

SUITES = {
    "h2o_groupby": range(1, 11), "h2o_join": range(1, 6),
    "pdsh": range(1, 23), "tpcds": range(1, 100), "clickbench": range(43),
}


def quote(label):
    # Mermaid numeric entities avoid interpreting quotes, markup or syntax.
    return ''.join(f"#{ord(c)};" if c in '"<>&#' else c for c in label)


def mermaid(plan):
    """Convert the root-first, two-space-indented explain tree to data flow."""
    lines = ["flowchart BT"]
    stack = []
    children = {}
    for i, line in enumerate(plan.strip().splitlines()):
        depth = (len(line) - len(line.lstrip())) // 2
        if depth > len(stack):
            raise ValueError(f"Invalid explain indentation: {line}")
        label = line.strip()
        style = "boundary" if "materialize" in label else "stream"
        if label.startswith("SCAN"):
            style = "scan"
        lines.append(f'  n{i}["{quote(label)}"]:::{style}')
        if depth:
            parent = stack[depth - 1]
            children[parent] = children.get(parent, 0) + 1
            edge = ""
            parent_label = plan.strip().splitlines()[parent].strip()
            if parent_label.startswith("JOIN"):
                edge = "|left|" if children[parent] == 1 else "|right|"
            lines.append(f"  n{i} -->{edge} n{parent}")
        stack[depth:] = [i]
    lines += [
        '  result["Materialized result DataFrame"]:::result',
        '  n0 --> result',
        '  classDef scan fill:#e0efff,stroke:#3266a1,color:#142e4c',
        '  classDef stream fill:#e0f4eb,stroke:#388467,color:#153e30',
        '  classDef boundary fill:#fff0d9,stroke:#b77c29,color:#593910',
        '  classDef result fill:#eae4fa,stroke:#7a61aa,color:#352450',
    ]
    return '\n'.join(lines)


def source_for(suite, q):
    name = "h2o" if suite.startswith("h2o") else suite
    path = ROOT / "benchmarks" / "suites" / f"{name}.mojo"
    text = path.read_text()
    if suite.startswith("h2o"):
        function = "groupby" if suite == "h2o_groupby" else "join"
        text = text.split(f"def {function}(", 1)[1].split('\ndef ', 1)[0]
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("    if "):
            continue
        matches = re.findall(r'(?:q|query) == "(q\d+)"', line)
        matches += [f"q{n}" for n in re.findall(r'n == (\d+)', line)]
        if q in matches:
            end = i + 1
            while end < len(lines):
                if re.match(r"    (?:if |raise )|^def ", lines[end]):
                    break
                end += 1
            return '\n'.join(lines[i:end]).rstrip()
    return "No Mojo translation for this query."


def eager_plan(suite, q):
    """Explicit eager operator sequences; these are not lazy explain output."""
    n = int(q[1:])
    if suite == "h2o_join":
        other, key, how = [
            ("small", "id1", "inner"), ("medium", "id2", "inner"),
            ("medium", "id2", "left"), ("medium", "id5", "inner"),
            ("big", "id3", "inner"),
        ][n - 1]
        return f"JOIN {how} on {key} [materialize]\n  SCAN x\n  SCAN {other}"
    labels = [
        "GROUP_BY id1 AGG sum(v1)", "GROUP_BY id1, id2 AGG sum(v1)",
        "GROUP_BY id3 AGG sum(v1), mean(v3)",
        "GROUP_BY id4 AGG mean(v1), mean(v2), mean(v3)",
        "GROUP_BY id6 AGG sum(v1), sum(v2), sum(v3)",
        "GROUP_BY id4, id5 AGG median(v3), std(v3)",
        "GROUP_BY id3 AGG max(v1) - min(v2)", "",
        "GROUP_BY id2, id4 AGG corr(v1, v2) squared",
        "GROUP_BY id1, id2, id3, id4, id5, id6 AGG sum(v3), len(v1)",
    ]
    if n == 8:
        return ("SELECT id6, v3 as largest2_v3 [materialize]\n"
                "  FILTER ordinal rank(v3 descending) over id6 <= 2 [materialize]\n"
                "    FILTER v3 is not null [materialize]\n      SCAN x")
    return labels[n - 1] + " [materialize]\n  SCAN x"


def parse_output(text):
    records = {}
    for block in text.split("QUERY\t")[1:]:
        q, body = block.split('\n', 1)
        body = body.split("END_QUERY", 1)[0]
        if "ERROR\t" in body:
            records[q] = {"error": body.split("ERROR\t", 1)[1].strip()}
        else:
            original, optimized = body.split("ORIGINAL\n", 1)[1].split("OPTIMIZED\n", 1)
            records[q] = {"original": original.strip(), "optimized": optimized.strip()}
    return records


def build_catalog(work, root):
    mojo = shutil.which("mojo") or str(ROOT / ".pixi/envs/default/bin/mojo")
    # ClickBench collects only at return sites (including its top helper).
    # Produce an inspection-only module without changing the benchmark source.
    click = (ROOT / "benchmarks/suites/clickbench.mojo").read_text()
    click = click.split('\ndef main()', 1)[0]
    if click.count('raises -> DataFrame:') != 2:
        raise ValueError("ClickBench signatures changed; review the plan adapter")
    click = click.replace('raises -> DataFrame:', 'raises -> LazyFrame:')
    click = click.replace('.collect()', '')
    (work / "clickbench_plan.mojo").write_text(click)
    binary = work / "query_plans"
    subprocess.run([mojo, "build", "-I", str(ROOT), "-I", str(ROOT / "benchmarks/suites"),
                    "-I", str(work), str(ROOT / "scripts/query_plans.mojo"),
                    "-o", str(binary)], check=True, cwd=ROOT)
    catalog = {}
    for suite, numbers in SUITES.items():
        queries = [f"q{i}" for i in numbers]
        if suite.startswith("h2o"):
            records = {q: {"original": eager_plan(suite, q), "optimized": eager_plan(suite, q)} for q in queries}
        else:
            if suite == "clickbench":
                paths = {"hits": root / "clickbench/hits_1.parquet"}
            else:
                tables = PDSH_TABLES if suite == "pdsh" else TPCDS_TABLES
                paths = {name: root / suite / "sf0.01" / f"{name}.parquet" for name in tables}
            missing = [str(p) for p in paths.values() if not p.is_file()]
            if missing:
                raise FileNotFoundError("Existing smoke/base benchmark data required: " + ', '.join(missing))
            print(f"Inspecting {suite} ({len(queries)} queries)", flush=True)
            env = dict(os.environ)
            env.setdefault("DATAFRAME_PARQUET_LIBRARY", str(ROOT / "build/dfparquet/libdfparquet.so"))
            output = subprocess.check_output([str(binary), suite, ','.join(queries), "0",
                                              *[f"{k}={v}" for k, v in paths.items()]],
                                             env=env, text=True, cwd=ROOT)
            records = parse_output(output)
            if set(records) != set(queries):
                raise ValueError(f"Incomplete plan capture for {suite}")
        for q, record in records.items():
            record["source"] = source_for(suite, q)
            record["eager"] = suite.startswith("h2o")
            for view in ("original", "optimized"):
                if view in record:
                    record[view + "_mermaid"] = mermaid(record[view])
        catalog[suite] = records
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--output", type=Path, default=ROOT / "docs/query-diagrams.html")
    parser.add_argument("--from-json", type=Path, help="Rerender a previously captured catalog without Mojo or data")
    args = parser.parse_args()
    work = ROOT / "build/query-plans"
    work.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        payload = json.loads(args.from_json.read_text())
    else:
        catalog = build_catalog(work, args.data_root)
        sources = sorted((ROOT / 'dataframe').glob('*.mojo')) + sorted((ROOT / 'benchmarks/suites').glob('*.mojo'))
        fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
        payload = {"catalog": catalog, "fingerprint": fingerprint,
                   "revision": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   "dirty": bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))}
        (work / 'plans.json').write_text(json.dumps(payload, indent=2) + '\n')
    template = (ROOT / 'scripts/query_diagrams.html').read_text()
    data = json.dumps(payload).replace('<', '\\u003c').replace('&', '\\u0026')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / 'docs/vendor'
    destination = args.output.parent / 'vendor'
    if destination.resolve() != vendor.resolve():
        shutil.copytree(vendor, destination, dirs_exist_ok=True)
    args.output.write_text(template.replace('__PLAN_DATA__', data))
    print(args.output)


if __name__ == '__main__':
    main()
