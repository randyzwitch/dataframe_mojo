#!/usr/bin/env python3
"""Keep a local history of CPU benchmark reports and rebuild its index.

    python3 benchmarks/tools/publish_report.py \\
        --results build/reports/cpu/results.json \\
        --html build/reports/cpu/results.html \\
        [--site ~/dataframe-reports] [--label NAME]

Writes `<site>/cpu/<YYYY-MM-DD>-<shortsha>[-<label>]/index.html`,
`results.json` and `summary.json` beside it, then regenerates
`<site>/index.html`: one row per report in `<site>/cpu/*`, newest first.
Publishing the same report twice replaces its directory, so reruns are safe.
The history is local; nothing here goes in the repository or is published.

With `--kind apple`, `--results` is the Apple timing receipt
(`performance.json`) and `--coverage` its exact CPU/Metal receipt
(`coverage.json`); the report files under `<site>/apple/<date>-<shortsha>/` and
the index gets its own table of Metal / CPU geometric means (below 1 means
Metal was faster), per suite and overall.

The summary (geometric mean of this library's time over the other engine's,
per suite) comes from the same code that renders the report. It needs the
oracle environment; without it the summary is left empty and the index still
lists the report.
"""

import argparse
import html
import json
from pathlib import Path
import re
import shutil
import sys

TOOLS = Path(__file__).resolve().parent

RULE = (
    "Every report ends with the queries-per-path table (fast-path coverage "
    "counted in distinct queries), so a path that only one query reaches is "
    "visible. A report without it is incomplete."
)


def report_id(provenance, label=""):
    date = provenance.get("utc", "")[:10] or "unknown-date"
    sha = provenance.get("revision", "")[:7] or "unknown"
    name = f"{date}-{sha}"
    if label:
        name += "-" + re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    return name


def summarize(result):
    """Per suite, the geometric mean of mojo/other for each other engine."""
    sys.path.insert(0, str(TOOLS))
    try:
        import bench_html
        import bench_suites
    except ImportError as error:
        print(f"summary skipped: {error}", file=sys.stderr)
        return {}
    cells = bench_suites.evaluate(result)
    engines = result.get("engines", list(bench_suites.ENGINES))
    out = {}
    for suite in bench_suites.SUITES:
        if not any(k[0] == suite for k in cells):
            continue
        _, summary = bench_html._suite_rows(
            suite, cells, engines, bench_suites.SUITES, bench_suites.VARIANTS
        )
        out[suite] = {
            engine: {"geomean": value, "queries": count}
            for engine, (value, count) in summary["geomean"].items()
        }
    return out


def summarize_apple(results, coverage):
    """Metal / CPU geometric means over admitted cases, from the renderer."""
    sys.path.insert(0, str(TOOLS))
    import apple_performance_html as apple

    metadata = apple.read(TOOLS / "metal_case_metadata.json")
    receipt = dict(apple.read(coverage), receipt_sha256=apple.sha(coverage))
    analysis = apple.analyze(apple.read(results), [receipt], metadata)
    return {
        "overall": analysis["policy_geometric_mean"],
        "suites": analysis["suite_policy_geometric_means"],
        "admitted": analysis["admitted_cases"],
        "planned": analysis["planned_cases"],
    }


def publish_apple(results, page, site, coverage, label="", summary=None):
    result = json.loads(Path(results).read_text())
    info = result["provenance"]
    name = report_id(
        {
            "utc": info.get("recorded_at_utc", ""),
            "revision": info.get("engine", {}).get("revision", ""),
        },
        label,
    )
    target = Path(site) / "apple" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copyfile(page, target / "index.html")
    shutil.copyfile(results, target / "performance.json")
    shutil.copyfile(coverage, target / "coverage.json")
    if summary is None:
        summary = summarize_apple(results, coverage)
    meta = {
        "id": name,
        "date": info.get("recorded_at_utc", "")[:10],
        "utc": info.get("recorded_at_utc", ""),
        "revision": info.get("engine", {}).get("revision", ""),
        "scale": info.get("scale", ""),
        "mojo": info.get("mojo", "").split(" (")[0],
        "platform": info.get("platform", ""),
        "summary": summary,
    }
    (target / "summary.json").write_text(json.dumps(meta, indent=1))
    write_index(site)
    return target


def publish(results, page, site, label="", summary=None):
    result = json.loads(Path(results).read_text())
    info = result["provenance"]
    name = report_id(info, label)
    target = Path(site) / "cpu" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copyfile(page, target / "index.html")
    shutil.copyfile(results, target / "results.json")
    if summary is None:
        summary = summarize(result)
    meta = {
        "id": name,
        "date": info.get("utc", "")[:10],
        "utc": info.get("utc", ""),
        "revision": info.get("revision", ""),
        "tier": info.get("tier", ""),
        "mojo": info.get("mojo", "").split(" (")[0],
        "versions": {
            e: info.get(e, "") for e in result.get("engines", []) if e != "mojo"
        },
        "engines": result.get("engines", []),
        "summary": summary,
    }
    (target / "summary.json").write_text(json.dumps(meta, indent=1))
    write_index(site)
    return target


def _ratio(entry):
    value = entry.get("geomean") if entry else None
    return "—" if value is None else f"{value:.2f}"


def write_index(site):
    site = Path(site)
    reports = []
    for path in sorted((site / "cpu").glob("*/summary.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except ValueError:
            continue
    reports.sort(key=lambda r: (r["utc"], r["id"]), reverse=True)
    apple_reports = []
    for path in sorted((site / "apple").glob("*/summary.json")):
        try:
            apple_reports.append(json.loads(path.read_text()))
        except ValueError:
            continue
    apple_reports.sort(key=lambda r: (r["utc"], r["id"]), reverse=True)
    apple_rows = []
    for r in apple_reports:
        s = r["summary"]
        suites = ", ".join(
            f"<b>{html.escape(k)}</b> {_ratio(dict(geomean=v))}"
            for k, v in s["suites"].items()
        )
        apple_rows.append(
            "<tr>"
            f"<td>{html.escape(r['date'])}</td>"
            f"<td><code>{html.escape(r['revision'][:10])}</code></td>"
            f"<td>{html.escape(r['scale'])}</td>"
            f"<td>{html.escape(r['mojo'])}; {html.escape(r['platform'])}</td>"
            f"<td>{_ratio(dict(geomean=s['overall']))} over "
            f"{s['admitted']} of {s['planned']} cases<br>{suites}</td>"
            f'<td><a href="apple/{html.escape(r["id"])}/">report</a></td>'
            "</tr>"
        )
    apple_section = ""
    if apple_rows:
        apple_section = f"""<h2>Apple CPU and Metal reports</h2>
<p>Forced Metal time divided by CPU time, geometric mean over the admitted
cases; below 1 means Metal was faster.</p>
<div class="scroll"><table>
<thead><tr><th>Date</th><th>Commit</th><th>Scale</th><th>Toolchain</th>
<th>Metal / CPU geometric mean</th><th>Report</th></tr></thead>
<tbody>
{chr(10).join(apple_rows)}
</tbody></table></div>
"""
    rows = []
    for r in reports:
        engines = ", ".join(
            f"{e} {v}" if v else e
            for e, v in [("mojo", r["mojo"])]
            + [(e, r["versions"].get(e, "")) for e in r["engines"] if e != "mojo"]
        )
        means = []
        for suite, per_engine in r["summary"].items():
            cell = ", ".join(
                f"{e} {_ratio(v)}" for e, v in per_engine.items()
            )
            means.append(f"<b>{html.escape(suite)}</b> {html.escape(cell)}")
        rows.append(
            "<tr>"
            f"<td>{html.escape(r['date'])}</td>"
            f"<td><code>{html.escape(r['revision'][:10])}</code></td>"
            f"<td>{html.escape(r['tier'])}</td>"
            f"<td>{html.escape(engines)}</td>"
            f"<td>{'<br>'.join(means) or '—'}</td>"
            f'<td><a href="cpu/{html.escape(r["id"])}/">report</a></td>'
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dataframe_mojo CPU benchmark reports</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:72rem;margin:0 auto;padding:1rem 16px;
background:#fff;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #ccc;vertical-align:top}}
.scroll{{overflow-x:auto}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#e6e6e6}}
th,td{{border-color:#444}}a{{color:#8ab4ff}}}}
</style></head><body>
<h1>dataframe_mojo CPU benchmark reports</h1>
<p>Newest first. Each report compares dataframe_mojo with Polars and DuckDB on
the external suites, run on one dedicated machine. The figures are the
geometric mean per suite of dataframe_mojo's time divided by the other
engine's; below 1 is faster.</p>
<p>{html.escape(RULE)}</p>
<div class="scroll"><table>
<thead><tr><th>Date</th><th>Commit</th><th>Tier</th><th>Engines</th>
<th>Geometric mean vs engine</th><th>Report</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>
{apple_section}</body></html>
"""
    (site / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument(
        "--site", type=Path, default=Path.home() / "dataframe-reports"
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--kind", choices=("cpu", "apple"), default="cpu")
    parser.add_argument(
        "--coverage", type=Path, help="exact receipt; required for --kind apple"
    )
    args = parser.parse_args()
    if args.kind == "apple":
        if args.coverage is None:
            parser.error("--kind apple needs --coverage")
        target = publish_apple(
            args.results, args.html, args.site, args.coverage, args.label
        )
    else:
        target = publish(args.results, args.html, args.site, args.label)
    print(f"published {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
