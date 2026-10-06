"""Render a benchmark-suite result as one self-contained HTML page.

`scripts/bench_suites.py` writes it beside the Markdown report
(`results.html` next to `results.md`) so the results can be read in a
browser or published, without VS Code. It shows the same numbers as the
Markdown report: medians over rounds of the fastest run, ratios of this
library's time to each other engine's, the worst data variant per query,
answer status, and fast-path coverage. The page needs no network access
beyond Google Fonts, and falls back to system fonts without it.
"""

import html
import math
import statistics

import bench_policy

# Ratio bands for the shaded cells: this library's time over the other
# engine's. Below 0.95 is a win; 0.95-1.25 is parity within noise.
BANDS = [
    (0.95, "win", "faster"),
    (1.25, "par", "within 25%"),
    (2.0, "s1", "1.25-2x"),
    (5.0, "s2", "2-5x"),
    (20.0, "s3", "5-20x"),
    (math.inf, "s4", "over 20x"),
]

STYLE = """
:root {
  --ground: #f4f6f5;
  --panel: #ffffff;
  --ink: #182022;
  --muted: #5b6769;
  --line: #d8dfdd;
  --accent: #1f5f7a;
  --win-bg: #d7efe2; --win-ink: #1d5c3d;
  --par-bg: #eceff0; --par-ink: #37474b;
  --s1-bg: #fbeccc; --s1-ink: #6b4a06;
  --s2-bg: #f8d9b8; --s2-ink: #7a3d0a;
  --s3-bg: #f3c1b3; --s3-ink: #7d2413;
  --s4-bg: #e39c93; --s4-ink: #5c0f08;
  --bad-bg: #f0d6e4; --bad-ink: #6b1c46;
  --warn-bg: #fff4d6; --warn-line: #e0c36b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #111719; --panel: #182125; --ink: #e3eaea; --muted: #9aa8aa;
    --line: #2b3a3e; --accent: #7cc0db;
    --win-bg: #173a2a; --win-ink: #9fe0bd;
    --par-bg: #243034; --par-ink: #c8d3d4;
    --s1-bg: #3d3212; --s1-ink: #f0d58f;
    --s2-bg: #45290f; --s2-ink: #f5bd8a;
    --s3-bg: #4d1f16; --s3-ink: #f4a794;
    --s4-bg: #5e1712; --s4-ink: #ffc2b8;
    --bad-bg: #44192f; --bad-ink: #f3b6d4;
    --warn-bg: #3a3214; --warn-line: #7d6a2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #111719; --panel: #182125; --ink: #e3eaea; --muted: #9aa8aa;
  --line: #2b3a3e; --accent: #7cc0db;
  --win-bg: #173a2a; --win-ink: #9fe0bd;
  --par-bg: #243034; --par-ink: #c8d3d4;
  --s1-bg: #3d3212; --s1-ink: #f0d58f;
  --s2-bg: #45290f; --s2-ink: #f5bd8a;
  --s3-bg: #4d1f16; --s3-ink: #f4a794;
  --s4-bg: #5e1712; --s4-ink: #ffc2b8;
  --bad-bg: #44192f; --bad-ink: #f3b6d4;
  --warn-bg: #3a3214; --warn-line: #7d6a2a;
}
body {
  background: var(--ground); color: var(--ink);
  font: 15px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  padding-inline: 16px; padding-block: 24px 48px;
}
main { max-width: 1180px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { text-wrap: balance; margin: 0; line-height: 1.2; }
h1 { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: 1.25rem; font-weight: 600; }
h3 { font-size: 1.05rem; font-weight: 600; }
p { margin: 0; max-width: 72ch; }
.muted { color: var(--muted); }
.eyebrow {
  font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); font-weight: 600;
}
header { display: grid; gap: 8px; }
.prov { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.85rem; color: var(--muted); }
.prov b { color: var(--ink); font-weight: 500; }
nav { display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.85rem; }
nav a {
  color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 10px; background: var(--panel);
}
nav a:hover, nav a:focus-visible { border-color: var(--accent); outline: none; }
section { display: grid; gap: 12px; }
.note {
  background: var(--warn-bg); border: 1px solid var(--warn-line);
  border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
}
.scroll { overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { padding: 6px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--line); }
thead th {
  font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--panel);
}
tbody tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums; }
td.q { font-family: "IBM Plex Mono", ui-monospace, monospace; font-weight: 600; }
td.status { white-space: normal; min-width: 14ch; color: var(--muted); font-size: 0.8rem; }
.chip {
  display: inline-block; min-width: 4.2em; text-align: right; border-radius: 4px;
  padding: 1px 6px; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums; font-size: 0.82rem;
}
.chip small { font-family: "IBM Plex Sans", system-ui, sans-serif; opacity: 0.8; margin-left: 4px; }
.win { background: var(--win-bg); color: var(--win-ink); }
.par { background: var(--par-bg); color: var(--par-ink); }
.s1 { background: var(--s1-bg); color: var(--s1-ink); }
.s2 { background: var(--s2-bg); color: var(--s2-ink); }
.s3 { background: var(--s3-bg); color: var(--s3-ink); }
.s4 { background: var(--s4-bg); color: var(--s4-ink); }
.bad { background: var(--bad-bg); color: var(--bad-ink); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: 0.8rem; color: var(--muted); align-items: center; }
.foot { font-size: 0.85rem; color: var(--muted); }
code { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.85em; }
"""


def _band(ratio):
    for limit, name, _ in BANDS:
        if ratio < limit:
            return name
    return "s4"


def _chip(ratio, label=""):
    if ratio is None:
        return '<span class="muted">—</span>'
    extra = f"<small>{html.escape(label)}</small>" if label else ""
    return f'<span class="chip {_band(ratio)}">{ratio:.2f}{extra}</span>'


def _ms(cell):
    if not cell or cell["status"] != "ok":
        return '<span class="muted">—</span>'
    ms = cell["ms"]
    text = f"{ms:,.1f}" if ms < 10_000 else f"{ms:,.0f}"
    return text


def _geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _suite_rows(suite, cells, all_engines, suites, variants_of):
    variants = [
        v for v in variants_of[suite] if any(k[0] == suite and k[1] == v for k in cells)
    ]
    engines = [e for e in all_engines if any(k[0] == suite and k[3] == e for k in cells)]
    others = [e for e in engines if e != "mojo"]
    base = variants[0] if variants else None
    rows, ratios, worsts = [], {e: [] for e in others}, {e: [] for e in others}
    counts = {"ok": 0, "unsupported": 0, "wrong answer": 0, "other": 0}
    for query in suites[suite]["queries"]:
        if not any(k[0] == suite and k[2] == query for k in cells):
            continue
        mojo = cells.get((suite, base, query, "mojo"))
        row = {"query": query, "ms": {}, "vs": {}, "worst": {}, "status": []}
        for engine in engines:
            cell = cells.get((suite, base, query, engine))
            row["ms"][engine] = cell
            if cell and cell["status"] != "ok":
                label = cell["status"]
                if cell.get("reason"):
                    label += f": {cell['reason']}"
                row["status"].append(f"{engine} {label}")
        for engine in others:
            other = cells.get((suite, base, query, engine))
            ratio = None
            if mojo and other and mojo["status"] == other["status"] == "ok":
                ratio = mojo["ms"] / other["ms"]
                ratios[engine].append(ratio)
            row["vs"][engine] = ratio
            worst = None
            if len(variants) > 1:
                for variant in variants:
                    m = cells.get((suite, variant, query, "mojo"))
                    o = cells.get((suite, variant, query, engine))
                    if m and o and m["status"] == o["status"] == "ok":
                        r = m["ms"] / o["ms"]
                        if worst is None or (isinstance(worst[0], float) and r > worst[0]):
                            worst = (r, variant)
                    elif m and m["status"] != "ok":
                        worst = (m["status"], variant)
                        break
                if worst and isinstance(worst[0], float):
                    worsts[engine].append(worst[0])
            row["worst"][engine] = worst
        status = mojo["status"] if mojo else "not run"
        counts[status if status in counts else "other"] += 1
        rows.append(row)
    summary = {
        "engines": engines,
        "others": others,
        "variants": variants,
        "geomean": {e: (_geomean(ratios[e]), len(ratios[e])) for e in others},
        "worst": {e: _geomean(worsts[e]) for e in others},
        "counts": counts,
    }
    return rows, summary


def _suite_table(suite, rows, summary, suites):
    engines, others, variants = summary["engines"], summary["others"], summary["variants"]
    head = ["<th>Query</th>"]
    head += [f'<th class="num">{html.escape(e)} ms</th>' for e in engines]
    head += [f"<th>vs {html.escape(e)}</th>" for e in others]
    if len(variants) > 1:
        head += [f"<th>worst vs {html.escape(e)}</th>" for e in others]
    head.append("<th>Status</th>")
    body = []
    for row in rows:
        cells = [f'<td class="q">{html.escape(row["query"])}</td>']
        cells += [f'<td class="num">{_ms(row["ms"][e])}</td>' for e in engines]
        cells += [f"<td>{_chip(row['vs'][e])}</td>" for e in others]
        if len(variants) > 1:
            for e in others:
                worst = row["worst"][e]
                if worst is None:
                    cells.append('<td><span class="muted">—</span></td>')
                elif isinstance(worst[0], float):
                    cells.append(f"<td>{_chip(worst[0], worst[1])}</td>")
                else:
                    cells.append(
                        f'<td><span class="chip bad">{html.escape(worst[0])}'
                        f"<small>{html.escape(worst[1])}</small></span></td>"
                    )
        status = "; ".join(row["status"]) or "ok"
        cells.append(f'<td class="status">{html.escape(status)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render(result, cells, suites, variants_of, instrumented):
    info = result["provenance"]
    engines = result.get("engines", ["mojo", "polars", "duckdb"])
    when = info.get("utc", "")[:16].replace("T", " ")
    parts = [
        "<title>dataframe_mojo Benchmarks</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
        f"<style>{STYLE}</style>",
        "<main>",
        "<header>",
        '<div class="eyebrow">External benchmark suites</div>',
        "<h1>dataframe_mojo against Polars and DuckDB</h1>",
        '<div class="prov">'
        + "".join(
            f"<span>{html.escape(k)} <b>{html.escape(str(v))}</b></span>"
            for k, v in [
                ("revision", info["revision"][:10] + (" (dirty)" if info.get("dirty") else "")),
                ("tier", info.get("tier", "full")),
                ("scale", info.get("scale", "")),
                ("threads", info.get("threads", "")),
                ("rounds × runs", f"{info.get('rounds')} × {info.get('reps')}"),
                ("Mojo", info.get("mojo", "").split(" (")[0]),
                ("Polars", info.get("polars", "")),
                ("DuckDB", info.get("duckdb", "")),
                ("CPU", info.get("cpu", "")),
                ("measured", when + " UTC"),
            ]
        )
        + "</div>",
        "<p class=\"muted\">Times are medians over rounds of the fastest run, in "
        "milliseconds. A ratio is dataframe_mojo's time divided by the other "
        "engine's: below 1 is faster. Geometric means use only queries every "
        "engine answered correctly. Answers are checked against DuckDB.</p>",
        '<div class="legend">'
        + "".join(f'<span class="chip {name}">{label}</span>' for _, name, label in BANDS)
        + "</div>",
        "</header>",
    ]
    parts.append(
        '<div class="note">'
        + html.escape(bench_policy.report_note(result))
        + '</div>'
    )
    evaluation = info.get("evaluation")
    if evaluation:
        parts.append(
            '<p class="foot">Local benchmark manifest: <code>'
            + html.escape(evaluation["manifest_sha256"])
            + '</code>. Source hashes and exposure evidence are recorded '
            'in the raw JSON.</p>'
        )
    load = result.get("load") or []
    threads = info.get("threads") or 0
    if load and max(load) > threads + 4:
        parts.append(
            f'<div class="note"><b>Busy host.</b> The one-minute load average '
            f"reached {max(load):.0f} (median {statistics.median(load):.0f}) against "
            f"{threads} benchmark threads, so treat small differences as noise.</div>"
        )
    cached = sorted({r["engine"] for r in result["runs"] if r.get("cached")})
    if cached:
        parts.append(
            '<div class="note">Times for '
            + html.escape(", ".join(cached))
            + " come from the reference cache (same data, engine version and "
            "thread count, possibly measured on an earlier day).</div>"
        )
    present = [s for s in suites if any(k[0] == s for k in cells)]
    tables = {}
    for suite in present:
        tables[suite] = _suite_rows(suite, cells, engines, suites, variants_of)
    parts.append(
        "<nav>"
        + "".join(f'<a href="#{s}">{html.escape(s)}</a>' for s in present)
        + ('<a href="#coverage">coverage</a>' if result.get("trace") else "")
        + ('<a href="#joins">joins</a>' if result.get("operators") else "")
        + "</nav>"
    )
    # Summary across suites.
    others = [e for e in engines if e != "mojo" and not e.startswith("base@")]
    head = "<th>Suite</th><th>Role</th><th class=\"num\">Queries</th><th>Answered</th>"
    head += "".join(
        f"<th>vs {html.escape(e)}</th><th>worst variant vs {html.escape(e)}</th>"
        for e in others
    )
    body = []
    for suite in present:
        _, summary = tables[suite]
        counts = summary["counts"]
        answered = f"{counts['ok']} ok"
        if counts["unsupported"]:
            answered += f", {counts['unsupported']} unsupported"
        if counts["wrong answer"]:
            answered += f", {counts['wrong answer']} wrong"
        if counts["other"]:
            answered += f", {counts['other']} failed"
        row = (
            f'<td class="q"><a href="#{suite}">{html.escape(suite)}</a></td>'
            f"<td>{'development' if suites[suite]['role'] == 'dev' else 'held out'}</td>"
            f'<td class="num">{sum(counts.values())}</td><td>{answered}</td>'
        )
        for e in others:
            g = summary["geomean"].get(e, (None, 0))[0]
            w = summary["worst"].get(e) if len(summary["variants"]) > 1 else None
            row += f"<td>{_chip(g)}</td><td>{_chip(w) if w else '<span class="muted">—</span>'}</td>"
        body.append(f"<tr>{row}</tr>")
    parts += [
        "<section>",
        "<h2>Summary</h2>",
        '<div class="scroll"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>",
        '<p class="muted">Tune on the development suites. Use held-out results '
        'to validate the completed change; do not use their per-query timings '
        'to choose optimizations or tune thresholds (docs/benchmarks.md).</p>',
        "</section>",
    ]
    for suite in present:
        rows, summary = tables[suite]
        variants = ", ".join(summary["variants"])
        parts += [
            f'<section id="{suite}">',
            f'<div class="eyebrow">{"development" if suites[suite]["role"] == "dev" else "held out"}</div>',
            f"<h2>{html.escape(suite)}</h2>",
            f'<p class="muted">{html.escape(suites[suite]["source"])}. '
            f"Data variants: {html.escape(variants)}; the query table shows "
            f"<code>{html.escape(summary['variants'][0])}</code>, and the worst "
            "variant beside it.</p>",
            _suite_table(suite, rows, summary, suites),
            "</section>",
        ]
    trace = result.get("trace") or {}
    if trace:
        hits = {}
        for query, paths in trace.items():
            for path in paths:
                hits.setdefault(path, []).append(query)
        body = []
        for path in sorted(hits, key=lambda p: (len(hits[p]), p)):
            distinct = len(bench_policy.distinct_queries(hits[path]))
            flag = ' <span class="chip s2">one query</span>' if distinct == 1 else ""
            body.append(
                f"<tr><td><code>{html.escape(path)}</code>{flag}</td>"
                f'<td class="num">{distinct}</td>'
                f'<td class="num">{len(hits[path])}</td>'
                f'<td class="status">{html.escape(", ".join(sorted(hits[path])[:4]))}</td></tr>'
            )
        never = [p for p in instrumented if p not in hits]
        parts += [
            '<section id="coverage">',
            "<h2>Fast-path coverage</h2>",
            '<p class="muted">Specialized paths each query took. A path only one '
            "query reaches may be shaped to that query; check that its trigger "
            "is a data property with real examples.</p>",
            '<div class="scroll"><table><thead><tr><th>Path</th>'
            '<th class="num">Distinct queries</th><th class="num">Query/variant cases</th><th>Examples</th></tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>",
        ]
        if never:
            parts.append(
                '<p class="foot">No query reached: '
                + ", ".join(f"<code>{html.escape(p)}</code>" for p in never)
                + ".</p>"
            )
        parts.append("</section>")
    if result.get("operators"):
        joins, larger = 0, []
        for query, rows in sorted(result["operators"].items()):
            for row in rows:
                if not row["operator"].startswith("JOIN") or row["builds"] == 0:
                    continue
                joins += 1
                if row["build_rows"] > row["input_rows"] > 0:
                    larger.append((row["build_rows"] - row["input_rows"], query, row))
        body = []
        for _, query, row in sorted(larger, key=lambda item: -item[0])[:20]:
            body.append(
                f"<tr><td>{html.escape(query)}</td>"
                f"<td>{html.escape(row['operator'][:60])}</td>"
                f"<td>{html.escape(row['executor'])}</td>"
                f"<td><code>{html.escape(row['algorithm'])}</code></td>"
                f'<td class="num">{row["input_rows"]:,}</td>'
                f'<td class="num">{row["build_rows"]:,}</td>'
                f'<td class="num">{row["output_rows"]:,}</td></tr>'
            )
        parts += [
            '<section id="joins">',
            "<h2>Join builds larger than their probe</h2>",
            f'<p class="muted">Of {joins} joins that built an index, {len(larger)} '
            "indexed more rows than probed them. Observed counts from the lazy "
            "execution report (<code>LazyFrame.profile</code>); the largest first.</p>",
            '<div class="scroll"><table><thead><tr><th>Query</th><th>Join</th>'
            '<th>Executor</th><th>Index</th><th class="num">Probe rows</th>'
            '<th class="num">Build rows</th><th class="num">Output rows</th></tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>",
            "</section>",
        ]
    parts += [
        '<p class="foot">Generated by <code>scripts/bench_suites.py</code>. '
        "Raw samples are in <code>results.json</code> beside this page.</p>",
        "</main>",
    ]
    return "\n".join(parts) + "\n"
