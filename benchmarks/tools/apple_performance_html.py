"""Render immutable Apple timing/coverage receipts; never run a query.

Only explicitly admitted CPU/Metal pairs enter ratios. Observed timings for
wrong, unsupported, contaminated or unverified cases remain visible separately.
"""
import argparse
import collections
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
import bench_apple_performance as collector
import bench_html
import bench_suites as suites
import metal_query_report as coverage

STATUSES = {
    "ok",
    "unsupported",
    "execution_failed",
    "input_failed",
    "answer_failed",
    "process_failed",
    "timeout",
    "protocol_failed",
}


def read(path):
    def reject(value):
        raise ValueError("Nonstandard JSON numeric constant: " + value)

    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    return type(value) is int and value > 0


def validate(data):
    if data.get("format") != collector.FORMAT:
        raise ValueError("Unknown timing receipt format")
    settings = data["provenance"]
    for field in ("rounds", "repetitions", "warmups", "threads"):
        if not positive(settings.get(field)):
            raise ValueError("Invalid timing setting: " + field)
    if settings.get("scale") not in suites.SCALES:
        raise ValueError("Unknown timing scale")
    if not settings.get("engine", {}).get("revision") or not settings.get("mojo"):
        raise ValueError("Missing engine/SDK identity")
    if (
        not settings.get("engine", {}).get("package_sha256")
        or not settings.get("harness_revision")
        or not settings.get("harness_sha256")
    ):
        raise ValueError("Missing engine/harness source manifest")
    declared = {coverage.key(c): c for c in data["cases"]}
    if len(declared) != len(data["cases"]):
        raise ValueError("Duplicate planned case")
    for suite, variant, query in declared:
        if (
            suite not in suites.SUITES
            or variant not in suites.VARIANTS[suite]
            or query not in suites.SUITES[suite]["queries"]
        ):
            raise ValueError("Timing case outside retained catalog")
        path = "benchmarks/suites/" + suites.SUITES[suite]["runner"] + ".mojo"
        if not settings.get("staged", {}).get("original_source_sha256", {}).get(path):
            raise ValueError("Missing canonical query source identity")
    seen = set()
    for run in data["runs"]:
        key = coverage.key(run)
        round_index = run.get("round")
        if (
            key not in declared
            or run.get("engine") not in collector.ENGINES
            or type(round_index) is not int
            or not 0 <= round_index < settings["rounds"]
        ):
            raise ValueError("Unexpected timing worker")
        identity = key + (run["engine"], round_index)
        if identity in seen:
            raise ValueError("Duplicate timing worker")
        seen.add(identity)
        if run.get("inputs") != declared[key].get("inputs"):
            raise ValueError("Timing worker inputs differ from its declared case")
        for label in ("host_before", "host_after"):
            host = run.get(label)
            if host is not None and (
                not isinstance(host.get("compilers"), list)
                or not isinstance(host.get("load_average"), list)
                or len(host["load_average"]) != 3
                or any(
                    not isinstance(v, (int, float))
                    or isinstance(v, bool)
                    or not math.isfinite(v)
                    or v < 0
                    for v in host["load_average"]
                )
            ):
                raise ValueError("Invalid host observation")
        if run.get("status") not in STATUSES:
            raise ValueError("Unknown timing worker status")
        samples = set()
        limits = {
            "cold": 1,
            "warmup": settings["warmups"],
            "warm": settings["repetitions"],
        }
        for sample in run.get("samples", []):
            phase, index = sample.get("phase"), sample.get("index")
            if (
                phase not in limits
                or type(index) is not int
                or not 0 <= index < limits[phase]
                or (phase, index) in samples
            ):
                raise ValueError("Duplicate or invalid raw sample")
            samples.add((phase, index))
            if "ns" in sample and (
                type(sample["ns"]) is not int or not 0 <= sample["ns"] <= (1 << 64) - 1
            ):
                raise ValueError("Invalid raw duration")
        if run["status"] == "ok" and (
            len(samples) != sum(limits.values())
            or any("ns" not in s or "summary" not in s for s in run.get("samples", []))
        ):
            raise ValueError("Successful worker lacks complete timing/answer evidence")
    return data


def statistics_for(runs, engine, phase):
    values = []
    minima = []
    for run in runs:
        if run["engine"] != engine:
            continue
        observed = [
            s["ns"]
            for s in run.get("samples", [])
            if s.get("phase") == phase and "ns" in s
        ]
        values.extend(observed)
        if observed:
            minima.append(min(observed))
    if not values:
        return None
    return dict(
        median_ns=statistics.median(values),
        minimum_ns=min(values),
        maximum_ns=max(values),
        samples=len(values),
        observed_rounds=len(minima),
        median_round_minimum_ns=statistics.median(minima),
    )


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def analyze(data, strict_receipts, metadata):
    validate(data)
    settings = data["provenance"]
    rows = []
    for case in data["cases"]:
        runs = [r for r in data["runs"] if coverage.key(r) == coverage.key(case)]
        confirmed = collector.strict_basis(
            case,
            strict_receipts,
            settings["engine"],
            settings.get("staged", {}).get("original_source_sha256", {}),
            settings["mojo"],
            settings["threads"],
            metadata,
        )
        oracle = data.get("oracle", {}).get("/".join(coverage.key(case)), {})
        reference = (
            oracle.get("answer", {}) if oracle.get("status") == "completed" else {}
        )
        if "summary" in reference:
            reference = dict(
                reference,
                summary=dict(
                    reference["summary"],
                    values=[float(v) for v in reference["summary"]["values"]],
                ),
            )
        verified_runs = [
            dict(r, oracle=collector.oracle_checks(r, reference)) for r in runs
        ]
        basis = collector.ratio_basis(
            verified_runs, confirmed, settings["rounds"], settings["repetitions"]
        )
        if basis.startswith("eligible") and any(
            "busy_host" not in r
            or "compiler_overlap" not in r
            or "load_average" not in r.get("host_before", {})
            or "compilers" not in r.get("host_after", {})
            for r in runs
        ):
            basis = "missing_host_observations"
        if basis.startswith("eligible") and any(
            r["host_before"].get("compilers")
            or r["host_after"].get("compilers")
            or r["host_before"]["load_average"][0] > settings["threads"]
            for r in runs
        ):
            basis = "observed_compiler_or_busy_host"
        if not data.get("engine_unchanged_after") or not data.get(
            "inputs_unchanged_after"
        ):
            basis = "missing_or_failed_final_engine_input_identity_check"
        if settings.get("engine", {}).get("status"):
            basis = "dirty_engine"
        if basis.startswith("eligible") and case.get("strict_basis", {}).get(
            "receipt_sha256"
        ) != confirmed.get("receipt_sha256"):
            basis = "recorded_and_supplied_strict_receipts_disagree"
        row = dict(
            case,
            confirmed_strict_basis=confirmed,
            ratio_basis=basis,
            runs=runs,
            times={
                engine: {
                    phase: statistics_for(runs, engine, phase)
                    for phase in ("cold", "warm")
                }
                for engine in collector.ENGINES
            },
            ratio=None,
            policy_ratio=None,
        )
        if basis.startswith("eligible"):
            cpu = row["times"]["cpu"]["warm"]
            metal = row["times"]["metal"]["warm"]
            row["ratio"] = metal["median_ns"] / cpu["median_ns"]
            row["policy_ratio"] = (
                metal["median_round_minimum_ns"] / cpu["median_round_minimum_ns"]
            )
        rows.append(row)
    admitted = [r for r in rows if r["ratio"] is not None]
    worst = {}
    for row in admitted:
        key = (row["suite"], row["query"])
        if key not in worst or row["ratio"] > worst[key]["ratio"]:
            worst[key] = row
    catalog = {
        (s, v, q)
        for s, spec in suites.SUITES.items()
        for v in suites.VARIANTS[s]
        for q in spec["queries"]
    }
    observed = {coverage.key(r) for r in rows}
    return dict(
        rows=rows,
        admitted_cases=len(admitted),
        admitted_query_ids=len(worst),
        geometric_mean=geomean([r["ratio"] for r in admitted]),
        policy_geometric_mean=geomean([r["policy_ratio"] for r in admitted]),
        worst_variants=sorted(
            worst.values(), key=lambda r: r["ratio"], reverse=True
        ),
        suite_policy_geometric_means={
            suite: geomean(
                [r["policy_ratio"] for r in admitted if r["suite"] == suite]
            )
            for suite in sorted({r["suite"] for r in admitted})
        },
        suite_geometric_means={
            suite: geomean([r["ratio"] for r in admitted if r["suite"] == suite])
            for suite in sorted({r["suite"] for r in admitted})
        },
        planned_cases=len(rows),
        planned_query_ids=len({(r["suite"], r["query"]) for r in rows}),
        catalog_cases=len(catalog),
        catalog_query_ids=sum(len(s["queries"]) for s in suites.SUITES.values()),
        unplanned_cases=sorted(catalog - observed),
        worker_statuses=dict(collections.Counter(r["status"] for r in data["runs"])),
    )


def coverage_evidence(paths, metadata, catalog_path=None):
    if not paths:
        return None
    cases, identity, sources, receipts = coverage.load_receipts(paths)
    definitions = coverage.catalog()
    catalog_receipt = None
    if catalog_path is not None:
        catalog_data = read(catalog_path)
        if catalog_data.get("format") != "dataframe-metal-suite-summary-v1":
            raise ValueError("Unknown pinned catalog report format")
        definitions = catalog_data["catalog"]
        catalog_receipt = dict(path=str(catalog_path), sha256=sha(catalog_path))
    for item in definitions.values():
        if sources.get(item["source_path"]) != item["source_sha256"]:
            raise ValueError(
                "Coverage query source differs from catalog; supply its source-pinned summary via --baseline-catalog/--final-catalog"
            )
    summary = coverage.summarize(cases, definitions, sources, metadata)
    raw = [read(p) for p in paths]
    inputs = {
        "/".join(coverage.key(c)): {
            k: v["sha256"] for k, v in c.get("inputs", {}).items()
        }
        for c in cases.values()
    }
    adapter = [r["provenance"].get("derived_source_sha256") for r in raw]
    audit_identity = {
        "/".join(coverage.key(c)): c["execution_audit"] for c in summary["cases"]
    }
    return dict(
        summary=summary,
        catalog_receipt=catalog_receipt,
        identity=identity,
        receipts=receipts,
        input_hashes=inputs,
        derived_workers=adapter,
        execution_audits=audit_identity,
        metadata_hashes=sorted(
            {r["provenance"].get("case_metadata_sha256", "unavailable") for r in raw}
        ),
    )


def coverage_comparison(before, after):
    if before is None or after is None:
        return "Baseline or final coverage is unavailable; no improvement count is inferred."
    fields = ("mojo", "threads", "scale", "driver_sha256", "comparison")
    same = all(before["identity"].get(k) == after["identity"].get(k) for k in fields)
    # Suite receipts can be split differently; compare unique worker manifests.
    manifests = lambda item: {
        json.dumps(v, sort_keys=True) for v in item["derived_workers"]
    }
    same = (
        same
        and manifests(before) == manifests(after)
        and before["input_hashes"] == after["input_hashes"]
        and before["execution_audits"] == after["execution_audits"]
        and before["metadata_hashes"] == after["metadata_hashes"]
    )
    if not same:
        return "Coverage identities, inputs or derived adapters differ. Counts describe each receipt separately; they are not attributed to kernel gains. Cross-run the final harness on the initial engine for like-for-like counts."
    if not before["summary"]["complete"] or not after["summary"]["complete"]:
        return "Coverage identities match, but a grid is partial; no complete improvement count is inferred."
    difference = (
        after["summary"]["known_hybrid_and_unknown_audits_excluded"]["matching_cases"]
        - before["summary"]["known_hybrid_and_unknown_audits_excluded"][
            "matching_cases"
        ]
    )
    return f"Matching input/SDK/thread/adapter identities: conservative exact matching cases changed by {difference:+d}. Distinct query IDs and variants remain separate below."


def basis_label(value):
    return {
        "eligible_separate_exact_and_external_complete_evidence": "Admitted: matching exact receipt and independent complete-result checks",
        "unverified_or_hybrid_exact_basis": "Exact validation unavailable, failed, or hybrid",
        "incomplete_rounds": "Incomplete paired rounds",
        "failed_or_contaminated_run": "Failed or contaminated worker",
        "external_answer_policy_not_verified": "Independent DuckDB answer policy not verified",
        "incomplete_warm_samples": "Incomplete warm samples",
        "missing_host_observations": "Host observations missing",
        "observed_compiler_or_busy_host": "Observed compiler activity or busy host",
        "missing_or_failed_final_engine_input_identity_check": "Final engine/input identity checks missing or failed",
        "dirty_engine": "Engine working tree was dirty",
        "recorded_and_supplied_strict_receipts_disagree": "Recorded and supplied exact receipts disagree",
    }.get(value, value.replace("_", " "))


def e(value):
    return html.escape(str(value), quote=True)


def coverage_section(title, item):
    if item is None:
        return f"<section><h2>{e(title)}</h2><p>Unmeasured / receipt not supplied.</p></section>"
    s = item["summary"]
    conservative = s["known_hybrid_and_unknown_audits_excluded"]
    raw = s["raw_adapter"]
    values = [
        ("Engine", item["identity"]["engine_revision"]),
        (
            "Scale / threads",
            f"{item['identity'].get('scale')} / {item['identity'].get('threads')}",
        ),
        (
            "Catalog query IDs / variant cases",
            f"{s['catalog_queries']} / {s['catalog_cases']}",
        ),
        (
            "CPU-translated query IDs / cases",
            f"{s['translated_queries']} / {s['translated_cases']}",
        ),
        (
            "Measured / unmeasured cases",
            f"{raw['measured_cases']} / {len(s['unmeasured_cases'])}",
        ),
        (
            "Raw exact matching / runnable cases",
            f"{raw['matching_cases']} / {raw['runnable_cases']}",
        ),
        (
            "Conservative exact matching / runnable cases",
            f"{conservative['matching_cases']} / {conservative['runnable_cases']}",
        ),
        (
            "All-variant matching / runnable query IDs",
            f"{len(conservative['all_variants_matching_queries'])} / {len(conservative['all_variants_runnable_queries'])}",
        ),
        (
            "Matching populated / empty / unknown shape",
            f"{conservative['matching_populated_cases']} / {conservative['matching_empty_cases']} / {conservative['matching_output_shape_unknown']}",
        ),
        ("Hybrid or unknown audit exclusions", conservative["excluded_cases"]),
        ("Raw statuses", json.dumps(raw["statuses"], sort_keys=True)),
        ("Grid", "complete" if s["complete"] else "partial; no complete percentage"),
    ]
    rows = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in values)
    details = json.dumps(
        dict(
            identity=item["identity"],
            receipts=item["receipts"],
            catalog_receipt=item.get("catalog_receipt"),
            summary=s,
        ),
        indent=2,
    )
    return f"<section><h2>{e(title)}</h2><table>{rows}</table><details><summary>Exact coverage cases, reasons and provenance</summary><pre>{e(details)}</pre></details></section>"


def classify(row):
    """Short status word and reason for one case: ok, wrong answer,
    unsupported or unverified."""
    if row["ratio"] is not None:
        return "ok", ""
    strict = row.get("confirmed_strict_basis", {})
    measured = strict.get("status", "unmeasured")
    reasons = [
        r.get("reason", "")
        for r in row["runs"]
        if r.get("status") != "ok" and r.get("reason")
    ]
    detail = basis_label(row["ratio_basis"])
    if reasons:
        detail += "; " + reasons[0]
    if measured in ("unsupported", "reference_unsupported") or any(
        r.get("status") == "unsupported" for r in row["runs"]
    ):
        return "unsupported", detail
    if (
        measured == "wrong_answer"
        or row["ratio_basis"] == "external_answer_policy_not_verified"
        or any(r.get("status") == "answer_failed" for r in row["runs"])
    ):
        return "wrong answer", detail + f" (exact receipt: {measured})"
    return "unverified", detail + f" (exact receipt: {measured})"


def _ms(stats):
    if not stats:
        return '<span class="muted">—</span>'
    ms = stats["median_round_minimum_ns"] / 1e6
    return f"{ms:,.1f}" if ms < 10_000 else f"{ms:,.0f}"


def _suite_section(suite, rows):
    variants = [v for v in suites.VARIANTS[suite] if any(r["variant"] == v for r in rows)]
    base = variants[0]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["query"]][r["variant"]] = r
    counts = collections.Counter(classify(r)[0] for r in rows)
    body, base_ratios, worst_ratios = [], [], []
    for query in suites.SUITES[suite]["queries"]:
        if query not in by:
            continue
        cases = by[query]
        main = cases.get(base)
        if main is None:
            continue
        status, reason = classify(main)
        cpu = main["times"]["cpu"]["warm"]
        metal = main["times"]["metal"]["warm"]
        ratio = main["policy_ratio"]
        if ratio is not None:
            base_ratios.append(ratio)
        worst = None
        for variant, case in cases.items():
            if case["policy_ratio"] is not None and (
                worst is None or case["policy_ratio"] > worst[0]
            ):
                worst = (case["policy_ratio"], variant)
        if worst:
            worst_ratios.append(worst[0])
        others = [
            f"{v} {classify(c)[0]}"
            for v, c in cases.items()
            if v != base and classify(c)[0] != "ok"
        ]
        note = status if status == "ok" else status
        title = e(reason) if reason else ""
        small = f"<small>{e('; '.join(others))}</small>" if others else ""
        worst_cell = (
            "<td>" + bench_html._chip(worst[0], worst[1]) + "</td>"
            if len(variants) > 1 and worst
            else '<td><span class="muted">—</span></td>'
        )
        body.append(
            f'<tr><td class="q">{e(query)}</td>'
            f'<td class="num">{_ms(cpu)}</td>'
            f'<td class="num">{_ms(metal) if status == "ok" else "<span class=muted>—</span>"}</td>'
            f"<td>{bench_html._chip(ratio)}</td>"
            + (worst_cell if len(variants) > 1 else "")
            + f'<td class="status" title="{title}">{e(note)}{small}</td></tr>'
        )
    head = (
        '<th>Query</th><th class="num">cpu ms</th><th class="num">metal ms</th>'
        "<th>vs cpu</th>"
        + ("<th>worst vs cpu (variant)</th>" if len(variants) > 1 else "")
        + "<th>Status</th>"
    )
    mean = bench_html._geomean(base_ratios)
    worst_mean = bench_html._geomean(worst_ratios)
    all_mean = bench_html._geomean([r["policy_ratio"] for r in rows if r["policy_ratio"]])
    summary = (
        f"Metal / CPU geometric mean {bench_html._chip(mean)} over {len(base_ratios)} "
        f"queries in <code>{e(base)}</code>"
        + (f"; worst variant per query {bench_html._chip(worst_mean)}" if len(variants) > 1 and worst_mean else "")
        + (f"; all {len(rows)} cases {bench_html._chip(all_mean)}" if len(variants) > 1 and all_mean else "")
        + f". Cases: {counts['ok']} admitted, {counts['wrong answer']} wrong answer, "
        f"{counts['unsupported']} unsupported, {counts['unverified']} unverified."
    )
    spec = suites.SUITES[suite]
    return (
        f'<section id="{e(suite)}">'
        f'<div class="eyebrow">{"development" if spec.get("role") == "dev" else "held out"}</div>'
        f"<h2>{e(suite)}</h2>"
        f'<p class="muted">{e(spec.get("source", ""))}. Data variants: {e(", ".join(variants))}; '
        f"the table shows <code>{e(base)}</code>"
        + (", and the worst variant beside it" if len(variants) > 1 else "")
        + f".</p><p>{summary}</p>"
        '<div class="scroll"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div></section>"
    )


def render(data, analysis, before=None, after=None, receipt_path="", receipt_sha="", timing_coverage=()):
    p = data["provenance"]
    rows = analysis["rows"]
    present = [s for s in suites.SUITES if any(r["suite"] == s for r in rows)]
    memory = p.get("physical_memory_bytes")
    memory_label = f"{memory / (1024 ** 3):.0f} GiB" if positive(memory) else "unavailable"
    oracle_version = p.get("oracle_runtime", {}).get("duckdb_version", "unavailable")
    measured = p.get("recorded_at_utc", "unavailable")
    if measured[:4].isdigit():
        measured = measured[:16].replace("T", " ") + " UTC"
    prov = "".join(
        f"<span>{e(k)} <b>{e(v)}</b></span>"
        for k, v in [
            ("revision", p["engine"]["revision"][:10]),
            ("scale", p["scale"]),
            ("threads", p["threads"]),
            ("rounds × runs", f"{p['rounds']} × {p['repetitions']} (+{p['warmups']} warmup)"),
            ("Mojo", p["mojo"].split(" (")[0]),
            ("machine", p.get("platform", "unavailable")),
            ("memory", memory_label),
            ("DuckDB", oracle_version),
            ("measured", measured),
        ]
    )
    groups = collections.defaultdict(list)
    summary_rows = []
    for suite in present:
        suite_rows = [r for r in rows if r["suite"] == suite]
        classes = [classify(r) for r in suite_rows]
        counts = collections.Counter(c[0] for c in classes)
        for r, (status, reason) in zip(suite_rows, classes):
            if status != "ok":
                groups[(status, reason.split(" (")[0])].append(
                    f"{r['suite']}/{r['variant']}/{r['query']}"
                )
        admitted = [r for r in suite_rows if r["policy_ratio"] is not None]
        worst = {}
        for r in admitted:
            if r["query"] not in worst or r["policy_ratio"] > worst[r["query"]]:
                worst[r["query"]] = r["policy_ratio"]
        spec = suites.SUITES[suite]
        summary_rows.append(
            f'<tr><td class="q"><a href="#{e(suite)}">{e(suite)}</a></td>'
            f"<td>{'development' if spec.get('role') == 'dev' else 'held out'}</td>"
            f'<td class="num">{len(suite_rows)}</td>'
            f"<td>{counts['ok']} admitted, {counts['wrong answer']} wrong answer, "
            f"{counts['unsupported']} unsupported, {counts['unverified']} unverified</td>"
            f"<td>{bench_html._chip(bench_html._geomean([r['policy_ratio'] for r in admitted]))}</td>"
            f"<td>{bench_html._chip(bench_html._geomean(list(worst.values())))}</td></tr>"
        )
    total = collections.Counter(classify(r)[0] for r in rows)
    overall = analysis["policy_geometric_mean"]
    raw = analysis["geometric_mean"]
    excluded = "".join(
        f"<details><summary><b>{e(status)}</b>: {e(reason)} — {len(cases)} cases</summary>"
        f"<p class='status'>{e(', '.join(cases))}</p></details>"
        for (status, reason), cases in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1])))
    ) or "<p class='muted'>None.</p>"
    provenance = json.dumps(
        dict(
            receipt=dict(path=str(receipt_path), sha256=receipt_sha),
            supplied_timing_coverage_receipts=timing_coverage,
            engine=p["engine"]["revision"],
            harness_revision=p.get("harness_revision"),
            harness_status=p.get("harness_status"),
            harness_sha256=p.get("harness_sha256"),
            mojo=p["mojo"],
            platform=p.get("platform"),
            binaries_sha256=p.get("binaries_sha256"),
            case_metadata_sha256=p.get("case_metadata_sha256"),
            timing_scope=p.get("timing_scope"),
            metal_execution_policy=p.get("metal_execution_policy"),
            oracle_policy=p.get("oracle_policy"),
            query_environment=p.get("query_environment"),
            worker_outcomes=analysis["worker_statuses"],
            final_engine_input_checks={
                k: data.get(k, "unavailable")
                for k in ("engine_unchanged_after", "inputs_unchanged_after")
            },
            aborted=data.get("aborted"),
        ),
        indent=2,
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>dataframe_mojo Metal against CPU</title>
<style>{bench_html.STYLE}
details{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 14px}}
summary{{cursor:pointer}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:0.75rem;max-height:480px;overflow:auto}}
td.status small{{display:block}} p.status{{color:var(--muted);font-size:0.8rem;overflow-wrap:anywhere;max-width:none}}
</style></head><body><main>
<header>
<div class="eyebrow">Apple silicon, external benchmark suites</div>
<h1>dataframe_mojo: forced Metal against CPU</h1>
<div class="prov">{prov}</div>
<p class="muted">Times are medians over rounds of the fastest run, in milliseconds, for the complete query call returning a host DataFrame (loading excluded). A ratio is Metal's time divided by CPU's: below 1 means Metal is faster. Metal here is forced for every collect, not chosen by a cost model, and may rerun on CPU after a data fault.</p>
<div class="legend">{''.join(f'<span class="chip {name}">{label}</span>' for _, name, label in bench_html.BANDS)}</div>
</header>
<div class="note"><b>Admitted</b> means the case matches an exact CPU/Metal receipt (full schema and row multiset), agrees with DuckDB, ran every round on a quiet host and has a verified runtime audit. Only admitted cases have a ratio. Status words: <b>ok</b> admitted; <b>wrong answer</b> Metal's result differed; <b>unsupported</b> no Metal translation or the query was rejected; <b>unverified</b> the evidence is incomplete or hybrid. Hover a status for its reason.</div>
<nav>{''.join(f'<a href="#{e(s)}">{e(s)}</a>' for s in present)}<a href="#excluded">excluded cases</a><a href="#provenance">provenance</a></nav>
<section><h2>Summary</h2>
<p>Overall Metal / CPU geometric mean {bench_html._chip(overall)} over {analysis['admitted_cases']} admitted cases ({analysis['admitted_query_ids']} query IDs) of {len(rows)} planned: {total['ok']} admitted, {total['wrong answer']} wrong answer, {total['unsupported']} unsupported, {total['unverified']} unverified. On the raw median of all warm samples the figure is {'unavailable' if raw is None else f'{raw:.3g}'}.</p>
<div class="scroll"><table><thead><tr><th>Suite</th><th>Role</th><th class="num">Cases</th><th>Outcome</th><th>Metal / CPU</th><th>Worst variant</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
{''.join(_suite_section(s, [r for r in rows if r['suite'] == s]) for s in present)}
<section id="excluded"><h2>Excluded cases</h2><p class="muted">Cases without a ratio, grouped by status and reason. Their raw timings are in <code>performance.json</code>.</p>{excluded}</section>
<section id="provenance"><h2>Provenance and audit detail</h2>
<p>{e(data.get('aborted', 'No collector abort recorded.'))} All scheduled workers attempted: {e(data.get('all_scheduled_workers_attempted', 'unavailable'))}. Catalog: {analysis['catalog_query_ids']} query IDs / {analysis['catalog_cases']} cases; planned here {analysis['planned_query_ids']} / {analysis['planned_cases']}.</p>
<p class="muted">{e(coverage_comparison(before, after))}</p>
{coverage_section('Baseline exact coverage', before)}{coverage_section('Final exact coverage', after)}
<details><summary>Recorded identities, hashes and policies</summary><pre>{e(provenance)}</pre></details>
<p class="foot">Generated by <code>benchmarks/tools/apple_performance_html.py</code>. The raw samples, host observations and per-run evidence are in <code>performance.json</code> beside this page.</p></section>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", type=Path)
    parser.add_argument("--baseline-coverage", type=Path, action="append", default=[])
    parser.add_argument("--final-coverage", type=Path, action="append", default=[])
    parser.add_argument(
        "--strict-coverage", type=Path, action="append", default=[],
        help="Exact receipts matching the timing inputs; separate from baseline/final support counts.",
    )
    parser.add_argument(
        "--case-metadata", type=Path, default=TOOLS / "metal_case_metadata.json"
    )
    parser.add_argument("--baseline-catalog", type=Path)
    parser.add_argument("--final-catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (
        [args.timings, args.case_metadata]
        + args.baseline_coverage
        + args.final_coverage
        + args.strict_coverage
        + [p for p in (args.baseline_catalog, args.final_catalog) if p is not None]
    )
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error("HTML output cannot overwrite an input receipt or metadata")
    metadata = read(args.case_metadata)
    coverage.validate_metadata(metadata)
    before = coverage_evidence(args.baseline_coverage, metadata, args.baseline_catalog)
    after = coverage_evidence(args.final_coverage, metadata, args.final_catalog)
    data = read(args.timings)
    supplied = [
        dict(read(p), receipt_sha256=sha(p))
        for p in args.baseline_coverage + args.final_coverage + args.strict_coverage
    ]
    analysis = analyze(data, supplied, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(data, analysis, before, after, args.timings, sha(args.timings),
               timing_coverage=[dict(path=str(p), sha256=sha(p)) for p in args.strict_coverage])
    )
    print(
        f"Rendered {analysis['planned_cases']} planned cases; {analysis['admitted_cases']} admitted pairs. No queries executed."
    )


if __name__ == "__main__":
    main()
