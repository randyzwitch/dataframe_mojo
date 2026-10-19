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
            row["ratio"] = cpu["median_ns"] / metal["median_ns"]
            row["policy_ratio"] = (
                cpu["median_round_minimum_ns"] / metal["median_round_minimum_ns"]
            )
        rows.append(row)
    admitted = [r for r in rows if r["ratio"] is not None]
    worst = {}
    for row in admitted:
        key = (row["suite"], row["query"])
        if key not in worst or row["ratio"] < worst[key]["ratio"]:
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
        worst_variants=sorted(worst.values(), key=lambda r: r["ratio"]),
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
        "eligible_separate_exact_and_external_summary_evidence": "Admitted: matching exact receipt and independent summary checks",
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


def milliseconds(value):
    return f"{value / 1e6:.4g} ms"


def timing_cell(stats, phase):
    if stats is None:
        return "<span class='muted'>unmeasured</span>"
    headline = stats["median_ns"]
    policy = (
        "median of round minima "
        + milliseconds(stats["median_round_minimum_ns"])
        + "<br>"
    ) if phase == "warm" else ""
    return f"<strong>raw median {milliseconds(headline)}</strong><small>{policy}range {milliseconds(stats['minimum_ns'])}–{milliseconds(stats['maximum_ns'])}<br>{stats['samples']} samples / {stats['observed_rounds']} observed rounds</small>"


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


def render(data, analysis, before=None, after=None, receipt_path="", receipt_sha="", timing_coverage=()):
    p = data["provenance"]
    ratio = (
        "unavailable" if analysis["geometric_mean"]
        is None else f"{analysis['geometric_mean']:.3g}×"
    )
    policy_ratio = (
        "unavailable" if analysis["policy_geometric_mean"]
        is None else f"{analysis['policy_geometric_mean']:.3g}×"
    )
    rows = []
    for row in analysis["rows"]:
        key = "/".join(coverage.key(row))
        statuses = {
            engine: dict(
                collections.Counter(
                    r["status"] for r in row["runs"] if r["engine"] == engine
                )
            )
            for engine in collector.ENGINES
        }
        status_text = "; ".join(
            engine
            + ": "
            + (
                json.dumps(statuses[engine], sort_keys=True) if statuses[
                    engine
                ] else "unmeasured"
            )
            for engine in collector.ENGINES
        )
        strict = row.get("confirmed_strict_basis", {})
        peaks = {
            engine: max(
                (
                    r.get("peak_rss_bytes") or 0
                    for r in row["runs"]
                    if r["engine"] == engine
                ),
                default=0,
            )
            for engine in collector.ENGINES
        }
        memory_text = "; ".join(
            engine
            + " peak RSS "
            + (f"{value / (1024 ** 3):.3g} GiB" if value else "unavailable")
            for engine, value in peaks.items()
        )
        reasons = "\n".join(
            f"{r['engine']} round {r['round']}: {r['status']} {r.get('reason', '')} | log {r.get('log', 'unavailable')}"
            for r in row["runs"]
        )
        details = json.dumps(
            dict(
                recorded_strict_basis=row.get("strict_basis", {}),
                confirmed_strict_basis=strict,
                ratio_basis=row["ratio_basis"],
                runs=row["runs"],
            ),
            indent=2,
        )
        search = (
            key
            + " "
            + status_text
            + " "
            + row["ratio_basis"]
            + " "
            + str(strict.get("status", "unmeasured"))
        )
        cells = "".join(
            "<td>" + timing_cell(row["times"][engine][phase], phase) + "</td>"
            for phase in ("cold", "warm")
            for engine in collector.ENGINES
        )
        value = (
            "excluded" if row["ratio"]
            is None else f"{row['ratio']:.3g}×<small>round-minimum policy {row['policy_ratio']:.3g}×</small>"
        )
        rows.append(
            f"<tr data-search='{e(search.lower())}'><th>{e(key)}</th>{cells}<td>{value}</td><td>{e(status_text)}<small>exact basis: {e(strict.get('status', 'unmeasured'))}<br>{e(basis_label(row['ratio_basis']))}<br>{e(memory_text)}</small><details><summary>Evidence / failures</summary><pre>{e(reasons)}</pre><pre>{e(details)}</pre></details></td></tr>"
        )
    worst = "".join(
        f"<tr><td>{e(r['suite'] + '/' + r['query'])}</td><td>{e(r['variant'])}</td><td>{r['ratio']:.3g}×</td><td>{sum(x['suite'] == r['suite'] and x['query'] == r['query'] and x['ratio'] is not None for x in analysis['rows'])} / {len(suites.VARIANTS[r['suite']])}</td></tr>"
        for r in analysis["worst_variants"]
    )
    if not worst:
        worst = "<tr><td colspan='4'>No admitted paired cases.</td></tr>"
    provenance = json.dumps(
        dict(
            receipt=dict(path=str(receipt_path), sha256=receipt_sha),
            supplied_timing_coverage_receipts=timing_coverage,
            provenance=p,
            final_engine_input_checks={
                k: data.get(k, "unavailable")
                for k in ("engine_unchanged_after", "inputs_unchanged_after")
            },
        ),
        indent=2,
    )
    memory = p.get("physical_memory_bytes")
    memory_label = f"{memory / (1024 ** 3):.2f} GiB" if positive(
        memory
    ) else "unavailable"
    oracle_version = p.get("oracle_runtime", {}).get(
        "duckdb_version", "unavailable / not recorded"
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apple CPU and forced-Metal query report</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;color:#162436;background:#f4f6f9}}main{{max-width:1500px;margin:auto;padding:28px}}h1,h2{{line-height:1.2}}section,.intro{{background:white;padding:22px;margin:20px 0;border:1px solid #dce2ea;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #e0e5ec;padding:10px}}thead th{{background:#eaf0f7;position:sticky;top:0}}small{{display:block;color:#566477;margin-top:5px}}.scroll{{overflow:auto}}.muted{{color:#657286}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f1f4f8;padding:12px;max-height:500px;overflow:auto}}input{{font:inherit;padding:8px;width:min(90%,500px)}}.metric{{font-size:28px;font-weight:700}}details{{margin:10px 0}}summary{{cursor:pointer}}code{{overflow-wrap:anywhere}}@media print{{input{{display:none}}thead th{{position:static}}body{{background:white}}main{{padding:0}}}}
</style><main>
<h1>Apple CPU and forced-Metal query report</h1>
<div class="intro"><p>Recorded engine <code>{e(p['engine']['revision'])}</code>; scale <strong>{e(p['scale'])}</strong>, {p['threads']} threads, {p['rounds']} rounds × {p['repetitions']} warm repetitions, {p['warmups']} extra warmup(s). {e('Full/default-tier attempt.' if p.get('headline_full_tier_attempt') else 'Supplemental or partial-tier attempt; not a replacement for the full tier.')}</p>
<p><strong>SDK:</strong> {e(p['mojo'])}<br><strong>Machine:</strong> {e(p.get('platform', 'unavailable'))}; physical memory {memory_label}<br><strong>DuckDB:</strong> {e(oracle_version)}<br><strong>Recorded:</strong> {e(p.get('recorded_at_utc', 'unavailable'))}</p>
<p class="metric">{ratio} CPU / forced-Metal warm raw-median geometric mean</p><p>Median-of-round-minima policy geometric mean on the same admitted pairs: <strong>{policy_ratio}</strong>.</p>
<p>Only {analysis['admitted_cases']} admitted paired variant cases across {analysis['admitted_query_ids']} distinct query IDs enter this figure. A ratio above 1 means Metal was faster; below 1 means CPU was faster. No ratio is computed for wrong, unverified, hybrid, failed, incomplete or contaminated evidence.</p>
<p>Warm headline: median of all recorded warm samples, with raw ranges. The existing benchmark policy statistic—the median of each round's fastest complete collection—is labeled separately, including its ratio on the same admitted pairs. Cold: median of first query invocations in fresh workers after untimed table loading; filesystem/driver/compiler caches are not claimed cold. Warmups stay in raw evidence and never enter ratios. Timings displayed for excluded cases are observations, not validated speedups.</p>
<p><strong>Backend timing:</strong> Metal measurements cover the complete requested backend path. When the engine uses CPU data-fault adjudication, a query may rerun on CPU and that time is included. These figures are not isolated GPU-kernel timings or a certificate of GPU-only execution.</p>
<p>Separate exact CPU/GPU receipts and the existing DuckDB summary/CHECKS policy remain distinct. The external policy cannot turn a strict mismatch into a pass. Forced collect boundaries do not certify every query operation runs on GPU; source boundary audits remain attached. Load/compiler snapshots do not certify absence of all competing work. Per-worker peak RSS includes table loading and untimed summaries; it is not a GPU-only allocation peak.</p></div>
<section><h2>Measurement coverage</h2><p>Retained catalog: {analysis['catalog_query_ids']} query IDs / {analysis['catalog_cases']} variant cases. Planned here: {analysis['planned_query_ids']} IDs / {analysis['planned_cases']} cases. Worker outcomes: {e(json.dumps(analysis['worker_statuses'], sort_keys=True))}.</p>
<p>{e(data.get('aborted', 'No collector abort recorded.'))} All scheduled workers attempted: {e(data.get('all_scheduled_workers_attempted', 'unavailable'))}.</p><details><summary>{len(analysis['unplanned_cases'])} catalog cases not planned in this receipt</summary><pre>{e(chr(10).join('/'.join(k) for k in analysis['unplanned_cases']))}</pre></details></section>
<section><h2>Baseline and final exact coverage</h2><p>{e(coverage_comparison(before, after))}</p><p>Coverage collection_ns values are not used in any performance statistic.</p></section>
{coverage_section('Baseline exact coverage', before)}{coverage_section('Final exact coverage', after)}
<section><h2>Complete collection timings and exclusions</h2><label>Filter case or outcome <input id="filter" type="search" placeholder="e.g. h2o_groupby/q3, unsupported, wrong_answer"></label><small>Filtering changes visible rows only; report metrics keep their original denominator.</small><div class="scroll"><table id="cases"><thead><tr><th>Suite / variant / query</th><th>CPU cold</th><th>Metal cold</th><th>CPU warm</th><th>Metal warm</th><th>CPU / Metal</th><th>Outcomes and admission basis</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Worst admitted variant per query</h2><p>The minimum raw-median speedup among admitted variants is shown. Excluded or unmeasured variants are not silently treated as passing; the coverage column exposes how many variants entered.</p><table><thead><tr><th>Suite / query ID</th><th>Worst admitted variant</th><th>CPU / Metal warm</th><th>Admitted / catalog variants</th></tr></thead><tbody>{worst}</tbody></table></section>
<section><h2>Workload, machine, SDK and provenance</h2><p>No historical timing is invented or imported from coverage durations. Engine package identity and measurement-harness identity are recorded separately; changed adapters require a same-harness baseline cross-run.</p><details><summary>Full recorded provenance</summary><pre>{e(provenance)}</pre></details></section>
</main><script>document.getElementById('filter').addEventListener('input',function(){{let text=this.value.toLowerCase();document.querySelectorAll('#cases tbody tr').forEach(row=>{{row.hidden=!row.dataset.search.includes(text)}})}});</script></html>"""


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
