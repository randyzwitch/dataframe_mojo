#!/usr/bin/env python3
"""Summarize immutable forced-Metal coverage receipts; never run a query.

Counts come from the retained catalog and receipts, not a previous report.
Coverage collection_ns is deliberately excluded from performance statistics.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
import bench_suites as suites

STATUSES = {
    "ok",
    "wrong_answer",
    "unsupported",
    "execution_failed",
    "reference_unsupported",
    "reference_failed",
    "process_failed",
    "timeout",
    "protocol_failed",
}
RUNNABLE = {"ok", "wrong_answer"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def catalog(root=ROOT):
    """Read the supported suite dispatch grammar, failing closed on changes."""
    definitions = {
        "h2o_groupby": ("h2o", "groupby"),
        "h2o_join": ("h2o", "join"),
        "pdsh": ("pdsh", "plan"),
        "tpcds": ("tpcds", "plan"),
        "clickbench": ("clickbench", "query"),
    }
    result = {}
    for name, spec in suites.SUITES.items():
        module, function = definitions[name]
        path = root / "benchmarks/suites" / (module + ".mojo")
        source = path.read_text()
        match = re.search(r"(?ms)^def " + function + r"\(.*?(?=^def |\Z)", source)
        if not match:
            raise ValueError(f"Missing reviewed dispatch function: {path}:{function}")
        body = match.group()
        branches = list(re.finditer(r"(?m)^    if (?:q|query|n)\s*==.*:$", body))
        if not branches:
            raise ValueError(f"Unrecognized query dispatch: {path}:{function}")
        translated = set()
        dispatched = set()
        for index, branch in enumerate(branches):
            header = branch.group()
            ids = re.findall(r'"(q\d+)"', header)
            if module == "clickbench":
                ids = ["q" + n for n in re.findall(r"n\s*==\s*(\d+)", header)]
            if not ids:
                raise ValueError(f"Unrecognized dispatch header: {header}")
            dispatched.update(ids)
            end = branches[index + 1].start() if index + 1 < len(branches) else len(
                body
            )
            block = body[branch.end() : end]
            if not re.search(r"(?m)^        raise unsupported\(", block):
                translated.update(ids)
        if not dispatched <= set(spec["queries"]):
            raise ValueError(f"Dispatch outside declared catalog: {name}")
        result[name] = {
            "queries": list(spec["queries"]),
            "variants": list(suites.VARIANTS[name]),
            "translated_queries": [q for q in spec["queries"] if q in translated],
            "role": spec["role"],
            "source_path": str(path.relative_to(root)),
            "source_sha256": digest(path.read_bytes()),
        }
    return result


def key(case):
    return case["suite"], case["variant"], case["query"]


def load_receipts(paths, revision=None):
    """Reject mixed engine/workload measurements and duplicate case receipts."""
    cases = {}
    identity = None
    source_hashes = {}
    receipts = []
    fields = (
        "engine_revision",
        "implementation",
        "mojo",
        "threads",
        "scale",
        "repetitions",
        "driver_sha256",
        "case_metadata_sha256",
        "comparison",
    )
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("format") != "dataframe-metal-suite-coverage-v1":
            raise ValueError(f"Unknown receipt format: {path}")
        provenance = data["provenance"]
        if (
            provenance.get("engine_status")
            or provenance.get("implementation") != "mojo"
        ):
            raise ValueError(f"Dirty or non-Mojo measurement: {path}")
        if revision and provenance["engine_revision"] != revision:
            raise ValueError(f"Unexpected engine revision: {path}")
        observed = {f: provenance.get(f) for f in fields}
        if not observed["engine_revision"]:
            raise ValueError(f"Missing engine revision: {path}")
        if identity is not None and observed != identity:
            raise ValueError(f"Mixed measurement identity: {path}")
        identity = observed
        for source, value in provenance.get("original_source_sha256", {}).items():
            if source in source_hashes and source_hashes[source] != value:
                raise ValueError(f"Mixed workload source: {source}")
            source_hashes[source] = value
        for case in data["cases"]:
            case_key = key(case)
            if case_key in cases:
                raise ValueError(f"Duplicate case: {case_key}")
            if case["status"] not in STATUSES:
                raise ValueError(f"Unknown status: {case['status']}")
            if case["status"] == "ok" and (
                not isinstance(case.get("ordered_equal"), bool)
                or not isinstance(case.get("collections"), int)
                or case["collections"] < 1
            ):
                raise ValueError(
                    f"Successful case lacks comparison/repetition evidence: {case_key}"
                )
            cases[case_key] = dict(case)
        receipts.append({"path": str(path), "sha256": digest(path.read_bytes())})
    if identity is None:
        raise ValueError("At least one receipt is required")
    return cases, identity, source_hashes, receipts


def execution_audit(case, source_hashes, metadata):
    entry = case.get("execution_audit")
    if entry is None:
        entry = metadata.get("cases", {}).get(
            case["suite"] + "/" + case["variant"] + "/" + case["query"]
        )
    if entry is None:
        entry = metadata.get("cases", {}).get(case["suite"] + "/" + case["query"])
    if not entry:
        return {
            "execution_boundary": "gpu_collects",
            "audit_reason": "Adapter forces every collect; other eager operations are not certified.",
        }
    validate_metadata(
        {"format": "dataframe-metal-case-metadata-v1", "cases": {"selected": entry}}
    )
    result = dict(entry)
    if source_hashes.get(entry["source_path"]) != entry["source_sha256"]:
        result.update(
            execution_boundary="unknown",
            audit_reason="Execution audit source hash does not match receipt.",
        )
    return result


def validate_metadata(metadata):
    if metadata.get("format") != "dataframe-metal-case-metadata-v1":
        raise ValueError("Unknown case metadata format")
    for case, entry in metadata.get("cases", {}).items():
        if entry.get("execution_boundary") not in {"gpu_collects", "hybrid", "unknown"}:
            raise ValueError(f"Invalid execution boundary: {case}")
        if entry.get("workload_kind") not in {"canonical", "supplemental"}:
            raise ValueError(f"Invalid workload kind: {case}")
        if not all(
            entry.get(field)
            for field in ("audit_reason", "source_path", "source_sha256")
        ):
            raise ValueError(f"Missing execution audit evidence: {case}")


def cpu_repeat(case, receipt_path):
    """Read old v1 diagnostics without inventing missing observations."""
    result = dict(case.get("diagnostics", {}))
    log = receipt_path.parent / case.get("log", "")
    if log.is_file():
        for line in log.read_text().splitlines():
            if line.startswith("cpu_repeat_equal\t"):
                value = line.split("\t", 1)[1]
                if value in {"True", "False"}:
                    result["cpu_repeat_ordered_equal"] = value == "True"
    return result


def summarize(cases, definitions, source_hashes, metadata):
    expected = {
        (s, v, q)
        for s, d in definitions.items()
        for v in d["variants"]
        for q in d["queries"]
    }
    translated = {
        (s, v, q)
        for s, d in definitions.items()
        for v in d["variants"]
        for q in d["translated_queries"]
    }
    if set(cases) - expected:
        raise ValueError(f"Cases outside catalog: {sorted(set(cases) - expected)}")
    if any(k not in translated and c["status"] in RUNNABLE for k, c in cases.items()):
        raise ValueError(
            "Receipt reports a runnable query absent from the translated source dispatch"
        )
    enriched = []
    for case_key, case in cases.items():
        audit = execution_audit(case, source_hashes, metadata)
        if audit.get("workload_kind") == "supplemental":
            raise ValueError(
                "Supplemental witnesses cannot count as canonical catalog coverage"
            )
        enriched.append(dict(case, execution_audit=audit))

    def counts(rows, excluded=False):
        available = {key(c): c for c in rows}
        matching = {k for k, c in available.items() if c["status"] == "ok"}
        runnable = {k for k, c in available.items() if c["status"] in RUNNABLE}
        all_matching = []
        all_runnable = []
        for s, d in definitions.items():
            for q in d["translated_queries"]:
                variants = {(s, v, q) for v in d["variants"]}
                if variants <= matching:
                    all_matching.append(s + "/" + q)
                if variants <= runnable:
                    all_runnable.append(s + "/" + q)
        complete = expected <= set(cases)
        return {
            "measured_cases": len(rows),
            "statuses": dict(collections.Counter(c["status"] for c in rows)),
            "runnable_cases": len(runnable),
            "matching_cases": len(matching),
            "matching_empty_cases": sum(
                c["status"] == "ok" and c.get("output_rows") == 0 for c in rows
            ),
            "matching_populated_cases": sum(
                c["status"] == "ok"
                and isinstance(c.get("output_rows"), int)
                and c["output_rows"] > 0
                for c in rows
            ),
            "matching_output_shape_unknown": sum(
                c["status"] == "ok" and "output_rows" not in c for c in rows
            ),
            "all_variants_matching_queries": all_matching,
            "all_variants_runnable_queries": all_runnable,
            "matching_percent_of_translated_cases": 100
            * len(matching)
            / len(translated) if complete
            and translated else None,
            "excluded_cases": len(cases) - len(rows) if excluded else 0,
        }

    conservative = [
        c
        for c in enriched
        if c["execution_audit"]["execution_boundary"] not in {"hybrid", "unknown"}
    ]
    missing = collections.defaultdict(list)
    for c in enriched:
        if c["status"] == "unsupported":
            missing[c.get("reason", "")].append("/".join(key(c)))
    return {
        "catalog_queries": sum(len(d["queries"]) for d in definitions.values()),
        "catalog_cases": len(expected),
        "translated_queries": sum(
            len(d["translated_queries"]) for d in definitions.values()
        ),
        "translated_cases": len(translated),
        "complete": expected <= set(cases),
        "unmeasured_cases": ["/".join(k) for k in sorted(expected - set(cases))],
        "raw_adapter": counts(enriched),
        "known_hybrid_and_unknown_audits_excluded": counts(conservative, excluded=True),
        "exclusion_note": "The original translated denominator is retained. Forced collect coverage does not certify every eager query operation executes on GPU.",
        "unsupported_first_blockers": dict(missing),
        "by_suite": {
            s: {
                "catalog_cases": len(d["queries"]) * len(d["variants"]),
                "translated_cases": len(d["translated_queries"]) * len(d["variants"]),
                "measured_cases": sum(c["suite"] == s for c in enriched),
                "statuses": dict(
                    collections.Counter(
                        c["status"] for c in enriched if c["suite"] == s
                    )
                ),
            }
            for s, d in definitions.items()
        },
        "cpu_repeat_ordered_equality": dict(
            collections.Counter(
                str(
                    c.get("diagnostics", {}).get(
                        "cpu_repeat_ordered_equal", "unavailable"
                    )
                )
                for c in enriched
                if c["status"] == "wrong_answer"
            )
        ),
        "cases": enriched,
    }


def render(report):
    s = report["summary"]
    raw = s["raw_adapter"]
    conservative = s["known_hybrid_and_unknown_audits_excluded"]
    return "\n".join(
        [
            "Apple query coverage (exact comparator; not a performance report)",
            "Engine revision: " + report["measurement"]["engine_revision"],
            f"Catalog: {s['catalog_queries']} queries / {s['catalog_cases']} variant cases.",
            f"CPU translations: {s['translated_queries']} queries / {s['translated_cases']} variant cases.",
            f"Measured: {raw['measured_cases']}; unmeasured: {len(s['unmeasured_cases'])}.",
            "Raw statuses: " + json.dumps(raw["statuses"], sort_keys=True),
            f"Excluding known hybrid/unverified audits: {conservative['matching_cases']} exact matches, {conservative['runnable_cases']} runnable; {conservative['excluded_cases']} excluded.",
            f"Matching populated/empty/unknown output: {conservative['matching_populated_cases']}/{conservative['matching_empty_cases']}/{conservative['matching_output_shape_unknown']}.",
            "CPU repeat ordered equality: "
            + json.dumps(s["cpu_repeat_ordered_equality"], sort_keys=True),
            "Complete grid." if s[
                "complete"
            ] else "Partial grid: no complete coverage percentage claimed.",
            "Differences remain differences; CPU repeat instability does not excuse or classify their cause.",
            s["exclusion_note"],
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--revision")
    parser.add_argument(
        "--case-metadata", type=Path, default=TOOLS / "metal_case_metadata.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON report; text written beside it"
    )
    args = parser.parse_args()
    inputs = {p.resolve() for p in args.receipts} | {args.case_metadata.resolve()}
    if (
        args.output.resolve() in inputs
        or args.output.with_suffix(".txt").resolve() in inputs
    ):
        raise ValueError("Report output must not overwrite an immutable input")
    cases, identity, source_hashes, receipts = load_receipts(
        args.receipts, args.revision
    )
    for path in args.receipts:
        data = json.loads(path.read_text())
        for case in data["cases"]:
            cases[key(case)]["diagnostics"] = cpu_repeat(case, path)
    definitions = catalog()
    for definition in definitions.values():
        if source_hashes.get(definition["source_path"]) != definition["source_sha256"]:
            raise ValueError(
                "Catalog query sources differ from measurement: "
                + definition["source_path"]
            )
    metadata = json.loads(args.case_metadata.read_text())
    validate_metadata(metadata)
    report = {
        "format": "dataframe-metal-suite-summary-v1",
        "measurement": identity,
        "receipts": receipts,
        "catalog": definitions,
        "case_metadata_sha256": digest(args.case_metadata.read_bytes()),
        "summary": summarize(cases, definitions, source_hashes, metadata),
        "note": "No historical counts or coverage collection_ns are used as performance evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".txt").write_text(render(report))
    print(render(report), end="")


if __name__ == "__main__":
    main()
