"""Exposure and workload identity for benchmark reports (no timing policy)."""

import hashlib
import json

POLICY_VERSION = 1
DEVELOPMENT_NOTE = (
    "Development benchmark results: all bundled suites have influenced engine "
    "optimization. External origin and new data seeds do not make these "
    "independent validation workloads. No independent validation is claimed."
)
LEGACY_NOTE = (
    "Legacy result: this file has no recorded workload-exposure metadata. "
    "The bundled suites have been used for development; previous held-out "
    "labels do not establish independence. Raw measurements are unchanged. "
    "No independent validation is claimed."
)

# A documented lower bound on exposure, not a claim about the first exposure.
EXPOSURE_EVIDENCE = {
    "h2o_groupby": "docs/benchmarks.md already designates H2O as development",
    "h2o_join": "docs/benchmarks.md already designates H2O as development",
    "pdsh": "dataframe/lazy.mojo join ordering and streaming policy cite PDS-H queries",
    "clickbench": "dataframe/lazy.mojo group-by policy cites ClickBench per-query timings",
}


def record(root, selected, suites):
    """Record local source identity without pretending it pins upstream."""
    paths = sorted(
        p
        for p in (root / "benchmarks/suites").iterdir()
        if p.suffix in (".py", ".mojo")
    )
    paths += [
        root / "scripts" / name
        for name in ("bench_suites.py", "bench_html.py", "bench_policy.py")
    ]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    return {
        "policy_version": POLICY_VERSION,
        "independent_validation": False,
        "suites": {
            name: {
                "role": "development",
                "exposure": "used_for_development",
                "evidence": EXPOSURE_EVIDENCE[name],
                "source": suites[name]["source"],
                "upstream_revision": None,
            }
            for name in selected
        },
        "source_sha256": hashes,
        "manifest_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "upstream_note": (
            "Hashes identify the local query translations, generators and runner. "
            "Historical upstream revisions were not recorded; local hashes are "
            "not upstream commit pins. DuckDB's version identifies tpch_queries/dbgen."
        ),
    }


def report_note(result):
    if not result.get("provenance", {}).get("evaluation"):
        return LEGACY_NOTE
    return DEVELOPMENT_NOTE


def distinct_queries(cases):
    """A suite/query is one query even when several data variants reach it."""
    return {
        (parts[0], parts[-1]) for parts in (case.split("/") for case in cases)
    }
