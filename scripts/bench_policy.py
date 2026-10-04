"""Exposure and workload identity for benchmark reports (no timing policy)."""

import hashlib
import json

POLICY_VERSION = 2
POLICY_NOTE = (
    "Tune on H2O and mechanism benchmarks; use PDS-H and ClickBench as "
    "held-out validation. Some prior optimization work used holdout results; "
    "the development/holdout boundary is enforced going forward."
)
LEGACY_NOTE = (
    "Legacy result: workload hashes and exposure metadata were not recorded. "
    "Raw measurements are unchanged. "
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
        "policy": "development_tuning_holdout_validation",
        "policy_effective": "2026-10-04",
        "suites": {
            name: {
                "role": "development" if suites[name]["role"]
                == "dev" else "heldout",
                "intended_use": "tuning" if suites[name]["role"]
                == "dev" else "validation_only",
                "prior_exposure": "used_for_development" if suites[name]["role"]
                == "dev" else "some_prior_optimization_use",
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
        return LEGACY_NOTE + POLICY_NOTE
    return POLICY_NOTE


def distinct_queries(cases):
    """A suite/query is one query even when several data variants reach it."""
    return {
        (parts[0], parts[-1]) for parts in (case.split("/") for case in cases)
    }
