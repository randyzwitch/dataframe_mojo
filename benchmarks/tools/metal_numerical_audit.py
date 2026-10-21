#!/usr/bin/env python3
"""Compare retained CPU/Metal answer multisets under a pinned numeric policy.

This audit checks CPU/Metal agreement for explicitly registered wrong-answer
cases. It does not certify independent query correctness or row ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_POLICY = HERE / "metal_numerical_policy.json"
POLICY_FORMAT = "dataframe-metal-numerical-policy-v1"
ANSWER_FORMAT = "dataframe-answer-schema-v1"
EXPECTED_RTOL = 1e-9
EXPECTED_ATOL = 1e-12
EXPECTED_SOURCES = {
    "benchmarks/suites/h2o.mojo", "benchmarks/suites/pdsh.mojo", "benchmarks/suites/tpcds.mojo"
}

_spec = importlib.util.spec_from_file_location("metal_numerical_compare", HERE / "metal_numerical_compare.py")
_comparator = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_comparator)
compare_rows = _comparator.compare_rows


class AuditInputError(ValueError):
    """Raised when receipts or policy cannot be safely interpreted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict) or policy.get("format") != POLICY_FORMAT:
        raise AuditInputError("policy has unsupported format")
    if type(policy.get("rtol")) not in (int, float) or type(policy.get("atol")) not in (int, float):
        raise AuditInputError("policy rtol/atol must be numeric")
    if policy["rtol"] != EXPECTED_RTOL or policy["atol"] != EXPECTED_ATOL:
        raise AuditInputError("policy tolerances must be exactly rtol=1e-9 and atol=1e-12")
    if policy.get("comparison") != "abs(a-b) <= max(atol, rtol * max(abs(a), abs(b)))":
        raise AuditInputError("policy comparison expression is unsupported")
    pins = policy.get("source_sha256")
    if not isinstance(pins, dict) or set(pins) != EXPECTED_SOURCES:
        raise AuditInputError("policy must pin exactly the H2O, PDS-H, and TPC-DS source files")
    for path, digest in pins.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AuditInputError(f"invalid source hash for {path}")
    cases = policy.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise AuditInputError("policy cases must be a nonempty object")
    for identity, columns in cases.items():
        if not isinstance(identity, str) or identity.count("/") != 1 or not all(identity.split("/")):
            raise AuditInputError(f"invalid policy case identity {identity!r}")
        if not isinstance(columns, list) or not columns or any(not isinstance(col, str) or not col for col in columns):
            raise AuditInputError(f"policy columns for {identity} must be a nonempty string list")
        if len(set(columns)) != len(columns):
            raise AuditInputError(f"duplicate policy column for {identity}")
    return policy


def _resolve_path(raw: Any, root: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AuditInputError(f"missing {role} path")
    path = Path(raw)
    if path.is_file():
        return path
    parts = path.parts
    try:
        marker = len(parts) - 1 - tuple(reversed(parts)).index("answers")
    except ValueError as exc:
        raise AuditInputError(f"{role} path has no answers suffix: {raw}") from exc
    suffix = Path(*parts[marker + 1 :])
    resolved_root, resolved = root.resolve(), (root.resolve() / suffix).resolve()
    if not suffix.parts or resolved_root not in resolved.parents or not resolved.is_file():
        raise AuditInputError(f"{role} artifact not found under answer root: {raw}")
    return resolved


def _validate_descriptor(descriptor: Any, receipt_export: dict[str, Any], export: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(descriptor, dict) or descriptor != receipt_export.get("original_schema"):
        raise AuditInputError("schema sidecar differs from receipt original_schema")
    if descriptor.get("format") != ANSWER_FORMAT:
        raise AuditInputError("unsupported answer schema format")
    if (descriptor.get("engine") != export.get("engine") or type(descriptor.get("repetition")) is not int
            or descriptor.get("repetition") != export.get("repetition")):
        raise AuditInputError("schema engine/repetition differs from export identity")
    rows = descriptor.get("rows")
    if type(rows) is not int or rows < 0:
        raise AuditInputError("schema row count must be a nonnegative integer")
    columns = descriptor.get("columns")
    if not isinstance(columns, list):
        raise AuditInputError("schema columns must be a list")
    names, indexes = set(), set()
    for position, col in enumerate(columns):
        if not isinstance(col, dict) or type(col.get("index")) is not int or col["index"] < 0:
            raise AuditInputError("schema column has invalid index")
        if col["index"] in indexes or not isinstance(col.get("name"), str) or col["name"] in names:
            raise AuditInputError("schema column indexes and names must be unique")
        if col["index"] != position:
            raise AuditInputError("schema column indexes must follow column order")
        dtype = col.get("dtype")
        if (not isinstance(dtype, dict) or not isinstance(dtype.get("name"), str) or not dtype["name"]
                or not isinstance(dtype.get("encoded"), str) or not dtype["encoded"]):
            raise AuditInputError("schema column lacks logical dtype name/encoding")
        indexes.add(col["index"])
        names.add(col["name"])
    if indexes != set(range(len(columns))):
        raise AuditInputError("schema column indexes must be contiguous")
    return columns


def _physical_dtype_matches(name: str, arrow_type: Any, pa: Any) -> bool:
    aliases = {
        "string": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
        "bool": pa.types.is_boolean,
        "int8": pa.types.is_int8, "int16": pa.types.is_int16,
        "int32": pa.types.is_int32, "int64": pa.types.is_int64,
        "uint8": pa.types.is_uint8, "uint16": pa.types.is_uint16,
        "uint32": pa.types.is_uint32, "uint64": pa.types.is_uint64,
        "float32": pa.types.is_float32, "float64": pa.types.is_float64,
        "date": lambda t: pa.types.is_date32(t) or pa.types.is_date64(t),
    }
    if name in aliases:
        return aliases[name](arrow_type)
    match = re.fullmatch(r"(?:decimal32|decimal64|decimal128|decimal)\[(\d+),(\d+)\]", name)
    if match:
        # Parquet retains decimal precision/scale; Arrow may materialize the
        # value in native narrow storage or decimal128. The original Mojo
        # storage dtype remains exact in the hash-verified sidecar comparison.
        return (pa.types.is_decimal(arrow_type) and getattr(arrow_type, "bit_width", 128) in (32, 64, 128)
                and arrow_type.precision == int(match.group(1)) and arrow_type.scale == int(match.group(2)))
    match = re.fullmatch(r"datetime\[(s|ms|us|ns)\]", name)
    if match:
        return pa.types.is_timestamp(arrow_type) and arrow_type.unit == match.group(1)
    raise AuditInputError(f"unsupported logical dtype for physical Parquet check: {name}")


def _load_export(export: dict[str, Any], receipt: dict[str, Any], answer_root: Path, pa: Any) -> tuple[Any, dict[str, Any], list[str], list[tuple[Any, ...]]]:
    if export.get("status") != "written":
        raise AuditInputError("requested comparison export is not written")
    answer = _resolve_path(export.get("path"), answer_root, "answer")
    schema_path = _resolve_path(export.get("schema_path"), answer_root, "schema")
    if sha256(answer) != receipt.get("sha256"):
        raise AuditInputError(f"answer Parquet hash mismatch: {answer}")
    if sha256(schema_path) != receipt.get("schema_sha256"):
        raise AuditInputError(f"answer schema hash mismatch: {schema_path}")
    try:
        descriptor = json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"invalid schema sidecar {schema_path}: {exc}") from exc
    cols = _validate_descriptor(descriptor, receipt, export)
    table = pa.parquet.read_table(answer)
    names = [column["name"] for column in cols]
    if table.column_names != names or table.num_rows != descriptor["rows"]:
        raise AuditInputError("Parquet names or row count differ from schema sidecar")
    for field, col in zip(table.schema, cols):
        if not _physical_dtype_matches(col["dtype"]["name"], field.type, pa):
            raise AuditInputError(f"Parquet physical dtype differs for column {col['name']}: {field.type} vs {col['dtype']['name']}")
    rows = [tuple(row[name] for name in names) for row in table.to_pylist()]
    return table, descriptor, names, rows


def _case_policy(policy: dict[str, Any], case: dict[str, Any]) -> list[str] | None:
    # Registry keys identify query families; their values are eligible output
    # columns and apply across variants. Runtime dtype filtering keeps non-float64 exact.
    return policy["cases"].get(f"{case.get('suite')}/{case.get('query')}")


def _validate_export_identities(exports: Any, query: str) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(exports, list):
        raise AuditInputError(f"{query} lacks answer_exports")
    result = {}
    for entry in exports:
        if not isinstance(entry, dict):
            raise AuditInputError(f"{query} contains malformed answer export")
        engine, repetition = entry.get("engine"), entry.get("repetition")
        if type(engine) is not str or engine not in {"cpu", "metal", "cpu_repeat"} or type(repetition) is not int or repetition < 0:
            raise AuditInputError(f"{query} contains invalid export identity {(engine, repetition)!r}")
        key = engine, repetition
        if key in result:
            raise AuditInputError(f"{query} contains duplicate export identity {key}")
        result[key] = entry
    for key in (("cpu", 0), ("metal", 0)):
        if key not in result or result[key].get("status") != "written":
            raise AuditInputError(f"{query} requires written {key[0]} repetition {key[1]}")
    return result


def audit_coverage(coverage_path: Path, answer_root: Path, policy_path: Path, max_candidate_pairs: int = 1_000_000) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise AuditInputError("CLI requires PyArrow") from exc
    try:
        coverage = json.loads(coverage_path.read_text())
        policy = validate_policy(json.loads(policy_path.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"cannot load audit input: {exc}") from exc
    provenance = coverage.get("provenance", {}).get("evaluation", {})
    receipt_sources = provenance.get("source_sha256", {})
    for source, pinned in policy["source_sha256"].items():
        if receipt_sources.get(source) != pinned:
            raise AuditInputError(f"coverage source hash does not match policy pin: {source}")

    cases = coverage.get("cases")
    if not isinstance(cases, list):
        raise AuditInputError("coverage lacks cases list")
    seen = set()
    results = []
    for case in cases:
        if not isinstance(case, dict):
            raise AuditInputError("coverage contains malformed case")
        identity = case.get("suite"), case.get("variant"), case.get("query")
        if identity in seen:
            raise AuditInputError(f"duplicate coverage case {identity}")
        seen.add(identity)
        status = case.get("status")
        row = {"suite": identity[0], "variant": identity[1], "query": identity[2], "strict_status": status}
        if status == "ok":
            row.update(comparison_status="exact_match", basis="strict_receipt")
            results.append(row)
            continue
        if status in ("unsupported", "reference_unsupported"):
            row.update(comparison_status=status, basis="strict_receipt")
            results.append(row)
            continue
        if status != "wrong_answer":
            row.update(comparison_status="unverified", reason="unknown_strict_status")
            results.append(row)
            continue
        declared_tolerance = _case_policy(policy, case)
        if declared_tolerance is None:
            row.update(comparison_status="unverified", reason="unregistered_wrong_answer")
            results.append(row)
            continue
        exports = _validate_export_identities(case.get("answer_exports"), str(identity[2]))
        baseline, base_desc, names, base_rows = _load_export(exports[("cpu", 0)], exports[("cpu", 0)], answer_root, pa)
        base_columns = {col["name"]: col for col in base_desc["columns"]}
        missing = set(declared_tolerance) - set(base_columns)
        if missing:
            raise AuditInputError(f"policy names absent output columns: {sorted(missing)}")
        tolerant = []
        for name in declared_tolerance:
            dtype_name = base_columns[name]["dtype"]["name"]
            if dtype_name == "float64":
                tolerant.append(name)
            elif not (dtype_name.startswith("decimal") or re.fullmatch(r"(?:u?int)(?:8|16|32|64)", dtype_name)):
                raise AuditInputError(f"registered tolerance field has unsupported logical dtype: {name}={dtype_name}")
        comparisons = []
        for key, export in sorted(exports.items()):
            if key == ("cpu", 0) or export.get("status") != "written":
                continue
            table, descriptor, other_names, other_rows = _load_export(export, export, answer_root, pa)
            if other_names != names:
                raise AuditInputError("CPU/Metal column names differ")
            if descriptor["columns"] != base_desc["columns"]:
                raise AuditInputError("CPU/Metal logical dtype name/encoding differs")
            outcome = compare_rows(base_rows, other_rows, names, tolerant, policy["rtol"], policy["atol"], max_candidate_pairs)
            comparisons.append({
                "engine": key[0], "repetition": key[1],
                "comparison_status": {"equal": "tolerance_match", "different": "different", "unverified": "unverified"}[outcome["status"]],
                "answer_sha256": export["sha256"], "schema_sha256": export["schema_sha256"],
                "rows": outcome["actual_rows"], "comparison": outcome,
            })
        row.update(
            comparison_status=("tolerance_match" if all(item["comparison_status"] == "tolerance_match" for item in comparisons)
                               else "different" if any(item["comparison_status"] == "different" for item in comparisons)
                               else "unverified"),
            tolerant_columns=tolerant,
            baseline={"engine": "cpu", "repetition": 0, "answer_sha256": exports[("cpu", 0)]["sha256"],
                      "schema_sha256": exports[("cpu", 0)]["schema_sha256"], "rows": len(base_rows)},
            comparisons=comparisons,
            reason=None if comparisons else "no_written_comparison_exports",
        )
        results.append(row)
    return {
        "audit": "metal_numerical_answer_agreement",
        "engine_revision": coverage.get("provenance", {}).get("engine_revision"),
        "coverage_path": str(coverage_path), "coverage_sha256": sha256(coverage_path),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "comparator_sha256": sha256(HERE / "metal_numerical_compare.py"),
        "policy_path": str(policy_path), "policy_sha256": sha256(policy_path),
        "rtol": policy["rtol"], "atol": policy["atol"],
        "source_sha256": policy["source_sha256"],
        "comparison_scope": "Unordered CPU/Metal answer multiset agreement only; does not establish independent correctness or ordered behavior.",
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--answer-root", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_coverage(args.coverage, args.answer_root, args.policy)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    except (AuditInputError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
