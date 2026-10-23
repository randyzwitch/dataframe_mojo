#!/usr/bin/env python3
"""Independent exact-selection oracle for nine retained ClickBench queries.

This audit checks semantic selection evidence only. It does not change strict
coverage status, support other queries, or create performance eligibility.
The reference query computes each full relation from the hash-verified
source input; retained CPU/Metal exports are checked for exact row membership,
multiplicity, row count, and (where declared) descending rank sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence


JULY = "CounterID = 62 AND EventDate >= DATE '2013-07-01' AND EventDate <= DATE '2013-07-31'"
# IsRefresh sums are bounded by the checked row count; BIGINT matches retained Int64.
QUERY_SQL = {
    "q11": "SELECT MobilePhone, MobilePhoneModel, count(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel",
    "q17": "SELECT UserID, SearchPhrase, count(*) AS c FROM hits GROUP BY UserID, SearchPhrase",
    "q18": "SELECT UserID, minute(EventTime) AS m, SearchPhrase, count(*) AS c FROM hits GROUP BY UserID, minute(EventTime), SearchPhrase",
    "q30": "SELECT SearchEngineID, ClientIP, count(*) AS c, CAST(sum(IsRefresh) AS BIGINT) AS r, avg(ResolutionWidth) AS w FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP",
    "q31": "SELECT WatchID, ClientIP, count(*) AS c, CAST(sum(IsRefresh) AS BIGINT) AS r, avg(ResolutionWidth) AS w FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP",
    "q32": "SELECT WatchID, ClientIP, count(*) AS c, CAST(sum(IsRefresh) AS BIGINT) AS r, avg(ResolutionWidth) AS w FROM hits GROUP BY WatchID, ClientIP",
    "q38": f"SELECT URL, count(*) AS PageViews FROM hits WHERE {JULY} AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL",
    "q39": f"SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN SearchEngineID = 0 AND AdvEngineID = 0 THEN Referer ELSE '' END AS Src, URL AS Dst, count(*) AS PageViews FROM hits WHERE {JULY} AND IsRefresh = 0 GROUP BY ALL",
    "q40": f"SELECT URLHash, EventDate, count(*) AS PageViews FROM hits WHERE {JULY} AND IsRefresh = 0 AND TraficSourceID IN (-1,6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate",
}
RANK_COLUMN = {
    "q11": "u", "q17": None, "q18": "c", "q30": "c", "q31": "c",
    "q32": "c", "q38": "PageViews", "q39": "PageViews", "q40": "PageViews",
}
OFFSET = {"q38": 1000, "q39": 1000, "q40": 100}
LIMIT = 10
SUPPORTED_QUERIES = tuple(QUERY_SQL)
SOURCE_PATH = "benchmarks/suites/clickbench.mojo"
REVIEWED_SOURCE_SHA256 = "451f54c051715d0f4c3b72364037a4688a4b52fee8ec2be11b6a2b6c610c90ee"


class OracleInputError(ValueError):
    """Raised when an input cannot be checked without guessing or coercion."""


def _checked_nonnull_rows(rows: Iterable[Sequence[Any]], label: str) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    width: int | None = None
    column_types: list[type] | None = None
    for row_number, row in enumerate(rows):
        if not isinstance(row, (tuple, list)):
            raise OracleInputError(f"{label} row {row_number} is not a tuple/list")
        values = tuple(row)
        if width is None:
            width = len(values)
            column_types = [type(value) for value in values]
        if len(values) != width:
            raise OracleInputError(f"{label} row {row_number} has inconsistent width")
        assert column_types is not None
        for column, value in enumerate(values):
            if value is None:
                raise OracleInputError(f"{label} row {row_number}, column {column} is null")
            if type(value) is not column_types[column]:
                raise OracleInputError(f"{label} row {row_number}, column {column} changes type")
            if isinstance(value, float) and not math.isfinite(value):
                raise OracleInputError(f"{label} row {row_number}, column {column} is non-finite")
            if isinstance(value, Decimal) and not value.is_finite():
                raise OracleInputError(f"{label} row {row_number}, column {column} is non-finite")
        result.append(values)
    return result


def _typed_row(row: tuple[Any, ...]) -> tuple[tuple[type, Any], ...]:
    # Python considers True == 1 and 1 == 1.0. Exact row membership must not.
    return tuple((type(value), struct.pack("!d", value) if type(value) is float else value) for value in row)


def validate_source_identity(source_hashes: Any) -> str:
    """Require the independently reviewed ClickBench suite source revision."""
    if not isinstance(source_hashes, dict) or source_hashes.get(SOURCE_PATH) != REVIEWED_SOURCE_SHA256:
        raise OracleInputError(f"coverage source identity does not match reviewed hash for {SOURCE_PATH}")
    return REVIEWED_SOURCE_SHA256


def validate_selection(
    full_rows: Iterable[Sequence[Any]],
    actual_rows: Iterable[Sequence[Any]],
    sort_index: int | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    """Validate a selected relation against its full reference relation.

    Rows are compared exactly and with multiplicity. Rank ties are allowed:
    only the selected descending rank sequence is fixed, not row order among
    equal ranks. With ``sort_index=None``, any valid subset of the expected
    count is accepted, which models an unordered group head.

    Invalid input shapes, nulls, non-finite values, and invalid limit/offset
    arguments raise ``OracleInputError``; they never receive a pass.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise OracleInputError("limit must be a nonnegative integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise OracleInputError("offset must be a nonnegative integer")
    full = _checked_nonnull_rows(full_rows, "full relation")
    actual = _checked_nonnull_rows(actual_rows, "actual relation")
    if full and actual and len(full[0]) != len(actual[0]):
        raise OracleInputError("actual and full relation column counts differ")
    if full and actual:
        for i, (expected, observed) in enumerate(zip(full[0], actual[0])):
            if type(expected) is not type(observed):
                raise OracleInputError(f"actual column {i} dtype differs from reference")
    if sort_index is not None:
        if isinstance(sort_index, bool) or not isinstance(sort_index, int):
            raise OracleInputError("sort_index must be an integer or None")
        width = len(full[0]) if full else (len(actual[0]) if actual else None)
        if sort_index < 0 or (width is not None and sort_index >= width):
            raise OracleInputError("sort_index is outside the relation schema")
    expected_count = min(limit, max(0, len(full) - offset))
    full_counts = Counter(_typed_row(row) for row in full)
    actual_counts = Counter(_typed_row(row) for row in actual)
    invalid_rows = sum((actual_counts - full_counts).values())
    missing_rows = sum((full_counts - actual_counts).values())
    rank_correct: bool | None = None
    expected_ranks: list[Any] | None = None
    actual_ranks: list[Any] | None = None
    if sort_index is not None:
        try:
            expected_ranks = sorted((row[sort_index] for row in full), reverse=True)[offset : offset + limit]
            actual_ranks = [row[sort_index] for row in actual]
            rank_correct = actual_ranks == expected_ranks
        except TypeError as exc:
            raise OracleInputError("rank column values are not mutually orderable") from exc
    exact_membership = invalid_rows == 0
    count_correct = len(actual) == expected_count
    valid = exact_membership and count_correct and (rank_correct is None or rank_correct)
    return {
        "valid": valid,
        "exact_membership": exact_membership,
        "invalid_row_multiplicity": invalid_rows,
        "missing_full_row_multiplicity": missing_rows,
        "row_count_correct": count_correct,
        "expected_row_count": expected_count,
        "actual_row_count": len(actual),
        "full_row_count": len(full),
        "rank_sequence_correct": rank_correct,
        "expected_rank_sequence": expected_ranks,
        "actual_rank_sequence": actual_ranks,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_receipt_path(raw_path: str, root: Path, role: str) -> Path:
    """Resolve a receipt path by its declared suite suffix, never basename search."""
    path = Path(raw_path)
    if path.is_file():
        return path
    parts = path.parts
    if role == "input":
        try:
            marker = parts.index("clickbench")
        except ValueError as exc:
            raise OracleInputError(f"input receipt path has no clickbench suffix: {raw_path}") from exc
        suffix = Path(*parts[marker:])
    elif role in ("answer", "schema"):
        try:
            marker = len(parts) - 1 - tuple(reversed(parts)).index("answers")
        except ValueError as exc:
            raise OracleInputError(f"answer receipt path has no answers suffix: {raw_path}") from exc
        suffix = Path(*parts[marker + 1 :])
    else:
        raise OracleInputError(f"unknown receipt path role {role}")
    if not suffix.parts or any(part in ("..", "") for part in suffix.parts):
        raise OracleInputError(f"unsafe receipt suffix: {raw_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / suffix).resolve()
    if resolved_root not in resolved.parents:
        raise OracleInputError(f"receipt path escapes {role} root: {raw_path}")
    if not resolved.is_file():
        raise OracleInputError(f"receipt artifact not found at exact suffix {suffix} under {root}")
    return resolved


def _dtype_name(arrow_type: Any, pa: Any) -> str:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_floating(arrow_type):
        return {"double": "float64", "float": "float32"}.get(str(arrow_type), str(arrow_type))
    if pa.types.is_integer(arrow_type):
        return str(arrow_type)
    if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return f"datetime[{arrow_type.unit}]"
    return str(arrow_type)


def _check_schema_export(export: dict[str, Any], answer_root: Path, receipt_export: dict[str, Any], pa: Any) -> tuple[Path, dict[str, Any], Any]:
    answer = _resolve_receipt_path(export["path"], answer_root, "answer")
    schema_path = _resolve_receipt_path(export["schema_path"], answer_root, "schema")
    if _sha256(answer) != receipt_export.get("sha256"):
        raise OracleInputError(f"answer parquet hash differs from coverage receipt: {answer}")
    if _sha256(schema_path) != receipt_export.get("schema_sha256"):
        raise OracleInputError(f"answer schema hash differs from coverage receipt: {schema_path}")
    descriptor = json.loads(schema_path.read_text())
    original = receipt_export.get("original_schema")
    if not isinstance(original, dict):
        raise OracleInputError("coverage export lacks original_schema")
    if (
        descriptor != original
        or descriptor.get("engine") != export.get("engine")
        or descriptor.get("repetition") != export.get("repetition")
        or descriptor.get("format") != "dataframe-answer-schema-v1"
    ):
        raise OracleInputError(f"schema descriptor differs from receipt: {schema_path}")
    table = pa.parquet.read_table(answer)
    desc_cols = original.get("columns", [])
    expected_names = [column["name"] for column in desc_cols]
    expected_dtypes = [column["dtype"]["name"] for column in desc_cols]
    actual_names = table.column_names
    actual_dtypes = [_dtype_name(field.type, pa) for field in table.schema]
    if actual_names != expected_names or actual_dtypes != expected_dtypes:
        raise OracleInputError(f"Parquet schema does not match receipt for {answer}: {(actual_names, actual_dtypes)} != {(expected_names, expected_dtypes)}")
    if table.num_rows != original.get("rows"):
        raise OracleInputError(f"Parquet row count differs from receipt schema: {answer}")
    return answer, descriptor, table


def _query_reference(duckdb: Any, pa: Any, input_path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]], Any]:
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(input_path)).create_view("hits", replace=True)
        table = connection.execute(sql).to_arrow_table()
        names = table.column_names
        rows = [tuple(row[name] for name in names) for row in table.to_pylist()]
        return names, rows, table.schema
    finally:
        connection.close()


def audit_coverage(coverage_path: Path, answer_root: Path, input_root: Path) -> dict[str, Any]:
    try:
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise OracleInputError("CLI requires DuckDB and PyArrow in the oracle environment") from exc

    coverage = json.loads(coverage_path.read_text())
    cases = coverage.get("cases")
    if not isinstance(cases, list):
        raise OracleInputError("coverage receipt lacks cases list")
    selected: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str]] = set()
    for case in cases:
        if case.get("suite") != "clickbench" or case.get("query") not in SUPPORTED_QUERIES:
            continue
        identity = (case["suite"], case.get("variant", ""), case["query"])
        if identity in identities:
            raise OracleInputError(f"duplicate receipt identity {identity}")
        identities.add(identity)
        query = case["query"]
        if query in selected:
            raise OracleInputError(f"multiple receipt cases for supported query {query}")
        selected[query] = case
    missing = sorted(set(SUPPORTED_QUERIES) - set(selected))
    if missing:
        raise OracleInputError(f"coverage receipt missing supported queries: {missing}")

    source_hashes = coverage.get("provenance", {}).get("evaluation", {}).get("source_sha256", {})
    reviewed_source_hash = validate_source_identity(source_hashes)
    report_cases = []
    for query in SUPPORTED_QUERIES:
        case = selected[query]
        input_record = case.get("inputs", {}).get("hits")
        if not input_record or not input_record.get("sha256"):
            raise OracleInputError(f"{query} lacks source input path/hash")
        input_path = _resolve_receipt_path(input_record["path"], input_root, "input")
        source_sha = _sha256(input_path)
        if source_sha != input_record["sha256"]:
            raise OracleInputError(f"source input hash differs from receipt: {input_path}")
        names, full_rows, oracle_schema = _query_reference(duckdb, pa, input_path, QUERY_SQL[query])
        rank_name = RANK_COLUMN[query]
        if rank_name is not None and rank_name not in names:
            raise OracleInputError(f"{query} SQL rank column {rank_name} is absent; got {names}")
        rank_index = names.index(rank_name) if rank_name is not None else None
        offset = OFFSET.get(query, 0)
        exports = case.get("answer_exports")
        if not isinstance(exports, list):
            raise OracleInputError(f"{query} lacks answer_exports")
        receipt_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in exports:
            engine, repetition = entry.get("engine"), entry.get("repetition")
            if engine not in {"cpu", "metal", "cpu_repeat"}:
                raise OracleInputError(f"{query} has unknown export engine {engine!r}")
            if type(repetition) is not int or repetition < 0:
                raise OracleInputError(f"{query} has invalid export repetition {repetition!r}")
            key = (engine, repetition)
            if key in receipt_by_identity:
                raise OracleInputError(f"duplicate answer export identity for {query}: {key}")
            receipt_by_identity[key] = entry
        for required in (("cpu", 0), ("metal", 0)):
            if required not in receipt_by_identity or receipt_by_identity[required].get("status") != "written":
                raise OracleInputError(f"{query} requires written {required[0]} repetition {required[1]} export")
        cpu_original = receipt_by_identity[("cpu", 0)].get("original_schema")
        if not isinstance(cpu_original, dict):
            raise OracleInputError(f"{query} CPU export lacks original_schema")
        expected_names = [column["name"] for column in cpu_original.get("columns", [])]
        expected_dtypes = [column["dtype"]["name"] for column in cpu_original.get("columns", [])]
        oracle_names = list(oracle_schema.names)
        oracle_dtypes = [_dtype_name(field.type, pa) for field in oracle_schema]
        if oracle_names != names or oracle_names != expected_names or oracle_dtypes != expected_dtypes:
            raise OracleInputError(f"{query} independent SQL schema does not match retained output schema: {(oracle_names, oracle_dtypes)} != {(expected_names, expected_dtypes)}")
        written = sorted((e for e in exports if e.get("status") == "written"), key=lambda e: (e.get("engine") != "cpu", e.get("engine"), e.get("repetition", -1)))
        outcomes = []
        cpu_schema = None
        for export in written:
            receipt_export = receipt_by_identity[(export["engine"], export["repetition"])]
            answer_path, descriptor, table = _check_schema_export(export, answer_root, receipt_export, pa)
            if table.column_names != names:
                raise OracleInputError(f"{query}/{export['engine']} SQL names differ: {table.column_names} != {names}")
            if export["engine"] == "cpu":
                cpu_schema = table.schema
            elif cpu_schema is not None and not table.schema.equals(cpu_schema, check_metadata=False):
                raise OracleInputError(f"{query}/{export['engine']} schema differs from CPU schema")
            actual_rows = [tuple(row[name] for name in names) for row in table.to_pylist()]
            outcome = validate_selection(full_rows, actual_rows, rank_index, LIMIT, offset)
            outcomes.append({
                "engine": export["engine"],
                "repetition": export["repetition"],
                "strict_status": case.get("status"),
                "selection_valid": outcome["valid"],
                "validation": outcome,
                "answer_sha256": receipt_export["sha256"],
                "schema_sha256": receipt_export["schema_sha256"],
            })
        report_cases.append({
            "suite": "clickbench", "query": query, "variant": case.get("variant"),
            "strict_status": case.get("status"),
            "oracle_rows": len(full_rows), "expected_selected_rows": min(LIMIT, max(0, len(full_rows) - offset)),
            "limit": LIMIT, "offset": offset, "rank_column": rank_name,
            "sql_sha256": hashlib.sha256(QUERY_SQL[query].encode()).hexdigest(),
            "source_identity": SOURCE_PATH,
            "source_identity_sha256": reviewed_source_hash,
            "input_path": str(input_path), "input_sha256": source_sha,
            "oracle_schema": str(oracle_schema), "exports": outcomes,
        })
    return {
        "audit": "independent_clickbench_ordering_oracle",
        "oracle": f"DuckDB {duckdb.__version__}",
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "coverage_path": str(coverage_path),
        "coverage_sha256": _sha256(coverage_path),
        "scope": "Nine explicit ClickBench queries. Exact row membership/multiplicity, expected count, and descending rank sequence where specified. Ties among equal ranks are permitted. Unordered q17 accepts any valid subset. Does not change strict status or performance eligibility.",
        "supported_queries": list(SUPPORTED_QUERIES),
        "cases": report_cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path, help="coverage.json receipt")
    parser.add_argument("--answer-root", required=True, type=Path, help="the receipt's answers directory")
    parser.add_argument("--input-root", required=True, type=Path, help="root containing the exact clickbench/ input suffix")
    parser.add_argument("--output", required=True, type=Path, help="write JSON audit here")
    args = parser.parse_args(argv)
    try:
        report = audit_coverage(args.coverage, args.answer_root, args.input_root)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    except (OracleInputError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
