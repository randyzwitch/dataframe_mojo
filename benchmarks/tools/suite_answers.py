"""Complete, hash-pinned suite answers; summaries never certify correctness."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from functools import cmp_to_key
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from metal_numerical_compare import compare_rows, _exact_token, _perfect_match


FORMAT = "suite-answer-v1"


def tolerant_columns(suite, query, oracle):
    """Reuse only the reviewed aggregate/arithmetic field policy, never keys."""
    policy_path = Path(__file__).with_name("suite_numerical_policy.json")
    policy = json.loads(policy_path.read_text())
    from metal_numerical_audit import validate_policy

    validate_policy(policy)
    root = Path(__file__).resolve().parents[2]
    for source, digest in policy["source_sha256"].items():
        if sha256(root / source) != digest:
            raise ValueError("numerical policy query source identity changed")
    fields = policy["cases"].get(f"{suite}/{query}", [])
    if (
        fields
        and suite == "clickbench"
        and sha256(root / "benchmarks/suites/clickbench.mojo")
        != policy.get("clickbench_source_sha256")
    ):
        raise ValueError("ClickBench numerical policy source identity changed")
    sql_hash = policy.get("duckdb_sql_sha256", {}).get(f"{suite}/{query}")
    if (
        sql_hash
        and oracle.original_schema.get("oracle", {}).get("sql_sha256")
        != sql_hash
    ):
        raise ValueError("numerical policy independent SQL identity changed")
    return [
        n
        for n, t in zip(oracle.names, oracle.types)
        if n in fields and t == "double"
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_export(record, directory, query):
    """Accept only this worker's paths and record immutable content identity."""
    if (
        not isinstance(record, dict)
        or type(record.get("repetition")) is not int
    ):
        raise ValueError("invalid answer repetition")
    repetition = record["repetition"]
    if (
        repetition < 0
        or not isinstance(query, str)
        or not query.startswith("q")
        or not query[1:].isdigit()
    ):
        raise ValueError("invalid answer identity")
    directory = Path(directory).resolve()
    out = dict(record)
    for key, suffix in (("path", ".parquet"), ("schema_path", ".schema.json")):
        expected = directory / f"{query}-{repetition}{suffix}"
        if (
            not isinstance(record.get(key), str)
            or Path(record[key]).resolve() != expected
        ):
            raise ValueError("answer path is not owned by this worker")
        out[key] = str(expected)
        out[key + "_sha256"] = sha256(expected)
    out.update(query=query, format=FORMAT)
    return out


@dataclass
class Answer:
    names: list[str]
    types: list[str]
    rows: list[tuple[Any, ...]]
    original_schema: dict


def _logical_type(dtype, pa):
    # Arrow's large-string offset width is a transport detail. Preserve all
    # integer widths, decimal precision/scale, and temporal units/timezones.
    # Arrow decimal32/64/128 storage widths can represent the same logical
    # precision and scale; the original receipt still records their storage.
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"
    if pa.types.is_decimal(dtype):
        return f"decimal({dtype.precision}, {dtype.scale})"
    return str(dtype)


def load_answer(record):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from metal_numerical_audit import _physical_dtype_matches

    if record.get("format") != FORMAT:
        raise ValueError("missing complete answer format")
    for key in ("path", "schema_path"):
        if sha256(record[key]) != record.get(key + "_sha256"):
            raise ValueError("answer content hash changed")
    schema = json.loads(Path(record["schema_path"]).read_text())
    if (
        schema.get("format") != FORMAT
        or schema.get("query") != record.get("query")
        or type(schema.get("repetition")) is not int
        or schema["repetition"] != record.get("repetition")
        or type(schema.get("rows")) is not int
        or schema["rows"] < 0
    ):
        raise ValueError("answer schema identity mismatch")
    columns = schema.get("columns")
    if not isinstance(columns, list) or any(
        not isinstance(c, dict) or not isinstance(c.get("name"), str)
        for c in columns
    ):
        raise ValueError("invalid answer schema columns")
    names = [c["name"] for c in columns]
    if len(set(names)) != len(names):
        raise ValueError("duplicate answer column names")
    table = pq.read_table(record["path"])
    if names != table.column_names or schema["rows"] != table.num_rows:
        raise ValueError("Parquet names/height differ from original schema")
    for field, column in zip(table.schema, columns):
        if "arrow" in column:
            valid = column["arrow"] == str(field.type)
        else:
            valid = _physical_dtype_matches(column.get("dtype"), field.type, pa)
        if not valid:
            raise ValueError(
                f"Parquet dtype differs from original schema: {field.name}"
            )
    # Arrow scalar as_py preserves decimals and integers. Temporal values use
    # physical integers instead of Python datetime (which can lose nanoseconds).
    arrays = []
    for column, field in zip(table.columns, table.schema):
        if pa.types.is_temporal(field.type):
            storage = (
                pa.int32() if pa.types.is_date32(field.type)
                or pa.types.is_time32(field.type) else pa.int64()
            )
            column = column.cast(storage)
        arrays.append(column.to_pylist())
    rows = list(zip(*arrays)) if arrays else [()] * table.num_rows
    return Answer(
        names, [_logical_type(f.type, pa) for f in table.schema], rows, schema
    )


def compare_answers(actual, expected, *, tolerant_columns=(), ordered=False):
    """Compare complete schemas and cells, with explicit field-level policy."""
    if actual.names != expected.names or actual.types != expected.types:
        return {"status": "different", "reason": "schema_mismatch"}
    tolerant = list(tolerant_columns)
    if any(
        name not in expected.names
        or expected.types[expected.names.index(name)] != "double"
        for name in tolerant
    ):
        raise ValueError("tolerance requires a declared Float64 result column")
    if ordered:
        if len(actual.rows) != len(expected.rows):
            return {"status": "different", "reason": "row_count_mismatch"}
        for got, want in zip(actual.rows, expected.rows):
            result = compare_rows([want], [got], expected.names, tolerant)
            if result["status"] != "equal":
                return result
        return {"status": "equal", "reason": "ordered_rows_equal"}
    return compare_rows(expected.rows, actual.rows, expected.names, tolerant)


def _subset(actual, available, names, tolerant_columns, budget=1_000_000):
    """Injective row matching: a LIMIT never licenses duplicate fabrication."""
    tolerant = [i for i, n in enumerate(names) if n in tolerant_columns]
    exact = [i for i in range(len(names)) if i not in tolerant]

    def buckets(rows):
        result = defaultdict(list)
        for row in rows:
            result[tuple(_exact_token(row[i]) for i in exact)].append(row)
        return result

    candidates = buckets(available)
    for key, rows in buckets(actual).items():
        pool = candidates.get(key, [])
        if len(pool) < len(rows):
            return {
                "status": "different",
                "reason": "invalid_relation_membership_or_multiplicity",
            }
        if tolerant:
            matched, count, exhausted = _perfect_match(
                rows, pool, tolerant, 1e-9, 1e-12, budget
            )
            budget -= count
            if exhausted:
                return {
                    "status": "unverified",
                    "reason": "candidate_pair_budget_exceeded",
                }
            if not matched:
                return {
                    "status": "different",
                    "reason": "invalid_relation_values",
                }
    return {"status": "equal", "reason": "valid_relation_membership"}


def compare_selection(actual, oracle, *, tolerant_columns=()):
    """Full relation membership and exact ordered ranks, including hidden keys.

    Ties can choose different rows. Match each output position only against
    rows with the independently established rank at that position.
    """
    spec = oracle.original_schema.get("oracle")
    if not isinstance(spec, dict):
        return {
            "status": "unverified",
            "reason": "missing independent relation metadata",
        }
    width, order = spec.get("width"), spec.get("order")
    limit, offset = spec.get("limit"), spec.get("offset")
    if (
        type(width) is not int
        or width < 0
        or not isinstance(order, list)
        or len(oracle.names) != width + len(order)
        or any(
            not isinstance(o, dict)
            or type(o.get("descending")) is not bool
            or type(o.get("nulls_first")) is not bool
            for o in order
        )
        or (limit is not None and (type(limit) is not int or limit < 0))
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError("invalid independent selection metadata")
    expected = Answer(oracle.names[:width], oracle.types[:width], [], {})
    if actual.names != expected.names or actual.types != expected.types:
        return {"status": "different", "reason": "schema_mismatch"}
    if any(
        n not in expected.names
        or expected.types[expected.names.index(n)] != "double"
        for n in tolerant_columns
    ):
        raise ValueError("tolerance requires a declared Float64 result column")
    count = max(0, len(oracle.rows) - offset)
    if limit is not None:
        count = min(count, limit)
    if len(actual.rows) != count:
        return {"status": "different", "reason": "selection_row_count_mismatch"}
    if not order:
        rows = [r[:width] for r in oracle.rows]
        if limit is None and offset == 0:
            expected.rows = rows
            return compare_answers(
                actual, expected, tolerant_columns=tolerant_columns
            )
        return _subset(actual.rows, rows, expected.names, tolerant_columns)

    def compare(left, right):
        for i, rule in enumerate(order):
            a, b = left[width + i], right[width + i]
            if a is None or b is None:
                if a is b:
                    continue
                result = -1 if a is None else 1
                return result if rule["nulls_first"] else -result
            # DuckDB sorts NaN above every non-NaN numeric value.
            a_nan, b_nan = type(a) is float and math.isnan(a), type(
                b
            ) is float and math.isnan(b)
            if a_nan or b_nan:
                result = int(a_nan) - int(b_nan)
            else:
                result = (a > b) - (a < b)
            if result:
                return -result if rule["descending"] else result
        return 0

    full = sorted(oracle.rows, key=cmp_to_key(compare))
    selected = full[offset : offset + count]
    # Build pools only for ranks selected by LIMIT/OFFSET. Include entire tie
    # groups at both boundaries so legitimate peer choices remain valid.
    def rank(row):
        return tuple(_exact_token(v) for v in row[width:])

    selected_ranks = {rank(row) for row in selected}
    pools, actual_groups = defaultdict(list), defaultdict(list)
    for row in full:
        key = rank(row)
        if key in selected_ranks:
            pools[key].append(row[:width])
    for row, expected_row in zip(actual.rows, selected):
        actual_groups[rank(expected_row)].append(row)
    for key, rows in actual_groups.items():
        result = _subset(rows, pools[key], expected.names, tolerant_columns)
        if result["status"] != "equal":
            return result
    return {"status": "equal", "reason": "valid_selection_and_order"}
