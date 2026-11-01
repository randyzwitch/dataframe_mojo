"""The row-based answer comparison that suite_answers replaced, kept as an
independent reference for the columnar one.

It converts answers to Python rows exactly as load_answer used to and runs
the original row comparison. Only tests use it: on full-tier answers it
needed tens of GB and most of an hour.
"""
from collections import defaultdict
from decimal import Decimal
from functools import cmp_to_key
import math

from metal_numerical_compare import compare_rows, _exact_token, _perfect_match
from suite_schema import ALIASES, INTEGER_SUMS


def rows(answer):
    """Python rows of a columnar answer (temporal columns are already their
    physical integers)."""
    arrays = [column.to_pylist() for column in answer.table.columns]
    return list(zip(*arrays)) if arrays else [()] * answer.table.num_rows


class RowAnswer:
    def __init__(self, names, types, rows, original_schema):
        self.names, self.types = list(names), list(types)
        self.rows, self.original_schema = rows, original_schema


def from_columnar(answer):
    return RowAnswer(
        answer.names, answer.types, rows(answer), answer.original_schema
    )


def normalize(answer, suite, query):
    types, rows = list(answer.types), list(answer.rows)
    for position in INTEGER_SUMS.get((suite, query), ()):
        if position >= len(types):
            raise ValueError("declared integer reduction is missing")
        dtype = types[position]
        if dtype not in {
            "int64",
            "uint64",
            "decimal(38, 0)",
            "decimal128(38, 0)",
        }:
            raise ValueError("integer reduction has an unexpected dtype")
        converted = []
        for row in rows:
            values = list(row)
            value = values[position]
            if value is not None:
                if type(value) not in (int, Decimal) or value != int(value):
                    raise ValueError(
                        "integer reduction contains a non-integer value"
                    )
                values[position] = int(value)
            converted.append(tuple(values))
        rows = converted
        types[position] = "integer_sum"
    aliases = ALIASES.get((suite, query), {})
    names = [aliases.get(n, n) for n in answer.names]
    if len(set(names)) != len(names):
        raise ValueError("schema alias mapping produces duplicate columns")
    grouping = {("tpcds", "q27"): 2, ("tpcds", "q36"): 3}.get((suite, query))
    if grouping is not None:
        if grouping >= len(types) or types[grouping] not in {"int32", "int64"}:
            raise ValueError("grouping tag has an unexpected dtype")
        types[grouping] = "int64"
    if (suite, query) == ("clickbench", "q34"):
        if (
            not types
            or types[0] not in {"int32", "int64"}
            or any(r[0] != 1 for r in rows)
        ):
            raise ValueError("q34 literal column does not contain integer one")
        types[0] = "int64"
    return RowAnswer(names, types, rows, answer.original_schema)


def compare_answers(actual, expected, *, tolerant_columns=()):
    if actual.names != expected.names or actual.types != expected.types:
        return {"status": "different", "reason": "schema_mismatch"}
    tolerant = list(tolerant_columns)
    if any(
        name not in expected.names
        or expected.types[expected.names.index(name)] != "double"
        for name in tolerant
    ):
        raise ValueError("tolerance requires a declared Float64 result column")
    return compare_rows(expected.rows, actual.rows, expected.names, tolerant)


def _subset(actual, available, names, tolerant_columns, budget=1_000_000):
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
    expected = RowAnswer(oracle.names[:width], oracle.types[:width], [], {})
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
