"""Pure-Python unordered row comparison with bounded Float64 tolerance matching."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Iterable, Sequence


DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def _float_token(value: float) -> tuple[str, Any]:
    if math.isnan(value):
        return ("nan", None)
    if value == 0.0:
        return ("zero", 0.0)
    return ("float", value)


def _exact_token(value: Any) -> tuple[type, Any]:
    """Hashable exact identity, with DataFrame-style NaN and signed-zero equality."""
    if type(value) is float:
        return (float, _float_token(value))
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"exact comparison value is unhashable: {type(value).__name__}") from exc
    return (type(value), value)


def _close_value(left: Any, right: Any, rtol: float, atol: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if type(left) is not float or type(right) is not float:
        raise ValueError("tolerant columns require Float64 values or nulls")
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    # Explicitly symmetric threshold; signed zeros compare equal.
    return abs(left - right) <= max(atol, rtol * max(abs(left), abs(right)))


def _perfect_match(expected: Sequence[tuple[Any, ...]], actual: Sequence[tuple[Any, ...]],
                   tolerant_positions: Sequence[int], rtol: float, atol: float,
                   remaining_pairs: int) -> tuple[bool, int, bool]:
    """Find a bipartite perfect matching using iterative augmenting BFS."""
    adjacency: list[list[int]] = []
    pairs = 0
    for left_row in expected:
        neighbors = []
        for right_index, right_row in enumerate(actual):
            pairs += 1
            if pairs > remaining_pairs:
                return False, pairs - 1, True
            if all(_close_value(left_row[i], right_row[i], rtol, atol) for i in tolerant_positions):
                neighbors.append(right_index)
        adjacency.append(neighbors)

    left_match: list[int | None] = [None] * len(expected)
    right_match: list[int | None] = [None] * len(actual)
    for root in range(len(expected)):
        queue = deque([root])
        seen_left = {root}
        parent_right: dict[int, int] = {}
        free_right = None
        while queue and free_right is None:
            left = queue.popleft()
            for right in adjacency[left]:
                if right in parent_right:
                    continue
                parent_right[right] = left
                mate = right_match[right]
                if mate is None:
                    free_right = right
                    break
                if mate not in seen_left:
                    seen_left.add(mate)
                    queue.append(mate)
        if free_right is None:
            return False, pairs, False
        right = free_right
        while True:
            left = parent_right[right]
            previous_right = left_match[left]
            left_match[left] = right
            right_match[right] = left
            if previous_right is None:
                break
            right = previous_right
    return True, pairs, False


def compare_rows(
    expected: Iterable[Sequence[Any]],
    actual: Iterable[Sequence[Any]],
    columns: Sequence[str],
    tolerant_columns: Iterable[str],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_candidate_pairs: int = 1_000_000,
) -> dict[str, Any]:
    """Compare unordered row multisets, tolerating only named Float64 columns.

    Exact fields partition rows first. Within each exact-key bucket, approximate
    float rows are compared by bipartite perfect matching so nontransitive
    closeness cannot make greedy pairing accept or reject the wrong multiset.
    The candidate-pair budget is global; exhaustion returns ``unverified``.
    """
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("rtol and atol must be finite nonnegative values")
    if type(max_candidate_pairs) is not int or max_candidate_pairs < 0:
        raise ValueError("max_candidate_pairs must be a nonnegative integer")
    names = list(columns)
    if len(set(names)) != len(names):
        raise ValueError("columns must be unique")
    tolerant = set(tolerant_columns)
    unknown = tolerant - set(names)
    if unknown:
        raise ValueError(f"unknown tolerant columns: {sorted(unknown)}")
    tol_positions = [i for i, name in enumerate(names) if name in tolerant]
    exact_positions = [i for i, name in enumerate(names) if name not in tolerant]

    def materialize(rows: Iterable[Sequence[Any]], label: str) -> list[tuple[Any, ...]]:
        result = []
        for index, row in enumerate(rows):
            if not isinstance(row, (tuple, list)) or len(row) != len(names):
                raise ValueError(f"{label} row {index} width does not match columns")
            values = tuple(row)
            for pos in tol_positions:
                value = values[pos]
                if value is not None and type(value) is not float:
                    raise ValueError(f"{label} row {index} tolerant column {names[pos]} is not Float64 or null")
            result.append(values)
        return result

    left = materialize(expected, "expected")
    right = materialize(actual, "actual")
    if len(left) != len(right):
        return {"status": "different", "compared_rows": min(len(left), len(right)),
                "expected_rows": len(left), "actual_rows": len(right), "candidate_pairs": 0,
                "reason": "row_count_mismatch"}

    def bucketize(rows: Sequence[tuple[Any, ...]]) -> dict[tuple[Any, ...], list[tuple[Any, ...]]]:
        buckets: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            key = tuple(_exact_token(row[i]) for i in exact_positions)
            buckets[key].append(row)
        return buckets

    left_buckets, right_buckets = bucketize(left), bucketize(right)
    if set(left_buckets) != set(right_buckets):
        return {"status": "different", "compared_rows": len(left), "expected_rows": len(left),
                "actual_rows": len(right), "candidate_pairs": 0, "reason": "exact_key_mismatch"}
    for key in left_buckets:
        if len(left_buckets[key]) != len(right_buckets[key]):
            return {"status": "different", "compared_rows": len(left), "expected_rows": len(left),
                    "actual_rows": len(right), "candidate_pairs": 0, "reason": "exact_key_multiplicity_mismatch"}

    budget_left = max_candidate_pairs
    total_pairs = 0
    for key, left_group in left_buckets.items():
        right_group = right_buckets[key]
        if not tol_positions:
            continue
        matched, pairs, exhausted = _perfect_match(left_group, right_group, tol_positions, rtol, atol, budget_left)
        total_pairs += pairs
        budget_left -= pairs
        if exhausted:
            return {"status": "unverified", "compared_rows": 0, "expected_rows": len(left),
                    "actual_rows": len(right), "candidate_pairs": total_pairs,
                    "reason": "candidate_pair_budget_exceeded"}
        if not matched:
            return {"status": "different", "compared_rows": len(left), "expected_rows": len(left),
                    "actual_rows": len(right), "candidate_pairs": total_pairs,
                    "reason": "no_tolerant_perfect_matching"}
    return {"status": "equal", "compared_rows": len(left), "expected_rows": len(left),
            "actual_rows": len(right), "candidate_pairs": total_pairs, "reason": None}
