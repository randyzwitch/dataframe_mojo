"""The columnar answer check agrees with the row-based check it replaced.

Random relations, LIMIT/OFFSET selections and ORDER BY ranks are compared
by both implementations, with correct answers and with deliberate faults: a
float beyond tolerance, swapped ordered rows, a wrong decimal digit, an
extra null, a duplicated row. Status and reason must match exactly.
"""
from decimal import Decimal
from functools import cmp_to_key
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
sys.path.insert(0, str(ROOT / "tests/tooling"))

try:
    import duckdb  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:
    duckdb = None

if duckdb is not None:
    import row_answer_reference as reference  # row_answer_reference.py
    import suite_answers
    from suite_answer_fixture import rows_answer

NAMES = ["k", "s", "x", "d", "v"]
TYPES = ["int64", "string", "double", "decimal128(12, 2)", "double"]
TOLERANT = ["v"]
NAN, INF = float("nan"), float("inf")


def verdict(result):
    return result["status"], result["reason"]


def make_row(rng, narrow):
    if narrow:
        k = rng.choice([0, 1, 2, None])
        s = rng.choice(["a", "b", None])
        x = rng.choice([0.0, -0.0, NAN, 1.5, None])
        d = rng.choice([Decimal("1.10"), Decimal("-2.05"), None])
    else:
        k = rng.choice([None] + list(range(10_000)))
        s = rng.choice([None, "", "é", "a"]) or f"s{rng.randrange(1000)}"
        x = rng.choice([NAN, INF, -INF, 0.0, -0.0, None, rng.uniform(-9, 9)])
        d = Decimal(rng.randrange(-10**6, 10**6)).scaleb(-2)
    v = rng.choice(
        [None, NAN, INF, 0.0, -0.0, 1e-300]
        + [rng.uniform(-1e6, 1e6) for _ in range(6)]
    )
    return (k, s, x, d, v)


def order_key(rng, row):
    return (
        rng.choice([None, 1, 2, 3, row[0]]),
        rng.choice([None, NAN, -0.0, 0.0, 1.0, row[4]]),
    )


def nudge(row):
    """The same answer within tolerance."""
    v = row[4]
    if v is None or not math.isfinite(v) or v == 0:
        return row
    return row[:4] + (v * (1 + 1e-10),)


def faults(rng, rows):
    """Deliberately wrong (or legitimately equal) variants of an answer."""
    out = []
    if not rows:
        return [[make_row(rng, False)]]
    i = rng.randrange(len(rows))
    row = rows[i]
    if row[4] is not None and math.isfinite(row[4]) and row[4] != 0:
        out.append(rows[:i] + [row[:4] + (row[4] * (1 + 1e-6),)] + rows[i + 1 :])
    if len(rows) > 1:
        j = rng.randrange(len(rows))
        swapped = list(rows)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        out.append(swapped)
        out.append(rows[:i] + [rows[j]] + rows[i + 1 :])  # duplicate a row
    if row[3] is not None:
        out.append(
            rows[:i] + [row[:3] + (row[3] + Decimal("0.01"),) + row[4:]]
            + rows[i + 1 :]
        )
    column = rng.randrange(5)
    nulled = list(row)
    nulled[column] = None
    out.append(rows[:i] + [tuple(nulled)] + rows[i + 1 :])
    if row[2] is not None and not math.isnan(row[2]) and row[2] == 0:
        out.append(rows[:i] + [row[:2] + (-row[2],) + row[3:]] + rows[i + 1 :])
    out.append(rows[:-1])
    return out


def relation(rows, order, limit, offset):
    names = NAMES + [f"__suite_rank_{j}" for j in range(len(order))]
    types = TYPES + ["int64", "double"][: len(order)]
    spec = {"width": 5, "order": order, "limit": limit, "offset": offset}
    return rows_answer(rows, names, types, {"oracle": spec})


def expected_selection(rng, oracle_rows, order, limit, offset):
    """A correct output: rows in rank order, with ties broken at random."""
    rows = list(oracle_rows)
    rng.shuffle(rows)
    if order:

        def compare(left, right):
            for j, rule in enumerate(order):
                a, b = left[5 + j], right[5 + j]
                if a is None or b is None:
                    if a is b:
                        continue
                    result = -1 if a is None else 1
                    return result if rule["nulls_first"] else -result
                a_nan = type(a) is float and math.isnan(a)
                b_nan = type(b) is float and math.isnan(b)
                result = (
                    int(a_nan) - int(b_nan) if a_nan or b_nan
                    else (a > b) - (a < b)
                )
                if result:
                    return -result if rule["descending"] else result
            return 0

        rows.sort(key=cmp_to_key(compare))
    end = None if limit is None else offset + limit
    return [nudge(r[:5]) for r in rows[offset:end]]


@unittest.skipIf(duckdb is None, "DuckDB and PyArrow are required")
class ColumnarAgreementTests(unittest.TestCase):
    def check(self, actual_rows, oracle):
        actual = rows_answer(actual_rows, NAMES, TYPES)
        new = suite_answers.compare_selection(
            actual, oracle, tolerant_columns=TOLERANT
        )
        old = reference.compare_selection(
            reference.from_columnar(actual),
            reference.from_columnar(oracle),
            tolerant_columns=TOLERANT,
        )
        self.assertEqual(verdict(new), verdict(old), actual_rows)
        return verdict(new)

    def test_random_relations_and_faults(self):
        rng = random.Random(1041)
        seen = set()
        for case in range(160):
            narrow = case % 2 == 0
            height = rng.choice([0, 1, 2, 5, 20, 60])
            base = [make_row(rng, narrow) for _ in range(height)]
            kind = case % 4
            if kind == 0:
                order, limit, offset = [], None, 0
            elif kind == 1:
                order, limit, offset = [], rng.randrange(1, 8), rng.randrange(3)
            else:
                order = [
                    {
                        "descending": rng.random() < 0.5,
                        "nulls_first": rng.random() < 0.5,
                    }
                    for _ in range(rng.choice([1, 2]))
                ]
                limit = rng.choice([None, 1, 3, 10])
                offset = rng.randrange(3)
            oracle_rows = [
                r + order_key(rng, r)[: len(order)] for r in base
            ]
            oracle = relation(oracle_rows, order, limit, offset)
            good = expected_selection(rng, oracle_rows, order, limit, offset)
            with self.subTest(case=case):
                if not order and limit is None:
                    self.assertEqual(self.check(good, oracle)[0], "equal")
                for rows in faults(rng, good):
                    seen.add(self.check(rows, oracle))
        # The faults must reach each failure path, not only schema checks.
        for reason in [
            "exact_key_mismatch",
            "exact_key_multiplicity_mismatch",
            "no_tolerant_perfect_matching",
            "invalid_relation_membership_or_multiplicity",
            "invalid_relation_values",
            "selection_row_count_mismatch",
        ]:
            self.assertIn(("different", reason), seen)

    def test_named_faults(self):
        rows = [
            (1, "a", 0.5, Decimal("10.25"), 100.0),
            (2, "b", 1.5, Decimal("20.50"), 200.0),
            (3, "c", 2.5, Decimal("30.75"), 300.0),
        ]
        ordered = relation(
            [r + (r[0], r[4]) for r in rows],
            [{"descending": False, "nulls_first": False}] * 2,
            None,
            0,
        )
        whole = relation(rows, [], None, 0)
        self.assertEqual(self.check(rows[::-1], whole), ("equal", None))
        self.assertEqual(
            self.check(rows, ordered), ("equal", "valid_selection_and_order")
        )
        wrong_float = [rows[0], rows[1][:4] + (200.0 * (1 + 1e-8),), rows[2]]
        self.assertEqual(
            self.check(wrong_float, whole),
            ("different", "no_tolerant_perfect_matching"),
        )
        within = [rows[0], rows[1][:4] + (200.0 * (1 + 1e-10),), rows[2]]
        self.assertEqual(self.check(within, whole), ("equal", None))
        self.assertEqual(
            self.check([rows[1], rows[0], rows[2]], ordered),
            ("different", "invalid_relation_membership_or_multiplicity"),
        )
        digit = [rows[0], rows[1][:3] + (Decimal("20.51"), 200.0), rows[2]]
        self.assertEqual(
            self.check(digit, whole), ("different", "exact_key_mismatch")
        )
        null = [rows[0], (2, None) + rows[1][2:], rows[2]]
        self.assertEqual(
            self.check(null, whole), ("different", "exact_key_mismatch")
        )
        null_value = [rows[0], rows[1][:4] + (None,), rows[2]]
        self.assertEqual(
            self.check(null_value, whole),
            ("different", "no_tolerant_perfect_matching"),
        )

    def test_pair_budget_order_matches(self):
        # 1,001 rows sharing every exact field cost 1,002,001 pairs: over the
        # budget. A failing bucket visited first is "different"; visited
        # after the expensive bucket, the budget runs out first.
        rng = random.Random(7)
        big = [(1, "a", 0.0, Decimal("1.00"), rng.random()) for _ in range(1001)]
        small = [(2, "b", 0.0, Decimal("1.00"), 5.0)]
        wrong = [(2, "b", 0.0, Decimal("1.00"), 6.0)]
        for oracle_rows, expected in [
            (small + big, ("different", "no_tolerant_perfect_matching")),
            (big + small, ("unverified", "candidate_pair_budget_exceeded")),
        ]:
            oracle = relation(oracle_rows, [], None, 0)
            self.assertEqual(self.check(big + wrong, oracle), expected)


if __name__ == "__main__":
    unittest.main()
