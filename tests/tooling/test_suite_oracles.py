"""Independent SQL and semantic LIMIT/ORDER BY validation contracts."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/suites"))
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
from oracle_sql import full_relation
from suite_answers import Answer, compare_selection


class SQLTests(unittest.TestCase):
    def test_distinct_cannot_gain_duplicate_outputs_from_hidden_keys(self):
        with self.assertRaises(ValueError):
            full_relation("SELECT DISTINCT a FROM t ORDER BY b LIMIT 2", ["a"])
        sql, _ = full_relation(
            "SELECT DISTINCT a FROM t ORDER BY a LIMIT 2", ["a"]
        )
        self.assertIn('a AS "__suite_rank_0"', sql)

    def test_only_outer_clauses_are_changed(self):
        sql = "WITH x AS (SELECT a FROM y ORDER BY a LIMIT 1) SELECT a, count(*) AS c FROM x GROUP BY a ORDER BY c DESC LIMIT 10 OFFSET 3;"
        full, spec = full_relation(sql)
        self.assertIn("ORDER BY a LIMIT 1", full)
        self.assertIn('count(*) AS "__suite_rank_0"', full)
        self.assertNotIn("LIMIT 10", full)
        self.assertEqual(
            spec,
            {
                "limit": 10,
                "offset": 3,
                "order": [{"descending": True, "nulls_first": False}],
            },
        )

    def test_quoted_clauses_and_commas_do_not_change_parsing(self):
        sql = "SELECT regexp_replace(a, 'ORDER BY, LIMIT', 'x') AS k FROM t ORDER BY k NULLS FIRST LIMIT 2"
        full, spec = full_relation(sql)
        self.assertIn("'ORDER BY, LIMIT'", full)
        self.assertEqual(len(spec["order"]), 1)
        self.assertTrue(spec["order"][0]["nulls_first"])

    def test_output_alias_precedes_same_named_input(self):
        full, _ = full_relation(
            "SELECT a, sum(v) v FROM t GROUP BY a ORDER BY v", ["a", "v"]
        )
        self.assertIn('sum(v) AS "__suite_rank_0"', full)

    def test_source_identifier_is_not_mistaken_for_implicit_alias(self):
        full, _ = full_relation("SELECT a - b FROM t ORDER BY b", ["(a - b)"])
        self.assertIn('b AS "__suite_rank_0"', full)

    def test_star_ordinals_require_explicit_output_schema(self):
        full, _ = full_relation("SELECT * FROM t ORDER BY 2 DESC", ["a", "b"])
        self.assertIn('"b" AS "__suite_rank_0"', full)
        with self.assertRaises(ValueError):
            full_relation("SELECT * FROM t ORDER BY 2")

    def test_unhandled_forms_fail_closed(self):
        for sql in [
            "SELECT a FROM x UNION SELECT a FROM y ORDER BY a",
            "SELECT a FROM x LIMIT ?",
            "SELECT a FROM x OFFSET 3",
            "SELECT a FROM x /*",
            "SELECT a FROM (x",
        ]:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                full_relation(sql)


def relation(
    rows, *, order=(), limit=2, offset=0, width=2, names=("key", "value")
):
    return Answer(
        list(names) + [f"__suite_rank_{i}" for i in range(len(order))],
        ["int64"] * (width + len(order)),
        rows,
        {
            "oracle": {
                "width": width,
                "order": list(order),
                "limit": limit,
                "offset": offset,
            }
        },
    )


DESC = {"descending": True, "nulls_first": False}
ASC = {"descending": False, "nulls_first": False}


class SelectionTests(unittest.TestCase):
    def test_q67_rank_is_exact_even_when_sum_has_float_tolerance(self):
        names = ["i_category", "sumsales", "rk"]
        types = ["string", "double", "int64"]
        oracle = Answer(
            names,
            types,
            [("a", 1.0, 1)],
            {"oracle": {"width": 3, "order": [], "limit": None, "offset": 0}},
        )
        actual = Answer(names, types, [("a", 1.0 + 1e-10, 1)], {})
        self.assertEqual(
            compare_selection(actual, oracle, tolerant_columns=["sumsales"])[
                "status"
            ],
            "equal",
        )
        actual.rows = [("a", 1.0 + 1e-10, 999)]
        self.assertEqual(
            compare_selection(actual, oracle, tolerant_columns=["sumsales"])[
                "status"
            ],
            "different",
        )

    def test_q17_rejects_fabricated_groups_and_duplicate_overuse(self):
        full = relation([(1, 4), (2, 7), (3, 8)])
        for rows, status in [
            ([(1, 4), (3, 8)], "equal"),
            ([(1, 4), (2, 99)], "different"),
            ([(1, 4), (1, 4)], "different"),
            ([(1, 4)], "different"),
        ]:
            actual = Answer(["key", "value"], ["int64", "int64"], rows, {})
            self.assertEqual(compare_selection(actual, full)["status"], status)

    def test_top_k_allows_boundary_ties_but_requires_rank_order(self):
        full = relation(
            [(1, 9, 9), (2, 8, 8), (3, 8, 8), (4, 1, 1)], order=[DESC]
        )
        for rows, status in [
            ([(1, 9), (3, 8)], "equal"),
            ([(3, 8), (1, 9)], "different"),
            ([(1, 9), (4, 1)], "different"),
            ([(1, 9), (3, 9)], "different"),
        ]:
            self.assertEqual(
                compare_selection(
                    Answer(["key", "value"], ["int64", "int64"], rows, {}), full
                )["status"],
                status,
            )

    def test_offset_rank_sequence_and_tie_pool(self):
        full = relation(
            [(1, 9, 9), (2, 8, 8), (3, 8, 8), (4, 1, 1)], order=[DESC], offset=2
        )
        good = Answer(
            ["key", "value"], ["int64", "int64"], [(2, 8), (4, 1)], {}
        )
        self.assertEqual(compare_selection(good, full)["status"], "equal")
        good.rows = [(1, 9), (2, 8)]
        self.assertEqual(compare_selection(good, full)["status"], "different")

    def test_hidden_q24_key_still_checks_selection_and_multiplicity(self):
        full = relation(
            [(1, 20), (2, 10), (2, 30)], order=[ASC], width=1, names=["key"]
        )
        for rows, status in [
            ([(2,), (1,)], "equal"),
            ([(1,), (2,)], "different"),
            ([(2,), (2,)], "different"),
        ]:
            self.assertEqual(
                compare_selection(Answer(["key"], ["int64"], rows, {}), full)[
                    "status"
                ],
                status,
            )

    def test_multikey_order_and_null_placement(self):
        full = relation(
            [(1, 5, None, 5), (2, 4, None, 4), (3, 8, 1, 8)],
            order=[{"descending": True, "nulls_first": True}, DESC],
        )
        good = Answer(
            ["key", "value"], ["int64", "int64"], [(1, 5), (2, 4)], {}
        )
        self.assertEqual(compare_selection(good, full)["status"], "equal")
        good.rows.reverse()
        self.assertEqual(compare_selection(good, full)["status"], "different")

    def test_empty_and_short_relations_obey_limit(self):
        for rows in [[], [(1, 2)]]:
            actual = Answer(["key", "value"], ["int64", "int64"], rows, {})
            self.assertEqual(
                compare_selection(actual, relation(rows, limit=10))["status"],
                "equal",
            )


if __name__ == "__main__":
    unittest.main()
