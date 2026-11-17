"""List and struct group keys and distinct counts of list and struct values
in the aggregate table, against a model of which values are equal: lists
equal elementwise with nulls equal, an empty list apart from a null one, a
null struct apart from a struct of nulls, -0.0 equal to 0.0 and every NaN
one value. Each key column repeats a few patterns whose equal classes are
written out by hand. The eager group-by runs on eight workers (tables that
clear) and on one, the lazy one through the pipeline's table sink, with
and without maintain_order, alone and next to a flat key."""
# contract-group: reduce.grouped
# contract-group: expr.nested
from std.testing import assert_equal, assert_true

from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    StructColumn,
    col,
    lit,
)
from dataframe.storage.column import _pack_bits
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def lists_of(
    child: Series, lengths: List[Int], valid: List[Bool]
) raises -> Series:
    """Rows of `lengths[i]` consecutive child values each (none for a
    null row, `valid[i]` false)."""
    var offsets = List[Int64](capacity=len(lengths) + 1)
    offsets.append(0)
    var at = 0
    for i in range(len(lengths)):
        at += lengths[i]
        offsets.append(Int64(at))
    return Series(
        "key", ListColumn(offsets^, child.renamed("item"), _pack_bits(valid))
    )


def int_lists() raises -> Tuple[Series, List[Int]]:
    # [1, 2], [1, 2], [], null, [null], [1, null], [1, null], [2, 1], [1],
    # [1, 2, 3], []
    var child = Series(
        "item",
        Column[Int64](
            [1, 2, 1, 2, 0, 1, 0, 1, 0, 2, 1, 1, 1, 2, 3],
            [
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        ),
    )
    var lengths: List[Int] = [2, 2, 0, 0, 1, 2, 2, 2, 1, 3, 0]
    var valid: List[Bool] = [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var classes: List[Int] = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 1]
    return (lists_of(child, lengths, valid), classes^)


def string_lists() raises -> Tuple[Series, List[Int]]:
    # ["a"], ["a"], ["ab"], ["a", "b"], ["a", null], [], null, [""], [""],
    # [long], [long], ["a", ""] (apart from ["a", null])
    var long = String("a string longer than twelve bytes")
    var values: List[String] = [
        "a",
        "a",
        "ab",
        "a",
        "b",
        "a",
        "",
        "",
        "",
        long,
        long,
        "a",
        "",
    ]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var child = Series("item", Column[String](values^, valid^))
    var lengths: List[Int] = [1, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 2]
    var rows: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    var classes: List[Int] = [0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8]
    return (lists_of(child, lengths, rows), classes^)


def float_lists() raises -> Tuple[Series, List[Int]]:
    # [0.0], [-0.0], [nan], [nan], [1.5]
    var zero = Float64(0)
    var nan = zero / zero
    var child = Series("item", Column[Float64]([0.0, -0.0, nan, -nan, 1.5]))
    var lengths: List[Int] = [1, 1, 1, 1, 1]
    var valid: List[Bool] = [True, True, True, True, True]
    var classes: List[Int] = [0, 0, 1, 1, 2]
    return (lists_of(child, lengths, valid), classes^)


def structs_with_lists() raises -> Tuple[Series, List[Int]]:
    # null, {null, null}, {null, null}, {1, ["x"]}, {1, ["x"]}, {1, []},
    # {1, null}, {2, ["x"]}; the null struct's fields hold hidden values.
    var a = Series(
        "a",
        Column[Int64](
            [9, 0, 0, 1, 1, 1, 1, 2],
            [True, False, False, True, True, True, True, True],
        ),
    )
    var strings = Series("item", Column[String](["z", "x", "x", "x"]))
    var b = lists_of(
        strings,
        [1, 0, 0, 1, 1, 0, 0, 1],
        [True, False, False, True, True, True, False, True],
    ).renamed("b")
    var key = Series(
        "key",
        StructColumn(
            [a^, b^],
            _pack_bits([False, True, True, True, True, True, True, True]),
        ),
    )
    var classes: List[Int] = [0, 1, 1, 2, 2, 3, 4, 5]
    return (key^, classes^)


def deep_lists() raises -> Tuple[Series, List[Int]]:
    # Lists of struct{x: list[int64], y: str}:
    # [{[1], "a"}], [{[1], "a"}], [{[1], "a"}, null], [null],
    # [{null, null}], [], null, [{[], "a"}], [{[1, 2], "a"}]
    var ints = Series("item", Column[Int64]([1, 1, 1, 1, 2]))
    # Struct elements in order: 0 {[1],a} 1 {[1],a} 2 {[1],a} 3 null
    # 4 null 5 {null,null} 6 {[],a} 7 {[1,2],a}
    var x = lists_of(
        ints,
        [1, 1, 1, 0, 0, 0, 0, 2],
        [True, True, True, False, False, False, True, True],
    ).renamed("x")
    var y = Series(
        "y",
        Column[String](
            ["a", "a", "a", "", "", "", "a", "a"],
            [True, True, True, False, False, False, True, True],
        ),
    )
    var elements = Series(
        "item",
        StructColumn(
            [x^, y^],
            _pack_bits([True, True, True, False, False, True, True, True]),
        ),
    )
    var key = lists_of(
        elements,
        [1, 1, 2, 1, 1, 0, 0, 1, 1],
        [True, True, True, True, True, True, False, True, True],
    )
    var classes: List[Int] = [0, 0, 1, 2, 3, 4, 5, 6, 7]
    return (key^, classes^)


struct Case(Movable):
    var name: String
    var patterns: Series
    var classes: List[Int]

    def __init__(
        out self, name: String, var made: Tuple[Series, List[Int]]
    ) raises:
        self.name = name
        self.patterns = made[0].copy()
        self.classes = made[1].copy()


def cases() raises -> List[Case]:
    var out = List[Case]()
    out.append(Case("int lists", int_lists()))
    out.append(Case("string lists", string_lists()))
    out.append(Case("float lists", float_lists()))
    out.append(Case("structs with lists", structs_with_lists()))
    out.append(Case("lists of structs of lists", deep_lists()))
    return out^


def pattern_of(row: Int, count: Int) -> Int:
    return (row * 7919) % count


def other_of(row: Int, count: Int) -> Int:
    return (row * 31 + row // 5) % count


def frame(kase: Case, rows: Int) raises -> DataFrame:
    """`rows` rows of the case's patterns as `key`, another pick of them as
    `nk` (for distinct counts), the row as `v` and a flat key `f`."""
    var count = len(kase.classes)
    var picks = List[Int](capacity=rows)
    var others = List[Int](capacity=rows)
    var v = List[Int64](capacity=rows)
    var f = List[Int64](capacity=rows)
    for row in range(rows):
        picks.append(pattern_of(row, count))
        others.append(other_of(row, count))
        v.append(Int64(row))
        f.append(Int64(row % 3))
    return DataFrame(
        [
            kase.patterns.take(picks).renamed("key"),
            kase.patterns.take(others).renamed("nk"),
            Series("v", Column[Int64](v^)),
            Series("f", Column[Int64](f^)),
        ]
    )


def expressions() -> List[Expr]:
    return [
        col("v").sum().alias("s"),
        col("v").count().alias("n"),
        col("v").min().alias("first_row"),
        col("nk").n_unique().alias("u"),
    ]


def check(kase: Case, result: DataFrame, rows: Int, flat: Bool) raises:
    """Every group against the model: one group per class (and flat key),
    its sum, count and distinct `nk` classes, and its key equal to the
    first row's pattern."""
    var count = len(kase.classes)
    var width = 3 if flat else 1
    var classes = 0
    for c in kase.classes:
        classes = max(classes, c + 1)
    var groups = classes * width
    var sums = List[Int64](length=groups, fill=0)
    var counts = List[Int64](length=groups, fill=0)
    var seen = List[List[Bool]](
        length=groups, fill=List[Bool](length=classes, fill=False)
    )
    for row in range(rows):
        var g = kase.classes[pattern_of(row, count)] * width + (
            row % 3 if flat else 0
        )
        sums[g] += Int64(row)
        counts[g] += 1
        seen[g][kase.classes[other_of(row, count)]] = True
    var present = 0
    for g in range(groups):
        if counts[g] > 0:
            present += 1
    assert_equal(result.height(), present, kase.name + ": group count")
    var keys = result.column("key")
    for i in range(result.height()):
        var first = Int(result.item(i, "first_row").int64())
        var p = pattern_of(first, count)
        var g = kase.classes[p] * width + (first % 3 if flat else 0)
        assert_equal(result.item(i, "s").int64(), sums[g], kase.name)
        assert_equal(result.item(i, "n").int64(), counts[g], kase.name)
        var distinct = 0
        for one in seen[g]:
            if one:
                distinct += 1
        assert_equal(result.item(i, "u").int64(), Int64(distinct), kase.name)
        assert_true(
            keys.get(i) == kase.patterns.get(p),
            kase.name + ": a group's key differs from its rows'",
        )
        if flat:
            assert_equal(result.item(i, "f").int64(), Int64(first % 3))


def grouped(
    kase: Case, rows: Int, lazy: Bool, ordered: Bool, flat: Bool
) raises -> DataFrame:
    var names: List[String] = ["key"]
    if flat:
        names = ["f", "key"]
    var source = frame(kase, rows)
    if lazy:
        return (
            source.lazy()
            .filter(col("v") >= lit(Int64(0)))
            .group_by(names, maintain_order=ordered)
            .agg(expressions())
            .collect()
        )
    return source.group_by(names, maintain_order=ordered).agg(expressions())


def run_cases(threads: Int, rows: Int, lazy: Bool) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for kase in cases():
        for ordered in [False, True]:
            for flat in [False, True]:
                var observed = StrategyTrace()
                var result = grouped(kase, rows, lazy, ordered, flat)
                var path = (
                    "dataframe-path: pipeline.group_table\n" if lazy else (
                        "dataframe-path: group_by.aggregate_table\n"
                    )
                )
                assert_true(
                    path in observed.text(),
                    kase.name + ": nested keys did not reach the table",
                )
                _ = observed^
                check(kase, result, rows, flat)
                if ordered:
                    # First-occurrence order: each group's first row rises.
                    for i in range(1, result.height()):
                        assert_true(
                            result.item(i, "first_row").int64()
                            > result.item(i - 1, "first_row").int64()
                        )
    _ = trace^
    _ = workers^


def test_eager_nested_keys_on_tables_that_clear() raises:
    run_cases(8, 60_000, False)


def test_eager_nested_keys_on_one_worker() raises:
    run_cases(1, 2_000, False)


def test_pipeline_nested_keys_after_a_filter() raises:
    run_cases(8, 60_000, True)


def test_distinct_rows_and_counts_of_nested_values() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    for kase in cases():
        var rows = 20_000
        var source = frame(kase, rows)
        var classes = 0
        for c in kase.classes:
            classes = max(classes, c + 1)
        # Every class appears among the keys; `nk` picks some of them.
        var hit = List[Bool](length=classes, fill=False)
        for row in range(rows):
            hit[kase.classes[other_of(row, len(kase.classes))]] = True
        var picked = 0
        for one in hit:
            if one:
                picked += 1
        assert_equal(
            source.select(col("key").n_unique()).item(0, "key").int64(),
            Int64(classes),
            kase.name,
        )
        assert_equal(
            source.select(col("nk").n_unique()).item(0, "nk").int64(),
            Int64(picked),
            kase.name,
        )
        assert_equal(
            source.lazy()
            .select(col("nk").n_unique())
            .collect()
            .item(0, "nk")
            .int64(),
            Int64(picked),
            kase.name,
        )
        assert_equal(
            source.unique(["key"], maintain_order=True).height(),
            classes,
            kase.name,
        )
        assert_equal(
            source.lazy().unique(["key"]).collect().height(),
            classes,
            kase.name,
        )
        assert_equal(source.group_by("key").len().height(), classes, kase.name)
        # An empty frame groups to nothing, typed as its keys.
        var empty = source.clear().group_by("key").agg(expressions())
        assert_equal(empty.height(), 0)
        assert_true(empty.column("key").dtype() == kase.patterns.dtype())
    _ = workers^
