"""List element transforms: unique, sort, reverse, slices, gathers,
drop_nulls, shift, diff and concatenation, eager and lazy."""
# contract-group: expr.list
from std.math import isnan, nan as std_nan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    col,
    concat_list,
    lit,
)
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.string import StringBuilder
from tests.support.coverage_generators import chunk_layout, offset_layout

# A null element in the integer and float fixtures.
comptime N = -999


def int_lists(
    name: String,
    rows: List[List[Int]],
    valid: List[Bool],
    dtype: DataType = DataType.INT64,
) raises -> Series:
    """A list column; N marks a null element. A null row keeps a poisoned
    payload of two elements, which no result may show."""
    var offsets: List[Int64] = [0]
    var values = List[Int64]()
    var present = List[Bool]()
    for i in range(len(rows)):
        if not valid[i]:
            values.append(77)
            values.append(78)
            present.append(True)
            present.append(True)
        else:
            for value in rows[i]:
                values.append(Int64(0) if value == N else Int64(value))
                present.append(value != N)
        offsets.append(Int64(len(values)))
    var child = Series("item", Column[Int64](values^, present)).cast(dtype)
    var flags = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            flags[i // 8] |= UInt8(1) << UInt8(i % 8)
    return Series(name, ListColumn(offsets^, child^, flags^))


def float_lists(
    name: String, rows: List[List[Float64]], valid: List[Bool]
) raises -> Series:
    var offsets: List[Int64] = [0]
    var values = List[Float64]()
    var present = List[Bool]()
    for i in range(len(rows)):
        if valid[i]:
            for value in rows[i]:
                values.append(0.0 if value == Float64(N) else value)
                present.append(value != Float64(N))
        offsets.append(Int64(len(values)))
    var flags = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            flags[i // 8] |= UInt8(1) << UInt8(i % 8)
    return Series(
        name,
        ListColumn(
            offsets^, Series("item", Column[Float64](values^, present)), flags^
        ),
    )


def str_lists(
    name: String, rows: List[List[String]], valid: List[Bool]
) raises -> Series:
    """String lists; "~" marks a null element."""
    var offsets: List[Int64] = [0]
    var builder = StringBuilder()
    var count = 0
    for i in range(len(rows)):
        if valid[i]:
            for value in rows[i]:
                if value == "~":
                    builder.append_null()
                else:
                    builder.append(value)
                count += 1
        offsets.append(Int64(count))
    var flags = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            flags[i // 8] |= UInt8(1) << UInt8(i % 8)
    return Series(
        name, ListColumn(offsets^, Series("item", builder^.finish()), flags^)
    )


def layout(frame: DataFrame, kind: Int) raises -> DataFrame:
    return (
        frame.copy() if kind
        == 0 else offset_layout(frame) if kind
        == 1 else chunk_layout(frame)
    )


def check(frame: DataFrame, expr: Expr, expected: Series) raises:
    """expr gives expected eagerly and lazily, on every layout."""
    for kind in range(3):
        var source = layout(frame, kind)
        var eager = source.select_exprs([expr.alias("r")]).column("r")
        var lazy = (
            source.lazy().select_exprs([expr.alias("r")]).collect().column("r")
        )
        var want = expected.renamed("r")
        assert_true(eager.equals(want))
        assert_true(lazy.equals(want))
        # Small batches evaluate windows onto the shared child.
        for batch in [1, 3]:
            var batched = source.select_exprs(
                [expr.alias("r")], batch_size=batch
            ).column("r")
            assert_true(batched.equals(want))


def fixture() raises -> DataFrame:
    var valid: List[Bool] = [True, False, True, True]
    return DataFrame(
        [
            int_lists(
                "a", [[3, 1, N, 1, 2], [], [], [5]], valid, DataType.INT8
            ),
            int_lists("b", [[1], [2], [], []], [True, True, False, True]),
            Series("x", Column[Int64]([1, 0, 3, 4], [True, False, True, True])),
        ]
    )


def test_list_transforms_follow_row_validity() raises:
    var frame = fixture()
    var valid: List[Bool] = [True, False, True, True]
    var i8 = DataType.INT8
    check(
        frame,
        col("a").list().unique(maintain_order=True),
        int_lists("", [[3, 1, N, 2], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().n_unique(),
        Series("", Column[Int64]([4, 0, 0, 1], valid)),
    )
    check(
        frame,
        col("a").list().sort(),
        int_lists("", [[N, 1, 1, 2, 3], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().sort(descending=True, nulls_last=True),
        int_lists("", [[3, 2, 1, 1, N], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().reverse(),
        int_lists("", [[2, 1, N, 1, 3], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().slice(1, 2),
        int_lists("", [[1, N], [], [], []], valid, i8),
    )
    check(
        frame,
        col("a").list().slice(-2),
        int_lists("", [[1, 2], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().head(2),
        int_lists("", [[3, 1], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().tail(2),
        int_lists("", [[1, 2], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().gather([0, -1], null_on_oob=True),
        int_lists("", [[3, 2], [], [N, N], [5, 5]], valid, i8),
    )
    check(
        frame,
        col("a").list().gather_every(2, 1),
        int_lists("", [[1, 1], [], [], []], valid, i8),
    )
    check(
        frame,
        col("a").list().drop_nulls(),
        int_lists("", [[3, 1, 1, 2], [], [], [5]], valid, i8),
    )
    check(
        frame,
        col("a").list().shift(1),
        int_lists("", [[N, 3, 1, N, 1], [], [], [N]], valid, i8),
    )
    check(
        frame,
        col("a").list().shift(-2),
        int_lists("", [[N, 1, 2, N, N], [], [], [N]], valid, i8),
    )
    check(
        frame,
        col("a").list().diff(),
        int_lists("", [[N, -2, N, N, 1], [], [], [N]], valid, i8),
    )
    check(
        frame,
        col("a").list().diff(-1),
        int_lists("", [[2, N, N, -1, N], [], [], [N]], valid, i8),
    )


def test_list_concat_and_concat_list() raises:
    var frame = fixture()
    # Int8 and Int64 elements concatenate as Int64; a null list on either
    # side makes the row null.
    check(
        frame,
        col("a").list().concat(col("b")),
        int_lists(
            "", [[3, 1, N, 1, 2, 1], [], [], [5]], [True, False, False, True]
        ),
    )
    # A non-list operand adds one element, null included; a bare literal
    # has its default type, Int64.
    check(
        frame,
        concat_list([col("a"), col("x")]),
        int_lists(
            "", [[3, 1, N, 1, 2, 1], [], [3], [5, 4]], [True, False, True, True]
        ),
    )
    check(
        frame,
        col("a").list().concat(7),
        int_lists(
            "",
            [[3, 1, N, 1, 2, 7], [], [7], [5, 7]],
            [True, False, True, True],
        ),
    )
    check(
        frame,
        concat_list([col("x"), col("x")]),
        int_lists(
            "", [[1, 1], [N, N], [3, 3], [4, 4]], [True, True, True, True]
        ),
    )


def test_list_unsigned_diff_widens() raises:
    var frame = DataFrame([int_lists("u", [[1, 5, 2]], [True], DataType.UINT8)])
    check(
        frame,
        col("u").list().diff(),
        int_lists("", [[N, 4, -3]], [True], DataType.INT16),
    )


def test_list_compare_strings_floats_and_bools() raises:
    var strings = DataFrame(
        [str_lists("s", [["b", "~", "a", "b"], [], ["x"]], [True, True, False])]
    )
    check(
        strings,
        col("s").list().unique(maintain_order=True),
        str_lists("", [["b", "~", "a"], [], []], [True, True, False]),
    )
    check(
        strings,
        col("s").list().sort(nulls_last=True),
        str_lists("", [["a", "b", "b", "~"], [], []], [True, True, False]),
    )
    var nan = std_nan[DType.float64]()
    var floats = DataFrame(
        [
            float_lists(
                "f",
                [[1.0, Float64(N), 1.0, nan, nan, -0.0, 0.0]],
                [True],
            )
        ]
    )
    var unique = floats.select_exprs(
        [col("f").list().unique(maintain_order=True).alias("u")]
    )
    # NaN equals NaN and -0.0 equals 0.0, so four values remain.
    assert_equal(len(unique.item(0, "u").list()), 4)
    var sorted = floats.select_exprs([col("f").list().sort().alias("s")])
    var row = sorted.item(0, "s").list()
    assert_true(row.get(0).is_null())
    assert_true(isnan(row.get(6).float64()))
    var bools = DataFrame(
        [
            Series(
                "b",
                ListColumn(
                    [0, 4],
                    Series(
                        "item",
                        BoolColumn(
                            [True, False, False, True],
                            [True, False, True, True],
                        ),
                    ),
                ),
            )
        ]
    )
    var result = bools.select_exprs(
        [
            col("b").list().unique(maintain_order=True).alias("u"),
            col("b").list().sort().alias("s"),
            col("b").list().n_unique().alias("n"),
        ]
    )
    assert_equal(result.item(0, "n").int64(), 3)
    var u = result.item(0, "u").list()
    assert_equal(len(u), 3)
    assert_true(u.get(0).bool())
    assert_true(u.get(1).is_null())
    var s = result.item(0, "s").list()
    assert_true(s.get(0).is_null())
    assert_true(not s.get(1).bool())
    assert_true(s.get(3).bool())


def test_list_long_rows_match_series_operations() raises:
    # One row long enough to be split across workers, beside short rows.
    var long = List[Int]()
    for i in range(50000):
        long.append(N if i % 97 == 0 else (i * 7919) % 1013)
    var frame = DataFrame(
        [int_lists("a", [long.copy(), [2, 1], [1]], [True, True, True])]
    )
    var child = frame.column("a").list_column().row(0)
    var result = frame.select_exprs(
        [
            col("a").list().sort().alias("s"),
            col("a").list().unique(maintain_order=True).alias("u"),
            col("a").list().n_unique().alias("n"),
        ]
    )
    assert_true(result.item(0, "s").list().equals(child.sort(nulls_last=False)))
    assert_true(result.item(0, "u").list().equals(child.unique(True)))
    assert_equal(result.item(0, "n").int64(), Int64(len(child.unique(True))))
    assert_equal(result.item(1, "n").int64(), 2)


def test_list_transform_errors() raises:
    var frame = fixture()
    with assert_raises(contains="out of bounds"):
        _ = frame.select_exprs([col("a").list().gather([7])])
    with assert_raises(contains="out of bounds"):
        _ = frame.lazy().select_exprs([col("a").list().gather([7])]).collect()
    with assert_raises(contains="list.sort requires a list"):
        _ = frame.select_exprs([col("x").list().sort()])
    with assert_raises(contains="positive step"):
        _ = frame.select_exprs([col("a").list().gather_every(0)])
    var strings = DataFrame([str_lists("s", [["a"]], [True])])
    with assert_raises(contains="list.diff needs"):
        _ = strings.select_exprs([col("s").list().diff()])
    with assert_raises(contains="common type"):
        _ = strings.select_exprs(
            [col("s").list().concat(col("s").list().len())]
        )


def test_list_transforms_inside_aggregation() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([1, 1, 2, 1, 2])),
            Series("v", Column[Int64]([3, 1, 5, 3, 4])),
        ]
    )
    var result = (
        frame.group_by(["g"])
        .agg([col("v").implode().list().unique().list().sort().alias("v")])
        .sort(["g"])
    )
    assert_true(
        result.column("v").equals(
            int_lists("v", [[1, 3], [4, 5]], [True, True])
        )
    )
