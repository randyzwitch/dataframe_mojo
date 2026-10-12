"""Row-order semantics against scalar scans, membership and pairwise ranks."""
from std.math import abs, isnan, isinf
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.window_reference import (
    WindowCells,
    cumulative,
    shifted,
    changed,
    filled,
    rolling,
    ranked,
    interpolated,
)


def fixture(n: Int, null_mode: Int) -> WindowCells:
    var values = List[Float64]()
    var valid = List[Bool]()
    for row in range(n):
        values.append(Float64((row * 7 + 3) % 13 - 6))
        valid.append(null_mode == 0 or (null_mode == 1 and row % 5 >= 2))
    return WindowCells(values^, valid^)


def integer_frame(cells: WindowCells) raises -> DataFrame:
    var values = List[Int64]()
    for value in cells.values:
        values.append(Int64(value))
    return DataFrame([Series("x", Column[Int64](values^, cells.valid.copy()))])


def layout_of(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def assert_cells(
    actual: Series,
    expected: WindowCells,
    dtype: DataType,
    approximate: Bool = False,
    standard: Bool = False,
) raises:
    assert_true(actual.dtype() == dtype)
    assert_equal(len(actual), len(expected.values))
    for row in range(len(expected.values)):
        var value = actual.get(row)
        assert_equal(value.is_null(), not expected.valid[row])
        if not expected.valid[row]:
            continue
        if dtype == DataType.INT64:
            assert_equal(value.int64(), Int64(expected.values[row]))
        else:
            var observed = value.float64()
            var target = expected.values[row]
            if approximate and not isnan(target) and not isinf(target):
                # Two-pass and sliding variance may reassociate. The budget
                # is 1e-12 absolute + relative in variance units. Compare std
                # squared: sqrt is ill-conditioned at zero, so an absolute
                # std budget would falsely reject tiny variance roundoff.
                var compared = observed * observed if standard else observed
                var reference = target * target if standard else target
                if standard:
                    assert_true(observed >= 0)
                assert_true(
                    abs(compared - reference) <= 1e-12 + 1e-12 * abs(reference),
                    "Window value mismatch for "
                    + actual.name()
                    + " at row "
                    + String(row)
                    + ": observed "
                    + String(observed)
                    + ", expected "
                    + String(target),
                )
            else:
                assert_float(observed, target)


def test_window_scalar_models_have_literal_anchors() raises:
    var cells = WindowCells([2, 99, 4, 8, 99], [True, False, True, True, False])
    var sum = cumulative(cells, "sum", False)
    assert_equal(sum.values, [Float64(2), 0, 6, 14, 0])
    assert_equal(sum.valid, [True, False, True, True, False])
    assert_equal(
        cumulative(cells, "count", True).values, [Float64(3), 2, 2, 1, 0]
    )
    assert_equal(shifted(cells, -1).valid, [False, True, True, False, False])
    assert_equal(filled(cells, False, 1).values, [Float64(2), 2, 4, 8, 8])
    var mean = rolling(cells, 3, 2, "mean")
    assert_equal(mean.valid, [False, False, True, True, True])
    assert_equal(mean.values[2], Float64(3))
    assert_equal(mean.values[3], Float64(6))
    var ties = WindowCells([4, 2, 4, 99, 1], [True, True, True, False, True])
    assert_equal(
        ranked(ties, "average", False).values, [Float64(3.5), 2, 3.5, 0, 1]
    )
    assert_equal(ranked(ties, "ordinal", True).values, [Float64(1), 3, 2, 0, 4])
    var gap = WindowCells([0, 99, 6], [True, False, True])
    assert_equal(interpolated(gap, False, [0, 1, 3]).values, [Float64(0), 2, 6])
    assert_equal(interpolated(gap, True, [0, 1, 2]).values, [Float64(0), 6, 6])


def test_window_statistical_error_budget_rejects_material_and_nonfinite_errors() raises:
    var expected = WindowCells([0, 3], [True, True])
    assert_cells(
        Series("std", Column[Float64]([1e-7, 3 + 1e-13])),
        expected,
        DataType.FLOAT64,
        True,
        True,
    )
    var mutations: List[List[Float64]] = [
        [1e-3, 3],
        [0, 3 + 1e-6],
        [0, -3],
        [Float64(0) / Float64(0), 3],
    ]
    for values in mutations:
        with assert_raises():
            assert_cells(
                Series("std", Column[Float64](values.copy())),
                expected,
                DataType.FLOAT64,
                True,
                True,
            )


def test_cumulative_shift_change_and_fill_match_scalar_order() raises:
    for n in [0, 1, 9, 65]:
        for null_mode in range(3):
            var cells = fixture(n, null_mode)
            var original = integer_frame(cells)
            for layout in range(3):
                var frame = layout_of(original, layout)
                for reverse in [False, True]:
                    var actual = frame.select_exprs(
                        [
                            col("x").cum_sum(reverse).alias("sum"),
                            col("x").cum_min(reverse).alias("min"),
                            col("x").cum_max(reverse).alias("max"),
                            col("x").cum_count(reverse).alias("count"),
                        ]
                    )
                    assert_equal(
                        actual.columns(), ["sum", "min", "max", "count"]
                    )
                    assert_equal(actual.height(), n)
                    for name in ["sum", "min", "max", "count"]:
                        assert_cells(
                            actual.column(name),
                            cumulative(cells, name, reverse),
                            DataType.INT64,
                        )
                for distance in [-70, -2, -1, 0, 1, 3, 70]:
                    var actual = frame.select_exprs(
                        [
                            col("x").shift(distance).alias("shift"),
                            col("x").diff(distance).alias("diff"),
                            col("x").pct_change(distance).alias("pct"),
                        ]
                    )
                    assert_equal(actual.columns(), ["shift", "diff", "pct"])
                    assert_cells(
                        actual.column("shift"),
                        shifted(cells, distance),
                        DataType.INT64,
                    )
                    assert_cells(
                        actual.column("diff"),
                        changed(cells, distance, False),
                        DataType.INT64,
                    )
                    assert_cells(
                        actual.column("pct"),
                        changed(cells, distance, True),
                        DataType.FLOAT64,
                    )
                for limit in [-1, 0, 1, 2, 70]:
                    var actual = frame.select_exprs(
                        [
                            col("x").forward_fill(limit).alias("forward"),
                            col("x").backward_fill(limit).alias("backward"),
                        ]
                    )
                    assert_cells(
                        actual.column("forward"),
                        filled(cells, False, limit),
                        DataType.INT64,
                    )
                    assert_cells(
                        actual.column("backward"),
                        filled(cells, True, limit),
                        DataType.INT64,
                    )


def test_rolling_membership_min_samples_and_ddof_match_two_pass_reference() raises:
    for n in [0, 1, 9, 65]:
        for null_mode in range(3):
            var cells = fixture(n, null_mode)
            var original = integer_frame(cells)
            for layout in range(3):
                var frame = layout_of(original, layout)
                for width in [1, 3, 10]:
                    for needed in [-1, 0, 1, width]:
                        var actual = frame.select_exprs(
                            [
                                col("x")
                                .rolling_sum(width, needed)
                                .alias("sum"),
                                col("x")
                                .rolling_mean(width, needed)
                                .alias("mean"),
                                col("x")
                                .rolling_min(width, needed)
                                .alias("min"),
                                col("x")
                                .rolling_max(width, needed)
                                .alias("max"),
                            ]
                        )
                        assert_equal(actual.height(), n)
                        assert_equal(
                            actual.columns(), ["sum", "mean", "min", "max"]
                        )
                        for kind in ["sum", "mean", "min", "max"]:
                            assert_cells(
                                actual.column(kind),
                                rolling(cells, width, needed, kind),
                                DataType.FLOAT64 if kind
                                == "mean" else DataType.INT64,
                                True,
                            )
                        for ddof in [0, 1, 4]:
                            var moments = frame.select_exprs(
                                [
                                    col("x")
                                    .rolling_var(width, needed, ddof)
                                    .alias("var"),
                                    col("x")
                                    .rolling_std(width, needed, ddof)
                                    .alias("std"),
                                ]
                            )
                            for kind in ["var", "std"]:
                                try:
                                    assert_cells(
                                        moments.column(kind),
                                        rolling(
                                            cells, width, needed, kind, ddof
                                        ),
                                        DataType.FLOAT64,
                                        True,
                                        kind == "std",
                                    )
                                except error:
                                    raise Error(
                                        "rolling n="
                                        + String(n)
                                        + " null_mode="
                                        + String(null_mode)
                                        + " layout="
                                        + String(layout)
                                        + " width="
                                        + String(width)
                                        + " needed="
                                        + String(needed)
                                        + " ddof="
                                        + String(ddof)
                                        + ": "
                                        + String(error)
                                    )


def test_ranks_use_declared_ties_null_nan_and_descending_policy() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var cells = WindowCells(
        [4, nan, -0.0, 4, 0.0, -inf, inf, 99, nan, -2],
        [True, True, True, True, True, True, True, False, True, True],
    )
    for n in [0, 1, len(cells.values)]:
        var values = List[Float64]()
        var valid = List[Bool]()
        for row in range(n):
            values.append(cells.values[row])
            valid.append(cells.valid[row])
        var subset = WindowCells(values^, valid^)
        var original = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](subset.values.copy(), subset.valid.copy()),
                )
            ]
        )
        for layout in range(3):
            var frame = layout_of(original, layout)
            for method in ["average", "min", "max", "dense", "ordinal"]:
                for descending in [False, True]:
                    var actual = frame.select(col("x").rank(method, descending))
                    assert_equal(actual.columns(), ["x"])
                    assert_cells(
                        actual.column("x"),
                        ranked(subset, method, descending),
                        DataType.FLOAT64 if method
                        == "average" else DataType.INT64,
                    )
    var exact = DataFrame(
        [
            Series(
                "x", Column[UInt64]([UInt64.MAX, UInt64.MAX - 1, 0, UInt64.MAX])
            )
        ]
    )
    var expected = WindowCells([3, 2, 1, 4], [True, True, True, True])
    assert_cells(
        exact.select(col("x").rank("ordinal")).column("x"),
        expected,
        DataType.INT64,
    )
    var text = DataFrame(
        [
            Series(
                "x",
                Column[String](
                    ["é", "a", "é", "", "null"], [True, True, True, True, False]
                ),
            )
        ]
    )
    var text_order = WindowCells(
        [2, 1, 2, 0, 99], [True, True, True, True, False]
    )
    var boolean = DataFrame(
        [
            Series(
                "x",
                Column[Bool](
                    [True, False, True, False, True],
                    [True, True, True, True, False],
                ),
            )
        ]
    )
    var bool_order = WindowCells(
        [1, 0, 1, 0, 99], [True, True, True, True, False]
    )
    for layout in range(3):
        for method in ["average", "min", "max", "dense", "ordinal"]:
            for descending in [False, True]:
                var dtype = (
                    DataType.FLOAT64 if method == "average" else DataType.INT64
                )
                assert_cells(
                    layout_of(text, layout)
                    .select(col("x").rank(method, descending))
                    .column("x"),
                    ranked(text_order, method, descending),
                    dtype,
                )
                assert_cells(
                    layout_of(boolean, layout)
                    .select(col("x").rank(method, descending))
                    .column("x"),
                    ranked(bool_order, method, descending),
                    dtype,
                )


def test_interpolation_keeps_end_nulls_nearest_ties_and_coordinate_distances() raises:
    var cells = WindowCells(
        [99, 2, 99, 99, 8, 99, 14, 99],
        [False, True, False, False, True, False, True, False],
    )
    var positions: List[Float64] = [0, 1, 2, 4, 7, 8, 10, 12]
    var row_positions: List[Float64] = [0, 1, 2, 3, 4, 5, 6, 7]
    var original = integer_frame(cells).with_columns(
        col("x").cast(DataType.FLOAT64).alias("y")
    )
    original = original.hstack(
        DataFrame([Series("axis", Column[Float64](positions.copy()))])
    )
    for layout in range(3):
        var frame = layout_of(original, layout)
        var expressions: List[Expr] = [
            col("x").interpolate().alias("linear"),
            col("x").interpolate("nearest").alias("nearest"),
            col("y").interpolate().alias("float"),
        ]
        # The chunked coordinate path has its own strict failure witness,
        # allowing these independent unary-window assertions to keep running.
        if layout != 2:
            expressions.append(col("x").interpolate_by(col("axis")).alias("by"))
        var actual = frame.select_exprs(expressions)
        assert_cells(
            actual.column("linear"),
            interpolated(cells, False, row_positions),
            DataType.FLOAT64,
            True,
        )
        assert_cells(
            actual.column("nearest"),
            interpolated(cells, True, row_positions),
            DataType.INT64,
        )
        if layout != 2:
            assert_cells(
                actual.column("by"),
                interpolated(cells, False, positions),
                DataType.FLOAT64,
                True,
            )
        assert_cells(
            actual.column("float"),
            interpolated(cells, False, row_positions),
            DataType.FLOAT64,
            True,
        )
        var empty = frame.clear().select(col("x").interpolate())
        assert_equal(empty.height(), 0)
        assert_true(empty.column("x").dtype() == DataType.FLOAT64)


def test_coordinate_interpolation_accepts_chunked_equal_length_columns() raises:
    var frame = chunk_layout(
        DataFrame(
            [
                Series(
                    "x",
                    Column[Int64](
                        [99, 2, 99, 99, 8, 99, 14, 99],
                        [False, True, False, False, True, False, True, False],
                    ),
                ),
                Series("axis", Column[Float64]([0, 1, 2, 4, 7, 8, 10, 12])),
            ]
        )
    )
    assert_equal(frame.column("x").is_chunked(), True)
    assert_equal(frame.column("axis").is_chunked(), True)
    var actual = frame.select(col("x").interpolate_by(col("axis")))
    assert_cells(
        actual.column("x"),
        WindowCells(
            [0, 2, 3, 5, 8, 10, 14, 0],
            [False, True, True, True, True, True, True, False],
        ),
        DataType.FLOAT64,
        True,
    )


def test_cut_qcut_edges_labels_nulls_and_breakpoint_struct_are_explicit() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var original = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [-inf, -1, 0, 1, 2, inf, nan, 99],
                    [True, True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    for layout in range(3):
        var frame = layout_of(original, layout)
        for left_closed in [False, True]:
            var result = frame.select(
                col("x").cut([0, 1], ["low", "mid", "high"], left_closed, True)
            )
            assert_equal(result.columns(), ["x"])
            var dtype = result.column("x").dtype()
            assert_true(
                dtype
                == DataType.struct(
                    ["breakpoint", "category"],
                    [DataType.FLOAT64, DataType.STRING],
                )
            )
            for row in range(8):
                assert_equal(
                    result.item(row, "x").struct_field("category").is_null(),
                    row >= 6,
                )
                assert_equal(
                    result.item(row, "x").struct_field("breakpoint").is_null(),
                    row >= 6,
                )
                if row >= 6:
                    continue
                var x = frame.item(row, "x").float64()
                var bin = (
                    Int(x >= 0)
                    + Int(x >= 1) if left_closed else Int(x > 0)
                    + Int(x > 1)
                )
                var labels: List[String] = ["low", "mid", "high"]
                var upper: List[Float64] = [0, 1, inf]
                assert_equal(
                    result.item(row, "x").struct_field("category").string(),
                    labels[bin],
                )
                assert_float(
                    result.item(row, "x").struct_field("breakpoint").float64(),
                    upper[bin],
                )
        var default_labels = frame.select(col("x").cut([0, 1]))
        var literal_labels: List[String] = [
            "(-inf, 0]",
            "(-inf, 0]",
            "(-inf, 0]",
            "(0, 1]",
            "(1, inf]",
            "(1, inf]",
        ]
        for row in range(6):
            assert_equal(
                default_labels.item(row, "x").string(), literal_labels[row]
            )
    var quantiles = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [0, 2, 4, 6, 8, 99], [True, True, True, True, True, False]
                ),
            )
        ]
    )
    for layout in range(3):
        for left_closed in [False, True]:
            var result = layout_of(quantiles, layout).select(
                col("x").qcut(
                    [0.25, 0.5, 0.75], ["a", "b", "c", "d"], left_closed
                )
            )
            var labels: List[String] = ["a", "b", "c", "d"]
            for row in range(5):
                var x = row * 2
                var bin = (
                    Int(x >= 2)
                    + Int(x >= 4)
                    + Int(x >= 6) if left_closed else Int(x > 2)
                    + Int(x > 4)
                    + Int(x > 6)
                )
                assert_equal(result.item(row, "x").string(), labels[bin])
            assert_true(result.item(5, "x").is_null())
    var same = DataFrame([Series("x", Column[Int64]([1, 1, 1]))])
    with assert_raises():
        _ = same.select(col("x").qcut([0.25, 0.5, 0.75]))
    var duplicate_allowed = same.select(
        col("x").qcut([0.25, 0.5, 0.75], allow_duplicates=True)
    )
    for row in range(3):
        assert_equal(duplicate_allowed.item(row, "x").string(), "(-inf, 1]")


def test_partition_windows_reset_reassociate_and_group_aggregates_broadcast() raises:
    var cells = WindowCells(
        [3, 10, 99, 1, 4, 2, 5, 20, 99],
        [True, True, False, True, True, True, True, True, False],
    )
    var keys: List[Int64] = [0, 1, 0, 2, 1, 2, 0, 1, 2]
    var frame = integer_frame(cells).hstack(
        DataFrame(
            [
                Series(
                    "k",
                    Column[String](
                        [
                            "a",
                            "b",
                            "a",
                            "hidden",
                            "b",
                            "hidden",
                            "a",
                            "b",
                            "hidden",
                        ],
                        [
                            True,
                            True,
                            True,
                            False,
                            True,
                            False,
                            True,
                            True,
                            False,
                        ],
                    ),
                ),
                Series(
                    "constant",
                    Column[Bool](
                        [True, True, True, True, True, True, True, True, True]
                    ),
                ),
            ]
        )
    )
    for layout in range(3):
        var input = layout_of(frame, layout)
        for composite in [False, True]:
            var partition_names: List[String] = [
                "k",
                "constant",
            ] if composite else ["k"]
            var actual = input.select_exprs(
                [
                    col("x").cum_sum().over(partition_names).alias("sum"),
                    col("x")
                    .cum_sum(True)
                    .over(partition_names)
                    .alias("reverse"),
                    col("x").shift().over(partition_names).alias("shift"),
                    col("x").rank().over(partition_names).alias("rank"),
                    col("x")
                    .rolling_mean(2, 1)
                    .over(partition_names)
                    .alias("rolling"),
                    col("x").forward_fill().over(partition_names).alias("fill"),
                    col("x").sum().over(partition_names).alias("broadcast"),
                ]
            )
            assert_equal(actual.height(), len(keys))
            assert_equal(
                actual.columns(),
                [
                    "sum",
                    "reverse",
                    "shift",
                    "rank",
                    "rolling",
                    "fill",
                    "broadcast",
                ],
            )
            for group in range(3):
                var indices = List[Int]()
                var values = List[Float64]()
                var valid = List[Bool]()
                var total = Int64(0)
                for row in range(len(keys)):
                    if keys[row] == Int64(group):
                        indices.append(row)
                        values.append(cells.values[row])
                        valid.append(cells.valid[row])
                        if cells.valid[row]:
                            total += Int64(cells.values[row])
                var subset = WindowCells(values^, valid^)
                var expected: List[WindowCells] = [
                    cumulative(subset, "sum", False),
                    cumulative(subset, "sum", True),
                    shifted(subset, 1),
                    ranked(subset, "average", False),
                    rolling(subset, 2, 1, "mean"),
                    filled(subset, False, -1),
                ]
                var names: List[String] = [
                    "sum",
                    "reverse",
                    "shift",
                    "rank",
                    "rolling",
                    "fill",
                ]
                for column in range(len(names)):
                    var restored = actual.column(names[column]).take(indices)
                    assert_cells(
                        restored,
                        expected[column],
                        DataType.FLOAT64 if column == 3
                        or column == 4 else DataType.INT64,
                        True,
                    )
                for row in indices:
                    assert_equal(actual.item(row, "broadcast").int64(), total)
        var grouped = input.group_by("k", maintain_order=True).agg(
            col("x").cum_sum().max().alias("peak")
        )
        assert_equal(grouped.height(), 3)
        for row in range(3):
            var peaks: List[Int64] = [8, 34, 3]
            assert_equal(grouped.item(row, "peak").int64(), peaks[row])


def test_cumulative_checks_each_prefix_rolling_checks_final_window_and_skips_nulls() raises:
    var original = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1, -1]))])
    for layout in range(3):
        var frame = layout_of(original, layout)
        with assert_raises():
            _ = frame.select(col("x").cum_sum())
        assert_equal(
            frame.select(col("x").cum_sum(True)).item(0, "x").int64(), Int64.MAX
        )
        assert_equal(
            frame.select(col("x").rolling_sum(3)).item(2, "x").int64(),
            Int64.MAX,
        )
        with assert_raises():
            _ = frame.select(col("x").rolling_sum(2))
    var poisoned = DataFrame(
        [Series("x", Column[Int64]([1, Int64.MAX, 2], [True, False, True]))]
    )
    var expected = WindowCells([1, 0, 3], [True, False, True])
    assert_cells(
        poisoned.select(col("x").cum_sum()).column("x"),
        expected,
        DataType.INT64,
    )


def test_window_binding_and_arguments_are_validated_on_empty_inputs() raises:
    var original = DataFrame(
        [Series("x", Column[Int64]([1])), Series("s", Column[String](["a"]))]
    )
    for empty in [False, True]:
        var frame = original.clear() if empty else original.copy()
        var invalid: List[Expr] = [
            col("x").rolling_sum(0),
            col("x").rolling_mean(-1),
            col("x").rolling_min(2, -2),
            col("x").rolling_var(2, 3),
            col("x").rolling_std(2, ddof=-1),
            col("x").forward_fill(-2),
            col("x").backward_fill(-2),
            col("x").rank("random"),
            col("x").interpolate("spline"),
            col("x").interpolate_by(lit(1)),
            col("x").interpolate_by(col("s")),
            col("s").interpolate(),
            col("s").cum_sum(),
            col("s").rolling_mean(2),
            lit(1).shift(),
            col("x").sum().cum_sum(),
            col("x").cum_sum().over("missing"),
            col("x").cut([0, 1], ["one"]),
            col("s").cut([1]),
        ]
        for expression in invalid:
            with assert_raises():
                _ = frame.select(expression)
    # Edge uniqueness and qcut probability bounds are data-dependent checks;
    # assert them with values rather than prescribing undocumented empty rules.
    for expression in [
        col("x").cut([1, 1]),
        col("x").cut([2, 1]),
        col("x").qcut([-0.1]),
        col("x").qcut([1.1]),
    ]:
        with assert_raises():
            _ = original.select(expression)
