"""Logical types retain exact ordered cells, metadata and partition association."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout
from tests.support.selection_types_reference import (
    assert_selection_cell,
    decimal_text,
    selection_column,
    selection_decimal,
    selection_dtype,
    selection_valid,
)
from tests.support.extended_window_reference import (
    extended_counts,
    extended_decimal_totals,
    extended_group,
    extended_partitions,
    extended_ranks,
    extended_sources,
)
from tests.support.window_reference import WindowCells, rolling


def extended_frame(
    kind: Int, n: Int, offset: Int, mode: Int, chunked: Bool = False
) raises -> DataFrame:
    var rows = List[Int64]()
    var keys = List[Int64]()
    var key_valid = List[Bool]()
    for row in range(offset, offset + n):
        rows.append(Int64(row))
        keys.append(Int64(row % 3))
        key_valid.append(row % 7 != 3)
    var frame = DataFrame(
        [
            Series("row", Column[Int64](rows^)),
            Series("g", Column[Int64](keys^, key_valid^)),
            selection_column(kind, n, offset, mode),
        ]
    )
    return chunk_layout(frame) if chunked else frame^


def extended_evaluate(
    frame: DataFrame, expression: Expr, partitioned: Bool
) raises -> DataFrame:
    var selected = expression.over("g") if partitioned else expression.copy()
    return frame.select_exprs([col("row"), selected.alias("out")])


def assert_extended_association(actual: DataFrame, n: Int, offset: Int) raises:
    var names: List[String] = ["row", "out"]
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), n)
    assert_true(actual.column("row").dtype() == DataType.INT64)
    for i in range(n):
        assert_equal(actual.item(i, "row").int64(), Int64(i + offset))


def assert_extended_sources(
    actual: DataFrame, kind: Int, rows: List[Int], offset: Int, mode: Int
) raises:
    assert_extended_association(actual, len(rows), offset)
    var dtype = actual.column("out").dtype()
    if kind == 13:
        # The expression binder reads categoricals as strings once an
        # operation is applied; windows retain the decoded values and order.
        assert_true(dtype == DataType.STRING)
    else:
        assert_true(
            dtype == selection_dtype(kind),
            "window logical dtype mismatch for kind "
            + String(kind)
            + ", n="
            + String(len(rows))
            + ": observed "
            + dtype.name()
            + ", expected "
            + selection_dtype(kind).name(),
        )
    for i in range(len(rows)):
        if rows[i] < 0:
            assert_true(actual.item(i, "out").is_null())
        else:
            assert_selection_cell(actual.item(i, "out"), kind, rows[i], mode)


def test_extended_window_scalar_sources_and_ranks_have_literal_anchors() raises:
    # Binary row ranks 3,1,3,2,0,2,1,4,3; row2 is null in mixed mode.
    assert_equal(
        extended_sources(12, 9, 0, 1, False, "shift", distance=2),
        List[Int]([-1, -1, 0, 1, 2, 3, 4, 5, 6]),
    )
    assert_equal(
        extended_sources(12, 9, 0, 1, False, "cum_min"),
        List[Int]([0, 1, -1, 1, 4, 4, 4, -1, 4]),
    )
    assert_equal(
        extended_sources(12, 9, 0, 1, False, "cum_max", reverse=True),
        List[Int]([8, 8, -1, 8, 8, 8, 8, -1, 8]),
    )
    assert_equal(
        extended_sources(
            12, 9, 0, 1, False, "rolling_min", window=3, minimum=2
        ),
        List[Int]([-1, 1, 1, 1, 4, 4, 4, 6, 6]),
    )
    assert_equal(
        extended_sources(12, 9, 0, 1, False, "forward", limit=0),
        List[Int]([0, 1, -1, 3, 4, 5, 6, -1, 8]),
    )
    var counts: List[Int64] = [1, 2, 2, 3, 4, 5, 6, 6, 7]
    var dense: List[Float64] = [4, 2, 0, 3, 1, 3, 2, 0, 4]
    var ordinal: List[Float64] = [6, 2, 0, 4, 1, 5, 3, 0, 7]
    assert_equal(extended_counts(9, 0, 1, False, False), counts)
    assert_equal(
        extended_ranks(12, 9, 0, 1, False, "dense", False),
        dense,
    )
    assert_equal(
        extended_ranks(12, 9, 0, 1, False, "ordinal", False),
        ordinal,
    )
    assert_equal(extended_group(3), 3)
    assert_equal(extended_group(10), 3)


def test_binary_categorical_decimal_geometry_shift_fill_and_counts_preserve_typed_cells() raises:
    for kind in [12, 13, 14, 15, 16, 28]:
        for n in [0, 1, 9, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    for partitioned in [False, True]:
                        var frame = extended_frame(
                            kind, n, offset, mode, n == 65
                        )
                        for distance in [-66, -3, -1, 0, 1, 3, 66]:
                            assert_extended_sources(
                                extended_evaluate(
                                    frame, col("x").shift(distance), partitioned
                                ),
                                kind,
                                extended_sources(
                                    kind,
                                    n,
                                    offset,
                                    mode,
                                    partitioned,
                                    "shift",
                                    distance=distance,
                                ),
                                offset,
                                mode,
                            )
                        for limit in [-1, 0, 1, 3]:
                            assert_extended_sources(
                                extended_evaluate(
                                    frame,
                                    col("x").forward_fill(limit),
                                    partitioned,
                                ),
                                kind,
                                extended_sources(
                                    kind,
                                    n,
                                    offset,
                                    mode,
                                    partitioned,
                                    "forward",
                                    limit=limit,
                                ),
                                offset,
                                mode,
                            )
                            assert_extended_sources(
                                extended_evaluate(
                                    frame,
                                    col("x").backward_fill(limit),
                                    partitioned,
                                ),
                                kind,
                                extended_sources(
                                    kind,
                                    n,
                                    offset,
                                    mode,
                                    partitioned,
                                    "backward",
                                    limit=limit,
                                ),
                                offset,
                                mode,
                            )
                        for reverse in [False, True]:
                            var actual = extended_evaluate(
                                frame, col("x").cum_count(reverse), partitioned
                            )
                            var expected = extended_counts(
                                n, offset, mode, partitioned, reverse
                            )
                            assert_extended_association(actual, n, offset)
                            assert_true(
                                actual.column("out").dtype() == DataType.INT64
                            )
                            for i in range(n):
                                assert_equal(
                                    actual.item(i, "out").int64(), expected[i]
                                )


def test_extended_cumulative_and_rolling_extrema_preserve_exact_types_and_partition_order() raises:
    for kind in [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        28,
    ]:
        for n in [0, 9, 65]:
            for mode in range(3):
                for partitioned in [False, True]:
                    var frame = extended_frame(kind, n, 7, mode, n == 65)
                    for reverse in [False, True]:
                        assert_extended_sources(
                            extended_evaluate(
                                frame, col("x").cum_min(reverse), partitioned
                            ),
                            kind,
                            extended_sources(
                                kind,
                                n,
                                7,
                                mode,
                                partitioned,
                                "cum_min",
                                reverse=reverse,
                            ),
                            7,
                            mode,
                        )
                        assert_extended_sources(
                            extended_evaluate(
                                frame, col("x").cum_max(reverse), partitioned
                            ),
                            kind,
                            extended_sources(
                                kind,
                                n,
                                7,
                                mode,
                                partitioned,
                                "cum_max",
                                reverse=reverse,
                            ),
                            7,
                            mode,
                        )
                    for size in [1, 3, 7, 66]:
                        for minimum in [0, 1, size, size + 1]:
                            assert_extended_sources(
                                extended_evaluate(
                                    frame,
                                    col("x").rolling_min(size, minimum),
                                    partitioned,
                                ),
                                kind,
                                extended_sources(
                                    kind,
                                    n,
                                    7,
                                    mode,
                                    partitioned,
                                    "rolling_min",
                                    window=size,
                                    minimum=minimum,
                                ),
                                7,
                                mode,
                            )
                            assert_extended_sources(
                                extended_evaluate(
                                    frame,
                                    col("x").rolling_max(size, minimum),
                                    partitioned,
                                ),
                                kind,
                                extended_sources(
                                    kind,
                                    n,
                                    7,
                                    mode,
                                    partitioned,
                                    "rolling_max",
                                    window=size,
                                    minimum=minimum,
                                ),
                                7,
                                mode,
                            )


def test_extended_rank_methods_compare_values_and_reassociate_noncontiguous_partitions() raises:
    for kind in [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        28,
    ]:
        for n in [0, 9, 65]:
            for mode in range(3):
                for partitioned in [False, True]:
                    var frame = extended_frame(kind, n, 7, mode, n == 65)
                    for descending in [False, True]:
                        for method in [
                            "average",
                            "min",
                            "max",
                            "dense",
                            "ordinal",
                        ]:
                            var actual = extended_evaluate(
                                frame,
                                col("x").rank(method, descending),
                                partitioned,
                            )
                            var expected = extended_ranks(
                                kind,
                                n,
                                7,
                                mode,
                                partitioned,
                                method,
                                descending,
                            )
                            assert_extended_association(actual, n, 7)
                            assert_true(
                                actual.column("out").dtype()
                                == (
                                    DataType.FLOAT64 if method
                                    == "average" else DataType.INT64
                                )
                            )
                            for row in range(n):
                                var valid = selection_valid(row + 7, mode)
                                assert_equal(
                                    actual.item(row, "out").is_null(), not valid
                                )
                                if valid:
                                    if method == "average":
                                        assert_equal(
                                            actual.item(row, "out").float64(),
                                            expected[row],
                                        )
                                    else:
                                        assert_equal(
                                            actual.item(row, "out").int64(),
                                            Int64(expected[row]),
                                        )


def test_decimal_cumulative_prefixes_are_exact_promoted_and_partitioned_at_every_width() raises:
    for kind in [14, 15, 16]:
        for n in [0, 1, 9, 65]:
            for mode in range(3):
                for partitioned in [False, True]:
                    var frame = extended_frame(kind, n, 7, mode, n == 65)
                    for reverse in [False, True]:
                        var actual = extended_evaluate(
                            frame, col("x").cum_sum(reverse), partitioned
                        )
                        var totals = extended_decimal_totals(
                            kind, n, 7, mode, partitioned, reverse
                        )
                        assert_extended_association(actual, n, 7)
                        assert_true(
                            actual.column("out").dtype()
                            == DataType.decimal(38, 2)
                        )
                        for row in range(n):
                            assert_equal(
                                actual.item(row, "out").is_null(),
                                not selection_valid(row + 7, mode),
                            )
                            if selection_valid(row + 7, mode):
                                assert_equal(
                                    String(actual.item(row, "out")),
                                    decimal_text(totals[row]),
                                )
                        var shifted = extended_sources(
                            kind, n, 7, mode, partitioned, "shift"
                        )
                        var differences = extended_evaluate(
                            frame, col("x").diff(), partitioned
                        )
                        # A difference is a subtraction: one more integer
                        # digit, at most 18 for 64-bit inputs, as DuckDB
                        # types it.
                        assert_true(
                            differences.column("out").dtype()
                            == DataType.decimal(
                                10 if kind == 14 else 18 if kind == 15 else 38,
                                2,
                                128 if kind == 16 else 64,
                            )
                        )
                        for row in range(n):
                            var valid = (
                                selection_valid(row + 7, mode)
                                and shifted[row] >= 0
                            )
                            if shifted[row] >= 0:
                                valid = valid and selection_valid(
                                    shifted[row], mode
                                )
                            assert_equal(
                                differences.item(row, "out").is_null(),
                                not valid,
                            )
                            if valid:
                                assert_equal(
                                    String(differences.item(row, "out")),
                                    decimal_text(
                                        selection_decimal(kind, row + 7)
                                        - selection_decimal(kind, shifted[row])
                                    ),
                                )


def decimal_quarter_frame(kind: Int) raises -> DataFrame:
    var units: List[Int128] = [-125, 250, 99, 450, 0, -275, 99, 125]
    var valid: List[Bool] = [True, True, False, True, True, True, False, True]
    var series: Series
    if kind == 14:
        var narrow = List[Int32]()
        for value in units:
            narrow.append(Int32(value))
        series = Series("x", Column[Int32](narrow^, valid.copy())).with_dtype(
            selection_dtype(kind)
        )
    elif kind == 15:
        var narrow = List[Int64]()
        for value in units:
            narrow.append(Int64(value))
        series = Series("x", Column[Int64](narrow^, valid.copy())).with_dtype(
            selection_dtype(kind)
        )
    else:
        series = Series("x", Column[Int128](units^, valid.copy())).with_dtype(
            selection_dtype(kind)
        )
    return DataFrame([series^])


def test_decimal_rolling_mean_and_moments_use_float_results_and_independent_window_membership() raises:
    var input = WindowCells(
        [-1.25, 2.5, 99, 4.5, 0, -2.75, 99, 1.25],
        [True, True, False, True, True, True, False, True],
    )
    for kind in [14, 15, 16]:
        var frame = decimal_quarter_frame(kind)
        for size in [1, 3, 8]:
            for minimum in [0, 1, size]:
                for ddof in [0, 1, 3]:
                    var actual = frame.select_exprs(
                        [
                            col("x").rolling_mean(size, minimum).alias("mean"),
                            col("x")
                            .rolling_var(size, minimum, ddof)
                            .alias("var"),
                            col("x")
                            .rolling_std(size, minimum, ddof)
                            .alias("std"),
                        ]
                    )
                    for name in ["mean", "var", "std"]:
                        var expected = rolling(input, size, minimum, name, ddof)
                        assert_true(
                            actual.column(name).dtype() == DataType.FLOAT64
                        )
                        for row in range(8):
                            var cell = actual.item(row, name)
                            assert_equal(
                                cell.is_null(), not expected.valid[row]
                            )
                            if expected.valid[row]:
                                var observed = cell.float64()
                                var target = expected.values[row]
                                # Compare variance units for std near zero.
                                if name == "std":
                                    assert_true(observed >= 0)
                                    observed *= observed
                                    target *= target
                                assert_true(
                                    abs(observed - target)
                                    <= 1e-12 + 1e-12 * abs(target)
                                )


def test_extended_windows_reject_nested_and_non_numeric_operations_even_when_empty() raises:
    for n in [0, 9]:
        for kind in [26, 27]:
            var frame = extended_frame(kind, n, 7, 1)
            var expressions: List[Expr] = [
                col("x").shift(),
                col("x").forward_fill(),
                col("x").backward_fill(),
                col("x").cum_count(),
                col("x").cum_min(),
                col("x").cum_max(),
                col("x").rank(),
                col("x").rolling_min(3, 1),
                col("x").rolling_max(3, 1),
                col("x").interpolate("nearest"),
            ]
            for expression in expressions:
                with assert_raises():
                    _ = frame.select(expression)
                with assert_raises():
                    _ = frame.lazy().select(expression).collect_schema()
        for kind in [10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 28]:
            var frame = extended_frame(kind, n, 7, 1)
            var expressions: List[Expr] = [
                col("x").cum_sum(),
                col("x").rolling_sum(3, 1),
                col("x").rolling_mean(3, 1),
                col("x").rolling_var(3, 1),
                col("x").rolling_std(3, 1),
                col("x").interpolate("nearest"),
            ]
            for expression in expressions:
                with assert_raises():
                    _ = frame.select(expression)
                with assert_raises():
                    _ = frame.lazy().select(expression).collect_schema()
        for kind in [14, 15, 16]:
            var message = String()
            try:
                _ = extended_frame(kind, n, 7, 1).select(
                    col("x").rolling_sum(3, 1)
                )
            except error:
                message = String(error)
            assert_true(
                "rolling_sum over a decimal column is not supported yet"
                in message
            )


def test_decimal_pct_change_obeys_documented_float64_result_contract() raises:
    var decimal_results = 0
    for kind in [14, 15, 16]:
        var actual = decimal_quarter_frame(kind).select(
            col("x").pct_change().alias("out")
        )
        if actual.column("out").dtype() == DataType.decimal(38, 2):
            decimal_results += 1
            continue
        assert_true(actual.column("out").dtype() == DataType.FLOAT64)
        assert_equal(actual.item(1, "out").float64(), Float64(-3))
    assert_equal(decimal_results, 0, "decimal pct_change must return Float64")
