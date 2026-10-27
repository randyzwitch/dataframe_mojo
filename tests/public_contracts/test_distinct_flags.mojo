"""Distinct predicates use value equality across batches and partitions."""
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StructColumn,
    ListColumn,
    col,
)
from dataframe.bool_column import BoolColumn
from dataframe.dtype import CategoricalDictionary
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    fixture_codes,
    reduction_validity,
)
from tests.support.group_strategy_reference import ScopedEnvironment


def _expressions() -> List[Expr]:
    return [
        col("x").is_duplicated().alias("duplicated"),
        col("x").is_unique().alias("unique"),
        col("x").is_first_distinct().alias("first"),
        col("x").is_last_distinct().alias("last"),
    ]


def _check(
    actual: DataFrame, codes: List[Int64], groups: List[Int64] = List[Int64]()
) raises:
    assert_equal(actual.height(), len(codes))
    for k in range(4):
        var flags = actual.column(
            [String("duplicated"), "unique", "first", "last"][k]
        ).rechunk()
        assert_equal(flags.dtype(), DataType.BOOL)
        assert_equal(flags.null_count(), 0)
        for row in range(len(codes)):
            var count = 0
            var first = row
            var last = row
            for other in range(len(codes)):
                if codes[other] == codes[row] and (
                    len(groups) == 0 or groups[other] == groups[row]
                ):
                    count += 1
                    first = min(first, other)
                    last = max(last, other)
            var expected = (
                count
                > 1 if k
                == 0 else count
                == 1 if k
                == 1 else row
                == first if k
                == 2 else row
                == last
            )
            assert_equal(flags._data[BoolColumn]._get(row), expected)


def _all_layouts(source: Series, codes: List[Int64]) raises:
    var raw = DataFrame([source.copy()])
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        _check(frame.select_exprs(_expressions(), batch_size=2), codes)
        for optimize in [False, True]:
            _check(
                frame.lazy()
                .select_exprs(_expressions())
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=True,
                    batch_size=2,
                ),
                codes,
            )
    _check(raw.clear().select_exprs(_expressions()), List[Int64]())


def test_distinct_flags_cover_scalar_types_null_patterns_and_layouts() raises:
    for mode in range(3):
        var valid = reduction_validity(mode)
        for source in typed_reduction_columns(valid):
            var codes = fixture_codes(source.dtype())
            for row in range(len(codes)):
                if not valid[row]:
                    codes[row] = -1
            _all_layouts(source, codes)
    for n in [0, 1, 2, 7, 8, 9, 17, 65]:
        var values = List[Int64]()
        var valid = List[Bool]()
        var codes = List[Int64]()
        for row in range(n):
            values.append(Int64(row % 5))
            valid.append(row % 7 != 0)
            codes.append(Int64(row % 5 if valid[row] else -1))
        _all_layouts(Series("x", Column[Int64](values^, valid)), codes)
        _all_layouts(
            Series.full_null("x", DataType.NULL, n),
            List[Int64](length=n, fill=-1),
        )


def test_distinct_flags_normalize_nan_signed_zero_and_dictionary_labels() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    _all_layouts(
        Series(
            "x",
            Column[Float64](
                [nan, -0.0, 0.0, inf, -inf, nan, 7.0, 8.0],
                [True, True, True, True, True, True, False, True],
            ),
        ),
        [1, 2, 2, 3, 4, 1, -1, 5],
    )
    _all_layouts(
        Series(
            "x",
            Column[Float32](
                [
                    Float32(nan),
                    -0.0,
                    0.0,
                    Float32(inf),
                    Float32(-inf),
                    Float32(nan),
                    7.0,
                    8.0,
                ],
                [True, True, True, True, True, True, False, True],
            ),
        ),
        [1, 2, 2, 3, 4, 1, -1, 5],
    )
    var dictionary = CategoricalDictionary()
    for label in [String("same"), "different", "same", "unique"]:
        dictionary.append(label)
    var categorical = Series(
        "x",
        Column[UInt32](
            [0, 2, 1, 3, 0, 99], [True, True, True, True, True, False]
        ),
    ).with_dtype(DataType.categorical(dictionary^))
    _all_layouts(categorical, [1, 1, 2, 3, 1, -1])
    var wide = Int128(1) << 80
    _all_layouts(
        Series(
            "x", Column[Int128]([1, wide + 1, 1, wide + 1, wide + 2])
        ).with_dtype(DataType.decimal(30, 0)),
        [1, 2, 1, 2, 3],
    )


def test_distinct_flags_restart_per_window_partition_and_remain_filterable() raises:
    var codes: List[Int64] = [1, 1, 2, 1, -1, -1, 3, -1, 1]
    var groups: List[Int64] = [0, 1, 0, 0, 0, 1, 1, 0, 1]
    var raw = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [1, 1, 2, 1, 99, 99, 3, 99, 1],
                    [True, True, True, True, False, False, True, False, True],
                ),
            ),
            Series("g", Column[Int64](groups.copy())),
        ]
    )
    var windows = List[Expr]()
    for expression in _expressions():
        windows.append(expression.over("g"))
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        _check(frame.select_exprs(windows, batch_size=2), codes, groups)
        _check(
            frame.lazy()
            .select_exprs(windows)
            .collect(engine="cpu", batch_size=2),
            codes,
            groups,
        )
    var selected = (
        raw.lazy()
        .filter(col("x").is_first_distinct())
        .collect(engine="cpu", batch_size=2)
    )
    assert_equal(selected.height(), 4)
    assert_equal(selected.column("g").int64()._get(0), 0)
    var cumulative = raw.select(
        col("x").is_first_distinct().cum_sum().alias("count")
    )
    assert_equal(cumulative.column("count")._data[Column[UInt32]]._get(8), 4)


def test_distinct_flags_keep_null_structs_separate_from_valid_null_fields() raises:
    var fields: List[Series] = [
        Series(
            "field",
            Column[Int64](
                [1, 2, 0, 0, 1, 0], [True, True, False, False, True, False]
            ),
        )
    ]
    var source = Series("x", StructColumn(fields^, [UInt8(0b111100)], length=6))
    # First two parents are null, with deliberately unequal hidden fields.
    _all_layouts(source, [-1, -1, 0, 0, 1, 0])
    var unsupported = DataFrame(
        [
            Series(
                "x",
                ListColumn([Int64(0), 1], Series("item", Column[Int64]([1]))),
            )
        ]
    )
    with assert_raises(contains="does not support list"):
        _ = unsupported.lazy().select(col("x").is_unique()).collect_schema()


def test_distinct_flags_high_cardinality_parallel_encoding_keeps_first_last_rows() raises:
    var values = List[Int64]()
    var first = List[Bool]()
    var last = List[Bool]()
    var n = 200000
    for row in range(n):
        values.append(Int64(row % 100002))
        first.append(row < 100002)
        last.append(row + 100002 >= n)
    var frame = DataFrame([Series("x", Column[Int64](values^))])
    for workers in [String("1"), "4"]:
        var scoped = ScopedEnvironment("DATAFRAME_THREADS", workers)
        var result = frame.select_exprs(_expressions(), batch_size=17)
        for row in range(n):
            var unique = row >= n - 100002 and row < 100002
            assert_equal(
                result.column("first")._data[BoolColumn]._get(row), first[row]
            )
            assert_equal(
                result.column("last")._data[BoolColumn]._get(row), last[row]
            )
            assert_equal(
                result.column("unique")._data[BoolColumn]._get(row), unique
            )
            assert_equal(
                result.column("duplicated")._data[BoolColumn]._get(row),
                not unique,
            )
        _ = scoped^
