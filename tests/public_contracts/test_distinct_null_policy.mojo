"""SQL distinct counts and the existing null-inclusive policy share kernels."""
# contract-group: reduce.distinct_nulls
from std.testing import assert_equal
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    fixture_codes,
    reduction_validity,
)
from tests.support.group_strategy_reference import ScopedEnvironment
from tests.public_contracts.test_text_functions import _source


def _expressions() -> List[Expr]:
    return [
        col("x").n_unique().alias("inclusive"),
        col("x").n_unique(ignore_nulls=True).alias("sql"),
        col("x").n_unique(ignore_nulls=False).alias("explicit"),
    ]


def _counts(values: List[Int64], valid: List[Bool]) -> List[Int64]:
    var seen = Dict[Int64, Bool]()
    var null = False
    for i in range(len(values)):
        if valid[i]:
            seen[values[i]] = True
        else:
            null = True
    return [Int64(len(seen) + Int(null)), Int64(len(seen))]


def _expected(counts: List[Int64]) raises -> DataFrame:
    return DataFrame(
        [
            Series("inclusive", Column[Int64]([counts[0]])),
            Series("sql", Column[Int64]([counts[1]])),
            Series("explicit", Column[Int64]([counts[0]])),
        ]
    )


def test_distinct_null_policy_across_types_layouts_empty_and_lazy() raises:
    for mode in range(3):
        var valid = reduction_validity(mode)
        for original in typed_reduction_columns(valid):
            var expected = _expected(
                _counts(fixture_codes(original.dtype()), valid)
            )
            var raw = DataFrame([original.copy()])
            for layout in range(3):
                var frame = raw.copy() if layout == 0 else (
                    offset_layout(raw) if layout == 1 else chunk_layout(raw)
                )
                assert_frame(
                    frame.select_exprs(_expressions(), batch_size=2), expected
                )
                for optimize in [False, True]:
                    for streaming in [False, True]:
                        assert_frame(
                            frame.lazy()
                            .select_exprs(_expressions())
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=2,
                            ),
                            expected,
                        )
                assert_frame(
                    frame.clear().select_exprs(_expressions()),
                    _expected([0, 0]),
                )
                assert_equal(
                    frame.lazy().select_exprs(_expressions()).collect_schema(),
                    [
                        String("inclusive: int64"),
                        String("sql: int64"),
                        String("explicit: int64"),
                    ],
                )


def test_distinct_null_policy_grouped_windows_views_and_masks() raises:
    var values: List[String] = [
        "a long external string é",
        "ignored",
        "b",
        "a long external string é",
        "c",
        "ignored",
        "ignored",
    ]
    var valid: List[Bool] = [True, False, True, True, True, False, False]
    var groups: List[Int64] = [0, 0, 1, 1, 2, 2, 3]
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2, 3])),
            Series("inclusive", Column[Int64]([2, 2, 2, 1])),
            Series("sql", Column[Int64]([1, 2, 1, 0])),
            Series("explicit", Column[Int64]([2, 2, 2, 1])),
        ]
    )
    for layout in range(4):
        var text = _source(values, valid, layout).column("s")
        text._name = "x"
        for categorical in [False, True]:
            var x = text.cast(
                DataType.CATEGORICAL
            ) if categorical else text.copy()
            var frame = DataFrame(
                [Series("g", Column[Int64](groups.copy())), x^]
            )
            assert_frame(
                frame.group_by("g", maintain_order=True).agg(_expressions()),
                expected,
            )
            for streaming in [False, True]:
                assert_frame(
                    frame.lazy()
                    .group_by("g", maintain_order=True)
                    .agg(_expressions())
                    .collect(
                        engine="cpu",
                        streaming=streaming,
                        batch_size=2,
                    ),
                    expected,
                )
            var window = frame.select(
                col("x").n_unique(ignore_nulls=True).over("g").alias("sql")
            )
            assert_frame(
                window,
                DataFrame(
                    [Series("sql", Column[Int64]([1, 1, 2, 2, 1, 1, 0]))]
                ),
            )
    var numeric = DataFrame(
        [
            Series("x", Column[String](["1", "bad", "2", "bad"])),
            Series("keep", Column[Bool]([True, False, True, False])),
        ]
    )
    var masked = when(col("keep")).then(col("x").cast(DataType.INT64)).end()
    assert_frame(
        numeric.select_exprs(
            [
                masked.n_unique().alias("inclusive"),
                masked.n_unique(ignore_nulls=True).alias("sql"),
                masked.n_unique(ignore_nulls=False).alias("explicit"),
            ]
        ),
        _expected([3, 2]),
    )
    var floats = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [
                        0.0,
                        -0.0,
                        Float64("nan"),
                        Float64("nan"),
                        Float64("inf"),
                        -Float64("inf"),
                        2.0,
                    ],
                    [True, True, True, False, True, True, False],
                ),
            )
        ]
    )
    assert_frame(floats.select_exprs(_expressions()), _expected([5, 4]))


def test_distinct_null_policy_partitioned_workers_and_stream_merges() raises:
    var n = 131_073
    var values = List[Int64](capacity=n)
    var groups = List[Int64](capacity=n)
    var valid = List[Bool](capacity=n)
    var per_group = List[Dict[Int64, Bool]](length=8, fill=Dict[Int64, Bool]())
    var nulls = List[Bool](length=8, fill=False)
    for i in range(n):
        var g = i % 8
        var value = Int64(i % 509)
        var live = g != 7 and i % 13 != 0
        values.append(value)
        groups.append(Int64(g))
        valid.append(live)
        if live:
            per_group[g][value] = True
        else:
            nulls[g] = True
    var sql = List[Int64]()
    var inclusive = List[Int64]()
    for g in range(8):
        sql.append(Int64(len(per_group[g])))
        inclusive.append(Int64(len(per_group[g]) + Int(nulls[g])))
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7])),
            Series("inclusive", Column[Int64](inclusive.copy())),
            Series("sql", Column[Int64](sql.copy())),
            Series("explicit", Column[Int64](inclusive.copy())),
        ]
    )
    var input = Series("x", Column[Int64](values.copy(), valid.copy()))
    for workers in [1, 2]:
        var setting = ScopedEnvironment("DATAFRAME_THREADS", String(workers))
        for dtype in [
            DataType.INT64,
            DataType.FLOAT64,
            DataType.STRING,
            DataType.CATEGORICAL,
        ]:
            var x = input.cast(DataType.STRING).cast(
                dtype
            ) if dtype.is_categorical() else input.cast(dtype)
            var raw = DataFrame([Series("g", Column[Int64](groups.copy())), x^])
            for layout in range(2):
                var frame = raw.copy() if layout == 0 else chunk_layout(raw)
                assert_frame(
                    frame.select_exprs(_expressions()),
                    _expected(_counts(values, valid)),
                )
                assert_frame(
                    frame.group_by("g", maintain_order=True).agg(
                        _expressions()
                    ),
                    expected,
                )
                assert_frame(
                    frame.lazy()
                    .group_by("g", maintain_order=True)
                    .agg(_expressions())
                    .collect(
                        engine="cpu",
                        streaming=True,
                        batch_size=127,
                    ),
                    expected,
                )
        _ = setting^
