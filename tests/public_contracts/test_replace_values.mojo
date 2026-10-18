"""replace and replace_strict recode values as Polars does (#789)."""
# contract-group: expr.null_logic
# contract-group: execution.backends
from std.collections import Dict
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    null,
    when,
)
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _layouts(frame: DataFrame) raises -> List[DataFrame]:
    return [frame.copy(), offset_layout(frame), chunk_layout(frame)]


def _shown(series: Series) raises -> List[String]:
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def _eager_and_lazy(frame: DataFrame, expr: Expr) raises -> List[Series]:
    """The expression's column from with_columns, eager and lazy, which
    must agree."""
    var eager = frame.with_columns(expr.alias("out")).column("out")
    var lazy = (
        frame.lazy().with_columns(expr.alias("out")).collect().column("out")
    )
    assert_equal(_shown(eager), _shown(lazy))
    assert_true(eager.dtype() == lazy.dtype())
    return [eager^, lazy^]


def _ints() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "a",
                Column[Int64]([1, 2, 0, 3, 2], [True, True, False, True, True]),
            ),
            Series("b", Column[Int64]([10, 20, 30, 40, 50])),
        ]
    )


def test_replace_keeps_unmatched_values_and_the_dtype() raises:
    for frame in _layouts(_ints()):
        var exprs: List[Expr] = [
            col("a").replace(2, 100),
            col("a").replace([2, 3], [100, 200]),
            col("a").replace([2, 3], 0),
            col("a").replace([7], [8]),
            col("a").replace(List[Expr](), List[Expr]()),
            # A null in old matches null input: a typed fill_null.
            col("a").replace([null("int64")], [lit(Int64(0))]),
            col("a").replace([null("int64"), lit(Int64(2))], [0, 9]),
            col("a").replace([2, 3], [null("int64"), lit(Int64(7))]),
        ]
        var expected: List[List[String]] = [
            ["1", "100", "null", "3", "100"],
            ["1", "100", "null", "200", "100"],
            ["1", "0", "null", "0", "0"],
            ["1", "2", "null", "3", "2"],
            ["1", "2", "null", "3", "2"],
            ["1", "2", "0", "3", "2"],
            ["1", "9", "0", "3", "9"],
            ["1", "null", "null", "7", "null"],
        ]
        for at in range(len(exprs)):
            for result in _eager_and_lazy(frame, exprs[at]):
                assert_true(result.dtype() == DataType.INT64)
                assert_equal(_shown(result), expected[at])
        var mapping = Dict[Int, Int]()
        mapping[2] = 100
        mapping[1] = -1
        for result in _eager_and_lazy(frame, col("a").replace(mapping)):
            assert_equal(_shown(result), ["-1", "100", "null", "3", "100"])


def test_replace_casts_values_to_the_input_dtype() raises:
    var frame = DataFrame([Series("a", Column[Int8]([1, 2, 3]))])
    var result = frame.with_columns(col("a").replace([2], [100]).alias("out"))
    assert_true(result.column("out").dtype() == DataType.INT8)
    assert_equal(_shown(result.column("out")), ["1", "100", "3"])
    # Values that do not fit the input dtype raise, as Polars' strict cast.
    with assert_raises(contains="old values cannot be cast to int8"):
        _ = frame.with_columns(col("a").replace([1000], [1]).alias("out"))
    with assert_raises(contains="new values cannot be cast to int8"):
        _ = frame.with_columns(col("a").replace([1], [1000]).alias("out"))
    with assert_raises(contains="must not contain duplicates"):
        _ = frame.with_columns(col("a").replace([1, 1], [5, 6]).alias("out"))
    with assert_raises(contains="same length as old or length 1"):
        _ = frame.with_columns(col("a").replace([1, 2], [5, 6, 7]).alias("out"))
    with assert_raises(contains="must be literals"):
        _ = frame.with_columns(col("a").replace([col("a")], [1]).alias("out"))


def test_replace_strings_floats_bools_and_dates() raises:
    var states = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["CA", "NY", "", "TX", "WA"],
                    [True, True, False, True, True],
                ),
            )
        ]
    )
    for frame in _layouts(states):
        for result in _eager_and_lazy(
            frame,
            col("s").replace(["CA", "NY"], ["California", "New York"]),
        ):
            assert_true(result.dtype() == DataType.STRING)
            assert_equal(
                _shown(result), ["California", "New York", "null", "TX", "WA"]
            )
        var fixes = Dict[String, String]()
        fixes["TX"] = "Texas"
        for result in _eager_and_lazy(frame, col("s").replace(fixes)):
            assert_equal(_shown(result), ["CA", "NY", "null", "Texas", "WA"])
        for result in _eager_and_lazy(
            frame, col("s").replace([null("string")], [lit("?")])
        ):
            assert_equal(_shown(result), ["CA", "NY", "?", "TX", "WA"])
    # Every NaN matches NaN and -0.0 matches 0.0, as join keys do.
    var nan = Float64(0) / Float64(0)
    var floats = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [1.0, nan, -0.0, 0.0, 0.0, 2.5],
                    [True, True, True, True, False, True],
                ),
            )
        ]
    )
    for frame in _layouts(floats):
        var swapped = _eager_and_lazy(
            frame, col("f").replace([nan, 0.0], [9.0, 7.0])
        )
        for result in swapped:
            assert_equal(
                _shown(result), ["1.0", "9.0", "7.0", "7.0", "null", "2.5"]
            )
        var narrow = frame.with_columns(
            col("f").cast(DataType.FLOAT32).replace([2.5], [3.5]).alias("out")
        )
        assert_true(narrow.column("out").dtype() == DataType.FLOAT32)
        assert_equal(narrow.item(5, "out").float32(), Float32(3.5))
        assert_true(isnan(narrow.item(1, "out").float32()))
    var bools = DataFrame(
        [Series("b", Column[Bool]([True, False, True], [True, True, False]))]
    )
    for frame in _layouts(bools):
        for result in _eager_and_lazy(frame, col("b").replace([True], [False])):
            assert_true(result.dtype() == DataType.BOOL)
            assert_equal(_shown(result), ["false", "false", "null"])
        for result in _eager_and_lazy(
            frame, col("b").replace_strict([True, False], ["yes", "no"])
        ):
            assert_equal(_shown(result), ["yes", "no", "null"])
    var dates = DataFrame(
        [
            Series(
                "d", Column[Int64]([19723, 19724, 0], [True, True, False])
            ).with_dtype(DataType.DATE)
        ]
    )
    for frame in _layouts(dates):
        # Old and new values given as text are cast to Date.
        for result in _eager_and_lazy(
            frame, col("d").replace(["2024-01-01"], ["2000-01-01"])
        ):
            assert_true(result.dtype() == DataType.DATE)
            assert_equal(_shown(result), ["2000-01-01", "2024-01-02", "null"])
        for result in _eager_and_lazy(
            frame,
            col("d").replace_strict(
                ["2024-01-01"], ["new year"], default="other"
            ),
        ):
            assert_equal(_shown(result), ["new year", "other", "other"])


def test_replace_strict_maps_every_value_and_may_change_dtype() raises:
    for frame in _layouts(_ints()):
        var full = _eager_and_lazy(
            frame, col("a").replace_strict([1, 2, 3], ["x", "y", "z"])
        )
        for result in full:
            assert_true(result.dtype() == DataType.STRING)
            assert_equal(_shown(result), ["x", "y", "null", "z", "y"])
        # With a default, unmatched values and null input take it.
        for result in _eager_and_lazy(
            frame, col("a").replace_strict([1, 2], [10, 20], default=-1)
        ):
            assert_true(result.dtype() == DataType.INT64)
            assert_equal(_shown(result), ["10", "20", "-1", "-1", "20"])
        for result in _eager_and_lazy(
            frame, col("a").replace_strict([1, 2], [10, 20], default=1.5)
        ):
            assert_true(result.dtype() == DataType.FLOAT64)
            assert_equal(_shown(result), ["10.0", "20.0", "1.5", "1.5", "20.0"])
        for result in _eager_and_lazy(
            frame,
            col("a").replace_strict([1, 2], ["x", "y"], default="o"),
        ):
            assert_equal(_shown(result), ["x", "y", "o", "o", "y"])
        # The default may be any expression; a row-valued one per row.
        for result in _eager_and_lazy(
            frame, col("a").replace_strict([1], [100], default=col("b"))
        ):
            assert_equal(_shown(result), ["100", "20", "30", "40", "50"])
        for result in _eager_and_lazy(
            frame, col("a").replace_strict([1], [100], default=col("b").max())
        ):
            assert_equal(_shown(result), ["100", "50", "50", "50", "50"])
        for result in _eager_and_lazy(
            frame,
            col("a").replace_strict(
                [1, 2, 3], [10, 20, 30], return_dtype=DataType.INT8
            ),
        ):
            assert_true(result.dtype() == DataType.INT8)
            assert_equal(_shown(result), ["10", "20", "null", "30", "20"])
        for result in _eager_and_lazy(
            frame,
            col("a").replace_strict(
                [null("int64"), lit(Int64(1)), lit(Int64(2)), lit(Int64(3))],
                [0, 10, 20, 30],
            ),
        ):
            assert_equal(_shown(result), ["10", "20", "0", "30", "20"])
        var mapping = Dict[Int, String]()
        mapping[1] = "one"
        mapping[2] = "two"
        for result in _eager_and_lazy(
            frame, col("a").replace_strict(mapping, default="many")
        ):
            assert_equal(_shown(result), ["one", "two", "many", "many", "two"])
        with assert_raises(contains="incomplete mapping"):
            _ = frame.with_columns(
                col("a").replace_strict([1, 2], [10, 20]).alias("out")
            )
        with assert_raises(contains="incomplete mapping"):
            _ = (
                frame.lazy()
                .with_columns(
                    col("a").replace_strict([1, 2], [10, 20]).alias("out")
                )
                .collect()
            )
        with assert_raises(contains="cannot be cast to int8"):
            _ = frame.with_columns(
                col("a")
                .replace_strict(
                    [1, 2, 3], [10, 20, 300], return_dtype=DataType.INT8
                )
                .alias("out")
            )
        with assert_raises(contains="cannot use values of type string"):
            _ = frame.with_columns(
                col("a").replace_strict(["1"], [5], default=0).alias("out")
            )
    var codes = Dict[String, Int]()
    codes["CA"] = 1
    codes["NY"] = 2
    var states = DataFrame([Series("s", Column[String](["NY", "CA", "NY"]))])
    var mapped = states.with_columns(
        col("s").replace_strict(codes).alias("out")
    )
    assert_true(mapped.column("out").dtype() == DataType.INT64)
    assert_equal(_shown(mapped.column("out")), ["2", "1", "2"])


def test_categorical_replace_recodes_the_dictionary() raises:
    # Enough rows that the expression runs once per dictionary value.
    var abc: List[String] = ["a", "b", "c"]
    var values = List[String]()
    var valid = List[Bool]()
    for i in range(64):
        values.append(abc[i % 3])
        valid.append(i % 7 != 0)
    var frame = DataFrame(
        [Series("c", Column[String](values^, valid)).cast("categorical")]
    )
    for layout in _layouts(frame):
        for result in _eager_and_lazy(
            layout, col("c").replace(["a", "c"], ["z", "b"])
        ):
            assert_true(result.dtype().is_categorical())
            assert_equal(len(result.dtype().dictionary()[]), 2)
            var recoded: List[String] = ["z", "b", "b"]
            for i in range(64):
                var expected = "null" if i % 7 == 0 else recoded[i % 3]
                assert_equal(String(result.get(i)), expected)
        for result in _eager_and_lazy(
            layout, col("c").replace_strict(["a", "b", "c"], [1, 2, 3])
        ):
            assert_true(result.dtype() == DataType.INT64)
            for i in range(64):
                var expected = "null" if i % 7 == 0 else String(i % 3 + 1)
                assert_equal(String(result.get(i)), expected)
        # Values of the dictionary that no row holds still need no mapping.
        var kept = layout.filter(col("c") != "c")
        var strict = kept.with_columns(
            col("c").replace_strict(["a", "b"], ["x", "y"]).alias("out")
        )
        assert_true(strict.column("out").dtype() == DataType.STRING)
        with assert_raises(contains="incomplete mapping"):
            _ = layout.with_columns(
                col("c").replace_strict(["a", "b"], ["x", "y"]).alias("out")
            )
    # Few rows: the strings are recoded row by row, then encoded.
    var small = DataFrame(
        [Series("c", Column[String](["a", "b"])).cast("categorical")]
    )
    var recoded = small.with_columns(col("c").replace("a", "b").alias("out"))
    assert_true(recoded.column("out").dtype().is_categorical())
    assert_equal(_shown(recoded.column("out")), ["b", "b"])
    # Inside a larger expression the categorical is read as its strings.
    var lengths = small.with_columns(
        col("c").replace("a", "zz").str().len_chars().alias("out")
    )
    assert_equal(_shown(lengths.column("out")), ["2", "1"])


def test_replace_in_group_by_agg_and_large_inputs() raises:
    # Two streamed batches of 16K rows and three with_columns batches.
    var n = 20_000
    var keys = List[Int64](capacity=n)
    var codes = List[Int64](capacity=n)
    var labels = List[String](capacity=n)
    var xyz: List[String] = ["x", "y", "z"]
    for i in range(n):
        keys.append(Int64(i % 4))
        codes.append(Int64(i % 10))
        labels.append(xyz[i % 3])
    var frame = DataFrame(
        [
            Series("g", Column[Int64](keys^)),
            Series("k", Column[Int64](codes^)),
            Series("s", Column[String](labels^)),
        ]
    )
    var old = List[Expr]()
    var new = List[Expr]()
    for value in range(10):
        old.append(value)
        new.append(value * 100)
    var expected_total = Int64(0)
    for i in range(n):
        expected_total += Int64(i % 10) * 100
    var recoded = col("k").replace_strict(old, new)
    # Streamed and materialized reductions, eager and lazy.
    var total = frame.select(recoded.sum().alias("t"))
    assert_equal(total.item(0, "t").int64(), expected_total)
    var lazy_total = frame.lazy().select(recoded.sum().alias("t")).collect()
    assert_equal(lazy_total.item(0, "t").int64(), expected_total)
    var rows = frame.with_columns(recoded.alias("r"))
    for i in range(0, n, 997):
        assert_equal(rows.item(i, "r").int64(), Int64(i % 10) * 100)
    var grouped = (
        frame.group_by("g", maintain_order=True)
        .agg(
            [
                recoded.sum().alias("t"),
                col("s")
                .replace_strict(["x", "y", "z"], [1, 2, 3])
                .max()
                .alias("m"),
            ]
        )
        .sort("g")
    )
    var lazy_grouped = (
        frame.lazy()
        .group_by("g")
        .agg(
            [
                recoded.sum().alias("t"),
                col("s")
                .replace_strict(["x", "y", "z"], [1, 2, 3])
                .max()
                .alias("m"),
            ]
        )
        .sort("g")
        .collect()
    )
    assert_true(grouped.equals(lazy_grouped))
    for g in range(4):
        var sum = Int64(0)
        for i in range(g, n, 4):
            sum += Int64(i % 10) * 100
        assert_equal(grouped.item(g, "t").int64(), sum)
        assert_equal(grouped.item(g, "m").int64(), Int64(3))
    # Within over(), and under a when branch.
    var windowed = frame.with_columns(
        col("k").replace([0], [1000]).max().over("g").alias("w")
    )
    assert_equal(windowed.item(0, "w").int64(), Int64(1000))
    var branch = frame.with_columns(
        when(col("g") == 0)
        .then(col("s").replace("x", "X"))
        .otherwise(lit("-"))
        .alias("b")
    )
    assert_equal(_shown(branch.column("b").head(4)), ["X", "-", "-", "-"])


def test_replace_falls_back_to_the_cpu_on_an_accelerator() raises:
    var frame = _ints()
    var query = frame.lazy().select(
        col("a").replace_strict([1, 2, 3], [10, 20, 30]).alias("out")
    )
    var runtime = MetalRuntime()
    assert_true("unsupported" in query.explain(accelerator=runtime))
    var cpu = query.collect(engine="cpu")
    var auto = query.collect(engine="auto", accelerator=runtime)
    assert_true(cpu.equals(auto))
    assert_equal(_shown(auto.column("out")), ["10", "20", "null", "30", "20"])
