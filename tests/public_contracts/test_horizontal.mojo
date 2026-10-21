"""Row-wise reductions across columns follow Polars' null and type rules."""
# contract-group: expr.horizontal
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    all,
    all_horizontal,
    any_horizontal,
    by_dtype,
    col,
    exclude,
    lit,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    null,
    sum_horizontal,
)
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import chunk_layout


def _value(column: Int, row: Int) -> Int64:
    """A small signed value, different per column and row."""
    return Int64((row * 7 + column * 13) % 23 - 11)


def _mixed_frame(width: Int) raises -> DataFrame:
    """`width` columns alternating Int64 and Float64; row r enumerates the
    null patterns (bit c of r nulls column c), so every input is null in
    every position and every combination occurs."""
    var rows = 1 << width
    var columns = List[Series]()
    for c in range(width):
        var valid = List[Bool]()
        for r in range(rows):
            valid.append((r >> c) & 1 == 0)
        var name = "c" + String(c)
        if c % 2 == 0:
            var values = List[Int64]()
            for r in range(rows):
                values.append(_value(c, r))
            columns.append(Series(name, Column[Int64](values^, valid)))
        else:
            var values = List[Float64]()
            for r in range(rows):
                values.append(Float64(_value(c, r)) + 0.5)
            columns.append(Series(name, Column[Float64](values^, valid)))
    return DataFrame(columns^)


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _check_reference(result: DataFrame, width: Int) raises:
    """Every result row against a model computed from the null pattern."""
    for r in range(1 << width):
        var total = Float64(0)
        var count = 0
        var least = Float64(0)
        var most = Float64(0)
        for c in range(width):
            if (r >> c) & 1 == 1:
                continue
            var v = Float64(_value(c, r)) + (0.5 if c % 2 == 1 else 0.0)
            least = v if count == 0 or v < least else least
            most = v if count == 0 or v > most else most
            total += v
            count += 1
        var complete = count == width
        assert_true(_close(result.item(r, "sum").float64(), total))
        assert_equal(result.item(r, "sum_strict").is_null(), not complete)
        if complete:
            assert_true(_close(result.item(r, "sum_strict").float64(), total))
        assert_equal(result.item(r, "mean").is_null(), count == 0)
        if count > 0:
            assert_true(
                _close(result.item(r, "mean").float64(), total / Float64(count))
            )
        assert_equal(result.item(r, "mean_strict").is_null(), not complete)
        assert_equal(result.item(r, "min").is_null(), count == 0)
        assert_equal(result.item(r, "max").is_null(), count == 0)
        if count > 0:
            assert_equal(result.item(r, "min").float64(), least)
            assert_equal(result.item(r, "max").float64(), most)


def _reductions(inputs: List[Expr]) raises -> List[Expr]:
    return [
        sum_horizontal(inputs).alias("sum"),
        sum_horizontal(inputs, ignore_nulls=False).alias("sum_strict"),
        mean_horizontal(inputs).alias("mean"),
        mean_horizontal(inputs, ignore_nulls=False).alias("mean_strict"),
        min_horizontal(inputs).alias("min"),
        max_horizontal(inputs).alias("max"),
    ]


def test_numeric_reductions_match_a_model_for_every_null_pattern() raises:
    for width in [2, 3, 10]:
        var frame = _mixed_frame(width)
        var inputs = List[Expr]()
        for c in range(width):
            inputs.append(col("c" + String(c)))
        var exprs = _reductions(inputs)
        for layout in range(2):
            var source = frame.copy() if layout == 0 else chunk_layout(frame)
            var eager = source.select_exprs(exprs)
            for name in ["sum", "sum_strict", "mean", "min", "max"]:
                assert_equal(eager.column(name).dtype(), DataType.FLOAT64)
            _check_reference(eager, width)
            _check_reference(source.lazy().select_exprs(exprs).collect(), width)
        # Selectors among the inputs expand into more inputs: all() is the
        # same list of columns.
        _check_reference(frame.select_exprs(_reductions([all()])), width)


def test_integer_inputs_keep_integer_sums_and_skip_nulls() raises:
    var frame = DataFrame(
        [
            Series(
                "a", Column[Int64]([1, 0, 3, 0], [True, False, True, False])
            ),
            Series(
                "b", Column[Int32]([10, 20, 0, 0], [True, True, False, False])
            ),
        ]
    )
    var out = frame.select_exprs(
        [
            sum_horizontal([col("a"), col("b")]),
            sum_horizontal([col("a"), col("b")], ignore_nulls=False).alias(
                "strict"
            ),
            min_horizontal([col("a"), col("b")]).alias("min"),
            max_horizontal([col("a"), col("b")]).alias("max"),
            mean_horizontal([col("a"), col("b")]).alias("mean"),
            sum_horizontal([col("b")]).alias("one"),
        ]
    )
    # Named after the first input, typed as the supertype (Polars).
    assert_equal(out.columns(), ["a", "strict", "min", "max", "mean", "one"])
    assert_equal(out.column("a").dtype(), DataType.INT64)
    assert_equal(out.column("one").dtype(), DataType.INT32)
    assert_equal(out.column("mean").dtype(), DataType.FLOAT64)
    var sums: List[Int64] = [11, 20, 3, 0]
    var mins: List[Int64] = [1, 20, 3, -1]
    var maxs: List[Int64] = [10, 20, 3, -1]
    var means: List[Float64] = [5.5, 20, 3, -1]
    var ones: List[Int64] = [10, 20, 0, 0]
    for r in range(4):
        assert_equal(out.item(r, "a").int64(), sums[r])
        assert_equal(out.item(r, "strict").is_null(), r != 0)
        assert_equal(out.item(r, "one").int32(), Int32(ones[r]))
        assert_equal(out.item(r, "min").is_null(), r == 3)
        assert_equal(out.item(r, "mean").is_null(), r == 3)
        if r < 3:
            assert_equal(out.item(r, "min").int64(), mins[r])
            assert_equal(out.item(r, "max").int64(), maxs[r])
            assert_equal(out.item(r, "mean").float64(), means[r])
    assert_equal(out.item(0, "strict").int64(), 11)
    # Integer sums keep `+`'s overflow check.
    var small = DataFrame([Series("x", Column[Int8]([100]))])
    with assert_raises(contains="overflow"):
        _ = small.select(sum_horizontal([col("x"), col("x")]))


def _dtype_of(frame: DataFrame, expr: Expr) raises -> DataType:
    return frame.select(expr.alias("out")).column("out").dtype()


def test_mixed_types_take_the_polars_supertype() raises:
    var frame = DataFrame(
        [
            Series("i8", Column[Int8]([1])),
            Series("i16", Column[Int16]([2])),
            Series("i32", Column[Int32]([3])),
            Series("i64", Column[Int64]([4])),
            Series("u8", Column[UInt8]([5])),
            Series("u32", Column[UInt32]([6])),
            Series("u64", Column[UInt64]([7])),
            Series("f32", Column[Float32]([1.5])),
            Series("f64", Column[Float64]([2.5])),
            Series("b", Column[Bool]([True])),
            Series("s", StringColumn(["x"])),
        ]
    )
    var cases: List[Tuple[String, String, DataType]] = [
        ("i8", "u8", DataType.INT16),
        ("i8", "i32", DataType.INT32),
        ("u32", "i16", DataType.INT64),
        ("u32", "i32", DataType.INT64),
        ("i64", "u64", DataType.FLOAT64),
        ("i16", "f32", DataType.FLOAT32),
        ("i32", "f32", DataType.FLOAT64),
        ("f32", "f64", DataType.FLOAT64),
        ("b", "i8", DataType.INT8),
        ("b", "f32", DataType.FLOAT32),
    ]
    for entry in cases:
        var pair: List[Expr] = [col(entry[0]), col(entry[1])]
        assert_equal(_dtype_of(frame, sum_horizontal(pair)), entry[2])
        assert_equal(_dtype_of(frame, min_horizontal(pair)), entry[2])
        assert_equal(_dtype_of(frame, max_horizontal(pair)), entry[2])
        assert_equal(
            _dtype_of(frame, mean_horizontal(pair)),
            DataType.FLOAT32 if entry[2]
            == DataType.FLOAT32 else DataType.FLOAT64,
        )
    var values = frame.select_exprs(
        [
            sum_horizontal([col("i8"), col("u8")]).alias("i16"),
            max_horizontal([col("i64"), col("u64")]).alias("f64"),
            sum_horizontal([col("b"), col("b")]).alias("bools"),
            mean_horizontal([col("b"), col("f32")]).alias("f32"),
        ]
    )
    assert_equal(values.item(0, "i16").int16(), 6)
    assert_equal(values.item(0, "f64").float64(), 7)
    assert_equal(values.column("bools").dtype(), DataType.UINT32)
    assert_equal(values.item(0, "bools").uint32(), 2)
    assert_equal(values.item(0, "f32").float32(), 1.25)
    # Untyped numbers adopt the inputs' type, as Polars' dynamic literals.
    assert_equal(
        _dtype_of(frame, sum_horizontal([Expr(1), col("i16")])), DataType.INT16
    )
    assert_equal(
        _dtype_of(frame, sum_horizontal([Expr(1.5), col("f32")])),
        DataType.FLOAT32,
    )
    assert_equal(
        _dtype_of(frame, sum_horizontal([Expr(1.5), col("i16")])),
        DataType.FLOAT64,
    )
    assert_equal(
        _dtype_of(frame, sum_horizontal([Expr(1), Expr(2)])), DataType.INT64
    )
    with assert_raises(contains="sum_horizontal requires inputs with a common"):
        _ = frame.select(sum_horizontal([col("s"), col("i8")]))
    with assert_raises(contains="sum_horizontal requires numeric"):
        _ = frame.select(sum_horizontal([col("s"), col("s")]))
    with assert_raises(contains="mean_horizontal requires numeric"):
        _ = frame.select(mean_horizontal([col("s")]))
    with assert_raises(contains="any_horizontal requires Bool or numeric"):
        _ = frame.select(any_horizontal([col("s")]))
    with assert_raises(contains="requires at least one expression"):
        _ = sum_horizontal(List[Expr]())


def test_decimals_take_duckdb_types() raises:
    var d = DataType.decimal(10, 2, 64)
    var e = DataType.decimal(5, 1, 32)
    var frame = DataFrame(
        [
            Series(
                "d", Column[Int64]([150, 0, 225], [True, False, True])
            ).with_dtype(d),
            Series(
                "e", Column[Int32]([15, 25, 0], [True, True, False])
            ).with_dtype(e),
            Series("i", Column[Int64]([1, 2, 3])),
            Series("f", Column[Float64]([1.25, 0.5, 9])),
        ]
    )
    var out = frame.select_exprs(
        [
            min_horizontal([col("d"), col("e")]).alias("min"),
            max_horizontal([col("i"), col("d")]).alias("max"),
            sum_horizontal([col("d"), col("e")]).alias("sum"),
            sum_horizontal([col("d"), col("e"), col("d")]).alias("sum3"),
            sum_horizontal([col("i"), col("d")]).alias("sum_int"),
            max_horizontal([col("d"), col("f")]).alias("float"),
            mean_horizontal([col("d"), col("e")]).alias("mean"),
        ]
    )
    # DuckDB: greatest(d, e) is DECIMAL(10,2), greatest(bigint, d)
    # DECIMAL(21,2); d + e is DECIMAL(11,2), d + e + d DECIMAL(12,2) and
    # bigint + d DECIMAL(22,2).
    assert_equal(out.column("min").dtype().precision(), 10)
    assert_equal(out.column("min").dtype().scale(), 2)
    assert_equal(out.column("max").dtype().precision(), 21)
    assert_equal(out.column("sum").dtype().precision(), 11)
    assert_equal(out.column("sum3").dtype().precision(), 12)
    assert_equal(out.column("sum_int").dtype().precision(), 22)
    assert_equal(out.column("sum_int").dtype().scale(), 2)
    assert_equal(out.column("float").dtype(), DataType.FLOAT64)
    assert_equal(out.column("mean").dtype(), DataType.FLOAT64)
    var texts: List[String] = ["1.50", "2.50", "2.25"]
    var maxs: List[String] = ["1.50", "2.00", "3.00"]
    var sums: List[String] = ["3.00", "2.50", "2.25"]
    var means: List[Float64] = [1.5, 2.5, 2.25]
    for r in range(3):
        assert_equal(String(out.item(r, "min")), texts[r])
        assert_equal(String(out.item(r, "max")), maxs[r])
        assert_equal(String(out.item(r, "sum")), sums[r])
        assert_equal(out.item(r, "mean").float64(), means[r])
    assert_equal(out.item(1, "float").float64(), 0.5)


def test_any_and_all_follow_kleene_logic() raises:
    # Rows enumerate (false, true, null) for each of three inputs.
    var states: List[Int] = [0, 1, -1]
    var a = List[Bool]()
    var av = List[Bool]()
    var b = List[Bool]()
    var bv = List[Bool]()
    var c = List[Int64]()
    var cv = List[Bool]()
    for x in states:
        for y in states:
            for z in states:
                a.append(x == 1)
                av.append(x >= 0)
                b.append(y == 1)
                bv.append(y >= 0)
                # A number reads as true when nonzero.
                c.append(Int64(5) if z == 1 else Int64(0))
                cv.append(z >= 0)
    var input = DataFrame(
        [
            Series("a", Column[Bool](a, av)),
            Series("b", Column[Bool](b^, bv)),
            Series("c", Column[Int64](c^, cv)),
        ]
    )
    for layout in range(2):
        var frame = input.copy() if layout == 0 else chunk_layout(input)
        var out = frame.select_exprs(
            [
                any_horizontal([col("a"), col("b"), col("c")]).alias("any"),
                all_horizontal([col("a"), col("b"), col("c")]).alias("all"),
                any_horizontal([col("a"), col("b")]).alias("any2"),
                all_horizontal([col("a")]).alias("one"),
            ]
        )
        for row in range(27):
            var x = states[row // 9]
            var y = states[(row // 3) % 3]
            var z = states[row % 3]
            var any = (
                1 if x == 1
                or y == 1
                or z == 1 else (-1 if x < 0 or y < 0 or z < 0 else 0)
            )
            var every = (
                0 if x == 0
                or y == 0
                or z == 0 else (-1 if x < 0 or y < 0 or z < 0 else 1)
            )
            var any2 = 1 if x == 1 or y == 1 else (-1 if x < 0 or y < 0 else 0)
            var expected: List[Int] = [any, every, any2, x]
            for k in range(4):
                var cell = out.item(row, out.columns()[k])
                assert_equal(cell.is_null(), expected[k] < 0)
                if expected[k] >= 0:
                    assert_equal(cell.bool(), expected[k] == 1)
        # As predicates they keep the true rows.
        var anyrows = 0
        var allrows = 0
        for row in range(27):
            anyrows += Int(
                states[row // 9] == 1
                or states[(row // 3) % 3] == 1
                or states[row % 3] == 1
            )
            allrows += Int(
                states[row // 9] == 1
                and states[(row // 3) % 3] == 1
                and states[row % 3] == 1
            )
        var inputs: List[Expr] = [col("a"), col("b"), col("c")]
        assert_equal(frame.filter(any_horizontal(inputs)).height(), anyrows)
        assert_equal(frame.filter(all_horizontal(inputs)).height(), allrows)
        assert_equal(
            frame.lazy().filter(any_horizontal(inputs)).collect().height(),
            anyrows,
        )


def test_min_and_max_skip_nan_and_order_strings_bools_and_dates() raises:
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64]([nan, 1, nan, 0], [True, True, True, False]),
            ),
            Series(
                "y",
                Column[Float64]([3, nan, nan, nan], [True, True, False, True]),
            ),
            Series(
                "s",
                StringColumn(["b", "a", "", "c"], [True, True, False, True]),
            ),
            Series(
                "t",
                StringColumn(["a", "", "z", "d"], [True, False, True, True]),
            ),
            Series(
                "p",
                Column[Bool](
                    [True, False, False, True], [True, False, True, False]
                ),
            ),
            Series(
                "q",
                Column[Bool](
                    [False, False, True, False], [False, True, True, False]
                ),
            ),
            Series(
                "d", Column[Int64]([10, 0, 30, 5], [True, False, True, True])
            ).with_dtype(DataType.DATE),
            Series(
                "e", Column[Int64]([20, 7, 0, 1], [True, True, False, True])
            ).with_dtype(DataType.DATE),
        ]
    )
    var out = frame.select_exprs(
        [
            min_horizontal([col("x"), col("y")]).alias("fmin"),
            max_horizontal([col("x"), col("y")]).alias("fmax"),
            min_horizontal([col("s"), col("t")]).alias("smin"),
            max_horizontal([col("s"), col("t")]).alias("smax"),
            min_horizontal([col("p"), col("q")]).alias("bmin"),
            max_horizontal([col("p"), col("q")]).alias("bmax"),
            min_horizontal([col("d"), col("e")]).alias("dmin"),
            max_horizontal([col("d"), col("e")]).alias("dmax"),
        ]
    )
    # Polars' min_ignore_nan: NaN gives way to any number, and stays only
    # when a row has nothing else.
    assert_equal(out.item(0, "fmin").float64(), 3)
    assert_equal(out.item(0, "fmax").float64(), 3)
    assert_equal(out.item(1, "fmin").float64(), 1)
    assert_equal(out.item(1, "fmax").float64(), 1)
    assert_true(isnan(out.item(2, "fmin").float64()))
    assert_true(isnan(out.item(3, "fmax").float64()))
    var smin: List[String] = ["a", "a", "z", "c"]
    var smax: List[String] = ["b", "a", "z", "d"]
    for r in range(4):
        assert_equal(out.item(r, "smin").string(), smin[r])
        assert_equal(out.item(r, "smax").string(), smax[r])
    assert_equal(out.item(0, "bmin").bool(), True)
    assert_equal(out.item(1, "bmin").bool(), False)
    assert_equal(out.item(2, "bmin").bool(), False)
    assert_equal(out.item(2, "bmax").bool(), True)
    assert_true(out.item(3, "bmax").is_null())
    assert_equal(out.column("dmin").dtype(), DataType.DATE)
    var dmin: List[Int64] = [10, 7, 30, 1]
    var dmax: List[Int64] = [20, 7, 30, 5]
    for r in range(4):
        assert_equal(out.item(r, "dmin").to_physical(), dmin[r])
        assert_equal(out.item(r, "dmax").to_physical(), dmax[r])


def test_selectors_expand_into_inputs_and_name_the_result() raises:
    var frame = DataFrame(
        [
            Series("k", StringColumn(["x", "y"])),
            Series("a", Column[Int64]([1, 2])),
            Series("b", Column[Int64]([10, 20], [True, False])),
            Series("f", Column[Float64]([0.5, 0.25])),
            Series("p", Column[Bool]([True, False])),
        ]
    )
    var out = frame.select_exprs(
        [
            sum_horizontal([by_dtype(["int64"])]),
            sum_horizontal([col("f"), by_dtype(["int64"])]).alias("mixed"),
            max_horizontal([exclude(["k", "p"])]).alias("max"),
            any_horizontal([by_dtype(["bool"]), col("a") > 1]).alias("any"),
        ]
    )
    # A selector first gives the result its first match's name.
    assert_equal(out.columns(), ["a", "mixed", "max", "any"])
    assert_equal(out.item(0, "a").int64(), 11)
    assert_equal(out.item(1, "a").int64(), 2)
    assert_equal(out.item(0, "mixed").float64(), 11.5)
    assert_equal(out.item(0, "max").float64(), 10)
    assert_equal(out.item(1, "max").float64(), 2)
    assert_equal(out.item(1, "any").bool(), True)
    # A selector matching nothing: any is false, all is true, the rest raise.
    var empty = frame.select_exprs(
        [
            any_horizontal([by_dtype(["date"])]),
            all_horizontal([by_dtype(["date"])]),
        ]
    )
    assert_equal(empty.columns(), ["any_horizontal", "all_horizontal"])
    assert_equal(empty.item(0, "any_horizontal").bool(), False)
    assert_equal(empty.item(0, "all_horizontal").bool(), True)
    var some = frame.select(
        sum_horizontal([col("a"), by_dtype(["date"])]).alias("s")
    )
    assert_equal(some.item(1, "s").int64(), 2)
    with assert_raises(contains="matched no columns"):
        _ = frame.select(sum_horizontal([by_dtype(["date"])]))


def test_with_columns_lazy_and_grouped_contexts() raises:
    var frame = DataFrame(
        [
            Series("g", StringColumn(["x", "x", "y"])),
            Series("a", Column[Int64]([1, 2, 3], [True, True, False])),
            Series("b", Column[Int64]([10, 0, 30], [True, False, True])),
        ]
    )
    var total = sum_horizontal([col("a"), col("b")]).alias("t")
    var eager = frame.with_columns([total.copy()])
    var lazy = frame.lazy().with_columns([total.copy()]).collect()
    var sums: List[Int64] = [11, 2, 30]
    for r in range(3):
        assert_equal(eager.item(r, "t").int64(), sums[r])
        assert_equal(lazy.item(r, "t").int64(), sums[r])
    # Inside agg the inputs may be aggregates.
    var agg: List[Expr] = [
        max_horizontal([col("a").sum(), col("b").max()]).alias("m"),
        mean_horizontal([col("a").min(), col("b").min()]).alias("avg"),
    ]
    var grouped = frame.group_by(["g"]).agg(agg).sort("g")
    var lazy_grouped = frame.lazy().group_by(["g"]).agg(agg).collect().sort("g")
    for table in [grouped^, lazy_grouped^]:
        assert_equal(table.item(0, "m").int64(), 10)
        assert_equal(table.item(1, "m").int64(), 30)
        assert_equal(table.item(0, "avg").float64(), 5.5)
        assert_equal(table.item(1, "avg").float64(), 30)
    # One-row inputs broadcast.
    var broadcast = frame.select(min_horizontal([col("b"), lit(Int64(5))]))
    var mins: List[Int64] = [5, 5, 5]
    for r in range(3):
        assert_equal(broadcast.item(r, "b").int64(), mins[r])
    var scalar = frame.select(
        mean_horizontal([col("a").sum(), col("b").sum()]).alias("x")
    )
    assert_equal(scalar.height(), 1)
    assert_equal(scalar.item(0, "x").float64(), 21.5)


def test_accelerators_leave_horizontal_reductions_to_the_cpu() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Float64]([1.5, 2.5], [True, False])),
            Series("b", Column[Float64]([4, 1])),
        ]
    )
    var query = frame.lazy().select(
        max_horizontal([col("a"), col("b")]).alias("m")
    )
    var runtime = MetalRuntime()
    assert_true("unsupported" in query.explain(accelerator=runtime))
    var out = query.collect(accelerator=runtime, engine="auto")
    assert_equal(out.item(0, "m").float64(), 4)
    assert_equal(out.item(1, "m").float64(), 1)
