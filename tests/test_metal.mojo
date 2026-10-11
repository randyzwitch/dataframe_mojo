"""Metal preflight is CPU-only; installed runtimes also execute parity checks."""
from std.memory import bitcast

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, when
from dataframe.dtype import DataType
from dataframe.expr import Expr
from dataframe.metal import MetalRuntime, metal_installed


def test_configuration_and_semantic_rejections_without_device() raises:
    with assert_raises(contains="nonnegative"):
        _ = MetalRuntime(-1)
    var runtime = MetalRuntime()
    var wide = DataFrame([Series("x", Column[Float64]([1, 2]))])
    assert_true(
        "Float64 requires CPU"
        in runtime.describe(wide.lazy().select(col("x") + 1))
    )
    with assert_raises(contains="Float64 requires CPU"):
        _ = wide.lazy().select(col("x") + 1).collect(accelerator=runtime)
    var narrow = DataFrame([Series("x", Column[Float32]([1, 2]))])
    with assert_raises(contains="Float64 accumulation"):
        _ = narrow.lazy().select(col("x").sum()).collect(accelerator=runtime)
    # Auto is conservative until matched end-to-end cost evidence exists.
    var query = narrow.lazy().select(col("x") + 1)
    assert_true(
        query.collect(engine="auto", accelerator=runtime).equals(
            query.collect(engine="cpu")
        )
    )


def test_native_resident_parity_and_profile() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ],
                    [
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
        ]
    ).slice(3, 8)
    var query = (
        frame.lazy()
        .with_columns(((col("x") + 1) * 2).alias("y"))
        .filter((col("y") > 6) & col("b").fill_null(lit(True)))
        .select_exprs([col("y"), (col("x") != 5).alias("flag"), col("b")])
        .head(3)
    )
    var cpu = query.collect(engine="cpu")
    assert_true(query.collect(engine="accel").equals(cpu))
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(cpu))
    assert_equal(measured[1].column("executor").string()._get(0), "metal")
    assert_equal(measured[1].column("synchronizations").int64()._get(0), 1)
    assert_true(measured[1].column("download_bytes").int64()._get(0) > 0)
    var again = query.profile(accelerator=runtime)
    assert_true(again[0].equals(cpu))
    assert_true(again[1].column("pipeline_cache_hit").bool()._get(0))
    assert_equal(
        again[1].column("pipeline_compile_ms").float64()._get(0), Float64(0)
    )
    var denied = MetalRuntime(memory_limit_bytes=0)
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=denied)


def test_native_integer_precision_and_empty_results() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var integers = DataFrame(
        [Series("x", Column[Int64]([9007199254740993, -9007199254740993, 1]))]
    )
    var query = integers.lazy().select((col("x") + lit(Int64(2))).alias("y"))
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var narrow = DataFrame([Series("x", Column[Int32]([2147483647, 1, -1]))])
    var sum = narrow.lazy().select_exprs(
        [
            col("x").sum().alias("s"),
            col("x").count().alias("n"),
            col("x").len().alias("l"),
        ]
    )
    assert_true(
        sum.collect(accelerator=runtime).equals(sum.collect(engine="cpu"))
    )
    with assert_raises(contains="overflow"):
        _ = narrow.lazy().select(col("x") + 1).collect(accelerator=runtime)
    var empty = (
        narrow.head(0)
        .lazy()
        .select_exprs([col("x"), (col("x") > 1).alias("b")])
    )
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var no_rows = (
        narrow.lazy().filter(col("x") < lit(Int32(-10))).select(col("x"))
    )
    assert_true(
        no_rows.collect(accelerator=runtime).equals(
            no_rows.collect(engine="cpu")
        )
    )
    var empty_sum = narrow.head(0).lazy().select(col("x").sum())
    assert_true(
        empty_sum.collect(accelerator=runtime).equals(
            empty_sum.collect(engine="cpu")
        )
    )


def _native_extended_integer[D: DType]() raises:
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]](
                    [
                        Scalar[D](0),
                        Scalar[D](1),
                        Scalar[D](2),
                        Scalar[D](3),
                        Scalar[D](4),
                    ],
                    [True, False, True, True, True],
                ),
            )
        ]
    ).slice(1, 4)
    var query = (
        frame.lazy()
        .filter(col("x") > lit(Scalar[D](2)))
        .select_exprs(
            [
                (col("x") + lit(Scalar[D](1))).alias("y"),
                (col("x") == lit(Scalar[D](3))).alias("flag"),
            ]
        )
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var count = frame.lazy().select_exprs(
        [col("x").count().alias("n"), col("x").len().alias("l")]
    )
    assert_true(
        count.collect(accelerator=runtime).equals(count.collect(engine="cpu"))
    )
    comptime if D != DType.uint64:
        var sum = frame.lazy().select(col("x").sum())
        assert_true(
            sum.collect(accelerator=runtime).equals(sum.collect(engine="cpu"))
        )
    var empty = frame.head(0).lazy().select(col("x"))
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )


def test_native_extended_integer_widths() raises:
    if not metal_installed():
        return
    _native_extended_integer[DType.int8]()
    _native_extended_integer[DType.int16]()
    _native_extended_integer[DType.uint8]()
    _native_extended_integer[DType.uint16]()
    _native_extended_integer[DType.uint32]()
    _native_extended_integer[DType.uint64]()
    var runtime = MetalRuntime()
    var wide = DataFrame(
        [Series("x", Column[UInt64]([UInt64.MAX, UInt64.MAX - 1]))]
    )
    var query = wide.lazy().select_exprs(
        [col("x"), (col("x") == lit(UInt64.MAX)).alias("eq")]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    with assert_raises(contains="overflow"):
        _ = (
            wide.lazy()
            .select(col("x") + lit(UInt64(1)))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="exact wide accumulation"):
        _ = wide.lazy().select(col("x").sum()).collect(accelerator=runtime)


def test_native_mixed_types_preserve_values_and_output_dtypes() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [0.25, 1.5, 2.75, 4, 5.5], [True, True, False, True, True]
                ),
            ),
            Series(
                "u",
                Column[UInt64]([UInt64.MAX, 1, UInt64.MAX - 1, 2, UInt64.MAX]),
            ),
            Series("n", Column[Int8]([-128, -2, 0, 2, 126])),
            Series("b", Column[Bool]([True, False, True, False, True])),
        ]
    )
    var query = (
        frame.lazy()
        .filter(col("u") > lit(UInt64(2)))
        .select_exprs(
            [
                (col("f") + 1).alias("plus"),
                col("u"),
                (col("n") + lit(Int8(1))).alias("n"),
                col("b"),
                (col("u") == lit(UInt64.MAX)).alias("max"),
            ]
        )
    )
    var result = query.profile(accelerator=runtime)
    assert_true(result[0].equals(query.collect(engine="cpu")))
    assert_equal(result[1].column("executor").string()._get(0), "metal")
    assert_equal(result[0].column("u").uint64()._get(0), UInt64.MAX)
    var counts = frame.lazy().select_exprs(
        [
            col("f").count().alias("f"),
            col("u").count().alias("u"),
            col("n").sum().alias("n"),
        ]
    )
    assert_true(
        counts.collect(accelerator=runtime).equals(counts.collect(engine="cpu"))
    )
    var empty = (
        frame.slice(0, 0)
        .lazy()
        .select_exprs([col("f"), col("u"), col("n"), col("b")])
    )
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var overflow = (
        DataFrame(
            [
                Series("f", Column[Float32]([1])),
                Series("n", Column[Int8]([127])),
            ]
        )
        .lazy()
        .select_exprs([col("f"), col("n") + lit(Int8(1))])
    )
    with assert_raises(contains="overflow"):
        _ = overflow.collect(accelerator=runtime)
    # The observable checked boundary depends on each expression dtype,
    # including when the first physical column is Float32.
    var unsafe = (
        frame.lazy()
        .with_columns((col("n") + lit(Int8(1))).alias("n"))
        .filter(col("f") > 1)
    )
    with assert_raises(contains="terminal projections"):
        _ = unsafe.collect(accelerator=runtime)


def test_native_numeric_casts() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [-128.5, -0.5, 1.5, 255.0], [True, True, False, True]
                ),
            ),
            Series("u", Column[UInt64]([UInt64.MAX, 0, 128, 255])),
            Series("b", Column[Bool]([True, False, True, False])),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").cast("int8", strict=False).alias("i8"),
            col("x").cast("uint8", strict=False).alias("u8"),
            col("x").cast("bool").alias("truth"),
            col("u").cast("int64", strict=False).alias("i64"),
            col("u").cast("float32").alias("f32"),
            col("b").cast("uint64").alias("u64"),
        ]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var filtered = (
        frame.lazy()
        .with_columns(col("x").cast("uint8", strict=False).alias("casted"))
        .filter(col("casted") > lit(UInt8(0)))
        .select_exprs([col("casted"), col("u")])
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    with assert_raises(contains="strict cast failed"):
        _ = (
            frame.lazy()
            .select(col("u").cast("int64"))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="Float64 requires CPU"):
        _ = (
            frame.lazy()
            .select(col("u").cast("float64"))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="strict casts require terminal"):
        _ = (
            frame.lazy()
            .with_columns(col("u").cast("int64").alias("i"))
            .filter(col("b"))
            .collect(accelerator=runtime)
        )


def _native_extrema[D: DType]() raises:
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x", Column[Scalar[D]]([1, 0, 3, 2], [True, False, True, True])
            )
        ]
    )
    for count in range(5):
        var query = (
            frame.slice(0, count)
            .lazy()
            .select_exprs(
                [
                    col("x").min().alias("lo"),
                    col("x").max().alias("hi"),
                    col("x").count().alias("n"),
                ]
            )
        )
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
    var filtered = (
        frame.lazy()
        .filter(col("x") > lit(Scalar[D](1)))
        .select_exprs([col("x").min().alias("lo"), col("x").max().alias("hi")])
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    var nulls = DataFrame(
        [Series("x", Column[Scalar[D]]([1, 2], [False, False]))]
    )
    var empty = nulls.lazy().select_exprs(
        [col("x").min().alias("lo"), col("x").max().alias("hi")]
    )
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )


def test_native_extrema_all_supported_types() raises:
    if not metal_installed():
        return
    _native_extrema[DType.int8]()
    _native_extrema[DType.int16]()
    _native_extrema[DType.int32]()
    _native_extrema[DType.int64]()
    _native_extrema[DType.uint8]()
    _native_extrema[DType.uint16]()
    _native_extrema[DType.uint32]()
    _native_extrema[DType.uint64]()
    _native_extrema[DType.float32]()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series("b", Column[Bool]([True, False, True], [True, True, False])),
            Series("u", Column[UInt64]([UInt64.MAX, 0, UInt64.MAX - 1])),
            Series("i", Column[Int8]([-128, 127, 0])),
            Series(
                "f",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(0x7FC00000)), -0.0, 0.0]
                ),
            ),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("b").min().alias("bmin"),
            col("b").max().alias("bmax"),
            col("u").min().alias("umin"),
            col("u").max().alias("umax"),
            col("i").min().alias("imin"),
            col("i").max().alias("imax"),
            col("i").sum().alias("isum"),
            col("f").min().alias("fmin"),
            col("f").max().alias("fmax"),
        ]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


def _native_fixed_logical(dtype: DataType) raises:
    var x = Series(
        "x",
        Column[Int64](
            [-200, -100, 0, 100, 200], [True, False, True, True, False]
        ),
    )
    var y = Series("y", Column[Int64]([200, 100, 1, 100, -200]))
    if dtype.physical() == DataType.INT32:
        x = Series(
            "x",
            Column[Int32](
                [-200, -100, 0, 100, 200], [True, False, True, True, False]
            ),
        )
        y = Series("y", Column[Int32]([200, 100, 1, 100, -200]))
    x = x.with_dtype(dtype)
    y = y.with_dtype(dtype)
    var frame = DataFrame(
        [x^, y^, Series("b", Column[Bool]([True, True, False, True, True]))]
    )
    var runtime = MetalRuntime()
    var projected = (
        frame.slice(1, 4)
        .lazy()
        .select_exprs(
            [
                col("x"),
                col("x").fill_null(col("y")).alias("filled"),
                col("x").is_null().alias("null"),
                (col("x") <= col("y")).alias("cmp"),
            ]
        )
    )
    var result = projected.collect(accelerator=runtime)
    assert_true(result.equals(projected.collect(engine="cpu")))
    assert_equal(result.column("x").dtype(), dtype)
    assert_equal(result.column("filled").dtype(), dtype)
    var filtered = (
        frame.lazy().filter(col("b")).select_exprs([col("x"), col("y")]).head(3)
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    var ordered = frame.slice(1, 4).lazy().sort("x")
    assert_true(
        ordered.collect(accelerator=runtime).equals(
            ordered.collect(engine="cpu")
        )
    )
    for n in range(6):
        var reduced = (
            frame.slice(0, n)
            .lazy()
            .select_exprs(
                [
                    col("x").min().alias("lo"),
                    col("x").max().alias("hi"),
                    col("x").count().alias("n"),
                ]
            )
        )
        assert_true(
            reduced.collect(accelerator=runtime).equals(
                reduced.collect(engine="cpu")
            )
        )


def test_native_fixed_logical_types_preserve_metadata() raises:
    if not metal_installed():
        return
    _native_fixed_logical(DataType.DATE)
    _native_fixed_logical(DataType.TIME)
    _native_fixed_logical(DataType.datetime("ns"))
    _native_fixed_logical(DataType.datetime("us", "America/New_York"))
    _native_fixed_logical(DataType.duration("ms"))
    _native_fixed_logical(DataType.decimal(9, 2, 32))
    _native_fixed_logical(DataType.decimal(18, 4, 64))
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([100])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("b", Column[Int32]([100])).with_dtype(
                DataType.decimal(9, 3, 32)
            ),
        ]
    )
    with assert_raises(contains="matching scales and storage widths"):
        _ = (
            frame.lazy()
            .select(col("a") == col("b"))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="Decimal128 accumulation"):
        _ = frame.lazy().select(col("a").sum()).collect(accelerator=runtime)


def _native_integer_row_ops[D: DType]() raises:
    var values = List[Scalar[D]]()
    var divisors = List[Scalar[D]]()
    comptime if D.is_signed():
        values = [-7, -1, 0, 1, 7]
        divisors = [3, -3, 0, -1, 2]
    else:
        values = [0, 1, 2, 7, 15]
        divisors = [3, 3, 0, 1, 2]
    var frame = DataFrame(
        [
            Series("x", Column[Scalar[D]](values^)),
            Series("y", Column[Scalar[D]](divisors^)),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            (col("x") // col("y")).alias("div"),
            (col("x") % col("y")).alias("mod"),
            col("x").pow(lit(Scalar[D](2))).alias("pow"),
            col("x").abs().alias("abs"),
            col("x").clip(lit(Scalar[D](0)), lit(Scalar[D](2))).alias("clip"),
            col("x").floor().alias("floor"),
            col("x").ceil().alias("ceil"),
            col("x").round(3).alias("round"),
        ]
    )
    var runtime = MetalRuntime()
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var ordered = frame.lazy().sort("x", descending=True)
    assert_true(
        ordered.collect(accelerator=runtime).equals(
            ordered.collect(engine="cpu")
        )
    )


def test_native_additional_integer_operators() raises:
    if not metal_installed():
        return
    _native_integer_row_ops[DType.int8]()
    _native_integer_row_ops[DType.int16]()
    _native_integer_row_ops[DType.int32]()
    _native_integer_row_ops[DType.int64]()
    _native_integer_row_ops[DType.uint8]()
    _native_integer_row_ops[DType.uint16]()
    _native_integer_row_ops[DType.uint32]()
    _native_integer_row_ops[DType.uint64]()
    var runtime = MetalRuntime()
    var frame = DataFrame([Series("x", Column[Int8]([Int8.MIN]))])
    with assert_raises(contains="overflow"):
        _ = frame.lazy().select(col("x").abs()).collect(accelerator=runtime)
    with assert_raises(contains="nonnegative exponent"):
        _ = (
            frame.lazy()
            .select(col("x").pow(lit(Int8(-1))))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="terminal projections"):
        _ = (
            frame.lazy()
            .with_columns(col("x").abs().alias("abs"))
            .filter(col("x") > lit(Int8(0)))
            .collect(accelerator=runtime)
        )


def test_native_conditional_masks_preserve_observable_errors() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "p",
                Column[Bool](
                    [False, False, False, True], [True, True, False, True]
                ),
            ),
            Series("n", Column[Int8]([127, -128, 1, 2])),
            Series(
                "u", Column[UInt64]([UInt64.MAX, UInt64.MAX, UInt64.MAX, 2])
            ),
            Series("f", Column[Float32]([1e-20, 1e-20, 1e-20, 1e20])),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            when(col("p"))
            .then(col("n").abs())
            .otherwise(col("n"))
            .alias("abs"),
            when(col("p"))
            .then(col("u").cast("int8"))
            .otherwise(lit(Int8(0)))
            .alias("cast"),
            when(col("p"))
            .then(col("f") * lit(Float32(1e-20)))
            .otherwise(col("f"))
            .alias("float"),
            when(col("p")).then(col("n")).alias("nullable"),
            when(col("p"))
            .then(
                when(col("n") > lit(Int8(1)))
                .then(col("n") + lit(Int8(1)))
                .otherwise(col("n"))
            )
            .otherwise(col("n"))
            .alias("nested"),
        ]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var bad = frame.lazy().select(
        when(~col("p")).then(col("n").abs()).otherwise(col("n"))
    )
    with assert_raises(contains="overflow"):
        _ = bad.collect(accelerator=runtime)


def test_native_float_value_operations_and_subnormal_comparisons() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [
                        tiny,
                        -tiny,
                        -0.0,
                        0.0,
                        0.5,
                        -0.5,
                        1.5,
                        -1.5,
                        bitcast[DType.float32](UInt32(0x7FC00000)),
                        bitcast[DType.float32](UInt32(0x7F800000)),
                        bitcast[DType.float32](UInt32(0xFF800000)),
                    ]
                ),
            )
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            (col("x") > lit(Float32(0))).alias("positive"),
            (col("x") < lit(Float32(0))).alias("negative"),
            (col("x") == lit(Float32(0))).alias("zero"),
            col("x").abs().alias("abs"),
            col("x").floor().alias("floor"),
            col("x").ceil().alias("ceil"),
            col("x").round().alias("round"),
            col("x").is_nan().alias("nan"),
            col("x").is_not_nan().alias("not_nan"),
            col("x").is_finite().alias("finite"),
            col("x").is_infinite().alias("infinite"),
            col("x").fill_nan(lit(Float32(2))).alias("fill"),
            col("x").clip(lit(Float32(-1)), lit(Float32(1))).alias("clip"),
        ]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var filtered = (
        frame.lazy().filter(col("x") > lit(Float32(0))).select(col("x"))
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    with assert_raises(contains="Float64 intermediate"):
        _ = frame.lazy().select(col("x").round(2)).collect(accelerator=runtime)


def test_native_stable_multikey_sort_between_resident_steps() raises:
    if not metal_installed():
        return
    var xs = List[Float32]()
    var us = List[UInt64]()
    var bs = List[Bool]()
    var ids = List[Int32]()
    var xv = List[Bool]()
    var uv = List[Bool]()
    var bv = List[Bool]()
    for i in range(521):
        var x = Float32(i % 23 - 11)
        if i % 29 == 0:
            x = bitcast[DType.float32](UInt32(0x7FC00000))
        elif i % 31 == 0:
            x = bitcast[DType.float32](UInt32(1))
        elif i % 37 == 0:
            x = -0.0
        xs.append(x)
        us.append(UInt64.MAX - UInt64(i % 17))
        bs.append(i % 2 == 0)
        ids.append(Int32(i))
        xv.append(i % 7 != 3)
        uv.append(i % 13 != 2)
        bv.append(i % 11 != 4)
    var frame = DataFrame(
        [
            Series("x", Column[Float32](xs^, xv)),
            Series("u", Column[UInt64](us^, uv)),
            Series("b", Column[Bool](bs^, bv)),
            Series("id", Column[Int32](ids^)),
        ]
    )
    var runtime = MetalRuntime()
    for descending in [False, True]:
        for nulls_last in [False, True]:
            var query = (
                frame.slice(3, 515)
                .lazy()
                .sort(
                    ["b", "x", "u"],
                    descending=[descending, not descending, descending],
                    nulls_last=[nulls_last, not nulls_last, nulls_last],
                )
                .select_exprs([col("id"), col("x"), col("u"), col("b")])
            )
            var measured = query.profile(accelerator=runtime)
            assert_true(measured[0].equals(query.collect(engine="cpu")))
            assert_equal(
                measured[1].column("synchronizations").int64()._get(0), 1
            )
    var resident = (
        frame.lazy()
        .with_columns(col("x").abs().alias("key"))
        .filter(col("id") >= 5)
        .sort(
            ["key", "u"],
            descending=[False, True],
            nulls_last=[True, False],
        )
        .filter(col("id") < 511)
        .sort("u")
        .select_exprs([col("id"), col("key"), col("u")])
    )
    assert_true(
        resident.collect(accelerator=runtime).equals(
            resident.collect(engine="cpu")
        )
    )
    var terminal = resident.select_exprs(
        [col("id"), (col("id") + lit(Int32(1))).alias("next")]
    )
    assert_true(
        terminal.collect(accelerator=runtime).equals(
            terminal.collect(engine="cpu")
        )
    )
    var top = frame.lazy().sort("x", descending=True).select(col("id")).head(9)
    assert_true(
        top.collect(accelerator=runtime).equals(top.collect(engine="cpu"))
    )
    var empty = frame.slice(0, 0).lazy().sort("u").select(col("id"))
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var all_filtered = frame.lazy().filter(col("id") < 0).sort("x")
    assert_true(
        all_filtered.collect(accelerator=runtime).equals(
            all_filtered.collect(engine="cpu")
        )
    )


def _native_groups[D: DType]() raises:
    var values = List[Scalar[D]]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for i in range(521):
        values.append(Scalar[D](i % 13))
        keys.append(Int32((i * 7) % 9))
        valid.append(i % 11 != 2)
    var frame = DataFrame(
        [
            Series("key", Column[Int32](keys^, valid)),
            Series("x", Column[Scalar[D]](values^, valid)),
        ]
    )
    var runtime = MetalRuntime()
    var expressions: List[Expr] = [
        col("x").min().alias("lo"),
        col("x").max().alias("hi"),
        col("x").count().alias("count"),
        col("x").len().alias("len"),
    ]
    for ordered in [False, True]:
        var query = (
            frame.slice(5, 513)
            .lazy()
            .group_by("key", maintain_order=ordered)
            .agg(expressions)
        )
        if not ordered:
            query = query.sort("key")
        var measured = query.profile(accelerator=runtime)
        assert_true(measured[0].equals(query.collect(engine="cpu")))
        assert_equal(measured[1].column("synchronizations").int64()._get(0), 1)
    var empty = frame.slice(0, 0).lazy().group_by("key").agg(expressions)
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )


def test_native_grouped_reductions() raises:
    if not metal_installed():
        return
    _native_groups[DType.float32]()
    _native_groups[DType.int8]()
    _native_groups[DType.int16]()
    _native_groups[DType.int32]()
    _native_groups[DType.int64]()
    _native_groups[DType.uint8]()
    _native_groups[DType.uint16]()
    _native_groups[DType.uint32]()
    _native_groups[DType.uint64]()
    var runtime = MetalRuntime()
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var frame = DataFrame(
        [
            Series(
                "key",
                Column[Float32](
                    [nan, 1, -0.0, nan, 0.0, 1, nan],
                    [True, False, True, True, True, True, False],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, True, False, False, True, False, True],
                    [True, False, True, True, True, True, False],
                ),
            ),
            Series(
                "x",
                Column[Int16](
                    [10, 20, 30, 40, 50, 60, 70],
                    [True, True, False, True, True, True, False],
                ),
            ),
            Series("date", Column[Int64]([1, 2, 3, 4, 5, 6, 7])).with_dtype(
                DataType.DATE
            ),
            Series(
                "decimal", Column[Int32]([100, 200, 300, 400, 500, 600, 700])
            ).with_dtype(DataType.decimal(9, 2, 32)),
        ]
    )
    var query = (
        frame.lazy()
        .group_by(["key", "b"], maintain_order=True)
        .agg(
            [
                col("x").sum().alias("sum"),
                col("b").count().alias("bool_count"),
                col("date").min().alias("date"),
                col("decimal").max().alias("decimal"),
            ]
        )
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var pipeline = (
        query.filter(col("sum") > 0)
        .sort("sum")
        .select_exprs([col("sum"), col("date"), col("decimal")])
        .head(3)
    )
    assert_true(
        pipeline.collect(accelerator=runtime).equals(
            pipeline.collect(engine="cpu")
        )
    )
    var twice = query.group_by("date", maintain_order=True).agg(
        col("sum").max()
    )
    assert_true(
        twice.collect(accelerator=runtime).equals(twice.collect(engine="cpu"))
    )
    var final_reduction = query.select_exprs(
        [
            col("sum").min().alias("min_sum"),
            col("decimal").max().alias("max_decimal"),
        ]
    )
    assert_true(
        final_reduction.collect(accelerator=runtime).equals(
            final_reduction.collect(engine="cpu")
        )
    )
    var empty_global = (
        frame.slice(0, 0)
        .lazy()
        .group_by(List[String]())
        .agg(
            [
                col("x").sum().alias("sum"),
                col("x").len().alias("len"),
                col("x").min().alias("min"),
            ]
        )
    )
    with assert_raises(contains="requires at least one key"):
        _ = empty_global.collect(accelerator=runtime)
    with assert_raises(contains="requires at least one key"):
        _ = empty_global.collect(engine="cpu")
    var keys_only = (
        frame.lazy()
        .group_by(["key", "b"], maintain_order=True)
        .agg(List[Expr]())
    )
    assert_true(
        keys_only.collect(accelerator=runtime).equals(
            keys_only.collect(engine="cpu")
        )
    )
    var global_query = frame.lazy().select(col("b").count())
    assert_true(
        global_query.collect(accelerator=runtime).equals(
            global_query.collect(engine="cpu")
        )
    )
    var overflow = (
        DataFrame(
            [
                Series("key", Column[Int32]([0, 0])),
                Series("x", Column[Int32]([Int32.MAX, 1])),
            ]
        )
        .lazy()
        .group_by("key")
        .agg(col("x").sum())
    )
    with assert_raises(contains="overflow"):
        _ = overflow.collect(accelerator=runtime)
    with assert_raises(contains="overflow"):
        _ = overflow.collect(engine="cpu")


def _native_extra_reductions[D: DType]() raises:
    var values = List[Scalar[D]]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for i in range(521):
        values.append(Scalar[D](i % 13))
        keys.append(Int32((i * 7) % 9))
        valid.append(i % 11 != 2)
    var frame = DataFrame(
        [
            Series("key", Column[Int32](keys^, valid)),
            Series("x", Column[Scalar[D]](values^, valid)),
        ]
    )
    var expressions: List[Expr] = [
        col("x").first().alias("first"),
        col("x").last().alias("last"),
        col("x").null_count().alias("nulls"),
        col("x").arg_min().alias("argmin"),
        col("x").arg_max().alias("argmax"),
    ]
    var runtime = MetalRuntime()
    var scalar = (
        frame.slice(5, 513)
        .lazy()
        .filter(col("key") >= 2)
        .select_exprs(expressions)
    )
    assert_true(
        scalar.collect(accelerator=runtime).equals(scalar.collect(engine="cpu"))
    )
    var grouped = (
        frame.slice(5, 513)
        .lazy()
        .group_by("key", maintain_order=True)
        .agg(expressions)
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var empty = frame.slice(0, 0).lazy().select_exprs(expressions)
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )


def test_native_additional_reductions() raises:
    if not metal_installed():
        return
    _native_extra_reductions[DType.float32]()
    _native_extra_reductions[DType.int8]()
    _native_extra_reductions[DType.int16]()
    _native_extra_reductions[DType.int32]()
    _native_extra_reductions[DType.int64]()
    _native_extra_reductions[DType.uint8]()
    _native_extra_reductions[DType.uint16]()
    _native_extra_reductions[DType.uint32]()
    _native_extra_reductions[DType.uint64]()
    var runtime = MetalRuntime()
    var expressions: List[Expr] = [
        col("b").any().alias("any"),
        col("b").all().alias("all"),
        col("b").any(ignore_nulls=False).alias("any_nulls"),
        col("b").all(ignore_nulls=False).alias("all_nulls"),
        col("b").first().alias("first"),
        col("b").last().alias("last"),
        col("b").null_count().alias("nulls"),
        col("b").arg_min().alias("argmin"),
        col("b").arg_max().alias("argmax"),
    ]
    for values in [
        List[Bool](),
        List[Bool]([False, True, False]),
        List[Bool]([False, False, False]),
        List[Bool]([True, True, True]),
    ]:
        var valid = List[Bool](length=len(values), fill=True)
        if len(values) > 0:
            valid[0] = False
        var keys = List[Int32](length=len(values), fill=0)
        var frame = DataFrame(
            [
                Series("key", Column[Int32](keys^)),
                Series("b", Column[Bool](values.copy(), valid)),
            ]
        )
        var scalar = frame.lazy().select_exprs(expressions)
        assert_true(
            scalar.collect(accelerator=runtime).equals(
                scalar.collect(engine="cpu")
            )
        )
        var grouped = frame.lazy().group_by("key").agg(expressions)
        assert_true(
            grouped.collect(accelerator=runtime).equals(
                grouped.collect(engine="cpu")
            )
        )
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var special = DataFrame(
        [
            Series("key", Column[Int32]([0, 0, 1, 1, 2, 2])),
            Series("x", Column[Float32]([nan, 1, nan, nan, -0.0, 0.0])),
        ]
    )
    var arg_exprs: List[Expr] = [
        col("x").arg_min().alias("argmin"),
        col("x").arg_max().alias("argmax"),
    ]
    var grouped = (
        special.lazy().group_by("key", maintain_order=True).agg(arg_exprs)
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    assert_equal(
        grouped.collect(accelerator=runtime).column("argmax").uint32()._get(0),
        UInt32(1),
    )
    var logical = DataFrame(
        [
            Series("key", Column[Int32]([0, 0, 1])),
            Series(
                "date", Column[Int64]([1, 2, 3], [False, True, True])
            ).with_dtype(DataType.DATE),
            Series(
                "decimal", Column[Int32]([100, 200, 300], [True, True, False])
            ).with_dtype(DataType.decimal(9, 2, 32)),
        ]
    )
    var logical_query = (
        logical.lazy()
        .group_by("key", maintain_order=True)
        .agg(
            [
                col("date").first().alias("date_first"),
                col("date").last().alias("date_last"),
                col("decimal").first().alias("decimal_first"),
                col("decimal").last().alias("decimal_last"),
                col("decimal").null_count().alias("decimal_nulls"),
            ]
        )
    )
    assert_true(
        logical_query.collect(accelerator=runtime).equals(
            logical_query.collect(engine="cpu")
        )
    )


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
