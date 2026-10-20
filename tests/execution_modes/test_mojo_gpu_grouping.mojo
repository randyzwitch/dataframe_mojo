"""Direct Mojo grouped reductions, nullable keys and stable representatives."""

from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    cov,
    corr,
    lit,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.rows import ROW_GROUP_HEADER
from dataframe.accelerator.gpu.runtime import (
    _grouped_integer_partial_nodes,
    _preflight,
)
from tests.execution_modes.test_backend_metal_moments import near


def _expressions() -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").mean().alias("mean"),
        col("x").min().alias("min"),
        col("x").max().alias("max"),
        col("x").count().alias("count"),
        col("x").len().alias("length"),
        col("x").null_count().alias("nulls"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
        col("x").arg_min().alias("arg_min"),
        col("x").arg_max().alias("arg_max"),
        col("flag").any(ignore_nulls=False).alias("any"),
        col("flag").all(ignore_nulls=False).alias("all"),
        col("x").var().alias("variance"),
        col("x").std().alias("standard"),
        col("x").skew(False).alias("skew"),
        col("x").kurtosis(True, False).alias("kurtosis"),
        cov(col("x"), col("y")).alias("covariance"),
        corr(col("x"), col("y")).alias("correlation"),
    ]


def _check(query: LazyFrame) raises -> DataFrame:
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    near(actual, query.collect(engine="cpu", streaming=False))
    return actual^


def test_grouped_integer_metadata_certificate_accepts_normal_groupby_agg() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 1, 1])),
            Series("x", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    var plan = _preflight(
        frame.lazy()
        .group_by("k")
        .agg([col("x").sum().alias("sum"), col("x").count().alias("count")])
    )
    var found = False
    for step in plan.steps:
        if step.grouped:
            assert_equal(plan.code[4 * step.start], Int64(ROW_GROUP_HEADER))
            var candidates = _grouped_integer_partial_nodes(
                plan, step.start, step.nodes
            )
            assert_equal(len(candidates), 2)
            found = True
    assert_true(found)


def _layout_case[D: DType](n: Int, chunked: Bool) raises:
    var keys = List[Scalar[D]]()
    var second = List[Int32]()
    var xs = List[Int64]()
    var ys = List[Float32]()
    var flags = List[Bool]()
    var kv = List[Bool]()
    var sv = List[Bool]()
    var xv = List[Bool]()
    var yv = List[Bool]()
    for row in range(n + 6):
        var key = Scalar[D](row % 5)
        comptime if D == DType.float32:
            var encodings: List[UInt32] = [
                0,
                0x80000000,
                0x7FC00001,
                0xFFC00002,
                0x3F800000,
            ]
            key = bitcast[DType.float32](encodings[row % 5]).cast[D]()
        elif D == DType.float64:
            var encodings: List[UInt64] = [
                0,
                0x8000000000000000,
                0x7FF8000000000001,
                0xFFF8000000000002,
                0x3FF0000000000000,
            ]
            key = bitcast[DType.float64](encodings[row % 5]).cast[D]()
        keys.append(key)
        second.append(Int32(row % 3))
        xs.append(Int64(row % 11 - 5))
        ys.append(Float32(row % 11 - 5) * 2 + Float32(row % 3))
        flags.append(row % 2 == 0)
        kv.append(row % 13 != 0)
        sv.append(row % 17 != 0)
        xv.append(row % 7 != 0)
        yv.append(row % 5 != 0)
    var columns = List[Series](
        [
            Series("k", Column[Scalar[D]](keys^, kv^)),
            Series("j", Column[Int32](second^, sv^)),
            Series("x", Column[Int64](xs^, xv.copy())),
            Series("y", Column[Float32](ys^, yv^)),
            Series("flag", Column[Bool](flags^, xv^)),
        ]
    )
    for i in range(len(columns)):
        columns[i] = columns[i].slice(3, n)
        if chunked:
            columns[i] = Series._from_chunks(
                [
                    columns[i].slice(0, n // 2),
                    columns[i].slice(n // 2, 0),
                    columns[i].slice(n // 2, n - n // 2),
                ]
            )
    var frame = DataFrame(columns^)
    var result = _check(
        frame.lazy()
        .group_by(["k", "j"], maintain_order=True)
        .agg(_expressions())
    )
    assert_true(result.height() <= n)
    if n == 0:
        assert_equal(result.height(), 0)
    else:
        var total = Int64(0)
        for row in range(result.height()):
            total += result.item(row, "length").int64()
        assert_equal(total, Int64(n))


# requires: mojo_gpu
def test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts() raises:
    comptime types = (DType.int8, DType.uint64, DType.float32, DType.float64)
    comptime for i in range(len(types)):
        comptime D = types[i]
        for n in List[Int]([0, 1, 1027]):
            _layout_case[D](n, False)
            _layout_case[D](n, True)


# requires: mojo_gpu
def test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions() raises:
    var keys = List[UInt64]()
    for row in range(1027):
        keys.append(UInt64.MAX - UInt64(row))
    var frame = DataFrame([Series("k", Column[UInt64](keys.copy()))])
    var result = _check(
        frame.lazy().group_by("k", maintain_order=True).agg(List[Expr]())
    )
    assert_equal(result.height(), 1027)
    assert_true(result.column("k").equals(frame.column("k")))
    _ = _check(frame.lazy().group_by("k").agg(List[Expr]()).sort("k"))
    var source = DataFrame(
        [
            Series(
                "k",
                Column[Bool](
                    [True, False, True, False, True, False],
                    [True, True, False, True, True, False],
                ),
            ),
            Series("x", Column[Int64]([4, 3, 2, 1, 0, -1])),
        ]
    )
    result = _check(
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("x").mean().alias("total"), col("x").len().alias("length")])
        .filter(col("length") > lit(Int64(1)))
        .sort("total")
        .select_exprs([col("k"), col("total") * lit(Float64(2))])
    )
    assert_equal(result.height(), 3)


# requires: mojo_gpu
def test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int32]([2, 1, 2, 1, 2, 1])),
            Series(
                "x", Column[Int64]([Int64.MAX, 8, Int64.MAX, 2, -Int64.MAX, 4])
            ),
        ]
    )
    var result = _check(
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").sum().alias("sum"),
                col("x").arg_min().alias("min_at"),
                col("x").arg_max().alias("max_at"),
            ]
        )
    )
    assert_equal(result.item(0, "sum").int64(), Int64.MAX)
    assert_equal(result.item(0, "min_at").numeric[DType.uint32](), UInt32(2))
    assert_equal(result.item(0, "max_at").numeric[DType.uint32](), UInt32(0))
    assert_equal(result.item(1, "min_at").numeric[DType.uint32](), UInt32(1))
    var overflow = DataFrame(
        [
            Series("k", Column[Int32]([0, 0])),
            Series("x", Column[UInt64]([UInt64.MAX, 1])),
        ]
    )
    with assert_raises(contains="overflow"):
        _ = (
            overflow.lazy()
            .group_by("k")
            .agg(col("x").sum())
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    _ = _check(overflow.lazy().group_by("k").agg(col("x").sum(min_count=3)))


# requires: mojo_gpu
def test_mojo_gpu_grouping_integer_partials_shared_merge_and_fallback() raises:
    var rows = 10000
    var keys = List[Int32](length=rows, fill=0)
    var values = List[Int64](length=rows, fill=0)
    var valid = List[Bool](length=rows, fill=True)
    for row in range(rows):
        values[row] = Int64.MAX if row % 2 == 0 else -Int64.MAX
        if (row // 2) % 100 == 0:
            valid[row] = False
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Int64](values^, valid^)),
        ]
    )
    var query = (
        frame.lazy()
        .group_by("k")
        .agg(
            [
                col("x").sum().alias("sum"),
                col("x").count().alias("count"),
                col("x").min().alias("min"),
                col("x").max().alias("max"),
                col("x").sum(min_count=10000).alias("minimum_count"),
            ]
        )
    )
    var runtime = MetalRuntime()
    var measured = query.profile(engine="accel", accelerator=runtime)
    assert_true(
        measured[0].equals(query.collect(engine="cpu", streaming=False))
    )
    assert_equal(measured[0].height(), 1)
    assert_equal(measured[0].item(0, "sum").int64(), Int64(0))
    assert_equal(measured[0].item(0, "count").int64(), Int64(9900))
    assert_equal(measured[0].item(0, "min").int64(), -Int64.MAX)
    assert_equal(measured[0].item(0, "max").int64(), Int64.MAX)
    assert_true(measured[0].item(0, "minimum_count").is_null())
    assert_equal(
        measured[1].item(0, "grouped_int_algorithm").string(),
        "mojo_gpu_grouped_int_partials",
    )
    assert_equal(
        measured[1].item(0, "grouped_int_parallel_outputs").int64(),
        Int64(5),
    )
    assert_true(
        measured[1].item(0, "grouped_int_max_partials_per_group").int64() >= 2
    )
    assert_true(
        measured[1].item(0, "grouped_int_max_partial_states").int64() <= 4096
    )
    assert_equal(
        measured[1].item(0, "grouped_int_scratch_bytes").int64(),
        Int64(4096 * 8 * 8),
    )
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    var bounded = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="memory budget"):
        _ = query.collect(engine="accel", accelerator=bounded)
    var exact_runtime = MetalRuntime(memory_limit_bytes=peak)
    var exact = query.profile(engine="accel", accelerator=exact_runtime)
    assert_true(exact[0].equals(measured[0]))

    var distinct = List[Int32](length=4097, fill=0)
    var ones = List[Int64](length=4097, fill=1)
    for row in range(4097):
        distinct[row] = Int32(row)
    var high_cardinality = DataFrame(
        [
            Series("k", Column[Int32](distinct^)),
            Series("x", Column[Int64](ones^)),
        ]
    )
    var fallback = (
        high_cardinality.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").sum())
    )
    var fallback_profile = fallback.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        fallback_profile[0].equals(
            fallback.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(
        fallback_profile[1].item(0, "grouped_int_algorithm").string(),
        "legacy",
    )
    assert_equal(
        fallback_profile[1].item(0, "grouped_int_partial_tiles").int64(),
        Int64(0),
    )
    assert_equal(
        fallback_profile[1].item(0, "grouped_int_scratch_bytes").int64(),
        Int64(0),
    )

    var unsigned_values = List[UInt64](length=rows, fill=0)
    for row in range(rows):
        unsigned_values[row] = UInt64.MAX if row % 2 == 0 else UInt64(0)
    var unsigned_frame = DataFrame(
        [
            Series("k", Column[Int32](keys.copy())),
            Series("x", Column[UInt64](unsigned_values^)),
        ]
    )
    var unsigned_query = (
        unsigned_frame.lazy()
        .group_by("k")
        .agg(
            [
                col("x").sum(min_count=rows + 1).alias("sum"),
                col("x").count().alias("count"),
                col("x").min().alias("min"),
                col("x").max().alias("max"),
            ]
        )
    )
    var unsigned_profile = unsigned_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        unsigned_profile[0].equals(
            unsigned_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_true(unsigned_profile[0].item(0, "sum").is_null())
    assert_equal(unsigned_profile[0].item(0, "count").int64(), Int64(rows))
    assert_equal(unsigned_profile[0].item(0, "min").uint64(), UInt64(0))
    assert_equal(unsigned_profile[0].item(0, "max").uint64(), UInt64.MAX)
    assert_equal(
        unsigned_profile[1].item(0, "grouped_int_algorithm").string(),
        "mojo_gpu_grouped_int_partials",
    )
    assert_equal(
        unsigned_profile[1].item(0, "grouped_int_parallel_outputs").int64(),
        Int64(4),
    )

    var skew_rows = 10000
    var skew_keys = List[Int32](length=skew_rows, fill=0)
    var skew_values = List[Int64](length=skew_rows, fill=1)
    for row in range(9000, skew_rows):
        skew_keys[row] = Int32(1 + (row - 9000) // 250)
    var skewed = DataFrame(
        [
            Series("k", Column[Int32](skew_keys^)),
            Series("x", Column[Int64](skew_values^)),
        ]
    )
    var skew_query = (
        skewed.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").sum().alias("sum"))
    )
    var skew_profile = skew_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        skew_profile[0].equals(
            skew_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(skew_profile[0].height(), 5)
    assert_equal(skew_profile[0].item(0, "sum").int64(), Int64(9000))
    for group in range(1, 5):
        assert_equal(skew_profile[0].item(group, "sum").int64(), Int64(250))
    assert_true(
        skew_profile[1].item(0, "grouped_int_max_partials_per_group").int64()
        >= 2
    )

    var floats = List[Float64](length=rows, fill=0)
    var mixed = DataFrame(
        [
            Series("k", Column[Int32](keys.copy())),
            Series("x", Column[Int64](List[Int64](length=rows, fill=1)^)),
            Series("f", Column[Float64](floats^)),
        ]
    )
    var mixed_query = (
        mixed.lazy()
        .group_by("k")
        .agg(
            [
                col("x").sum().alias("integer_sum"),
                col("f").sum().alias("float_sum"),
            ]
        )
    )
    var mixed_profile = mixed_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        mixed_profile[0].equals(
            mixed_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(mixed_profile[0].item(0, "integer_sum").int64(), Int64(rows))
    assert_equal(
        mixed_profile[1].item(0, "grouped_int_parallel_outputs").int64(),
        Int64(1),
    )

    var overflowing = DataFrame(
        [
            Series("k", Column[Int32](List[Int32](length=rows, fill=0)^)),
            Series(
                "x",
                Column[Int64](List[Int64](length=rows, fill=Int64.MAX)^),
            ),
        ]
    )
    with assert_raises(contains="overflow"):
        _ = (
            overflowing.lazy()
            .group_by("k")
            .agg(col("x").sum())
            .collect(engine="accel", accelerator=MetalRuntime())
        )
    var overflowing_group = (
        overflowing.lazy().group_by("k").agg(col("x").sum().alias("sum"))
    )
    with assert_raises(contains="overflow"):
        _ = overflowing_group.collect(engine="cpu", streaming=False)
    with assert_raises(contains="overflow"):
        _ = overflowing_group.head(0).collect(
            engine="cpu", optimize=False, streaming=False
        )
    with assert_raises(contains="overflow"):
        _ = overflowing_group.filter(col("sum") < lit(Int64(0))).collect(
            engine="cpu", optimize=False, streaming=False
        )
    with assert_raises(contains="overflow"):
        _ = overflowing_group.head(0).collect(
            engine="accel", accelerator=MetalRuntime(), optimize=False
        )
    with assert_raises(contains="overflow"):
        _ = overflowing_group.filter(col("sum") < lit(Int64(0))).collect(
            engine="accel", accelerator=MetalRuntime(), optimize=False
        )

    var join_right = DataFrame(
        [
            Series("k", Column[Int32]([0])),
            Series("tag", Column[Int32]([1])),
        ]
    )
    var composed = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").sum().alias("total"))
        .join(join_right.lazy(), on="k")
        .filter(col("total") >= lit(Int64(0)))
        .select_exprs([(col("total") + lit(Int64(1))).alias("adjusted")])
    )
    var composed_profile = composed.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        composed_profile[0].equals(
            composed.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(composed_profile[0].item(0, "adjusted").int64(), Int64(1))
    assert_true("grouped_int_algorithm" in composed_profile[1].columns())
    var saw_grouped_partials = False
    for row in range(composed_profile[1].height()):
        saw_grouped_partials |= (
            composed_profile[1].item(row, "grouped_int_algorithm").string()
            == "mojo_gpu_grouped_int_partials"
        )
    assert_true(saw_grouped_partials)
