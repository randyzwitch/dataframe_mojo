"""Stable direct Mojo GPU ordering across public lazy-query compositions."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.support.key_kernel_reference import KeyCell, expected_order
from dataframe.accelerator.gpu.ordering import merge_grid


def _layout(source: Series, chunked: Bool) raises -> Series:
    if not chunked:
        return source.copy()
    var cut = len(source) // 2
    return Series._from_chunks(
        [
            source.slice(0, cut),
            source.slice(cut, 0),
            source.slice(cut, len(source) - cut),
        ]
    )


def _integer_case[D: DType](n: Int, chunked: Bool) raises:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var flags = List[Bool]()
    var flag_valid = List[Bool]()
    var ids = List[Int64]()
    var reference = List[List[KeyCell]]()
    for physical in range(n + 6):
        var value = Scalar[D](physical % 7)
        if physical % 13 == 0:
            value = Scalar[D].MAX
        elif physical % 13 == 1:
            value = Scalar[D].MIN
        var present = physical % 11 != 4
        var flag = physical % 3 == 0
        var flag_present = physical % 9 != 2
        values.append(value)
        valid.append(present)
        flags.append(flag)
        flag_valid.append(flag_present)
        ids.append(Int64(physical - 3))
        if physical >= 3 and physical < n + 3:
            reference.append(
                [
                    KeyCell(0, present, integer=Int128(value)),
                    KeyCell(0, flag_present, integer=Int128(flag)),
                ]
            )
    var frame = DataFrame(
        [
            _layout(
                Series("x", Column[Scalar[D]](values^, valid^)).slice(3, n),
                chunked,
            ),
            _layout(
                Series("flag", Column[Bool](flags^, flag_valid^)).slice(3, n),
                chunked,
            ),
            _layout(Series("id", Column[Int64](ids^)).slice(3, n), chunked),
        ]
    )
    for reverse in [False, True]:
        var directions: List[Bool] = [reverse, not reverse]
        var null_placement: List[Bool] = [not reverse, reverse]
        var query = frame.lazy().sort(
            ["x", "flag"],
            descending=directions,
            nulls_last=null_placement,
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        same(actual, query.collect(engine="cpu"))
        assert_equal(actual.column("x").dtype(), DataType.of(D))
        var expected = expected_order(reference, directions, null_placement)
        for row in range(n):
            assert_equal(actual.item(row, "id").int64(), Int64(expected[row]))
    # Exercise the integer radix certificate itself, including repeated keys
    # and stable null ties; the two-key cases above intentionally use merge.
    var direct_reference = List[List[KeyCell]]()
    for row in reference:
        direct_reference.append([row[0]])
    for reverse in [False, True]:
        var query = frame.lazy().sort(
            "x", descending=reverse, nulls_last=not reverse
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        same(actual, query.collect(engine="cpu"))
        var expected = expected_order(
            direct_reference, [reverse], [not reverse]
        )
        for row in range(n):
            assert_equal(actual.item(row, "id").int64(), Int64(expected[row]))


def _topk_composite_frame(rows: Int, chunked: Bool) raises -> DataFrame:
    var major = List[Int64]()
    var major_valid = List[Bool]()
    var float_words = List[UInt64]()
    var float_valid = List[Bool]()
    var wide = List[Int128]()
    var wide_valid = List[Bool]()
    var ids = List[Int64]()
    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0xFFF8000000000099,
        1,
        0x8000000000000001,
    ]
    var large = pow10(25)
    var decimal_keys: List[Int128] = [
        -2 * large,
        -large,
        -1,
        0,
        1,
        large,
        2 * large,
    ]
    for physical in range(rows + 6):
        major.append(Int64((physical * 17 + physical // 7) % 19 - 9))
        major_valid.append(physical % 97 != 0)
        float_words.append(
            encodings[(physical * 3 + physical // 11) % len(encodings)]
        )
        float_valid.append(physical % 113 != 0)
        wide.append(
            decimal_keys[(physical * 5 + physical // 13) % len(decimal_keys)]
        )
        wide_valid.append(physical % 131 != 0)
        ids.append(Int64(physical - 3))
    var float_values = List[Float64](capacity=len(float_words))
    for value in float_words:
        float_values.append(bitcast[DType.float64](value))
    var frame = DataFrame(
        [
            _layout(
                Series(
                    "major",
                    Column[Int64](major^, major_valid^),
                ).slice(3, rows),
                chunked,
            ),
            _layout(
                Series(
                    "minor",
                    Column[Float64](
                        float_values^,
                        float_valid^,
                    ),
                ).slice(3, rows),
                chunked,
            ),
            _layout(
                Series("wide", Column[Int128](wide^, wide_valid^))
                .with_dtype(DataType.decimal(38, 0))
                .slice(3, rows),
                chunked,
            ),
            _layout(Series("id", Column[Int64](ids^)).slice(3, rows), chunked),
        ]
    )
    return frame^


# requires: mojo_gpu
def test_mojo_gpu_ordering_topk_composite_stability_layout_and_budget() raises:
    var rows = 8_193
    var frame = _topk_composite_frame(rows, True)
    for k in [1, 17, 255, 256, 257, 1_024, 4_096]:
        var query = (
            frame.lazy()
            .sort(
                ["major", "minor", "wide"],
                descending=[True, False, True],
                nulls_last=[False, True, False],
            )
            .head(k)
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        same(actual, query.collect(engine="cpu", streaming=False))
        assert_equal(actual.height(), k)

    # Cover partial base runs and merge widths around the 256-row boundary.
    for boundary_rows in [31, 255, 257, 511, 513]:
        var boundary_frame = _topk_composite_frame(boundary_rows, True)
        for k in [1, 17, 255, 256]:
            if k >= boundary_rows:
                continue
            var boundary_query = (
                boundary_frame.lazy()
                .sort(
                    ["major", "minor", "wide"],
                    descending=[True, False, True],
                    nulls_last=[False, True, False],
                )
                .head(k)
            )
            var boundary_actual = boundary_query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            same(
                boundary_actual,
                boundary_query.collect(engine="cpu", streaming=False),
            )
            assert_equal(boundary_actual.height(), k)

    var selected = frame.lazy().sort("major").head(31)
    var result = selected.collect(accelerator=MetalRuntime(), engine="accel")
    same(result, selected.collect(engine="cpu", streaming=False))
    var profile = selected.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    var topk_launches = profile[1].item(0, "kernel_launches").int64()
    assert_equal(
        profile[1].item(0, "resident_output_bytes").int64(),
        Int64(31 * frame.width() * 21),
    )
    with assert_raises(contains="budget"):
        _ = selected.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        selected.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        result,
    )
    var full = frame.lazy().sort("major")
    var full_profile = full.profile(accelerator=MetalRuntime(), engine="accel")
    assert_true(
        topk_launches < full_profile[1].item(0, "kernel_launches").int64()
    )

    # Filters are before the sort in this plan. The exact one-step certificate
    # rejects pushdown and retains the ordinary full-capacity result path.
    var filtered = frame.lazy().filter(col("major") > 0).sort("major").head(31)
    var filtered_result = filtered.collect(
        accelerator=MetalRuntime(), engine="accel"
    )
    same(filtered_result, filtered.collect(engine="cpu", streaming=False))
    var filtered_profile = filtered.profile(
        accelerator=MetalRuntime(), engine="accel"
    )
    assert_true(
        filtered_profile[1].item(0, "resident_output_bytes").int64()
        > Int64(31 * frame.width() * 21)
    )


# requires: mojo_gpu
def test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability() raises:
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for i in range(len(types)):
        for n in [0, 257]:
            _integer_case[types[i]](n, False)
            _integer_case[types[i]](n, True)
    _integer_case[DType.uint64](1_027, True)


def _floating_case[D: DType](chunked: Bool) raises:
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0xFFF8000000000099,
        1,
        0x8000000000000001,
        0x0010000000000000,
        0x8010000000000000,
    ]
    comptime if D == DType.float32:
        words = [
            0,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC00042,
            0xFFC00099,
            1,
            0x80000001,
            0x00800000,
            0x80800000,
        ]
    var values = List[Scalar[D]]()
    var validity = List[Bool]()
    var ids = List[Int64]()
    var secondary = List[Int16]()
    var reference = List[List[KeyCell]]()
    for physical in range(263):
        var word = words[physical % len(words)]
        var value = Scalar[D](0)
        comptime if D == DType.float32:
            value = bitcast[DType.float32](UInt32(word)).cast[D]()
        else:
            value = bitcast[DType.float64](word).cast[D]()
        var present = physical % 9 != 4
        values.append(value)
        validity.append(present)
        ids.append(Int64(physical))
        secondary.append(Int16(physical % 3))
        if physical >= 3 and physical < 260:
            reference.append(
                [
                    KeyCell(1, present, floating=Float64(value)),
                    KeyCell(0, integer=Int128(physical % 3)),
                ]
            )
    var frame = DataFrame(
        [
            _layout(
                Series("x", Column[Scalar[D]](values^, validity^)).slice(
                    3, 257
                ),
                chunked,
            ),
            _layout(Series("id", Column[Int64](ids^)).slice(3, 257), chunked),
            _layout(
                Series("k", Column[Int16](secondary^)).slice(3, 257), chunked
            ),
        ]
    )
    for descending in [False, True]:
        for nulls_last in [False, True]:
            var query = frame.lazy().sort(
                ["x", "k"],
                descending=[descending, not descending],
                nulls_last=[nulls_last, not nulls_last],
            )
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            same(actual, query.collect(engine="cpu"))
            var expected = expected_order(
                reference,
                [descending, not descending],
                [nulls_last, not nulls_last],
            )
            for row in range(actual.height()):
                var source = Int(actual.item(row, "id").int64())
                assert_equal(source, expected[row] + 3)
                if actual.item(row, "x").is_null():
                    continue
                comptime if D == DType.float32:
                    assert_equal(
                        bitcast[DType.uint32](
                            actual.column("x").float32()._get(row)
                        ),
                        UInt32(words[source % len(words)]),
                    )
                else:
                    assert_equal(
                        bitcast[DType.uint64](
                            actual.column("x").float64()._get(row)
                        ),
                        words[source % len(words)],
                    )


# requires: mojo_gpu
def test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls() raises:
    _floating_case[DType.float32](False)
    _floating_case[DType.float32](True)
    _floating_case[DType.float64](False)
    _floating_case[DType.float64](True)


# requires: mojo_gpu
def test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery() raises:
    for dtype in [
        DataType.DATE,
        DataType.datetime("us", "UTC"),
        DataType.duration("ns"),
    ]:
        var logical = DataFrame(
            [
                Series(
                    "key",
                    Column[Int64]([4, -1, 4, 0], [True, True, True, False]),
                ).with_dtype(dtype),
                Series("id", Column[Int64]([0, 1, 2, 3])),
            ]
        )
        var ordered = logical.lazy().sort("key")
        var result = ordered.collect(accelerator=MetalRuntime(), engine="accel")
        same(result, ordered.collect(engine="cpu"))
        assert_equal(result.column("key").dtype(), dtype)
        for row in range(4):
            var expected: List[Int64] = [1, 0, 2, 3]
            assert_equal(result.item(row, "id").int64(), expected[row])
    var source = DataFrame(
        [
            Series("x", Column[Int64]([9, 4, 7, 1, 7, 3])),
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5])),
            Series(
                "keep", Column[Bool]([False, True, True, False, True, True])
            ),
        ]
    )
    var runtime = MetalRuntime()
    var selected = (
        source.lazy()
        .filter(col("keep"))
        .sort("x", descending=True)
        .slice(1, 2)
        .select(["id", "x"])
    )
    var actual = selected.collect(accelerator=runtime, engine="accel")
    same(actual, selected.collect(engine="cpu"))
    assert_equal(actual.item(0, "id").int64(), Int64(4))
    assert_equal(actual.item(1, "id").int64(), Int64(1))
    var reduced = selected.select_exprs(
        [col("x").sum().alias("s"), col("id").first().alias("id")]
    )
    var result = reduced.collect(accelerator=runtime, engine="accel")
    same(result, reduced.collect(engine="cpu"))
    assert_equal(result.item(0, "s").int64(), Int64(11))
    assert_equal(result.item(0, "id").int64(), Int64(4))
    var chained = (
        source.lazy().sort("x").sort("keep", descending=True).select(["id"])
    )
    same(
        chained.collect(accelerator=runtime, engine="accel"),
        chained.collect(engine="cpu"),
    )
    var chained_radix = (
        source.lazy().sort("x").sort("id", descending=True).select(["id", "x"])
    )
    var radix_actual = chained_radix.collect(
        accelerator=runtime, engine="accel"
    )
    same(radix_actual, chained_radix.collect(engine="cpu"))
    var radix_profile = chained_radix.profile(
        accelerator=runtime, engine="accel"
    )
    var radix_peak = Int(
        radix_profile[1].item(0, "peak_requested_bytes").int64()
    )
    assert_true(radix_profile[1].item(0, "kernel_launches").int64() >= 100)
    with assert_raises(contains="budget"):
        _ = chained_radix.collect(
            accelerator=MetalRuntime(memory_limit_bytes=radix_peak - 1),
            engine="accel",
        )
    same(
        chained_radix.collect(
            accelerator=MetalRuntime(memory_limit_bytes=radix_peak),
            engine="accel",
        ),
        radix_actual,
    )
    var no_rows = source.lazy().filter(col("x") < 0).sort("x")
    assert_equal(
        no_rows.collect(accelerator=runtime, engine="accel").height(), 0
    )
    var overflow = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 1, 2])),
            Series("id", Column[Int64]([0, 1, 2])),
        ]
    )
    # Checked key errors are observed even when the key is later dropped:
    # the projection step reports them before the sort runs.
    var bad = (
        overflow.lazy()
        .with_columns((col("x") + 1).alias("key"))
        .sort("key")
        .select(["id"])
    )
    with assert_raises(contains="overflow"):
        _ = bad.collect(accelerator=runtime, engine="accel")
    var recovered = (
        overflow.lazy()
        .filter(col("id") > 0)
        .with_columns((col("x") + 1).alias("key"))
        .sort("key", descending=True)
        .select(["id"])
    )
    var safe = recovered.collect(accelerator=runtime, engine="accel")
    assert_equal(safe.item(0, "id").int64(), Int64(2))
    assert_equal(safe.item(1, "id").int64(), Int64(1))
    var profile = selected.profile(accelerator=runtime, engine="accel")
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    assert_true(peak > 0)
    with assert_raises(contains="budget"):
        _ = selected.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        selected.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        actual,
    )


def _tiled_integer_boundary(rows: Int) raises:
    var keys = List[Int64](length=rows, fill=0)
    var secondary = List[Int32](length=rows, fill=0)
    var valid = List[Bool](length=rows, fill=True)
    var ids = List[Int64](length=rows, fill=0)
    for row in range(rows):
        # Many equal composite keys straddle run and output-tile boundaries.
        keys[row] = Int64((rows - row + row // 13) % 19 - 9)
        secondary[row] = Int32((row * 7 + row // 5) % 5)
        valid[row] = row % 97 != 0
        ids[row] = Int64(row)
    var frame = DataFrame(
        [
            Series("key", Column[Int64](keys^, valid^)),
            Series("secondary", Column[Int32](secondary^)),
            Series("id", Column[Int64](ids^)),
        ]
    )
    var query = frame.lazy().sort(
        ["key", "secondary"],
        descending=[True, False],
        nulls_last=[True, False],
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    same(actual, query.collect(engine="cpu", streaming=False))


# requires: mojo_gpu
def test_mojo_gpu_ordering_tiled_merge_boundaries_ties_nulls_and_float_encodings() raises:
    assert_equal(merge_grid(1024, 512), 4)
    assert_equal(merge_grid(1025, 1024), 1)
    assert_equal(merge_grid(2047, 512), 8)
    assert_equal(merge_grid(2048, 1024), 1)
    assert_equal(merge_grid(4096, 2048), 2)
    assert_equal(merge_grid(4097, 4096), 4)
    for rows in [
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        8191,
        8192,
        8193,
        16385,
    ]:
        _tiled_integer_boundary(rows)

    # More than 256 radix tiles forces the recursive histogram-summary carry
    # path. The dominant key deliberately creates long, skewed stable runs.
    var radix_rows = 65_537
    var radix_keys = List[Int16](length=radix_rows, fill=0)
    var radix_valid = List[Bool](length=radix_rows, fill=True)
    var radix_ids = List[Int64](length=radix_rows, fill=0)
    for row in range(radix_rows):
        radix_keys[row] = Int16(0 if row % 101 else (row % 7) - 3)
        radix_valid[row] = row % 509 != 0
        radix_ids[row] = Int64(row)
    var radix_frame = DataFrame(
        [
            Series("key", Column[Int16](radix_keys^, radix_valid^)),
            Series("id", Column[Int64](radix_ids^)),
        ]
    )
    for reverse in [False, True]:
        var query = radix_frame.lazy().sort(
            "key", descending=reverse, nulls_last=not reverse
        )
        same(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu", streaming=False),
        )

    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000011,
        0xFFF8000000000022,
        1,
        0x8000000000000001,
        0x0010000000000000,
        0x8010000000000000,
    ]
    var floats = List[Float64](length=4097, fill=0)
    var secondary = List[Int16](length=4097, fill=0)
    var valid = List[Bool](length=4097, fill=True)
    var ids = List[Int64](length=4097, fill=0)
    for row in range(4097):
        floats[row] = bitcast[DType.float64](
            encodings[(row * 5 + row // 7) % len(encodings)]
        )
        secondary[row] = Int16((row * 3 + row // 11) % 7)
        valid[row] = row % 113 != 0
        ids[row] = Int64(row)
    var frame = DataFrame(
        [
            Series("key", Column[Float64](floats^, valid^)),
            Series("secondary", Column[Int16](secondary^)),
            Series("id", Column[Int64](ids^)),
        ]
    )
    var query = frame.lazy().sort(
        ["key", "secondary"],
        descending=[True, False],
        nulls_last=[False, True],
    )
    same(
        query.collect(accelerator=MetalRuntime(), engine="accel"),
        query.collect(engine="cpu", streaming=False),
    )
