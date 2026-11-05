"""Typed-column footprint checks and native Float32 transfer roundtrips."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, Series
from dataframe_metal.accelerator.gpu.query_scope import Scope
from dataframe_metal.accelerator.gpu.typed_storage import (
    native_column_bytes,
    check_native_column_capacity,
    upload_float32,
    readback_float32,
    wrap_native,
    _append_native_bits,
    _NativeReservation,
    NativeColumnAny,
)
from dataframe_metal.accelerator.gpu.erased_input import numeric_views
from tests.support.metal_outcomes import require_metal
from dataframe_metal.exact_float import finish_float_bins
from std.memory import bitcast


# requires: metal
def test_erased_native_views_keep_typed_reservations_through_completion() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2, 3])),
            Series("l", Column[Int64]([4, 5, 6])),
            Series("f", Column[Float32]([7, 8, 9])),
        ]
    )
    var scope = Scope(budget=48)
    var ctx = scope.context()
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var owners = List[NativeColumnAny]()
    var views = numeric_views(frame, "i", scope, False, empty, owners)
    views.append(
        numeric_views(frame, "l", scope, False, empty, owners)[0].copy()
    )
    views.append(
        numeric_views(frame, "f", scope, False, empty, owners)[0].copy()
    )
    assert_equal(scope.info()[0], 48)
    var tail = views[0].view(1, 2)
    _ = views^
    assert_equal(scope.info()[0], 48)
    with assert_raises(contains="memory budget"):
        _ = scope.check(1)
    var read = List[Int32](length=2, fill=0)
    ctx.enqueue_copy(
        read.unsafe_ptr(), tail.values.create_sub_buffer[DType.int32](0, 2)
    )
    scope.synchronize()
    assert_equal(read[0], Int32(2))
    assert_equal(read[1], Int32(3))
    _ = tail^
    _ = owners^
    assert_equal(scope.info()[0], 0)


def test_float32_exact_bins_normalize_carries_sign_and_rounding() raises:
    var bins = List[Int128](length=9, fill=0)
    bins[4] = Int128(1) << 21
    assert_equal(finish_float_bins(bins, False, False, False), Float64(1))
    bins[4] = -bins[4]
    assert_equal(finish_float_bins(bins, False, False, False), Float64(-1))
    bins = List[Int128](length=9, fill=0)
    bins[0] = -Int128(4294967295)
    bins[1] = 1
    assert_equal(
        finish_float_bins(bins, False, False, False),
        Float64(bitcast[DType.float32](UInt32(1))),
    )
    bins = List[Int128](length=9, fill=0)
    bins[7] = Int128(0xE0000000) * 2
    bins[8] = Int128(0x1FFFFF) * 2
    assert_equal(
        finish_float_bins(bins, False, False, False),
        Float64(Float32.MAX_FINITE) * 2,
    )
    bins = List[Int128](length=9, fill=0)
    bins[1] = Int128(1) << 21
    bins[0] = 1
    var scale = bitcast[DType.float64](UInt64(1023 - 149) << 52)
    assert_equal(
        finish_float_bins(bins, False, False, False),
        Float64(Int64(1) << 53) * scale,
    )
    bins[0] = 3
    assert_equal(
        finish_float_bins(bins, False, False, False),
        Float64((Int64(1) << 53) + 4) * scale,
    )


def test_native_bitmap_append_matches_bits_at_every_alignment_and_vector_tail() raises:
    var source = List[UInt8]()
    for i in range(64):
        source.append(UInt8((i * 37 + 93) % 256))
    for prefix in range(8):
        for offset in range(16):
            for count in [
                0,
                1,
                7,
                8,
                9,
                15,
                16,
                17,
                127,
                128,
                129,
                255,
                256,
                257,
            ]:
                var bits = List[UInt8](length=(prefix + 7) // 8, fill=255)
                var expected = List[UInt8](
                    length=(prefix + count + 7) // 8, fill=0
                )
                for i in range(prefix):
                    if i % 3 != 1:
                        expected[0] |= UInt8(1) << UInt8(i)
                    else:
                        bits[0] &= ~(UInt8(1) << UInt8(i))
                var before = bits.copy()
                for i in range(count):
                    if (
                        source[(offset + i) // 8] >> UInt8((offset + i) % 8)
                    ) & 1:
                        expected[(prefix + i) // 8] |= UInt8(1) << UInt8(
                            (prefix + i) % 8
                        )
                _append_native_bits(bits, prefix, source, offset, count)
                assert_equal(bits, before if count == 0 else expected)


# requires: metal
def test_native_wrapped_validity_preserves_page_and_bit_offsets_without_uploads() raises:
    _ = require_metal()
    var values = List[Float32]()
    var valid = List[Bool]()
    for i in range(400_013):
        values.append(Float32(i % 31))
        valid.append(i % 11 != 3)
    var source = Series("x", Column[Float32](values^, valid))
    # Cross bitmap pages and exercise each possible first-bit offset.
    for offset in [1, 7, 32_771, 262_149]:
        var view = source.slice(offset, 1003)
        var scope = Scope()
        var wrapped = wrap_native[DType.float32](
            scope, view, wrap_validity=True
        )
        assert_equal(Bool(wrapped), True)
        var restored = Series("x", readback_float32(wrapped.value()))
        assert_equal(restored.equals(view), True)
        assert_equal(scope.transfers()[0], 0)


def test_native_float32_column_capacity_includes_only_present_sidecars() raises:
    assert_equal(native_column_bytes[DType.float32](0), 0)
    assert_equal(native_column_bytes[DType.float32](5), 20)
    assert_equal(native_column_bytes[DType.float32](5, 2), 22)
    assert_equal(native_column_bytes[DType.float32](5, 2, 5), 42)
    with assert_raises(contains="allocation size overflow"):
        _ = native_column_bytes[DType.float32](Int.MAX)


def test_native_column_capacity_checks_budget_before_allocation() raises:
    var scope = Scope(budget=41)
    with assert_raises(contains="memory budget"):
        _ = check_native_column_capacity[DType.float32](scope, 5, 2, 5)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 0)
    assert_equal(
        check_native_column_capacity[DType.float32](Scope(), 5, 2, 5), 42
    )


def test_native_reservation_precedes_allocation_and_releases_shared_scope() raises:
    var scope = Scope(budget=8)
    var copied = scope.copy()
    var reservation = _NativeReservation(scope, 8)
    assert_equal(copied.info()[0], 8)
    with assert_raises(contains="memory budget"):
        _ = _NativeReservation(copied, 1)
    assert_equal(scope.info()[0], 8)
    assert_equal(scope.initialized(), False)
    _ = reservation^
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 8)
    var healthy = _NativeReservation(copied, 8)
    assert_equal(scope.info()[0], 8)
    _ = healthy^
    assert_equal(scope.info()[0], 0)


# requires: metal
def test_native_float32_upload_roundtrips_slice_and_packed_nulls() raises:
    _ = require_metal()
    var source = Series(
        "x",
        Column[Float32](
            [99, 1, 2, 3, 4, 88],
            [True, False, True, False, True, True],
        ),
    ).slice(1, 4)
    var scope = Scope()
    var resident = upload_float32(scope, source)
    assert_equal(resident.length, 4)
    assert_equal(resident.validity_bit_offset, 1)
    assert_equal(scope.transfers()[0], 17)
    var sliced = resident.view(1, 2)
    var result = readback_float32(sliced)
    assert_equal(result._get(0), Float32(2))
    assert_equal(result._get(1), Float32(3))
    assert_equal(result.is_valid(0), True)
    assert_equal(result.is_valid(1), False)
    assert_equal(scope.transfers()[1], 9)
    assert_equal(scope.info()[0], 17)
    _ = sliced^
    _ = resident^
    assert_equal(scope.info()[0], 0)


# requires: metal
def test_native_float32_upload_rechunks_only_validity_for_chunked_series() raises:
    _ = require_metal()
    var first = Series(
        "x", Column[Float32]([90, 1, 2], [True, False, True])
    ).slice(1, 2)
    var second = Series(
        "x", Column[Float32]([3, 4, 5, 99], [False, True, True, True])
    ).slice(0, 3)
    var source = Series._from_chunks([first^, second^])
    var scope = Scope()
    var resident = upload_float32(scope, source)
    assert_equal(resident.length, 5)
    assert_equal(resident.validity_bit_offset, 0)
    assert_equal(scope.transfers()[0], 21)
    var result = readback_float32(resident)
    assert_equal(result._get(0), Float32(1))
    assert_equal(result._get(1), Float32(2))
    assert_equal(result._get(2), Float32(3))
    assert_equal(result._get(3), Float32(4))
    assert_equal(result._get(4), Float32(5))
    assert_equal(result.is_valid(0), False)
    assert_equal(result.is_valid(1), True)
    assert_equal(result.is_valid(2), False)
    assert_equal(result.is_valid(3), True)
    assert_equal(result.is_valid(4), True)
    assert_equal(scope.transfers()[1], 21)


# requires: metal
def test_native_float32_empty_and_budget_failure_allocate_nothing() raises:
    _ = require_metal()
    var empty = Series("x", Column[Float32](List[Float32]()))
    var scope = Scope()
    var resident = upload_float32(scope, empty)
    assert_equal(resident.length, 0)
    assert_equal(resident.has_validity(), False)
    assert_equal(resident.has_errors(), False)
    assert_equal(scope.transfers()[0], 0)
    assert_equal(scope.initialized(), False)
    var empty_result = readback_float32(resident)
    assert_equal(len(empty_result), 0)
    assert_equal(scope.initialized(), False)

    var exact = Scope(budget=13)
    var nullable = Series("x", Column[Float32]([1, 2, 3], [True, False, True]))
    var exact_owner = upload_float32(exact, nullable)
    assert_equal(exact_owner.retained_bytes(), 13)
    assert_equal(exact.info()[0], 13)
    _ = exact_owner^
    assert_equal(exact.info()[0], 0)

    var too_small = Scope(budget=12)
    with assert_raises(contains="memory budget"):
        _ = upload_float32(too_small, nullable)
    assert_equal(too_small.initialized(), False)
    assert_equal(too_small.info()[0], 0)


# requires: metal
def test_native_float32_views_retain_one_owner_and_readback_checks_budget() raises:
    _ = require_metal()
    var source = Series("x", Column[Float32]([1, 2, 3, 4]))
    var scope = Scope()
    var original = upload_float32(scope, source)
    var view = original.view(1, 2)
    _ = original^
    assert_equal(scope.info()[0], 16)
    var result = readback_float32(view)
    assert_equal(result._get(0), Float32(2))
    assert_equal(result._get(1), Float32(3))
    _ = view^
    assert_equal(scope.info()[0], 0)

    var limited = Scope(budget=16)
    var retained = upload_float32(limited, source)
    with assert_raises(contains="memory budget"):
        _ = readback_float32(retained)
    assert_equal(limited.transfers()[1], 0)
    assert_equal(limited.info()[0], 16)
    _ = retained^
    assert_equal(limited.info()[0], 0)

    var all_valid = Series("x", Column[Float32]([1, 2, 3], [True, True, True]))
    var infallible = Scope()
    var native = upload_float32(infallible, all_valid)
    assert_equal(native.has_validity(), False)
    assert_equal(native.has_errors(), False)
    assert_equal(native.retained_bytes(), 12)
    assert_equal(infallible.transfers()[0], 12)


# requires: metal
def test_composed_exact_float_prefixes_cover_all_arities_and_wide_bins() raises:
    from dataframe import Expr, col
    from dataframe_metal.handoff import execute_timed

    _ = require_metal()
    for floating in [
        Float32(1.5),
        Float32.MAX_FINITE,
    ]:
        var frame = DataFrame(
            [
                Series(
                    "f", Column[Float32]([floating, floating], [True, False])
                ),
                Series("i", Column[Int32]([10, 20])),
                Series("l", Column[Int64]([4_000_000_000, 8_000_000_000])),
            ]
        )
        for arity in range(1, 5):
            for floats in range(1, arity + 1):
                var expressions = List[Expr]()
                # Present integer fields first to exercise alias remapping.
                for field in range(arity - 1, -1, -1):
                    var name = String("f") if field < floats else (
                        String("i") if field % 2 else String("l")
                    )
                    expressions.append(
                        (col(name) + field).mean().alias("mean" + String(field))
                    )
                expressions.append((col("f") + 0).count().alias("count"))
                var plan = frame.lazy().select_exprs(expressions)
                var expected = plan.collect(engine="cpu")
                for zero_copy in [True, False]:
                    var actual = execute_timed(plan, zero_copy=zero_copy)
                    assert_equal(actual[0].columns(), expected.columns())
                    for name in expected.columns():
                        assert_equal(
                            actual[0].item(0, name).is_null(),
                            expected.item(0, name).is_null(),
                        )
                        assert_equal(
                            actual[0].column(name).dtype(),
                            expected.column(name).dtype(),
                        )
                        assert_equal(
                            actual[0]
                            .column(name)
                            .equals(expected.column(name)),
                            True,
                        )


# requires: metal
def test_composed_float32_arithmetic_adjudicates_gradual_underflow() raises:
    from dataframe import Expr, col, lit
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for bits in [UInt32(1), UInt32(0x00800000), UInt32(0x00800001)]:
        var frame = DataFrame(
            [Series("f", Column[Float32]([bitcast[DType.float32](bits)]))]
        )
        var expressions: List[Expr]
        if bits == 1:
            expressions = [col("f") + 0, col("f") * 1]
        elif bits == 0x00800000:
            expressions = [col("f") * lit(Float32(0.5))]
        else:
            expressions = [
                col("f") - lit(bitcast[DType.float32](UInt32(0x00800000)))
            ]
        for expression in expressions:
            var plan = frame.lazy().select_exprs(
                [
                    expression.mean().alias("mean"),
                    col("f").count().alias("count"),
                ]
            )
            var expected = plan.collect(engine="cpu")
            with assert_raises(contains="CPU gradual underflow"):
                _ = execute_timed(plan)
            assert_equal(
                plan.collect(accelerator=runtime).equals(expected), True
            )


def test_float32_range_keys_match_float64_order_at_underflow_and_specials() raises:
    from dataframe_metal.accelerator.gpu.float_predicate import float_in_range

    var patterns = List[UInt32]()
    for bits in [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC12345,
    ]:
        patterns.append(UInt32(bits))
    for v in patterns:
        for lo in patterns:
            for hi in patterns:
                for inclusive in range(4):
                    var value = bitcast[DType.float32](v)
                    var low = bitcast[DType.float32](lo)
                    var high = bitcast[DType.float32](hi)
                    var x = Float64(value)
                    var a = Float64(low)
                    var b = Float64(high)
                    var above = x >= a if inclusive & 1 else x > a
                    var below = x <= b if inclusive & 2 else x < b
                    assert_equal(
                        float_in_range(value, low, high, UInt32(inclusive)),
                        above and below,
                    )


# requires: metal
def test_float32_underflow_predicates_match_cpu_in_all_reduction_paths() raises:
    from dataframe import Expr, col, lit
    from dataframe_metal.handoff import execute_timed

    _ = require_metal()
    var values = List[Float32]()
    var integers = List[Int32]()
    var floats = List[Float32]()
    for bits in [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
    ]:
        values.append(bitcast[DType.float32](UInt32(bits)))
        integers.append(Int32(len(integers) + 1))
        floats.append(Float32(len(floats) + 1) / 4)
    var frame = DataFrame(
        [
            Series("p", Column[Float32](values^)),
            Series("i", Column[Int32](integers^)),
            Series("f", Column[Float32](floats^)),
        ]
    )
    var tiny = Expr(Float64(bitcast[DType.float32](UInt32(1))))
    var negative_tiny = Expr(
        Float64(bitcast[DType.float32](UInt32(0x80000001)))
    )
    for predicate in [
        col("p") > 0,
        col("p") >= 0,
        col("p") < 0,
        col("p") <= 0,
        col("p") == 0,
        col("p") > tiny,
        col("p") == tiny,
        col("p") < negative_tiny,
        (col("p") >= negative_tiny) & (col("p") <= tiny),
    ]:
        var query = frame.lazy().filter(predicate)
        for mode in range(5):
            var expressions: List[Expr]
            if mode == 0:
                expressions = [col("i").sum().alias("s")]
            elif mode == 1:
                expressions = [
                    col("i").sum().alias("s"),
                    col("i").count().alias("c"),
                    col("i").mean().alias("m"),
                ]
            elif mode == 2:
                expressions = [
                    col("f").mean().alias("m"),
                    col("i").count().alias("c"),
                ]
            elif mode == 3:
                expressions = [(col("i") * 1).sum().alias("s")]
            else:
                expressions = [
                    (col("f") + 1).mean().alias("m"),
                    col("i").count().alias("c"),
                ]
            var plan = query.select_exprs(expressions)
            var expected = plan.collect(engine="cpu")
            for zero_copy in [True, False]:
                var actual = execute_timed(plan, zero_copy=zero_copy)
                assert_equal(actual[0].equals(expected), True)


# requires: metal
def test_single_float_composition_covers_mean_prefixes_and_native_sum_suffixes() raises:
    from dataframe import Expr, col
    from dataframe_metal.handoff import execute_timed
    from dataframe_metal.composition_plan import (
        plan_composition,
        single_float_composition,
    )

    _ = require_metal()
    for wide in [False, True]:
        var value = Float32.MAX_FINITE if wide else Float32(2)
        var frame = DataFrame(
            [
                Series("a", Column[Float32]([value, value], [True, False])),
                Series("b", Column[Float32]([1, 1])),
                Series("c", Column[Float32]([3, 4])),
                Series("d", Column[Float32]([5, 6])),
                Series("p", Column[Int32]([1, 1])),
            ]
        )
        for arity in range(1, 5):
            for means in range(arity + 1):
                for op in range(3):
                    var binary = col("a") + 0 if op == 0 else (
                        0 - col("a") if op == 1 else col("a") * col("b")
                    )
                    var expressions = List[Expr]()
                    for field in range(arity - 1, -1, -1):
                        var expression = binary.copy() if field == 0 else col(
                            "b" if field == 1 else ("c" if field == 2 else "d")
                        )
                        expressions.append(
                            (
                                expression.mean() if field
                                < means else expression.sum()
                            ).alias(String(field))
                        )
                    # COUNT shares the computed arithmetic state.
                    expressions.append(binary.count().alias("count"))
                    var plan = (
                        frame.lazy()
                        .filter(col("p") > 0)
                        .select_exprs(expressions)
                    )
                    var matched = plan_composition(plan)
                    assert_equal(
                        single_float_composition(matched.value()), True
                    )
                    assert_equal(
                        matched.value().fields[0].needs_mean, means > 0
                    )
                    var expected = plan.collect(engine="cpu")
                    for zero_copy in [True, False]:
                        var actual = execute_timed(plan, zero_copy=zero_copy)
                        assert_equal(actual[0].equals(expected), True)


# requires: metal
def test_single_float_composition_preserves_native_sum_underflow_and_native_means() raises:
    from dataframe import Expr, col
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series("a", Column[Float32]([2, 4])),
            Series("b", Column[Float32]([1, 1])),
            Series("c", Column[Float32]([tiny, tiny])),
        ]
    )
    var product = col("a") * col("b")
    var plan = frame.lazy().select_exprs(
        [product.mean().alias("m"), col("c").sum().alias("s")]
    )
    with assert_raises(contains="CPU gradual underflow"):
        _ = execute_timed(plan)
    assert_equal(
        plan.collect(accelerator=runtime).equals(plan.collect(engine="cpu")),
        True,
    )
    for variant in range(3):
        var expressions: List[Expr]
        if variant == 0:
            expressions = [product.sum().alias("s"), col("c").mean().alias("m")]
        elif variant == 1:
            expressions = [
                product.count().alias("c"),
                col("c").mean().alias("m"),
            ]
        else:
            expressions = [
                product.mean().alias("m"),
                col("c").count().alias("c"),
            ]
        var exact = frame.lazy().select_exprs(expressions)
        var actual = execute_timed(exact)
        assert_equal(actual[0].equals(exact.collect(engine="cpu")), True)


# requires: metal
def test_single_float_composition_reuses_float_predicates_and_nullable_operands() raises:
    from dataframe import Expr, col
    from dataframe_metal.handoff import execute_timed

    _ = require_metal()
    var frame = DataFrame(
        [
            Series("a", Column[Float32]([2, 4, 6], [True, False, True])),
            Series("b", Column[Float32]([1, 2, 3], [True, True, False])),
            Series(
                "c",
                Column[Float32](
                    [
                        bitcast[DType.float32](UInt32(1)),
                        bitcast[DType.float32](UInt32(1)),
                        bitcast[DType.float32](UInt32(1)),
                    ]
                ),
            ),
        ]
    )
    for predicate in ["a", "b", "c"]:
        for binary in [col("a") * col("b"), col("a") + 1, 1 - col("a")]:
            var plan = (
                frame.lazy()
                .filter(col(predicate) > 0)
                .select_exprs(
                    [
                        col("c").mean().alias("native_mean"),
                        binary.sum().alias("sum"),
                        binary.mean().alias("mean"),
                        col("b").count().alias("count"),
                    ]
                )
            )
            var expected = plan.collect(engine="cpu")
            for zero_copy in [True, False]:
                var actual = execute_timed(plan, zero_copy=zero_copy)
                assert_equal(actual[0].equals(expected), True)


def test_metal_region_fusion_recovers_cpu_common_expressions_without_pruning_work() raises:
    from dataframe import col
    from dataframe.lazy import WITH_COLUMNS
    from dataframe_metal.region_plan import normalize_region
    from dataframe_metal.composition_plan import plan_composition

    var frame = DataFrame(
        [
            Series("a", Column[Float32]([2, 4, 6])),
            Series("b", Column[Float32]([1, 2, 3])),
            Series("p", Column[Int32]([1, 1, 0])),
        ]
    )
    var value = col("a") * col("b")
    var query = (
        frame.lazy()
        .filter(col("p") > 0)
        .select_exprs(
            [
                value.sum().alias("s"),
                value.count().alias("c"),
                value.mean().alias("m"),
            ]
        )
    )
    var optimized = query._optimized()
    var shared = False
    for node in optimized._plan._nodes:
        shared = shared or node.kind == WITH_COLUMNS
    assert_equal(shared, True)
    var recovered = normalize_region(optimized)
    assert_equal(Bool(recovered), True)
    assert_equal(Bool(plan_composition(recovered.value())), True)
    assert_equal(
        recovered.value()
        .collect(engine="cpu", optimize=False)
        .equals(query.collect(engine="cpu")),
        True,
    )
    var unused = (
        frame.lazy()
        .with_columns((col("a") + col("b")).alias("unused"))
        .select(col("b").sum())
    )
    assert_equal(Bool(normalize_region(unused)), False)
    var dependent = (
        frame.lazy()
        .with_columns(
            [(col("a") + col("b")).alias("x"), (col("x") * 2).alias("y")]
        )
        .select_exprs([col("x").sum(), col("y").sum()])
    )
    assert_equal(Bool(normalize_region(dependent)), False)
    var before_filter = (
        frame.lazy()
        .with_columns(value.alias("v"))
        .filter(col("p") > 0)
        .select(col("v").sum())
    )
    assert_equal(Bool(normalize_region(before_filter)), False)


# requires: metal
def test_metal_composed_shared_arithmetic_stays_in_one_gpu_region() raises:
    from dataframe import col
    from dataframe_metal.handoff import supports, execute_timed
    from dataframe_metal.metal import MetalRuntime

    _ = require_metal()
    var frame = DataFrame(
        [
            Series("a", Column[Float32]([2, 4, 6], [True, False, True])),
            Series("b", Column[Float32]([1, 2, 3])),
            Series("p", Column[Int32]([1, 1, 0])),
        ]
    )
    var value = col("a") * col("b")
    var plan = (
        frame.lazy()
        .filter(col("p") > 0)
        .select_exprs(
            [
                value.sum().alias("s"),
                value.count().alias("c"),
                value.mean().alias("m"),
            ]
        )
    )
    var optimized = plan._optimized()
    assert_equal(supports(optimized), True)
    var expected = plan.collect(engine="cpu")
    var direct = execute_timed(optimized)
    assert_equal(direct[0].equals(expected), True)
    assert_equal(
        plan.collect(accelerator=MetalRuntime()).equals(expected), True
    )
    assert_equal(
        optimized.collect(accelerator=MetalRuntime()).equals(expected), True
    )


def _check_adjacent_native_chunks[D: DType]() raises:
    from std.memory import Pointer
    from dataframe import Column, Series
    from dataframe_metal.accelerator.gpu.typed_storage import (
        coalesced_native_chunks,
    )

    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for i in range(32):
        values.append(Scalar[D](i - 16))
        valid.append(i % 5 != 0)
    var source = Series("x", Column[Scalar[D]](values^, valid))
    var chunked = Series._from_chunks(
        [
            source.slice(3, 7),
            source.slice(10, 11),
            source.slice(21, 5),
        ]
    )
    var joined = coalesced_native_chunks[D](chunked)
    assert_equal(len(joined), 1)
    assert_equal(joined[0].equals(source.slice(3, 23)), True)
    var original = source.numeric[D]()
    var merged = joined[0].numeric[D]()
    assert_equal(
        Int(merged.unsafe_values()),
        Int(original.unsafe_values().unsafe_offset(3)),
    )
    assert_equal(
        Int(Pointer(to=merged._data[])), Int(Pointer(to=original._data[]))
    )
    assert_equal(
        Int(Pointer(to=merged._bits[])), Int(Pointer(to=original._bits[]))
    )
    # Storage identity and row adjacency are both required.
    var gaps = Series._from_chunks([source.slice(3, 7), source.slice(11, 5)])
    assert_equal(len(coalesced_native_chunks[D](gaps)), 2)
    var reverse = Series._from_chunks([source.slice(10, 5), source.slice(3, 7)])
    assert_equal(len(coalesced_native_chunks[D](reverse)), 2)
    var duplicate = Series._from_chunks(
        [source.slice(3, 7), source.slice(3, 7)]
    )
    assert_equal(len(coalesced_native_chunks[D](duplicate)), 2)
    var independent = Series(
        "x", Column[Scalar[D]](List[Scalar[D]](length=5, fill=0))
    )
    var separate = Series._from_chunks([source.slice(3, 7), independent^])
    assert_equal(len(coalesced_native_chunks[D](separate)), 2)
    # The same value allocation with a different bitmap must never coalesce.
    var other = Column[Scalar[D]](
        List[Scalar[D]](length=32, fill=0), List[Bool](length=32, fill=True)
    )
    var different = Column[Scalar[D]]._from_buffers(
        original._data, other._bits, 10, 11
    )
    var masks = Series._from_chunks(
        [source.slice(3, 7), Series("x", different^)]
    )
    var retained = coalesced_native_chunks[D](masks)
    assert_equal(len(retained), 2)
    assert_equal(retained[1].null_count(), 0)
    assert_equal(len(coalesced_native_chunks[D](source.slice(0, 0))), 0)
    assert_equal(source.n_chunks(), 1)
    assert_equal(chunked.n_chunks(), 3)


def test_native_chunk_coalescing_requires_shared_values_bitmaps_and_adjacent_rows() raises:
    _check_adjacent_native_chunks[DType.int32]()
    _check_adjacent_native_chunks[DType.int64]()
    _check_adjacent_native_chunks[DType.float32]()


# requires: metal
def test_native_coalesced_views_retain_one_typed_owner_through_readback() raises:
    from dataframe_metal.accelerator.gpu.erased_input import numeric_views
    from dataframe_metal.accelerator.gpu.typed_storage import NativeColumnAny

    var values = List[Float32]()
    var valid = List[Bool]()
    for i in range(32):
        values.append(Float32(i - 16))
        valid.append(i % 5 != 0)
    var source = Series("f", Column[Float32](values^, valid))
    var frame = DataFrame(
        [
            Series._from_chunks(
                [
                    source.slice(3, 7),
                    source.slice(10, 11),
                    source.slice(21, 5),
                ]
            )
        ]
    )
    var scope = Scope(budget=1 << 20)
    var ctx = scope.context()
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var owners = List[NativeColumnAny]()
    var views = numeric_views(frame, "f", scope, True, empty, owners)
    assert_equal(len(views), 1)
    assert_equal(len(owners), 1)
    assert_equal(views[0].rows, 23)
    var tail = views[0].view(1, 2)
    _ = views^
    assert_equal(scope.info()[0] > 0, True)
    var observed = List[Float32](length=2, fill=0)
    try:
        ctx.enqueue_copy(
            observed.unsafe_ptr(),
            tail.values.create_sub_buffer[DType.float32](0, 2),
        )
        scope.synchronize()
    except error:
        try:
            scope.synchronize()
        except:
            pass
        _ = (observed^, tail^, owners^, frame^, source^)
        raise error^
    assert_equal(observed[0], Float32(-12))
    assert_equal(observed[1], Float32(-11))
    _ = tail^
    _ = owners^
    assert_equal(scope.info()[0], 0)
    _ = (frame^, source^)
