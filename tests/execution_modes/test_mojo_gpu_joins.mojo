"""Direct Mojo joins preserve typed pairs, IEEE keys and bounded cardinality."""
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col
from dataframe.metal import MetalRuntime
from dataframe.partition import _mix, _combine
from tests.support.metal_outcomes import same_outcome
from tests.execution_modes.test_backend_metal_join_domains import (
    join_frame,
    pairs,
    expected_join,
)
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.support.coverage_generators import offset_layout, chunk_layout


def _null_join_pairs(
    how: String, nulls_equal: Bool
) -> Tuple[List[Int], List[Int]]:
    var left_valid: List[Bool] = [False, False, True]
    var right_valid: List[Bool] = [False, True, False]
    var left_key: List[Int] = [0, 0, 1]
    var right_key: List[Int] = [0, 1, 0]
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    var matched_right = List[Bool](length=3, fill=False)
    if how == "right":
        for r in range(3):
            var found = False
            for l in range(3):
                var equal = left_valid[l] and right_valid[r]
                if nulls_equal:
                    equal = (left_valid[l] == right_valid[r]) and (
                        not left_valid[l] or left_key[l] == right_key[r]
                    )
                if equal:
                    left_rows.append(l)
                    right_rows.append(r)
                    found = True
            if not found:
                left_rows.append(-1)
                right_rows.append(r)
        return (left_rows^, right_rows^)
    for l in range(3):
        var found = False
        for r in range(3):
            var equal = left_valid[l] and right_valid[r]
            if nulls_equal:
                equal = (left_valid[l] == right_valid[r]) and (
                    not left_valid[l] or left_key[l] == right_key[r]
                )
            if equal:
                found = True
                matched_right[r] = True
                if how != "semi" and how != "anti":
                    left_rows.append(l)
                    right_rows.append(r)
        if how == "semi" and found:
            left_rows.append(l)
            right_rows.append(-1)
        elif (how == "anti" and not found) or (
            not found and (how == "left" or how == "full")
        ):
            left_rows.append(l)
            right_rows.append(-1)
    if how == "full":
        for r in range(3):
            if not matched_right[r]:
                left_rows.append(-1)
                right_rows.append(r)
    return (left_rows^, right_rows^)


def _assert_native_null_join(
    how: String, nulls_equal: Bool, runtime: MetalRuntime
) raises:
    var left = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 1], [False, False, True])),
            Series("li", Column[Int32]([10, 11, 12])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0], [False, True, False])),
            Series("ri", Column[Int32]([20, 21, 22])),
        ]
    )
    var query = left.lazy().join(right.lazy(), on="k", how=how, coalesce=False)
    # The execution plan carries this existing low-level keyed-join option
    # independently from the join style. Public joins currently leave it at
    # the default (false); setting the plan bit exercises both supported
    # values through the same public lazy query and native join router.
    query._nodes[len(query._nodes) - 1].flags = [nulls_equal]
    var measured = query.profile(accelerator=runtime, engine="accel")
    var actual = measured[0].copy()
    var expected = _null_join_pairs(how, nulls_equal)
    assert_equal(actual.height(), len(expected[0]))
    var joined = False
    for row in range(measured[1].height()):
        joined |= (
            measured[1].item(row, "algorithm").string()
            == "mojo_gpu_native_hash_join"
        )
    assert_true(joined)
    for row in range(actual.height()):
        var left_row = expected[0][row]
        assert_equal(actual.item(row, "li").is_null(), left_row < 0)
        if left_row >= 0:
            assert_equal(actual.item(row, "li").int32(), Int32(10 + left_row))
        if how != "semi" and how != "anti":
            var right_row = expected[1][row]
            assert_equal(actual.item(row, "ri").is_null(), right_row < 0)
            if right_row >= 0:
                assert_equal(
                    actual.item(row, "ri").int32(), Int32(20 + right_row)
                )


def _typed_join[D: DType](runtime: MetalRuntime, dtype: DataType) raises:
    var left_rows: List[Int] = [0, 1, 2, 3, 4, 5, 6]
    var right_rows: List[Int] = [0, 1, 2, 3]
    var a = offset_layout(join_frame[D](left_rows, False, dtype))
    var b = chunk_layout(join_frame[D](right_rows, True, dtype))
    for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
        var expected_pairs = pairs[D](left_rows, right_rows, how)
        var query = a.lazy().join(
            b.lazy(), how="cross"
        ) if how == "cross" else a.lazy().join(
            b.lazy(), left_on=["k", "b"], right_on=["r", "b"], how=how
        )
        same(
            query.collect(accelerator=runtime, engine="accel"),
            expected_join[D](
                expected_pairs[0], expected_pairs[1], how, True, dtype
            ),
        )
    for empty_side in range(3):
        var left = List[Int]() if empty_side != 2 else left_rows.copy()
        var right = List[Int]() if empty_side != 1 else right_rows.copy()
        var empty_a = join_frame[D](left, False, dtype)
        var empty_b = join_frame[D](right, True, dtype)
        for how in ["left", "right", "full", "anti"]:
            var expected_pairs = pairs[D](left, right, how)
            var query = empty_a.lazy().join(
                empty_b.lazy(),
                left_on=["k", "b"],
                right_on=["r", "b"],
                how=how,
                coalesce=False,
            )
            same(
                query.collect(accelerator=runtime, engine="accel"),
                expected_join[D](
                    expected_pairs[0], expected_pairs[1], how, False, dtype
                ),
            )


# requires: mojo_gpu
def test_mojo_gpu_join_styles_typed_composite_keys_coalescing_and_empty_sides() raises:
    var runtime = MetalRuntime()
    comptime types = (
        DType.float32,
        DType.int32,
        DType.int64,
        DType.int8,
        DType.int16,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float64,
    )
    comptime for at in range(len(types)):
        _typed_join[types[at]](runtime, DataType.of(types[at]))
    _typed_join[DType.bool](runtime, DataType.BOOL)
    _typed_join[DType.int64](runtime, DataType.DATE)


# requires: mojo_gpu
def test_mojo_gpu_join_nan_zero_subnormal_keys_and_exact_payloads() raises:
    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000001,
        0xFFF8000000000002,
        1,
        0x8000000000000001,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
    ]
    var values = List[Float64]()
    var other = List[Float64]()
    var valid = List[Bool]()
    var other_valid = List[Bool]()
    var ids = List[Int32]()
    for row in range(len(encodings)):
        values.append(bitcast[DType.float64](encodings[row]))
        other.append(
            bitcast[DType.float64](encodings[len(encodings) - row - 1])
        )
        valid.append(row != 8)
        other_valid.append(row != 8)
        ids.append(Int32(row))
    var left = offset_layout(
        DataFrame(
            [
                Series("k", Column[Float64](values.copy(), valid)),
                Series("id", Column[Int32](ids.copy())),
            ]
        )
    )
    var right = chunk_layout(
        DataFrame(
            [
                Series("k", Column[Float64](other.copy(), other_valid)),
                Series("id", Column[Int32](ids.copy())),
            ]
        )
    )
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        var a = List[Int]()
        var b = List[Int]()
        var matched_right = List[Bool](length=len(other), fill=False)
        if how == "right":
            for r in range(len(other)):
                var found = False
                for l in range(len(values)):
                    if (
                        valid[l]
                        and other_valid[r]
                        and (
                            values[l] == other[r]
                            or (isnan(values[l]) and isnan(other[r]))
                        )
                    ):
                        a.append(l)
                        b.append(r)
                        found = True
                if not found:
                    a.append(-1)
                    b.append(r)
        else:
            for l in range(len(values)):
                var found = False
                for r in range(len(other)):
                    if (
                        valid[l]
                        and other_valid[r]
                        and (
                            values[l] == other[r]
                            or (isnan(values[l]) and isnan(other[r]))
                        )
                    ):
                        found = True
                        matched_right[r] = True
                        if how != "semi" and how != "anti":
                            a.append(l)
                            b.append(r)
                if (
                    (how == "semi" and found)
                    or (how == "anti" and not found)
                    or (not found and (how == "left" or how == "full"))
                ):
                    a.append(l)
                    b.append(-1)
            if how == "full":
                for r in range(len(other)):
                    if not matched_right[r]:
                        a.append(-1)
                        b.append(r)
        var query = left.lazy().join(
            right.lazy(), on="k", how=how, coalesce=False
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.height(), len(a))
        for row in range(len(a)):
            assert_equal(actual.item(row, "id").is_null(), a[row] < 0)
            if a[row] >= 0:
                assert_equal(
                    actual.item(row, "id").numeric[DType.int32](), Int32(a[row])
                )
                assert_equal(actual.item(row, "k").is_null(), not valid[a[row]])
                if valid[a[row]]:
                    assert_equal(
                        bitcast[DType.uint64](actual.item(row, "k").float64()),
                        encodings[a[row]],
                    )
            else:
                assert_true(actual.item(row, "k").is_null())
            if how != "semi" and how != "anti":
                assert_equal(actual.item(row, "id_right").is_null(), b[row] < 0)
                if b[row] >= 0:
                    assert_equal(
                        actual.item(row, "id_right").numeric[DType.int32](),
                        Int32(b[row]),
                    )
                    assert_equal(
                        actual.item(row, "k_right").is_null(),
                        not other_valid[b[row]],
                    )
                    if other_valid[b[row]]:
                        assert_equal(
                            bitcast[DType.uint64](
                                actual.item(row, "k_right").float64()
                            ),
                            encodings[len(encodings) - b[row] - 1],
                        )
        same(actual, query.collect(engine="cpu", streaming=False))


# requires: mojo_gpu
def test_mojo_gpu_native_hash_join_float32_encodings_and_nulls() raises:
    var left_bits: List[UInt32] = [
        0x00000000,
        0x80000000,
        0x7FC00001,
        0xFFC00002,
        0x00000001,
        0x80000001,
        0x40400000,
    ]
    var right_bits: List[UInt32] = [
        0x80000000,
        0x00000000,
        0xFFC00003,
        0x7FC01234,
        0x80000001,
        0x00000001,
        0x40A00000,
    ]
    var left_keys = List[Float32]()
    var right_keys = List[Float32]()
    for bits in left_bits:
        left_keys.append(bitcast[DType.float32](bits))
    for bits in right_bits:
        right_keys.append(bitcast[DType.float32](bits))
    var validity: List[Bool] = [True, True, True, True, True, True, False]
    var left_ids: List[Int32] = [10, 11, 12, 13, 14, 15, 16]
    var right_ids: List[Int32] = [20, 21, 22, 23, 24, 25, 26]
    var left = DataFrame(
        [
            Series("k", Column[Float32](left_keys^, validity.copy())),
            Series("left_id", Column[Int32](left_ids^)),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Float32](right_keys^, validity.copy())),
            Series("right_id", Column[Int32](right_ids^)),
        ]
    )
    var query = left.lazy().join(
        right.lazy(), on="k", how="inner", coalesce=False
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    var native_row = _profile_row_for_algorithm(
        measured[1], "mojo_gpu_native_hash_join"
    )
    var actual = measured[0].copy()
    assert_equal(actual.height(), 10)

    # The expected relation explicitly covers both zeros, distinct NaN
    # payloads, both subnormal signs, and excludes the null rows.
    var expected_left: List[Int] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5]
    var expected_right: List[Int] = [0, 1, 0, 1, 2, 3, 2, 3, 5, 4]
    var observed = List[Bool](length=10, fill=False)
    for row in range(actual.height()):
        var li = Int(actual.item(row, "left_id").int32()) - 10
        var ri = Int(actual.item(row, "right_id").int32()) - 20
        var expected_at = -1
        for at in range(len(expected_left)):
            if expected_left[at] == li and expected_right[at] == ri:
                expected_at = at
        assert_true(expected_at >= 0)
        assert_true(not observed[expected_at])
        observed[expected_at] = True
        assert_equal(
            bitcast[DType.uint32](actual.item(row, "k").float32()),
            left_bits[li],
        )
        assert_equal(
            bitcast[DType.uint32](actual.item(row, "k_right").float32()),
            right_bits[ri],
        )
    for found in observed:
        assert_true(found)
    assert_true(
        measured[1].item(native_row, "native_hash_phase_launches").int64() > 0
    )
    same(actual, query.collect(engine="cpu", streaming=False))
    print("PASS native Float32 NaN/zero/subnormal/null keys and payload bits")


# requires: mojo_gpu
def test_mojo_gpu_native_hash_join_null_equality_all_keyed_styles() raises:
    var runtime = MetalRuntime()
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        _assert_native_null_join(how, False, runtime)
        _assert_native_null_join(how, True, runtime)
    print(
        "PASS Mojo GPU native hash joins: null equality follows the keyed option for every join style"
    )


def _collision_fallback_query(how: String) raises -> LazyFrame:
    var keys = List[Int32]()
    var candidate = 0
    var outsider = Int32(-1)
    while len(keys) < 128 or outsider < 0:
        var hash_value = _mix(_combine(UInt64(1), UInt64(candidate)))
        if UInt32(hash_value) & UInt32(1023) == 0:
            if len(keys) < 128:
                keys.append(Int32(candidate))
            else:
                outsider = Int32(candidate)
        candidate += 1
    var build_valid = List[Bool](length=129, fill=True)
    var probe_valid: List[Bool] = [True, False]
    keys.append(0)
    build_valid[128] = False
    var probe_keys: List[Int32] = [outsider, 0]
    var build_ids = List[Int32](length=129, fill=Int32(0))
    for row in range(128):
        build_ids[row] = Int32(100 + row)
    build_ids[128] = 228
    var probe_ids: List[Int32] = [10, 11]
    var left = DataFrame(
        [
            Series("k", Column[Int32](probe_keys^, probe_valid^)),
            Series("li", Column[Int32](probe_ids^)),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int32](keys^, build_valid^)),
            Series("ri", Column[Int32](build_ids^)),
        ]
    )
    var query = left.lazy().join(right.lazy(), on="k", how=how, coalesce=False)
    query._nodes[len(query._nodes) - 1].flags = [True]
    return query^


# requires: mojo_gpu
def test_mojo_gpu_native_hash_join_collision_fallback_preserves_null_equal() raises:
    var runtime = MetalRuntime()
    for how in ["inner", "full"]:
        var query = _collision_fallback_query(how)
        var measured = query.profile(accelerator=runtime, engine="accel")
        var actual = measured[0].copy()
        assert_equal(actual.height(), 1 if how == "inner" else 130)
        var sorted_fallback = False
        for row in range(measured[1].height()):
            sorted_fallback |= (
                measured[1].item(row, "algorithm").string()
                == "mojo_gpu_sorted_join"
            )
        assert_true(sorted_fallback)
        var null_pair = False
        for row in range(actual.height()):
            if (
                not actual.item(row, "li").is_null()
                and actual.item(row, "li").int32() == 11
            ):
                assert_true(not actual.item(row, "ri").is_null())
                assert_equal(actual.item(row, "ri").int32(), Int32(228))
                null_pair = True
        assert_true(null_pair)
    print(
        "PASS Mojo GPU native hash join: probe exhaustion falls back with null-equal semantics"
    )


def _profile_row_for_algorithm(
    report: DataFrame, algorithm: String
) raises -> Int:
    for row in range(report.height()):
        if report.item(row, "algorithm").string() == algorithm:
            return row
    raise Error("Expected join algorithm is absent from profile")


# requires: mojo_gpu
def test_mojo_gpu_native_hash_join_empty_and_full_profile_launch_counts() raises:
    var runtime = MetalRuntime()
    var empty = DataFrame(
        [
            Series("k", Column[Int32](List[Int32]())),
            Series("x", Column[Float64](List[Float64]())),
        ]
    )
    var nonempty = DataFrame(
        [Series("k", Column[Int32]([1])), Series("x", Column[Float64]([2]))]
    )
    var both_empty = (
        empty.lazy()
        .join(empty.lazy(), on="k", how="inner", coalesce=False)
        .profile(accelerator=runtime, engine="accel")
    )
    assert_equal(both_empty[0].height(), 0)
    var both_empty_row = _profile_row_for_algorithm(
        both_empty[1], "mojo_gpu_sorted_join"
    )
    assert_equal(
        both_empty[1].item(both_empty_row, "algorithm").string(),
        "mojo_gpu_sorted_join",
    )

    var empty_probe = (
        empty.lazy()
        .join(nonempty.lazy(), on="k", how="inner", coalesce=False)
        .profile(accelerator=runtime, engine="accel")
    )
    assert_equal(empty_probe[0].height(), 0)
    var empty_probe_row = _profile_row_for_algorithm(
        empty_probe[1], "mojo_gpu_native_hash_join"
    )
    assert_equal(
        empty_probe[1]
        .item(empty_probe_row, "native_hash_phase_launches")
        .int64(),
        # initialize hash + build hash/members + three slot scans + offsets
        # + one key pack; the empty probe skips both probe kernels.
        Int64(8),
    )

    var empty_build = (
        nonempty.lazy()
        .join(empty.lazy(), on="k", how="inner", coalesce=False)
        .profile(accelerator=runtime, engine="accel")
    )
    assert_equal(empty_build[0].height(), 0)
    var empty_build_row = _profile_row_for_algorithm(
        empty_build[1], "mojo_gpu_native_hash_join"
    )
    assert_equal(
        empty_build[1]
        .item(empty_build_row, "native_hash_phase_launches")
        .int64(),
        # initialize hash + three slot scans + probe/hash scan/offsets
        # + one key pack; no build-row or pair-map kernels run.
        Int64(8),
    )

    var full_profile = (
        nonempty.lazy()
        .join(nonempty.lazy(), on="k", how="full", coalesce=False)
        .profile(accelerator=runtime, engine="accel")
    )
    assert_equal(full_profile[0].height(), 1)
    var full_row = _profile_row_for_algorithm(
        full_profile[1], "mojo_gpu_native_hash_join"
    )
    assert_equal(
        full_profile[1].item(full_row, "native_hash_phase_launches").int64(),
        # initialize/build/members (3), slot scans (3), probe/hash scan (2),
        # offsets (1), full unmatched path (4), map emit (1), key packs (2).
        Int64(16),
    )
    print("PASS native hash join empty/full profile launch counts")


# requires: mojo_gpu
def test_mojo_gpu_native_hash_join_mixed_payload_budget_and_recovery() raises:
    var left = DataFrame(
        [
            Series("k", Column[Float32]([1.0, 1.0, 2.0])),
            Series("amount", Column[Float64]([10.5, 11.5, 12.5])),
            Series("day", Column[Int64]([100, 101, 102])).with_dtype(
                DataType.DATE
            ),
            Series("small", Column[Int8]([1, 2, 3])),
            Series("left_id", Column[Int32]([10, 11, 12])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Float32]([1.0, 2.0, 2.0])),
            Series("amount_r", Column[Float32]([20.5, 21.5, 22.5])),
            Series("day_r", Column[Int64]([200, 201, 202])).with_dtype(
                DataType.DATE
            ),
            Series("small_r", Column[Int8]([4, 5, 6])),
            Series("right_id", Column[Int32]([20, 21, 22])),
        ]
    )
    var query = left.lazy().join(
        right.lazy(), on="k", how="full", coalesce=False
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    assert_equal(measured[0].height(), 4)
    var native_row = _profile_row_for_algorithm(
        measured[1], "mojo_gpu_native_hash_join"
    )
    var peak = Int(measured[1].item(native_row, "peak_requested_bytes").int64())
    assert_true(peak > 0)
    assert_equal(measured[0].column("day").dtype(), DataType.DATE)
    assert_equal(measured[0].column("day_r").dtype(), DataType.DATE)
    var cpu = query.collect(engine="cpu", streaming=False)
    same(measured[0], cpu)
    var expected_left_amount: List[Float64] = [10.5, 11.5, 12.5]
    var expected_right_amount: List[Float32] = [20.5, 21.5, 22.5]
    var expected_left_day: List[Int64] = [100, 101, 102]
    var expected_right_day: List[Int64] = [200, 201, 202]
    for row in range(measured[0].height()):
        var left_id = Int(measured[0].item(row, "left_id").int32())
        var right_id = Int(measured[0].item(row, "right_id").int32())
        var left_index = left_id - 10
        var right_index = right_id - 20
        assert_true(
            (left_index < 2 and right_index == 0)
            or (left_index == 2 and (right_index == 1 or right_index == 2))
        )
        assert_equal(
            measured[0].item(row, "k").float32(),
            Float32(1) if left_index < 2 else Float32(2),
        )
        assert_equal(
            measured[0].item(row, "k_right").float32(),
            Float32(1) if right_index == 0 else Float32(2),
        )
        assert_equal(
            measured[0].item(row, "amount").float64(),
            expected_left_amount[left_index],
        )
        assert_equal(
            measured[0].item(row, "amount_r").float32(),
            expected_right_amount[right_index],
        )
        assert_equal(
            measured[0].item(row, "day").to_physical(),
            expected_left_day[left_index],
        )
        assert_equal(
            measured[0].item(row, "day_r").to_physical(),
            expected_right_day[right_index],
        )
        assert_equal(
            measured[0].item(row, "small").int8(), Int8(left_index + 1)
        )
        assert_equal(
            measured[0].item(row, "small_r").int8(), Int8(right_index + 4)
        )
    var limited_runtime = MetalRuntime(memory_limit_bytes=peak)
    var bounded = query.profile(accelerator=limited_runtime, engine="accel")
    assert_equal(bounded[0].height(), 4)
    var bounded_peak = 0
    for row in range(bounded[1].height()):
        bounded_peak = max(
            bounded_peak,
            Int(bounded[1].item(row, "peak_requested_bytes").int64()),
        )
    assert_true(bounded_peak <= peak)
    same(bounded[0], cpu)
    var ids = List[Int32]()
    for row in range(512):
        ids.append(Int32(row))
    var source = DataFrame([Series("id", Column[Int32](ids^))])
    var too_large = source.lazy().join(source.lazy(), how="cross")
    with assert_raises(contains="memory budget"):
        _ = too_large.collect(accelerator=limited_runtime, engine="accel")
    var tiny = (
        left.head(1)
        .lazy()
        .join(right.head(1).lazy(), on="k", how="inner", coalesce=False)
    )
    var recovered = tiny.collect(accelerator=limited_runtime, engine="accel")
    same(recovered, tiny.collect(engine="cpu", streaming=False))
    print("PASS native hash join mixed payload budgets and recovery")


# requires: mojo_gpu
def test_mojo_gpu_join_bounded_cross_fanout_and_exact_memory_budget() raises:
    var ids = List[Int32]()
    for row in range(50_000):
        ids.append(Int32(row))
    var source = DataFrame([Series("id", Column[Int32](ids^))])
    var query = source.lazy().join(source.lazy(), how="cross").head(513)
    var runtime = MetalRuntime(memory_limit_bytes=8 * 1024 * 1024)
    var profile = query.profile(accelerator=runtime, engine="accel")
    assert_equal(profile[0].height(), 513)
    for row in range(513):
        assert_equal(
            profile[0].item(row, "id").numeric[DType.int32](), Int32(0)
        )
        assert_equal(
            profile[0].item(row, "id_right").numeric[DType.int32](), Int32(row)
        )
    var peak = 0
    for row in range(profile[1].height()):
        peak = max(
            peak, Int(profile[1].item(row, "peak_requested_bytes").int64())
        )
    assert_true(peak <= 8 * 1024 * 1024)
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        profile[0],
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
    )
    assert_equal(
        query.head(0).collect(accelerator=runtime, engine="accel").height(), 0
    )


# requires: mojo_gpu
def test_mojo_gpu_join_regions_consumers_retained_budget_and_preflight() raises:
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 2, 3])),
            Series("x", Column[Float64]([10, 20, 21, 30])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64]([2, 2, 4])),
            Series("y", Column[Float64]([200, 201, 400])),
        ]
    )
    var a = left.lazy().with_columns((col("x") + 1).alias("x"))
    var b = right.lazy().with_columns((col("y") + 1).alias("y"))
    var query = (
        a.join(b, on="k")
        .with_columns((col("x") + col("y")).alias("z"))
        .filter(col("z") > 222)
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    same(measured[0], query.collect(engine="cpu", streaming=False))
    assert_equal(measured[0].height(), 3)
    assert_equal(measured[0].item(0, "z").float64(), Float64(223))
    assert_equal(measured[0].item(1, "z").float64(), Float64(223))
    assert_equal(measured[0].item(2, "z").float64(), Float64(224))
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var joined = False
    var retained = False
    for row in range(measured[1].height()):
        var local = Int(measured[1].item(row, "peak_requested_bytes").int64())
        var live = Int(
            measured[1].item(row, "retained_intermediate_bytes").int64()
        )
        # A region's peak includes the query's peak so far (#740), which
        # already counts the intermediates retained across it.
        assert_true(peak >= local and local >= live)
        joined |= (
            measured[1].item(row, "algorithm").string()
            == "mojo_gpu_sorted_join"
        )
        retained |= live > 0
        assert_equal(
            measured[1].item(row, "runtime_implementation").string(), "mojo"
        )
    assert_true(joined and retained)
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        measured[0],
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
    )
    var reduced = query.select_exprs(
        [
            col("z").sum().alias("s"),
            col("z").mean().alias("m"),
            col("z").len().alias("n"),
        ]
    )
    var result = reduced.collect(accelerator=MetalRuntime(), engine="accel")
    same(result, reduced.collect(engine="cpu", streaming=False))
    assert_equal(result.item(0, "s").float64(), Float64(670))
    assert_equal(result.item(0, "n").int64(), Int64(3))
    var third = DataFrame(
        [Series("k", Column[Int64]([2])), Series("q", Column[Int64]([-10]))]
    )
    var chained = query.join(third.lazy(), on="k").select_exprs(
        [col("z"), col("q")]
    )
    same(
        chained.collect(accelerator=MetalRuntime(), engine="accel"),
        chained.collect(engine="cpu", streaming=False),
    )
    var floats = DataFrame([Series("v", Column[Float64]([1, 2, 3]))])
    var scalar = (
        floats.lazy()
        .select(col("v").sum())
        .with_columns((col("v") + 1).alias("v"))
    )
    assert_equal(
        scalar.collect(accelerator=MetalRuntime(), engine="accel")
        .item()
        .float64(),
        Float64(7),
    )
    var bad = DataFrame(
        [
            Series("k", Column[Int64]([1, 2])),
            Series("x", Column[Int64]([Int64.MAX, 1])),
        ]
    )
    var overflowing = (
        bad.lazy()
        .with_columns((col("x") + 1).alias("x"))
        .join(right.lazy(), on="k", how="left")
        .head(0)
    )
    same_outcome(overflowing)
    var unsupported = left.lazy().join(
        right.lazy().with_columns(col("y").cast(DataType.STRING).alias("y")),
        on="k",
    )
    with assert_raises(contains="unsupported"):
        _ = unsupported.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
    assert_true(
        "Mojo GPU joins" in MetalRuntime(device_id=999999).describe(query)
    )
    print(
        "PASS Mojo GPU DAG flows: branches, post-join rows/reductions, chained joins, scalar consumer, retained budget, preflight and checked errors"
    )


# requires: mojo_gpu
def test_mojo_gpu_join_empty_projection_preserves_cardinality_and_zero_column_contract() raises:
    var left = DataFrame([Series("k", Column[Int64]([1, 2, 2, 3]))])
    var right = DataFrame([Series("k", Column[Int64]([2, 2, 4]))])
    var styles: List[String] = [
        "inner",
        "left",
        "right",
        "full",
        "semi",
        "anti",
        "cross",
    ]
    var heights: List[Int] = [4, 6, 5, 7, 2, 2, 12]
    for at in range(len(styles)):
        var query = left.lazy().join(right.lazy(), how="cross") if styles[
            at
        ] == "cross" else left.lazy().join(right.lazy(), on="k", how=styles[at])
        var empty = query.select(List[String]())
        var actual = empty.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.shape(), (heights[at], 0))
        assert_equal(
            empty.head(1)
            .collect(accelerator=MetalRuntime(), engine="accel")
            .shape(),
            (1, 0),
        )
        assert_equal(
            empty.head(0)
            .collect(accelerator=MetalRuntime(), engine="accel")
            .shape(),
            (0, 0),
        )
        assert_true(actual.equals(empty.collect(engine="cpu", streaming=False)))
    # The shared/native join contract requires an internal output column.
    # A keyed empty projection still retains its keys inside the join; a
    # zero-column cross join has no such internal field and rejects before
    # device initialization, independently of its cardinality limit.
    var zero_left = DataFrame(List[Series](), height=50_000)
    var zero_right = DataFrame(List[Series](), height=50_000)
    var zero_cross = (
        zero_left.lazy().join(zero_right.lazy(), how="cross").head(513)
    )
    with assert_raises(contains="join requires 1"):
        _ = zero_cross.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
