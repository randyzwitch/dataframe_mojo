"""Dictionary-backed string keys preserve label identity in a Metal sum."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.storage.string import StringCodes
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.dictionary_pair_reduce import _combine_dictionary_stripes
from dataframe_metal.dictionary_group import (
    plan_dictionary_group,
    execute_dictionary_group,
    _signed_group_codes,
)
from dataframe_metal.accelerator.gpu.target import METAL_TARGET


def test_metal_dictionary_group_labels_nulls_and_order() raises:
    for repeated in [False, True]:
        var dictionary = CategoricalDictionary()
        dictionary.append("west")
        dictionary.append("west" if repeated else "east")
        var codes: List[UInt32] = [1, 0, 0, 1, 0]
        var dtype = DataType.categorical(dictionary^)
        var key = decode(
            Series(
                "k",
                Column[UInt32](codes.copy(), [True, False, True, True, True]),
            ).with_dtype(dtype)
        )
        var values = Series(
            "v",
            Column[Int64]([9, 4, 5, 3, 1], [True, True, False, True, False]),
        )
        var plain = (
            DataFrame([key.copy(), values.copy()])
            .lazy()
            .group_by("k")
            .agg([col("v").sum()])
        )
        assert_false(Bool(plan_dictionary_group(plain)))
        key._data[StringColumn]._codes = ArcPointer(
            StringCodes(ArcPointer(codes^), dtype)
        )
        var coded_frame = DataFrame([key^, values^])
        var plan = (
            coded_frame.lazy()
            .group_by("k", maintain_order=True)
            .agg([col("v").sum(min_count=1).alias("total")])
        )
        var matched = plan_dictionary_group(plan)
        if repeated:
            assert_false(Bool(matched))
        else:
            assert_true(Bool(matched))
            comptime if METAL_TARGET:
                var result = execute_dictionary_group(
                    plan, matched.value(), True
                )
                assert_true(result[0].equals(plan.collect(engine="cpu")))
                var plain_tail = DataFrame(
                    [
                        Series("k", Column[String](["other", "east"])),
                        Series("v", Column[Int64]([3, 2])),
                    ]
                )
                var mixed = (
                    coded_frame.vstack(plain_tail)
                    .lazy()
                    .group_by("k", maintain_order=True)
                    .agg([col("v").sum(min_count=1).alias("total")])
                )
                assert_false(Bool(plan_dictionary_group(mixed)))
                assert_true(
                    mixed.collect(accelerator=MetalRuntime()).equals(
                        mixed.collect(engine="cpu")
                    )
                )


def _coded_key(
    name: String,
    labels: List[String],
    var codes: List[UInt32],
    valid: List[Bool],
) raises -> Series:
    var dictionary = CategoricalDictionary()
    for label in labels:
        dictionary.append(label)
    var dtype = DataType.categorical(dictionary^)
    var key = decode(
        Series(name, Column[UInt32](codes.copy(), valid)).with_dtype(dtype)
    )
    key._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(codes^), dtype)
    )
    return key^


def test_metal_dictionary_pair_groups_by_both_labels() raises:
    var left = _coded_key(
        "l",
        ["west", "east"],
        [0, 1, 1, 0, 0, 1, 0],
        [True, True, False, True, False, True, True],
    )
    var right = _coded_key(
        "r",
        ["east", "west"],
        [0, 1, 0, 0, 1, 0, 0],
        [True, False, True, True, False, True, True],
    )
    # Independent bit offsets and chunk boundaries, including a repeated
    # tuple, nulls in either or both keys, and an all-null value group.
    var a = Series._from_chunks([left.slice(1, 3), left.slice(0, 4)])
    var b = Series._from_chunks([right.slice(2, 4), right.slice(0, 3)])
    var v = Series(
        "v",
        Column[Int32](
            [9, 4, 2, 1, 5, 0, 3], [True, False, True, True, True, False, True]
        ),
    )
    var frame = DataFrame([a^, b^, v^])
    var keys: List[String] = ["l", "r"]
    var total = col("v").sum(min_count=1).alias("total")
    var plan = (
        frame.lazy().group_by(keys, maintain_order=True).agg([total.copy()])
    )
    assert_true(
        plan_dictionary_group(plan).value().dense_dictionary(frame.height())
    )
    var direct = _signed_group_codes(left.slice(1, 3)).numeric[DType.int32]()
    ref string = left._data[StringColumn]
    assert_equal(
        Int(direct.unsafe_values()),
        Int(string._codes.value()[].codes.unsafe_ptr().unsafe_offset(1)),
    )
    assert_equal(direct.validity_offset(), 1)
    var incompatible = _coded_key(
        "r", ["west", "east"], [0, 1, 0], [True, True, True]
    )
    var mixed = frame.with_column(
        Series._from_chunks([right.slice(2, 4), incompatible^])
    )
    assert_false(
        Bool(
            plan_dictionary_group(
                mixed.lazy().group_by(keys).agg([total.copy()])
            )
        )
    )
    comptime if METAL_TARGET:
        var expected = plan.collect(engine="cpu")
        var profiled = plan.profile(accelerator=MetalRuntime())
        assert_true(profiled[0].equals(expected))
        var launches = Int64(0)
        for row in range(profiled[1].height()):
            launches += profiled[1].item(row, "gpu_completed_launches").int64()
        # Three intersections of the independent key and value chunk ranges.
        assert_equal(launches, Int64(2))
        var copied = execute_dictionary_group(
            plan, plan_dictionary_group(plan).value(), False
        )
        assert_true(copied[0].equals(expected))
        var declined = (
            mixed.lazy().group_by(keys, maintain_order=True).agg([total.copy()])
        )
        assert_true(
            declined.collect(accelerator=MetalRuntime()).equals(
                declined.collect(engine="cpu")
            )
        )


def test_metal_dense_pair_domain_and_wide_sums() raises:
    from std.testing import assert_raises

    var labels = List[String]()
    for i in range(255):
        labels.append(String(i))
    var keys: List[String] = ["a", "b"]
    var a = _coded_key("a", labels, [1, 1, 1, 0], [True, True, True, True])
    var b = _coded_key("b", labels, [0, 0, 0, 1], [True, True, True, True])
    var large = Int64(1) << 40
    var sums = Series(
        "v", Column[Int64]([large, -large, -1, 3], [True, True, True, False])
    )
    var frame = DataFrame([a.copy(), b.copy(), sums.copy()])
    var plan = (
        frame.lazy()
        .group_by(keys, maintain_order=True)
        .agg([col("v").sum(min_count=2).alias("s")])
    )
    var matched = plan_dictionary_group(plan)
    assert_true(matched.value().dense_dictionary(4))
    assert_false(matched.value().dense_dictionary(0))
    assert_false(matched.value().dense_dictionary(Int(Int32.MAX)))
    labels.append("outside")
    var wider = frame.with_column(
        _coded_key("a", labels, [1, 1, 1, 0], [True, True, True, True])
    )
    var sparse = (
        wider.lazy()
        .group_by(keys, maintain_order=True)
        .agg([col("v").sum(min_count=1)])
    )
    assert_false(plan_dictionary_group(sparse).value().dense_dictionary(4))
    comptime if METAL_TARGET:
        assert_true(
            plan.collect(accelerator=MetalRuntime()).equals(
                plan.collect(engine="cpu")
            )
        )
        assert_true(
            sparse.collect(accelerator=MetalRuntime()).equals(
                sparse.collect(engine="cpu")
            )
        )
        for negative in [False, True]:
            var boundary = Int64.MIN if negative else Int64.MAX
            var delta = Int64(-1) if negative else Int64(1)
            var bad = frame.with_column(
                Series("v", Column[Int64]([boundary, delta, 0, 0]))
            )
            var overflow = bad.lazy().group_by(keys).agg([col("v").sum()])
            with assert_raises(contains="int64 expression sum overflow"):
                _ = execute_dictionary_group(
                    overflow, plan_dictionary_group(overflow).value(), True
                )
            var narrow = frame.with_column(
                Series(
                    "v",
                    Column[Int32](
                        [
                            Int32.MIN if negative else Int32.MAX,
                            Int32(delta),
                            0,
                            0,
                        ]
                    ),
                )
            )
            var overflow32 = narrow.lazy().group_by(keys).agg([col("v").sum()])
            with assert_raises(contains="int32 expression sum overflow"):
                _ = execute_dictionary_group(
                    overflow32, plan_dictionary_group(overflow32).value(), True
                )


def test_dictionary_sums_keep_sparse_pairs_across_native_chunks() raises:
    from dataframe.execution_observation import DeviceObservation

    var labels = List[String]()
    for i in range(300):
        labels.append("value " + String(i))
    var l = _coded_key(
        "zone",
        labels,
        [299, 1, 7, 299, 0, 299, 1],
        [True, True, False, True, False, True, True],
    )
    var r = _coded_key(
        "kind",
        labels,
        [2, 0, 44, 2, 0, 2, 0],
        [True, False, True, True, False, True, False],
    )
    var v = Series("delta", Column[Int64]([11, 8, 3, -5, 4, 2, -4]))
    var frame = DataFrame(
        [
            Series._from_chunks([l.slice(0, 2), l.slice(3, 4)]),
            Series._from_chunks([r.slice(1, 3), r.slice(0, 3)]),
            Series._from_chunks([v.slice(2, 1), v.slice(0, 5)]),
        ]
    )
    var keys: List[String] = ["zone", "kind"]
    for count in [1, 2]:
        var selected = List[String]()
        for i in range(count):
            selected.append(keys[i].copy())
        var plan = (
            frame.lazy()
            .group_by(selected, maintain_order=True)
            .agg([col("delta").sum(min_count=1).alias("net")])
        )
        var matched = plan_dictionary_group(plan)
        assert_true(Bool(matched))
        assert_equal(
            matched.value().dense_dictionary(frame.height()), count == 1
        )
        comptime if METAL_TARGET:
            var expected = plan.collect(engine="cpu")
            for wrap in [False, True]:
                var obs = ArcPointer(DeviceObservation())
                plan._plan._device_observation = obs
                var actual = execute_dictionary_group(
                    plan, matched.value(), wrap
                )
                assert_true(actual[0].equals(expected))
                assert_true(obs[].completed_launches.load() > 0)
                if wrap:
                    assert_equal(obs[].upload_bytes.load(), Int64(0))


def test_sparse_dictionary_sum_all_null_tuple_and_highest_packed_code() raises:
    from dataframe.expr import SUM
    from dataframe_metal.dictionary_pair_reduce import execute_dictionary_sum
    from dataframe_metal.multi_plan import ReductionOutput

    comptime if METAL_TARGET:
        var hi = Int32(65534)
        var frame = DataFrame(
            [
                Series(
                    "a",
                    Column[Int32]([hi, 0, hi, hi], [True, False, True, False]),
                ),
                Series(
                    "b",
                    Column[Int32]([hi, hi, hi, 0], [True, False, True, False]),
                ),
                Series("v", Column[Int64]([9, 2, -4, -3])),
            ]
        )
        var output = ReductionOutput(0, "total", SUM, 1)
        var result = execute_dictionary_sum[False, True](
            frame, output, None, 65535, 65535, True
        )
        assert_equal(result[0].height(), 2)
        assert_equal(result[0].item(0, "a").int32(), hi)
        assert_equal(result[0].item(0, "b").int32(), hi)
        assert_equal(result[0].item(0, "total").int64(), Int64(5))
        assert_true(result[0].item(1, "a").is_null())
        assert_true(result[0].item(1, "b").is_null())
        assert_equal(result[0].item(1, "total").int64(), Int64(-1))


# requires: metal
def test_single_dictionary_sum_combines_full_width_addends_without_overflow() raises:
    comptime if METAL_TARGET:
        var keys = List[UInt32](length=129, fill=0)
        var valid = List[Bool](length=129, fill=True)
        var values = List[Int64](capacity=129)
        for row in range(128):
            values.append(Int64.MAX if row < 64 else -Int64.MAX)
        values.append(17)
        var key = _coded_key("key", ["hot"], keys^, valid^)
        var frame = DataFrame([key^, Series("v", Column[Int64](values^))])
        var plan = (
            frame.lazy()
            .group_by("key", maintain_order=True)
            .agg([col("v").sum(min_count=1).alias("total")])
        )
        var result = plan.collect(accelerator=MetalRuntime())
        assert_true(result.equals(plan.collect(engine="cpu")))
        assert_equal(
            result.column("total").numeric[DType.int64]().value(0), Int64(17)
        )


def test_dictionary_stripe_sampling_respects_keys_nulls_and_dispatch_boundaries() raises:
    var unique = List[Int32]()
    var local = List[Int32]()
    var striped = List[Int32]()
    var hot = List[Int32]()
    for i in range(1024):
        unique.append(Int32(i))
        local.append(Int32(i // 32))
        striped.append(Int32(i % 32))
        hot.append(1)
    var full: List[Int] = [1024]
    assert_false(
        _combine_dictionary_stripes(Series("key", Column[Int32](unique^)), full)
    )
    assert_false(
        _combine_dictionary_stripes(Series("key", Column[Int32](local^)), full)
    )
    var stripes = Series("key", Column[Int32](striped^))
    assert_true(_combine_dictionary_stripes(stripes, full))
    assert_false(
        _combine_dictionary_stripes(stripes, List[Int](length=32, fill=32))
    )
    var nulls = Series(
        "key", Column[Int32](hot^, List[Bool](length=1024, fill=False))
    )
    assert_true(_combine_dictionary_stripes(nulls, full))
    assert_false(_combine_dictionary_stripes(nulls.slice(0, 12), [12]))


# requires: metal
def test_single_dictionary_batches_offset_spans_for_direct_and_striped_keys() raises:
    from dataframe.execution_observation import DeviceObservation
    from dataframe.expr import SUM
    from dataframe_metal.dictionary_pair_reduce import execute_dictionary_sum
    from dataframe_metal.multi_plan import ReductionOutput

    comptime if METAL_TARGET:
        for striped in [False, True]:
            var keys = List[Series]()
            var start = 0
            for length in [193, 257, 79]:
                var data = List[Int32]()
                var valid = List[Bool]()
                var pad = 3 + start % 5
                for i in range(pad + length):
                    var row = start + i - pad
                    var key = Int32(row % 4 if striped else row % 197)
                    # Discover a fresh group in each of the second span and odd tail.
                    if row == 135 or row == 200 or row == 450:
                        key = Int32(row + 200)
                    data.append(key)
                    valid.append(row % 17 != 4)
                keys.append(
                    Series("k", Column[Int32](data^, valid^)).slice(pad, length)
                )
                start += length
            var values = List[Series]()
            start = 0
            for length in [135, 65, 329]:
                var data = List[Int64]()
                var valid = List[Bool]()
                var pad = 5 + start % 3
                for i in range(pad + length):
                    var row = start + i - pad
                    data.append(Int64(row % 13 - 6) * (Int64(1) << 34))
                    valid.append(row % 11 != 1)
                values.append(
                    Series("v", Column[Int64](data^, valid^)).slice(pad, length)
                )
                start += length
            var key = Series._from_chunks(keys^)
            var lengths: List[Int] = [135, 58, 7, 250, 79]
            assert_equal(_combine_dictionary_stripes(key, lengths), striped)
            var frame = DataFrame([key^, Series._from_chunks(values^)])
            var expected = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg([col("v").sum(min_count=1).alias("total")])
                .collect(engine="cpu")
            )
            var obs = ArcPointer(DeviceObservation())
            var actual = execute_dictionary_sum[True](
                frame, ReductionOutput(0, "total", SUM, 1), obs, 651, 0, True
            )
            assert_true(actual[0].equals(expected))
            assert_equal(obs[].completed_launches.load(), Int64(3))
