"""Dictionary-backed string keys preserve label identity in a Metal sum."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.storage.string import StringCodes
from dataframe_metal.metal import MetalRuntime
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
    assert_true(plan_dictionary_group(plan).value().dense_pair(frame.height()))
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
        assert_equal(launches, Int64(3))
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
    assert_true(matched.value().dense_pair(4))
    assert_false(matched.value().dense_pair(0))
    assert_false(matched.value().dense_pair(Int(Int32.MAX)))
    labels.append("outside")
    var wider = frame.with_column(
        _coded_key("a", labels, [1, 1, 1, 0], [True, True, True, True])
    )
    var sparse = (
        wider.lazy()
        .group_by(keys, maintain_order=True)
        .agg([col("v").sum(min_count=1)])
    )
    assert_false(plan_dictionary_group(sparse).value().dense_pair(4))
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
