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
    assert_true(Bool(plan_dictionary_group(plan)))
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
        assert_true(launches > 0)
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
