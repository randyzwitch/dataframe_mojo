"""Dictionary-backed string keys preserve label identity in a Metal sum."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.storage.string import StringCodes
from dataframe_metal.dictionary_group import (
    plan_dictionary_group,
    execute_dictionary_group,
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
        var plan = (
            DataFrame([key^, values^])
            .lazy()
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
