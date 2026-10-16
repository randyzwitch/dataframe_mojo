"""Resident union, distinct and slicing obey independently selected positions."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    StringColumn,
    col,
    lit,
)
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.metal_domain_checks import check_domain
from tests.support.metal_domain_reference import (
    slice_positions,
    distinct_positions,
)
from tests.support.coverage_compare import assert_frame


def resident_rows(indices: List[Int]) raises -> DataFrame:
    var raw_x: List[Int16] = [5, 1, 99, 2, 3, 1]
    var raw_s: List[String] = ["é", "a", "é", "b", "poison", "b"]
    var raw_f: List[UInt32] = [
        0x80000000,
        0x7FC00011,
        0,
        0x3F800000,
        1,
        0xBF800000,
    ]
    var id = List[Int32]()
    var x = List[Int16]()
    var x_bits = List[Bool]()
    var s = List[String]()
    var s_bits = List[Bool]()
    var f = List[Float32]()
    var u = List[UInt64]()
    var d = List[Int64]()
    var d_bits = List[Bool]()
    var decimal = List[Int32]()
    var flags = List[Bool]()
    for at in indices:
        id.append(Int32(at))
        x.append(raw_x[at])
        x_bits.append(at != 2)
        s.append(raw_s[at].copy())
        s_bits.append(at != 4)
        f.append(bitcast[DType.float32](raw_f[at]))
        u.append(UInt64.MAX - UInt64(at))
        d.append(Int64.MIN if at == 1 else Int64(at - 2))
        d_bits.append(at != 1)
        decimal.append(Int32((at - 2) * 100))
        flags.append(at % 2 == 0)
    return DataFrame(
        [
            Series("id", Column[Int32](id^)),
            Series("x", Column[Int16](x^, x_bits^)),
            Series("s", StringColumn(s^, s_bits^)),
            Series("f", Column[Float32](f^)),
            Series("__dataframe_concat_side", Column[UInt64](u^)),
            Series("date", Column[Int64](d^, d_bits^)).with_dtype(
                DataType.DATE
            ),
            Series("decimal", Column[Int32](decimal^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("x__dataframe_concat_right", Column[Bool](flags^)),
        ]
    )


def union_and_slice_cases(runtime: MetalRuntime, native: Bool) raises:
    var all_rows: List[Int] = [0, 1, 2, 3, 4, 5]
    var source = resident_rows(all_rows)
    for left_n in [0, 2, 6]:
        for right_n in [0, 3, 6]:
            var left_rows = slice_positions(all_rows, 0, left_n)
            var right_rows = slice_positions(all_rows, 0, right_n)
            var combined = left_rows.copy()
            combined.extend(right_rows.copy())
            var right = (
                source.slice(0, right_n)
                .lazy()
                .select(
                    [
                        "s",
                        "id",
                        "date",
                        "decimal",
                        "f",
                        "x",
                        "__dataframe_concat_side",
                        "x__dataframe_concat_right",
                    ]
                )
            )
            var plan = source.slice(0, left_n).lazy().concat(right)
            check_domain(
                plan,
                resident_rows(combined),
                runtime,
                native,
                submitted=len(combined) > 0,
            )
    var combined: List[Int] = [0, 1, 2, 3, 4, 5, 1, 2, 0, 1]
    var chain = (
        source.lazy()
        .concat(source.slice(1, 2).lazy())
        .concat(source.slice(0, 2).lazy())
    )
    for offset in [Int.MIN, -20, -10, -3, -1, 0, 1, 8, 10, 20, Int.MAX]:
        for length in [-1, 0, 1, 4, Int.MAX]:
            var expected_rows = slice_positions(combined, offset, length)
            check_domain(
                chain.slice(offset, length),
                resident_rows(expected_rows),
                runtime,
                native,
                submitted=len(expected_rows) > 0,
            )
    var after = chain.filter(col("x").is_not_null()).sort("id").slice(-3, 2)
    check_domain(
        after,
        resident_rows([3, 4]),
        runtime,
        native,
        resident=True,
        dictionary=True,
    )
    var sum = chain.slice(-3, 2).select_exprs(
        [col("x").sum().alias("sum"), col("x").count().alias("count")]
    )
    check_domain(
        sum,
        DataFrame(
            [
                Series("sum", Column[Int64]([5])),
                Series("count", Column[Int64]([1])),
            ]
        ),
        runtime,
        native,
        resident=True,
    )
    if native:
        with assert_raises(contains="Slice length must be nonnegative"):
            _ = chain.slice(1, -2).collect(accelerator=runtime)
        with assert_raises(contains="budget"):
            _ = chain.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
        check_domain(
            chain,
            resident_rows(combined),
            runtime,
            native,
            resident=True,
            dictionary=True,
        )


# requires: metal
def test_native_metal_nested_union_and_extreme_slices_preserve_all_typed_payloads_and_order() raises:
    union_and_slice_cases(require_metal(), True)


def distinct_cases(runtime: MetalRuntime, native: Bool) raises:
    var sequence: List[Int] = [0, 1, 2, 3, 4, 5, 1, 2, 0, 1]
    var keys: List[Int] = [0, 1, 0, 2, -1, 2, 1, 0, 0, 1]
    var source = resident_rows([0, 1, 2, 3, 4, 5])
    var chain = (
        source.lazy()
        .concat(source.slice(1, 2).lazy())
        .concat(source.slice(0, 2).lazy())
    )
    for keep in [String("first"), String("last"), String("none")]:
        var positions = distinct_positions(keys, keep)
        var rows = List[Int]()
        for position in positions:
            rows.append(sequence[position])
        var unique = chain.unique(["s"], keep=keep, maintain_order=True)
        check_domain(
            unique,
            resident_rows(rows),
            runtime,
            native,
            resident=True,
            dictionary=True,
        )
        var sliced = slice_positions(rows, -3, 2)
        check_domain(
            unique.slice(-3, 2),
            resident_rows(sliced),
            runtime,
            native,
            submitted=len(sliced) > 0,
        )
    var nan = bitcast[DType.float32](UInt32(0x7FC00035))
    var float_keys = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [nan, -0.0, nan, 0.0, 5, 99],
                    [True, True, True, True, True, False],
                ),
            )
        ]
    )
    for keep in [String("first"), String("last"), String("none")]:
        var values: List[Float32] = [nan, -0.0, 5, 0]
        var bits: List[Bool] = [True, True, True, False]
        if keep == "last":
            values = [nan, 0.0, 5, 0]
        elif keep == "none":
            values = [5, 0]
            bits = [True, False]
        check_domain(
            float_keys.lazy().unique(keep=keep, maintain_order=True),
            DataFrame([Series("f", Column[Float32](values^, bits^))]),
            runtime,
            native,
        )
    # `any` promises one member of each equality class, not a specific payload
    # representative. The selected row id lets us check the entire source row.
    if native:
        var result = chain.unique(
            ["s"], keep="any", maintain_order=True
        ).collect(accelerator=runtime)
        assert_equal(result.height(), 4)
        var seen: List[Bool] = [False, False, False, False]
        var ids = List[Int]()
        for row in range(result.height()):
            var id = Int(result.item(row, "id").int32())
            assert_true(id >= 0 and id < 6)
            var group = (
                0 if id == 0
                or id == 2 else 1 if id == 1 else 2 if id == 3
                or id == 5 else 3
            )
            assert_true(not seen[group])
            seen[group] = True
            ids.append(id)
        assert_frame(result, resident_rows(ids))


# requires: metal
def test_native_metal_distinct_keep_modes_use_logical_null_nan_zero_classes_and_current_slice_positions() raises:
    distinct_cases(require_metal(), True)


def verify_metal_resident_cpu_models() raises:
    var runtime = MetalRuntime()
    union_and_slice_cases(runtime, False)
    distinct_cases(runtime, False)
