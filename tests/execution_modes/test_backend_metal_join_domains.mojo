"""Native joins follow independent typed composite-key row-pair enumeration."""
# contract-group: execution.backends
from std.testing import assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, col
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import (
    require_metal,
    assert_native_report,
    NATIVE_NUMERIC,
)
from tests.support.metal_domain_checks import check_domain, assert_domain


def key_value[D: DType](row: Int, right: Bool) -> Scalar[D]:
    var left: List[Int] = [2, 1, 2, 3, 0, 7, 0]
    var other: List[Int] = [1, 2, 0, 9]
    return Scalar[D](other[row] if right else left[row])


def key_valid(row: Int, right: Bool) -> Bool:
    return row != (2 if right else 4)


def flag_value(row: Int, right: Bool) -> Bool:
    var left: List[Bool] = [False, True, False, True, True, False, True]
    var other: List[Bool] = [True, False, True, False]
    return other[row] if right else left[row]


def flag_valid(row: Int, right: Bool) -> Bool:
    return right or row != 6


def typed_key[
    D: DType
](
    name: String, positions: List[Int], right: Bool, dtype: DataType
) raises -> Series:
    var values = List[Scalar[D]]()
    var bits = List[Bool]()
    for row in positions:
        values.append(key_value[D](row, right) if row >= 0 else Scalar[D](0))
        bits.append(row >= 0 and key_valid(row, right))
    comptime if D == DType.bool:
        var booleans = List[Bool]()
        for value in values:
            booleans.append(Bool(value))
        return Series(name, Column[Bool](booleans^, bits^))
    else:
        return Series(name, Column[Scalar[D]](values^, bits^)).with_dtype(dtype)


def flags(name: String, positions: List[Int], right: Bool) raises -> Series:
    var values = List[Bool]()
    var bits = List[Bool]()
    for row in positions:
        values.append(row >= 0 and flag_value(row, right))
        bits.append(row >= 0 and flag_valid(row, right))
    return Series(name, Column[Bool](values^, bits^))


def payloads(name: String, positions: List[Int], right: Bool) raises -> Series:
    var values = List[Float32]()
    var bits = List[Bool]()
    for row in positions:
        values.append(Float32(row) + Float32(0.5))
        bits.append(row >= 0 and row != (2 if right else 1))
    return Series(name, Column[Float32](values^, bits^))


def ids(name: String, positions: List[Int], right: Bool) raises -> Series:
    var values = List[Int32]()
    var bits = List[Bool]()
    for row in positions:
        values.append(Int32(row + (10 if right else 0)))
        bits.append(row >= 0)
    return Series(name, Column[Int32](values^, bits^))


def join_frame[
    D: DType
](positions: List[Int], right: Bool, dtype: DataType) raises -> DataFrame:
    return DataFrame(
        [
            typed_key[D]("r" if right else "k", positions, right, dtype),
            flags("b", positions, right),
            payloads("v", positions, right),
            ids("id", positions, right),
        ]
    )


def pairs[
    D: DType
](left: List[Int], right: List[Int], how: String) -> Tuple[
    List[Int], List[Int]
]:
    var a = List[Int]()
    var b = List[Int]()
    var matched_right = List[Bool](length=len(right), fill=False)
    if how == "right":
        for r in right:
            var found = False
            for l in left:
                if (
                    key_valid(l, False)
                    and key_valid(r, True)
                    and flag_valid(l, False)
                    and flag_valid(r, True)
                    and key_value[D](l, False) == key_value[D](r, True)
                    and flag_value(l, False) == flag_value(r, True)
                ):
                    a.append(l)
                    b.append(r)
                    found = True
            if not found:
                a.append(-1)
                b.append(r)
        return (a^, b^)
    for l in left:
        var found = False
        for at in range(len(right)):
            var r = right[at]
            var equal = how == "cross" or (
                key_valid(l, False)
                and key_valid(r, True)
                and flag_valid(l, False)
                and flag_valid(r, True)
                and key_value[D](l, False) == key_value[D](r, True)
                and flag_value(l, False) == flag_value(r, True)
            )
            if equal:
                found = True
                matched_right[at] = True
                if how != "anti" and how != "semi":
                    a.append(l)
                    b.append(r)
        if how == "semi" and found:
            a.append(l)
            b.append(-1)
        elif (how == "anti" and not found) or (
            not found and how in ["left", "full"]
        ):
            a.append(l)
            b.append(-1)
    if how == "full":
        for at in range(len(right)):
            if not matched_right[at]:
                a.append(-1)
                b.append(right[at])
    return (a^, b^)


def expected_join[
    D: DType
](
    left: List[Int],
    right: List[Int],
    how: String,
    coalesce: Bool,
    dtype: DataType,
) raises -> DataFrame:
    var keys = List[Scalar[D]]()
    var key_bits = List[Bool]()
    var flag_values = List[Bool]()
    var flag_bits = List[Bool]()
    for at in range(len(left)):
        var use_right = (
            coalesce and how != "cross" and (how == "right" or left[at] < 0)
        )
        var row = right[at] if use_right else left[at]
        keys.append(key_value[D](row, use_right) if row >= 0 else Scalar[D](0))
        key_bits.append(row >= 0 and key_valid(row, use_right))
        flag_values.append(row >= 0 and flag_value(row, use_right))
        flag_bits.append(row >= 0 and flag_valid(row, use_right))
    var key_series: Series
    comptime if D == DType.bool:
        var booleans = List[Bool]()
        for value in keys:
            booleans.append(Bool(value))
        key_series = Series("k", Column[Bool](booleans^, key_bits^))
    else:
        key_series = Series(
            "k", Column[Scalar[D]](keys^, key_bits^)
        ).with_dtype(dtype)
    var columns: List[Series] = [
        key_series^,
        Series("b", Column[Bool](flag_values^, flag_bits^)),
        payloads("v", left, False),
        ids("id", left, False),
    ]
    if how != "semi" and how != "anti":
        if not coalesce or how == "cross":
            columns.append(typed_key[D]("r", right, True, dtype))
            columns.append(flags("b_right", right, True))
        columns.append(payloads("v_right", right, True))
        columns.append(ids("id_right", right, True))
    return DataFrame(columns^)


def typed_cases[
    D: DType
](runtime: MetalRuntime, native: Bool, dtype: DataType) raises:
    for empty_side in [-1, 0, 1]:
        var left: List[Int] = List[Int]() if empty_side == 0 else [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        var right: List[Int] = List[Int]() if empty_side == 1 else [0, 1, 2, 3]
        var a = join_frame[D](left, False, dtype)
        var b = join_frame[D](right, True, dtype)
        for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
            for coalesce in [False, True]:
                var rows = pairs[D](left, right, how)
                var query = a.lazy().join(
                    b.lazy(), how="cross"
                ) if how == "cross" else a.lazy().join(
                    b.lazy(),
                    left_on=["k", "b"],
                    right_on=["r", "b"],
                    how=how,
                    coalesce=coalesce,
                )
                check_domain(
                    query,
                    expected_join[D](rows[0], rows[1], how, coalesce, dtype),
                    runtime,
                    native,
                    submitted=len(left) > 0 and len(right) > 0,
                )


def all_typed_cases(runtime: MetalRuntime, native: Bool) raises:
    comptime for at in range(len(NATIVE_NUMERIC)):
        comptime D = NATIVE_NUMERIC[at]
        typed_cases[D](runtime, native, DataType.of(D))
    typed_cases[DType.bool](runtime, native, DataType.BOOL)
    typed_cases[DType.int64](runtime, native, DataType.DATE)
    typed_cases[DType.int32](runtime, native, DataType.decimal(9, 2, 32))


# requires: metal
def test_native_metal_typed_composite_join_styles_coalescing_and_empty_sides_match_scalar_pairs() raises:
    all_typed_cases(require_metal(), True)


def verify_metal_join_cpu_models() raises:
    all_typed_cases(MetalRuntime(), False)


# requires: metal
def test_native_metal_bounded_cross_product_heads_preserve_pair_order_without_materializing_fanout() raises:
    var runtime = require_metal()
    var values = List[Int32]()
    for row in range(40000):
        values.append(Int32(row))
    var source = DataFrame([Series("id", Column[Int32](values^))])
    var left = List[Int32](length=513, fill=0)
    var right = List[Int32]()
    for row in range(513):
        right.append(Int32(row))
    var expected = DataFrame(
        [
            Series("id", Column[Int32](left^)),
            Series("id_right", Column[Int32](right^)),
        ]
    )
    var budget = 8 * 1024 * 1024
    var bounded = MetalRuntime(memory_limit_bytes=budget)
    for shape in range(3):
        var query = source.lazy().join(source.lazy(), how="cross")
        if shape == 0:
            query = query.head(513)
        elif shape == 1:
            query = query.select_exprs([col("id"), col("id_right")]).head(513)
        else:
            query = query.head(513).select_exprs([col("id"), col("id_right")])
        var measured = query.profile(accelerator=bounded)
        assert_domain(measured[0], expected)
        assert_native_report(measured[1])
        for row in range(measured[1].height()):
            assert_true(
                measured[1].item(row, "query_peak_requested_bytes").int64()
                < Int64(budget)
            )
        with assert_raises(contains="memory budget"):
            _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    # CPU mode parity uses a bounded fixture. The large witness specifically
    # proves native allocation proportional to the requested prefix.
    var small = source.slice(0, 17)
    var small_left = List[Int32]()
    var small_right = List[Int32]()
    for a in range(17):
        for b in range(17):
            small_left.append(Int32(a))
            small_right.append(Int32(b))
    check_domain(
        small.lazy().join(small.lazy(), how="cross").head(513),
        DataFrame(
            [
                Series("id", Column[Int32](small_left^)),
                Series("id_right", Column[Int32](small_right^)),
            ]
        ),
        runtime,
        True,
    )
