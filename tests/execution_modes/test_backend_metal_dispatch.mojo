"""Native dispatch mechanisms must retain complete stable scalar semantics."""
from std.memory import bitcast
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.metal_domain_checks import check_domain


def predicate_case(runtime: MetalRuntime, native: Bool) raises:
    var keys = List[Int32]()
    var values = List[Int16]()
    var bits = List[Bool]()
    var total = Int64(0)
    var count = Int64(0)
    var length = Int64(0)
    for row in range(521):
        keys.append(Int32(row % 5))
        values.append(Int16(row % 13 if row % 7 != 3 else 99))
        bits.append(row % 7 != 3)
        if row >= 5 and row % 5 >= 2 and row % 5 <= 3:
            length += 1
            if row % 7 != 3:
                count += 1
                total += Int64(row % 13)
    var source = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Int16](values^, bits^)),
        ]
    ).slice(5, 516)
    var plan = (
        source.lazy()
        .filter(col("k") >= lit(Int32(2)))
        .filter(col("k") <= lit(Int32(3)))
        .select_exprs(
            [
                col("x").sum().alias("sum"),
                col("x").count().alias("count"),
                col("x").len().alias("len"),
            ]
        )
    )
    check_domain(
        plan,
        DataFrame(
            [
                Series("sum", Column[Int64]([total])),
                Series("count", Column[Int64]([count])),
                Series("len", Column[Int64]([length])),
            ]
        ),
        runtime,
        native,
        mechanism="mojo_gpu_reductions",
    )


def test_native_metal_filtered_scalar_reductions_observe_predicates_without_row_compaction() raises:
    predicate_case(require_metal(), True)


def hash_group_cases(runtime: MetalRuntime, native: Bool) raises:
    for n in [4095, 4096, 4097]:
        var keys = List[Int32]()
        var key_bits = List[Bool]()
        var flags = List[Bool]()
        var values = List[Int16]()
        var bits = List[Bool]()
        var order = List[Int]()
        var sums = List[Int64]()
        var counts = List[Int64]()
        var lengths = List[Int64]()
        for row in range(n):
            var key = -1 if row % 19 == 0 else row % 17
            var flag = row % 2
            var code = (key + 1) * 2 + flag
            var at = -1
            for candidate in range(len(order)):
                if order[candidate] == code:
                    at = candidate
                    break
            if at < 0:
                at = len(order)
                order.append(code)
                sums.append(0)
                counts.append(0)
                lengths.append(0)
            lengths[at] += 1
            if row % 7 != 3:
                sums[at] += Int64(row % 5)
                counts[at] += 1
            keys.append(Int32(max(0, key)))
            key_bits.append(key >= 0)
            flags.append(flag != 0)
            values.append(Int16(row % 5 if row % 7 != 3 else 99))
            bits.append(row % 7 != 3)
        var source = DataFrame(
            [
                Series("k", Column[Int32](keys^, key_bits^)),
                Series("flag", Column[Bool](flags^)),
                Series("x", Column[Int16](values^, bits^)),
            ]
        )
        var expected_keys = List[Int32]()
        var expected_valid = List[Bool]()
        var expected_flags = List[Bool]()
        for code in order:
            expected_keys.append(Int32(max(0, code // 2 - 1)))
            expected_valid.append(code // 2 > 0)
            expected_flags.append(code % 2 != 0)
        var expected = DataFrame(
            [
                Series("k", Column[Int32](expected_keys^, expected_valid^)),
                Series("flag", Column[Bool](expected_flags^)),
                Series("sum", Column[Int64](sums^)),
                Series("count", Column[Int64](counts^)),
                Series("len", Column[Int64](lengths^)),
            ]
        )
        var mechanism = String("mojo_gpu_grouped_rows")
        check_domain(
            source.lazy()
            .group_by(["k", "flag"], maintain_order=True)
            .agg(
                [
                    col("x").sum().alias("sum"),
                    col("x").count().alias("count"),
                    col("x").len().alias("len"),
                ]
            ),
            expected,
            runtime,
            native,
            mechanism=mechanism,
        )


def test_native_metal_group_dispatch_boundary_preserves_nullable_composite_first_seen_groups() raises:
    hash_group_cases(require_metal(), True)


def sort_rows(indices: List[Int]) raises -> DataFrame:
    var key = List[UInt64]()
    var bits = List[Bool]()
    var tie = List[Int8]()
    var id = List[Int32]()
    for row in indices:
        key.append(UInt64.MAX - UInt64(row % 17))
        bits.append(row % 23 != 0)
        tie.append(Int8(row % 5))
        id.append(Int32(row))
    return DataFrame(
        [
            Series("k", Column[UInt64](key^, bits^)),
            Series("t", Column[Int8](tie^)),
            Series("id", Column[Int32](id^)),
        ]
    )


def radix_sort_cases(runtime: MetalRuntime, native: Bool) raises:
    for n in [65535, 65536, 65537]:
        for composite in [False, True]:
            var input_rows = List[Int]()
            for row in range(n):
                input_rows.append(row)
            var expected_rows = List[Int]()
            # Enumerate finite key classes independently of merge/radix sorting.
            # A single key preserves source order within every null/value class;
            # the composite order uses the second key before the stable row id.
            for key in range(-1, 17):
                for tie in range(5 if composite else 1):
                    for row in range(n):
                        var candidate = -1 if row % 23 == 0 else row % 17
                        if candidate == key and (
                            not composite or row % 5 == tie
                        ):
                            expected_rows.append(row)
            # Both simple and composite keys use the actual stable Mojo
            # ordering kernels, retaining exact UInt64 words at every size.
            var mechanism = String("mojo_gpu_ordered_rows")
            var keys: List[String] = ["k", "t"] if composite else ["k"]
            var descending: List[Bool] = [True, False] if composite else [True]
            var nulls_last: List[Bool] = [False, True] if composite else [False]
            check_domain(
                sort_rows(input_rows)
                .lazy()
                .sort(keys, descending=descending, nulls_last=nulls_last),
                sort_rows(expected_rows),
                runtime,
                native,
                mechanism=mechanism,
            )


def test_native_metal_integer_sort_dispatch_boundary_keeps_exact_words_null_placement_and_stable_ties() raises:
    radix_sort_cases(require_metal(), True)


def float_join_cases(runtime: MetalRuntime, native: Bool) raises:
    var nan = bitcast[DType.float32](UInt32(0x7FC00031))
    var left = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [nan, -0.0, 1, nan, 99], [True, True, True, True, False]
                ),
            ),
            Series("a", Column[Int32]([0, 1, 2, 3, 4])),
        ]
    )
    var right = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [nan, 0.0, nan, 2, 99], [True, True, True, True, False]
                ),
            ),
            Series("b", Column[Int32]([10, 11, 12, 13, 14])),
        ]
    )
    var plan = left.lazy().join(right.lazy(), "k", how="full")
    var expected = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [nan, nan, -0.0, 1, nan, nan, 0, 2, 0],
                    [True, True, True, True, True, True, False, True, False],
                ),
            ),
            Series(
                "a",
                Column[Int32](
                    [0, 0, 1, 2, 3, 3, 4, 0, 0],
                    [True, True, True, True, True, True, True, False, False],
                ),
            ),
            Series(
                "b",
                Column[Int32](
                    [10, 12, 11, 0, 10, 12, 0, 13, 14],
                    [True, True, True, False, True, True, False, True, True],
                ),
            ),
        ]
    )
    check_domain(plan, expected, runtime, native)


def test_native_metal_float_join_keys_canonicalize_nan_and_signed_zero_without_matching_nulls() raises:
    float_join_cases(require_metal(), True)


def verify_metal_dispatch_cpu_models() raises:
    var runtime = MetalRuntime()
    predicate_case(runtime, False)
    hash_group_cases(runtime, False)
    radix_sort_cases(runtime, False)
    float_join_cases(runtime, False)
