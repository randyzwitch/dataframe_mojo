"""The radix-partitioned aggregate table against a scalar model of each group,
through both routes that dispatch to it: the
eager group-by over many groups, and a pipeline's grouped sink. Keys cover
short, long, empty, Unicode and null strings, null integers, -0.0/0.0 and NaN
floats and null Booleans; layouts cover contiguous, offset and unequally
chunked columns and strings carrying Parquet dictionary codes beside plain
ones. Worker counts cover tables that clear and keep their rows (three or
more) and tables that grow (two). Decimal sums and means (decimal32, 64
and 128 inputs) keep exact 128-bit totals on both routes and raise past
their precision."""
# contract-group: reduce.grouped
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_raises, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    concat,
    lit,
)
from tests.support.aggregate_table_reference import (
    aggregate_frame,
    assert_close,
    assert_groups,
    coded_strings,
    group_models,
    group_of,
    key_text,
    model_key_text,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def reductions() -> List[Expr]:
    return [
        col("x").sum(min_count=1).alias("x_sum"),
        col("x").mean().alias("x_mean"),
        col("x").min().alias("x_min"),
        col("x").max().alias("x_max"),
        col("x").count().alias("x_count"),
        col("w").sum(min_count=1).alias("w_sum"),
        col("w").null_count().alias("w_nulls"),
        col("x").len().alias("rows"),
        col("y").mean().alias("y_mean"),
        col("y").std().alias("y_std"),
        col("y").median().alias("y_median"),
        col("x").quantile(0.25).alias("x_q25"),
    ]


def eager_route(threads: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 72_000
    var groups = 6_000
    var raw = aggregate_frame(rows, groups)
    var models = group_models(rows, groups)
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var observed = StrategyTrace()
        var actual = frame.group_by(["s", "i", "f", "b"]).agg(reductions())
        assert_true(
            "dataframe-path: group_by.aggregate_table\n" in observed.text(),
            "the eager group-by did not reach the aggregate table",
        )
        _ = observed^
        assert_groups(actual, models, True)
    _ = trace^
    _ = workers^


def test_eager_route_with_tables_that_clear_matches_scalar_groups_on_every_layout() raises:
    eager_route(8)


def test_eager_route_with_tables_that_grow_matches_scalar_groups_on_every_layout() raises:
    eager_route(2)


def test_pipeline_sink_after_a_filter_matches_scalar_groups_on_every_layout() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 72_000
    var groups = 6_000
    var start = 9_001
    var raw = aggregate_frame(rows, groups)
    var models = group_models(rows, groups, start)
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var observed = StrategyTrace()
        var actual = (
            frame.lazy()
            .filter(col("r") >= lit(Int64(start)))
            .group_by(["s", "i", "f", "b"])
            .agg(reductions())
            .collect()
        )
        assert_true(
            "dataframe-path: pipeline.group_table\n" in observed.text(),
            "the pipeline did not sink into the aggregate table",
        )
        _ = observed^
        assert_groups(actual, models, True)
    _ = trace^
    _ = workers^


def test_dictionary_coded_and_plain_string_chunks_form_the_same_groups() raises:
    # The first half's keys carry dictionary codes and are hashed through
    # their dictionary's entries; the second half's are plain strings. A
    # group with rows in both halves must come back once.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 72_000
    var groups = 6_000
    var raw = aggregate_frame(rows, groups)
    var half = rows // 2
    var first = raw.slice(0, half).rechunk()
    var columns = first._columns.copy()
    columns[0] = coded_strings(first.column("s"))
    var coded = DataFrame(columns^, height=half)
    var mixed = concat([coded.copy(), raw.slice(half, rows - half).rechunk()])
    var models = group_models(rows, groups)
    assert_groups(
        mixed.group_by(["s", "i", "f", "b"]).agg(reductions()), models, True
    )
    assert_groups(
        mixed.lazy()
        .filter(col("r") >= lit(Int64(0)))
        .group_by(["s", "i", "f", "b"])
        .agg(reductions())
        .collect(),
        models,
        True,
    )
    _ = workers^


def test_tables_that_fill_many_times_merge_every_group_once() raises:
    # 60K groups over three workers: each worker's sink table fills,
    # grows or clears many times, and a group's rows end up in several
    # workers' partitions before the finish merges them.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "3")
    var rows = 240_000
    var groups = 60_000
    var frame = aggregate_frame(rows, groups)
    assert_groups(
        frame.group_by(["s", "i"]).agg(reductions()),
        group_models(rows, groups),
        False,
    )
    _ = workers^


def test_few_valued_keys_forming_many_groups_reach_the_table() raises:
    # A brand, a type and a size, each with few values, whose product is
    # under the few-groups bound but which form thousands of groups among
    # the rows (PDS-H q16's keys): the table takes them, as it does in a
    # pipeline, rather than per-range states that number each range's
    # strings again.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 120_000
    var brands = List[String]()
    var types = List[String]()
    var sizes = List[Int64]()
    var values = List[Int64]()
    for row in range(rows):
        brands.append("Brand#" + String(row % 25))
        types.append("STANDARD POLISHED TIN " + String((row // 25) % 150))
        sizes.append(Int64((row // 3750) % 8))
        values.append(Int64(row % 1000))
    var frame = DataFrame(
        [
            Series("brand", Column[String](brands^)),
            Series("type", Column[String](types^)),
            Series("size", Column[Int64](sizes^)),
            Series("v", Column[Int64](values^)),
        ]
    )
    var observed = StrategyTrace()
    var grouped = frame.group_by(["brand", "type", "size"]).agg(
        [col("v").sum().alias("total"), col("v").len().alias("rows")]
    )
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "few-valued keys forming many groups did not reach the table",
    )
    _ = observed^
    _ = trace^
    var totals = Dict[String, Int]()
    var counts = Dict[String, Int]()
    for row in range(rows):
        var key = (
            "Brand#"
            + String(row % 25)
            + "|STANDARD POLISHED TIN "
            + String((row // 25) % 150)
            + "|"
            + String((row // 3750) % 8)
        )
        totals[key] = totals.get(key, 0) + row % 1000
        counts[key] = counts.get(key, 0) + 1
    assert_equal(grouped.height(), len(totals))
    for r in range(grouped.height()):
        var key = (
            grouped.item(r, "brand").string()
            + "|"
            + grouped.item(r, "type").string()
            + "|"
            + String(grouped.item(r, "size").int64())
        )
        assert_true(key in totals, "unexpected group " + key)
        assert_equal(grouped.item(r, "total").int64(), Int64(totals[key]))
        assert_equal(grouped.item(r, "rows").int64(), Int64(counts[key]))
    _ = workers^


def decimal_value(row: Int) -> Int64:
    return Int64((row * 104729) % 200_001 - 100_000)


def decimal_frame(rows: Int, groups: Int) raises -> DataFrame:
    """Keys s and i of each row's group, then the same unscaled values as
    decimal32 (7, 2), decimal64 (15, 2) and decimal128 (30, 4) columns,
    each null on its own rows, and r, the row's index."""
    var keys = aggregate_frame(rows, groups)
    var d32 = List[Int32]()
    var v32 = List[Bool]()
    var d64 = List[Int64]()
    var v64 = List[Bool]()
    var d128 = List[Int128]()
    var v128 = List[Bool]()
    for row in range(rows):
        var value = decimal_value(row)
        d32.append(Int32(value))
        v32.append(row % 11 != 4)
        d64.append(value)
        v64.append(row % 13 != 6)
        d128.append(Int128(value) * 1_000_000_000_000_000)
        v128.append(row % 7 != 2)
    return DataFrame(
        [
            keys.column("s"),
            keys.column("i"),
            Series("d32", Column[Int32](d32^, v32)).with_dtype(
                DataType.decimal(7, 2, 32)
            ),
            Series("d64", Column[Int64](d64^, v64)).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("d128", Column[Int128](d128^, v128)).with_dtype(
                DataType.decimal(30, 4)
            ),
            keys.column("r"),
        ]
    )


def unscaled(column: Series, row: Int) raises -> Int128:
    var flat = column.rechunk() if column.is_chunked() else column.copy()
    return flat._data[Column[Int128]]._get(row)


def assert_decimal_groups(
    actual: DataFrame, rows: Int, groups: Int, start: Int
) raises:
    """Each group's decimal sums (exact, unscaled) and means against Int128
    scalar totals over rows `start` on, in any order."""
    var totals = List[List[Int128]](
        length=groups, fill=[Int128(0), Int128(0), Int128(0)]
    )
    var counts = List[List[Int]](length=groups, fill=[0, 0, 0])
    var seen_rows = List[Int](length=groups, fill=0)
    for row in range(start, rows):
        var g = group_of(row, groups)
        var value = Int128(decimal_value(row))
        seen_rows[g] += 1
        if row % 11 != 4:
            totals[g][0] += value
            counts[g][0] += 1
        if row % 13 != 6:
            totals[g][1] += value
            counts[g][1] += 1
        if row % 7 != 2:
            totals[g][2] += value * 1_000_000_000_000_000
            counts[g][2] += 1
    var lookup = Dict[String, Int]()
    var present = 0
    for g in range(groups):
        if seen_rows[g] > 0:
            lookup[model_key_text(g)] = g
            present += 1
    assert_equal(actual.height(), present)
    var names: List[String] = ["d32", "d64", "d128"]
    var scales: List[Int] = [2, 2, 4]
    for r in range(actual.height()):
        var text = key_text(actual.item(r, "s"), actual.item(r, "i"))
        assert_true(text in lookup, "unexpected group " + text)
        var g = lookup[text]
        for c in range(3):
            var total = actual.column(names[c] + "_sum")
            if counts[g][c] == 0:
                assert_true(total.get(r).is_null())
                continue
            assert_equal(unscaled(total, r), totals[g][c])
            var mean = actual.item(r, names[c] + "_mean").float64()
            var expected = (
                Float64(totals[g][c])
                / Float64(counts[g][c])
                / (10.0 ** Float64(scales[c]))
            )
            assert_close(mean, expected, names[c] + " mean of " + text)


def decimal_reductions() -> List[Expr]:
    var out = List[Expr]()
    for name in ["d32", "d64", "d128"]:
        out.append(col(name).sum(min_count=1).alias(name + "_sum"))
        out.append(col(name).mean().alias(name + "_mean"))
    return out^


def test_decimal_sums_and_means_match_int128_totals_on_both_routes() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 60_000
    var groups = 5_000
    var frame = decimal_frame(rows, groups)
    var observed = StrategyTrace()
    var eager = frame.group_by(["s", "i"]).agg(decimal_reductions())
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "decimal sums did not reach the aggregate table",
    )
    _ = observed^
    assert_decimal_groups(eager, rows, groups, 0)
    observed = StrategyTrace()
    var piped = (
        frame.lazy()
        .filter(col("r") >= lit(Int64(77)))
        .group_by(["s", "i"])
        .agg(decimal_reductions())
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.group_table\n" in observed.text(),
        "decimal sums did not reach the aggregate table sink",
    )
    _ = observed^
    assert_decimal_groups(piped, rows, groups, 77)
    _ = trace^
    _ = workers^


def test_decimal_sum_past_its_precision_raises() raises:
    # decimal(38, 0) values near 10^37: a group's total passes 38 digits,
    # within one worker's partial sum or when partitions merge.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 40_000
    var keys = List[Int64]()
    var values = List[Int128]()
    for row in range(rows):
        keys.append(Int64(row % 9_000))
        values.append(Int128(10) ** 37 * 6)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int128](values^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )
    with assert_raises():
        _ = frame.group_by(["k"]).agg([col("v").sum().alias("total")])
    _ = workers^


def narrow_frame(rows: Int, groups: Int) raises -> DataFrame:
    """Narrow integer and Float32 columns (whole-valued, so every route's
    float totals are exact), null in every row of each seventh group and in
    every fifth row of the others."""
    var keys = List[Int64]()
    var i8 = List[Int8]()
    var i16 = List[Int16]()
    var i32 = List[Int32]()
    var u32 = List[UInt32]()
    var f32 = List[Float32]()
    var valid = List[Bool]()
    for row in range(rows):
        var k = row % groups
        keys.append(Int64(k))
        i8.append(Int8(row % 100 - 50))
        i16.append(Int16(row % 3000 - 1500))
        i32.append(Int32(row * 7919 % 2_000_000 - 1_000_000))
        u32.append(UInt32(row % 70_000))
        f32.append(Float32(row % 64 - 20))
        valid.append(k % 7 != 0 and row % 5 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("i8", Column[Int8](i8^, valid)),
            Series("i16", Column[Int16](i16^, valid)),
            Series("i32", Column[Int32](i32^, valid)),
            Series("u32", Column[UInt32](u32^, valid)),
            Series("f32", Column[Float32](f32^, valid)),
        ]
    )


def narrow_reductions() -> List[Expr]:
    var out = List[Expr]()
    for name in ["i8", "i16", "i32", "u32", "f32"]:
        out.append(col(name).sum().alias(name + "_sum"))
        out.append(col(name).sum(min_count=1).alias(name + "_sum1"))
        out.append(col(name).mean().alias(name + "_mean"))
        out.append(col(name).count().alias(name + "_count"))
    out.append(col("k").len().alias("rows"))
    return out^


def test_results_written_from_group_rows_match_the_single_thread_route() raises:
    # Counts, sums and means are finished straight from the table's group
    # rows: each must have the type, value and validity the reducers give
    # (an Int8 or Int16 sum as Int64, an Int32, UInt32 or Float32 sum in its
    # own type, a group of nulls null under min_count and as a mean), in
    # any order and in first-occurrence order.
    var rows = 72_000
    var groups = 6_000
    var frame = narrow_frame(rows, groups)
    var single = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var expected = frame.group_by(["k"], maintain_order=True).agg(
        narrow_reductions()
    )
    _ = single^
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var ordered = frame.group_by(["k"], maintain_order=True).agg(
        narrow_reductions()
    )
    var unordered = frame.group_by(["k"]).agg(narrow_reductions())
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "narrow sums did not reach the aggregate table",
    )
    _ = observed^
    assert_frame(ordered, expected)
    assert_frame(unordered.sort("k"), expected)
    observed = StrategyTrace()
    var piped = (
        frame.lazy()
        .filter(col("k") >= lit(Int64(0)))
        .group_by(["k"])
        .agg(narrow_reductions())
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.group_table\n" in observed.text(),
        "narrow sums did not reach the aggregate table sink",
    )
    _ = observed^
    assert_frame(piped.sort("k"), expected)
    _ = trace^
    _ = workers^


def test_int32_sum_past_its_type_raises_from_group_rows() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var rows = 40_000
    var keys = List[Int64]()
    var values = List[Int32]()
    for row in range(rows):
        keys.append(Int64(row % 9_000))
        values.append(Int32(2_000_000_000))
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int32](values^)),
        ]
    )
    with assert_raises():
        _ = frame.group_by(["k"]).agg([col("v").sum().alias("total")])
    _ = workers^


def test_quantiles_from_group_rows_match_the_single_thread_route() raises:
    # A quantile's values sit in chains of segments behind three words of
    # the group's row; merging groups from several workers links the chains.
    # NaN sorts above every number, a group of nulls is null, and a
    # Float32 and an unsigned input read as Float64 (a group's values
    # arrive from every worker in any order).
    var rows = 72_000
    var groups = 6_000
    var keys = List[Int64](capacity=rows)
    var floats = List[Float64](capacity=rows)
    var float_valid = List[Bool](capacity=rows)
    var narrow = List[Float32](capacity=rows)
    var wide = List[UInt64](capacity=rows)
    var wide_valid = List[Bool](capacity=rows)
    for row in range(rows):
        var g = (row * 7919) % groups
        keys.append(Int64(g))
        var value = Float64((row * 104729) % 1999) / 8 - 100
        if row % 23 == 5:
            value = Float64(0) / Float64(0)
        floats.append(value)
        # Every tenth group has no valid float at all.
        float_valid.append(g % 10 != 3 and row % 11 != 4)
        narrow.append(Float32((row * 31) % 257) / 4)
        wide.append(UInt64((row * 977) % 65_537) * 4_000_000_007)
        wide_valid.append(row % 5 != 1)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("f", Column[Float64](floats^, float_valid)),
            Series("n", Column[Float32](narrow^)),
            Series("u", Column[UInt64](wide^, wide_valid)),
        ]
    )
    var aggregations: List[Expr] = [
        col("f").median().alias("f_median"),
        col("f").quantile(0.9).alias("f_q90"),
        col("f").quantile(0.3, "lower").alias("f_lower"),
        col("f").quantile(0.3, "higher").alias("f_higher"),
        col("f").quantile(0.3, "nearest").alias("f_nearest"),
        col("f").quantile(0.3, "midpoint").alias("f_midpoint"),
        col("n").median().alias("n_median"),
        col("u").quantile(0.75).alias("u_q75"),
    ]
    var single = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var expected = frame.group_by(["k"], maintain_order=True).agg(
        aggregations.copy()
    )
    _ = single^
    for threads in [2, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var ordered = frame.group_by(["k"], maintain_order=True).agg(
            aggregations.copy()
        )
        var unordered = frame.group_by(["k"]).agg(aggregations.copy())
        assert_true(
            "dataframe-path: group_by.aggregate_table\n" in observed.text(),
            "quantiles did not reach the aggregate table",
        )
        _ = observed^
        assert_frame(ordered, expected)
        assert_frame(unordered.sort("k"), expected.sort("k"))
        observed = StrategyTrace()
        var piped = (
            frame.lazy()
            .filter(col("k") >= lit(Int64(0)))
            .group_by(["k"])
            .agg(aggregations.copy())
            .collect()
        )
        assert_true(
            "dataframe-path: pipeline.group_table\n" in observed.text(),
            "quantiles did not reach the aggregate table sink",
        )
        _ = observed^
        assert_frame(piped.sort("k"), expected.sort("k"))
        _ = trace^
        _ = workers^
