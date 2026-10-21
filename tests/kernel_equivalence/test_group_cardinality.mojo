"""Cardinality routing and partitioned distinct counts have scalar witnesses."""
# contract-group: kernel.order_keys
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
)
from dataframe.binding import bind, BoundExpr
from dataframe.hash_agg import hash_aggregate, hash_agg_eligible
from tests.support.coverage_compare import assert_float
from dataframe.distinct import distinct_counts
from dataframe.hash_agg import _moderate_groups
from dataframe.string_view import StringViewBuilder
from tests.support.lifecycle_jobs import WorkerSetting


def _periodic(rows: Int, classes: Int) -> Series:
    var values = List[Int64]()
    for row in range(rows):
        values.append(Int64(row % classes))
    return Series("key", Column[Int64](values^))


def test_cardinality_sampling_rejects_period_aliasing_and_respects_the_strict_density_cutoff() raises:
    assert_false(_moderate_groups(_periodic(0, 1)))
    for n in [1000, 2000, 10000]:
        for delta in [-1, 0, 1]:
            var classes = n * 85 // 100 + delta
            assert_equal(_moderate_groups(_periodic(n, classes)), delta < 0)
    for rows in [262144, 524288, 1048576]:
        var source = _periodic(rows + 7, rows // 2).slice(7, rows)
        assert_false(_moderate_groups(source))
        var chunks = Series._from_chunks(
            [
                source.slice(0, 1),
                source.slice(1, 65537),
                source.slice(65538, rows - 65538),
            ]
        )
        assert_false(_moderate_groups(chunks))
        assert_true(_moderate_groups(_periodic(rows, 37)))


def _assert_counts(
    values: Series, labels: List[Int], classes: List[Int], group_count: Int
) raises:
    var grouped_seen = List[UInt8](length=group_count, fill=0)
    var global_seen = UInt8(0)
    for row in range(len(values)):
        var bit = UInt8(1) << UInt8(classes[row])
        grouped_seen[labels[row]] |= bit
        global_seen |= bit
    for grouped in [False, True]:
        var result = distinct_counts(values, labels, group_count, grouped)
        assert_true(
            Bool(result),
            "partitioned distinct route must accept primitive storage",
        )
        var actual = result.value().copy()
        assert_equal(len(actual), group_count if grouped else 1)
        for group in range(len(actual)):
            var bits = grouped_seen[group] if grouped else global_seen
            var expected = Int64(0)
            for cls in range(8):
                expected += Int64((bits >> UInt8(cls)) & 1)
            assert_equal(actual[group], expected)


def _numeric_distinct[D: DType](rows: Int) raises:
    comptime GROUPS = 65537
    var payload = List[Scalar[D]](length=7, fill=0)
    var valid = List[Bool](length=7, fill=False)
    var labels = List[Int]()
    var classes = List[Int]()
    for row in range(rows):
        var cls = row // GROUPS % 4
        var present = row % 17 != 3
        comptime if D.is_floating_point():
            var nan = Scalar[D](0) / Scalar[D](0)
            payload.append(
                nan if cls
                == 1 else Scalar[D](-0.0 if row % 2 else 0.0) if cls
                == 0 else Scalar[D](1.25 if cls == 2 else -2.5)
            )
        else:
            payload.append(
                Scalar[D](0) if cls
                == 0 else Scalar[D](1) if cls
                == 1 else Scalar[D].MAX if cls
                == 2 else Scalar[D].MIN
            )
            # Unsigned MIN equals zero, so its canonical class is zero.
            comptime if D.is_unsigned():
                if cls == 3:
                    cls = 0
        valid.append(present)
        labels.append(row * 8191 % GROUPS)
        classes.append(cls if present else 4)
    var source = Series("v", Column[Scalar[D]](payload^, valid)).slice(7, rows)
    _assert_counts(source, labels, classes, GROUPS + 1)
    if rows > 65538:
        var chunked = Series._from_chunks(
            [
                source.slice(0, 1),
                source.slice(1, 65537),
                source.slice(65538, rows - 65538),
            ]
        )
        _assert_counts(chunked, labels, classes, GROUPS + 1)


def test_shared_group_value_partitions_preserve_integer_bits_float_classes_and_nulls_across_workers() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        _numeric_distinct[DType.int8](262145)
        _numeric_distinct[DType.uint64](262145)
        _numeric_distinct[DType.float32](262145)
        _numeric_distinct[DType.float64](262145)
        _ = setting^


def test_partitioned_distinct_compares_native_view_string_bytes_across_pieces_and_declines_decimals() raises:
    comptime ROWS = 131073
    var builder = StringViewBuilder()
    var labels = List[Int]()
    var classes = List[Int]()
    for row in range(ROWS):
        var cls = row % 6
        var present = row % 19 != 7
        if present:
            var text = (
                "" if cls
                == 0 else "short" if cls
                == 1 else "a common prefix extending beyond inline storage:"
                + String(cls)
            )
            builder.append(StringSlice(text))
        else:
            builder.append_null()
        labels.append(row % 1031)
        classes.append(cls if present else 6)
    var source = Series("v", StringColumn(builder^.finish()))
    assert_true(Bool(source.string()._view_storage))
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        var chunked = Series._from_chunks(
            [source.slice(0, 65535), source.slice(65535, ROWS - 65535)]
        )
        _assert_counts(chunked, labels, classes, 1032)
        _ = setting^
    var decimal = Series("v", Column[Int128]([Int128(1)])).with_dtype(
        DataType.decimal(5, 2)
    )
    assert_false(Bool(distinct_counts(decimal, [0], 1, True)))


struct HashStats(Copyable, Movable):
    var first: Int
    var rows: Int64
    var count: Int64
    var total: Int128
    var least: Int64
    var most: Int64
    var float_count: Int64
    var float_total: Float64
    var float_least: Float64
    var float_most: Float64
    var nan: Bool

    def __init__(out self) raises:
        self.first = -1
        self.rows = 0
        self.count = 0
        self.total = 0
        self.least = Int64.MAX
        self.most = Int64.MIN
        self.float_count = 0
        self.float_total = 0
        self.float_least = Float64("inf")
        self.float_most = -Float64("inf")
        self.nan = False


def test_inplace_hash_aggregation_merges_complete_scalar_states_and_declines_unsupported_shapes() raises:
    comptime GROUPS = 8191
    comptime ROWS = 262145
    var expected = List[HashStats](length=GROUPS + 1, fill=HashStats())
    var keys = List[Int64](length=7, fill=Int64.MAX)
    var key_valid = List[Bool](length=7, fill=False)
    var integers = List[Int64](length=7, fill=Int64.MAX)
    var integer_valid = List[Bool](length=7, fill=False)
    var floats = List[Float64](length=7, fill=Float64("nan"))
    var float_valid = List[Bool](length=7, fill=False)
    for row in range(ROWS):
        var present_key = row % 31 != 2
        var group = row * 37 % GROUPS if present_key else GROUPS
        var iv = row % 7 != 3
        var fv = row % 11 != 4
        var integer = Int64(row % 29 - 14)
        var floating = (
            Float64("nan") if row % 257 == 3 else Float64(row % 41 - 20) / 4
        )
        keys.append(Int64(group) - 4000 if present_key else Int64.MAX)
        key_valid.append(present_key)
        integers.append(integer if iv else Int64.MAX)
        integer_valid.append(iv)
        floats.append(floating if fv else Float64("nan"))
        float_valid.append(fv)
        ref state = expected[group]
        if state.first < 0:
            state.first = row
        state.rows += 1
        if iv:
            state.count += 1
            state.total += Int128(integer)
            state.least = min(state.least, integer)
            state.most = max(state.most, integer)
        if fv:
            state.float_count += 1
            state.float_total += floating
            if isnan(floating):
                state.nan = True
            else:
                state.float_least = min(state.float_least, floating)
                state.float_most = max(state.float_most, floating)
    var frame = DataFrame(
        [
            Series("key", Column[Int64](keys^, key_valid^)).slice(7, ROWS),
            Series("i", Column[Int64](integers^, integer_valid^)).slice(
                7, ROWS
            ),
            Series("f", Column[Float64](floats^, float_valid^)).slice(7, ROWS),
        ]
    )
    var expressions: List[Expr] = [
        col("i").sum(),
        col("i").mean(),
        col("i").min(),
        col("i").max(),
        col("i").count(),
        col("i").len(),
        col("f").sum(),
        col("f").min(),
        col("f").max(),
    ]
    var bounds = List[BoundExpr]()
    for index in range(len(expressions)):
        bounds.append(
            bind(
                expressions[index].alias("out" + String(index)), frame._columns
            )
        )
    assert_true(hash_agg_eligible(frame.column("key"), bounds, frame._columns))
    var computed = List[BoundExpr]()
    computed.append(bind((col("i") + 1).sum(), frame._columns))
    assert_false(
        hash_agg_eligible(frame.column("key"), computed, frame._columns)
    )
    assert_false(hash_agg_eligible(frame.column("f"), bounds, frame._columns))
    assert_false(
        hash_agg_eligible(
            frame.column("key"), List[BoundExpr](), frame._columns
        )
    )
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        var output = hash_aggregate(
            frame.column("key"), bounds, frame._columns, workers
        )
        ref firsts = output[0]
        ref parts = output[1]
        assert_equal(len(firsts), GROUPS + 1)
        var seen = List[Bool](length=GROUPS + 1, fill=False)
        var cursor = 0
        for part in parts:
            assert_equal(len(part), len(bounds))
            for row in range(len(part[0])):
                var first = firsts[cursor]
                cursor += 1
                var group = first * 37 % GROUPS if first % 31 != 2 else GROUPS
                assert_false(seen[group])
                seen[group] = True
                ref state = expected[group]
                assert_equal(first, state.first)
                assert_equal(part[0].get(row).int64(), Int64(state.total))
                assert_float(
                    part[1].get(row).float64(),
                    Float64(state.total) / Float64(state.count),
                )
                assert_equal(part[2].get(row).int64(), state.least)
                assert_equal(part[3].get(row).int64(), state.most)
                assert_equal(part[4].get(row).int64(), state.count)
                assert_equal(part[5].get(row).int64(), state.rows)
                assert_float(part[6].get(row).float64(), state.float_total)
                assert_float(part[7].get(row).float64(), state.float_least)
                assert_float(
                    part[8].get(row).float64(),
                    Float64("nan") if state.nan else state.float_most,
                )
                for column in part:
                    assert_equal(len(column), len(part[0]))
                    assert_false(column.get(row).is_null())
        assert_equal(cursor, GROUPS + 1)
        _ = setting^
