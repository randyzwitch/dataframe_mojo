"""Partitioned distinct counts have scalar witnesses."""
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
from tests.support.coverage_compare import assert_float
from dataframe.kernels.distinct import distinct_counts
from dataframe.storage.string_view import StringViewBuilder
from tests.support.lifecycle_jobs import WorkerSetting


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
    var numbers: Int

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
        self.numbers = 0
