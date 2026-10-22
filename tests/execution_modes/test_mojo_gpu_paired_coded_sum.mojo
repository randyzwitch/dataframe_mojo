"""Bounded Cartesian dictionary sums preserve exact values and source order."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringCodes
from dataframe.lazy import LazyFrame
from dataframe_metal.accelerator.gpu.target import METAL_TARGET
from dataframe_metal.accelerator.gpu.paired_coded_sum import eligible, execute
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_compare import assert_frame


def coded(
    name: String, labels: List[String], codes: List[UInt32], valid: List[Bool]
) raises -> Series:
    var dictionary = CategoricalDictionary()
    for label in labels:
        dictionary.append(label)
    var values = List[String](capacity=len(codes))
    for row in range(len(codes)):
        values.append(labels[Int(codes[row])] if valid[row] else "")
    var strings = StringColumn(values, valid)
    strings._codes = ArcPointer(
        StringCodes(ArcPointer(codes.copy()), DataType.categorical(dictionary^))
    )
    return Series(name, strings^)


def plan(frame: DataFrame) raises -> LazyFrame:
    return (
        frame.lazy()
        .group_by(["region", "status"], maintain_order=True)
        .agg(col("events").sum().alias("total"))
    )


def fixture(rows: Int) raises -> DataFrame:
    var labels = List[String]()
    for index in range(13):
        labels.append("dimension_é_" + String(index))
    var left = List[UInt32]()
    var right = List[UInt32]()
    var sums = List[Int64]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    var value_valid = List[Bool]()
    for row in range(rows):
        left_valid.append(row % 7 != 0)
        right_valid.append(row % 11 != 0)
        value_valid.append(row % 13 != 0)
        left.append(UInt32(row % 13) if left_valid[row] else UInt32.MAX)
        right.append(
            UInt32((row // 13) % 13) if right_valid[row] else UInt32.MAX
        )
        sums.append(Int64(row % 17 - 8))
    return DataFrame(
        [
            coded("region", labels, left, left_valid),
            coded("status", labels, right, right_valid),
            Series("events", Column[Int64](sums^, value_valid)),
        ]
    )


def direct(
    query: LazyFrame, budget: Int = -1
) raises -> Tuple[DataFrame, DataFrame]:
    comptime if METAL_TARGET:
        return execute(query, 0, budget)
    else:
        raise Error("This witness requires Apple Metal")


def verify(frame: DataFrame) raises:
    var query = plan(frame)
    assert_true(eligible(query))
    var expected = frame.group_by(
        ["region", "status"], maintain_order=True
    ).agg(col("events").sum().alias("total"))
    var measured = query.profile(accelerator=require_metal())
    assert_frame(measured[0], expected)
    assert_equal(
        measured[1].item(0, "algorithm").string(),
        "mojo_gpu_paired_coded_int64_sum",
    )
    assert_equal(
        measured[1].item(0, "kernel_launches").int64(),
        Int64(3 if frame.height() else 0),
    )
    assert_equal(
        measured[1].item(0, "synchronizations").int64(),
        Int64(1 if frame.height() else 0),
    )
    assert_true("fault_adjudication" not in measured[1].columns())


# requires: metal
def test_mojo_gpu_paired_coded_sum_nulls_chunks_order_and_empty() raises:
    var frame = fixture(20003)
    verify(frame)
    verify(frame.slice(3, 19997))
    var columns = List[Series]()
    for column in frame._columns:
        columns.append(
            Series._from_chunks(
                [
                    column.slice(1, 7),
                    column.slice(8, 17),
                    column.slice(25, 19978),
                ]
            )
        )
    verify(DataFrame(columns^))
    verify(frame.slice(0, 0))


# requires: metal
def test_mojo_gpu_paired_coded_sum_wide_cancellation_and_overflow() raises:
    var frame = DataFrame(
        [
            coded(
                "region",
                ["r", "unused"],
                [0, 0, 0, 0],
                [True, True, True, True],
            ),
            coded(
                "status",
                ["s", "unused"],
                [0, 0, 0, 0],
                [True, True, True, True],
            ),
            Series("events", Column[Int64]([Int64.MAX, 1, -1, -Int64.MAX])),
        ]
    )
    verify(frame)
    assert_equal(direct(plan(frame))[0].item(0, "total").int64(), Int64(0))
    var cases: List[List[Int64]] = [[Int64.MAX, 1, 0, 0], [Int64.MIN, -1, 0, 0]]
    for values in cases:
        var bad = frame.with_column(
            Series("events", Column[Int64](values.copy()))
        )
        with assert_raises(contains="sum overflow"):
            _ = direct(plan(bad))
    with assert_raises(contains="budget"):
        _ = direct(plan(frame), 0)


# requires: metal
def test_mojo_gpu_paired_coded_sum_duplicate_labels_coalesce() raises:
    var frame = DataFrame(
        [
            coded("region", ["same", "same"], [0, 1, 0], [True, True, True]),
            coded("status", ["s"], [0, 0, 0], [True, True, True]),
            Series("events", Column[Int64]([7, 8, -2])),
        ]
    )
    var expected = DataFrame(
        [
            Series("region", StringColumn(["same"])),
            Series("status", StringColumn(["s"])),
            Series("total", Column[Int64]([13])),
        ]
    )
    assert_frame(direct(plan(frame))[0], expected)


def test_mojo_gpu_paired_coded_sum_domain_capacity_and_shape_guards() raises:
    var labels = List[String]()
    for index in range(128):
        labels.append(String(index))
    var left_labels = List[String]()
    for index in range(127):
        left_labels.append(labels[index])
    var frame = DataFrame(
        [
            coded("region", left_labels, [0], [True]),
            coded("status", left_labels, [0], [True]),
            Series("events", Column[Int64]([1])),
        ]
    )
    assert_true(eligible(plan(frame)))
    var large = frame.with_column(coded("region", labels, [0], [True]))
    assert_true(not eligible(plan(large)))
    assert_true(
        not eligible(
            frame.lazy()
            .group_by(["region", "status"])
            .agg(col("events").sum(min_count=1))
        )
    )
    assert_true(
        not eligible(
            frame.lazy()
            .filter(col("events") > lit(0))
            .group_by(["region", "status"])
            .agg(col("events").sum())
        )
    )
