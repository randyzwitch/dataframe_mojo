"""Small coded-key Int64 sums: exact wide arithmetic and real GPU routing."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringCodes
from dataframe_metal.accelerator.gpu.coded_group_sum import eligible, execute
from dataframe_metal.accelerator.gpu.target import METAL_TARGET
from dataframe.lazy import LazyFrame
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_compare import assert_frame


def gpu_execute(
    plan: LazyFrame, budget: Int
) raises -> Tuple[DataFrame, DataFrame]:
    comptime if METAL_TARGET:
        return execute(plan, 0, budget)
    else:
        raise Error("This witness requires Apple Metal")


def source(
    codes: List[UInt32],
    values: List[Int64],
    key_valid: List[Bool],
    value_valid: List[Bool],
) raises -> DataFrame:
    var dictionary = CategoricalDictionary()
    for name in ["z", "", "é🙂", "unused"]:
        dictionary.append(name)
    var strings = List[String](capacity=len(codes))
    for row in range(len(codes)):
        strings.append(
            String(dictionary.get(Int(codes[row]))) if key_valid[row] else ""
        )
    var key = StringColumn(strings^, key_valid)
    key._codes = ArcPointer(
        StringCodes(ArcPointer(codes.copy()), DataType.categorical(dictionary^))
    )
    return DataFrame(
        [
            Series("key", key^),
            Series("value", Column[Int64](values.copy(), value_valid.copy())),
            Series(
                "unused",
                StringColumn(
                    List[String](length=len(key_valid), fill="not a group key")
                ),
            ),
        ]
    )


def verify(frame: DataFrame) raises:
    var runtime = require_metal()
    var plan = (
        frame.lazy()
        .group_by("key", maintain_order=True)
        .agg(col("value").sum().alias("sum"))
    )
    assert_true(eligible(plan))
    var expected = frame.group_by("key", maintain_order=True).agg(
        col("value").sum().alias("sum")
    )
    var measured = plan.profile(accelerator=runtime)
    assert_frame(measured[0], expected)
    assert_equal(
        measured[1].item(0, "algorithm").string(), "mojo_gpu_coded_int64_sum"
    )
    assert_true("fault_adjudication" not in measured[1].columns())
    assert_equal(
        measured[1].item(0, "kernel_launches").int64(),
        Int64(2 if frame.height() else 0),
    )
    assert_equal(
        measured[1].item(0, "synchronizations").int64(),
        Int64(1 if frame.height() else 0),
    )
    assert_frame(plan.collect(accelerator=runtime), expected)


# requires: metal
def test_mojo_gpu_coded_sum_nulls_unused_entries_and_empty() raises:
    var frame = source(
        [2, 0, 2, UInt32.MAX, 1, 0, UInt32.MAX],
        [10, 20, -3, 7, 111, -5, -2],
        [True, True, True, False, True, True, False],
        [True, True, True, True, False, True, True],
    )
    verify(frame)
    var actual = (
        frame.lazy()
        .group_by("key", maintain_order=True)
        .agg(col("value").sum())
        .collect(accelerator=require_metal())
    )
    assert_equal(actual.height(), 4)
    assert_equal(actual.item(0, "key").string(), "é🙂")
    var sums: List[Int64] = [7, 15, 5, 0]
    for row in range(4):
        assert_equal(actual.item(row, "value").int64(), sums[row])
    assert_true(actual.item(2, "key").is_null())
    verify(frame.slice(1, 5))
    verify(frame.slice(0, 0))


# requires: metal
def test_mojo_gpu_coded_sum_chunks_offsets_wide_cancellation_and_reuse() raises:
    var n = 20004
    var codes = List[UInt32](capacity=n)
    var values = List[Int64](capacity=n)
    var keys = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for row in range(n):
        codes.append(UInt32((row // 2) % 3))
        values.append(Int64.MAX if row % 2 == 0 else -Int64.MAX)
        keys.append((row // 2) % 17 != 0)
        valid.append((row // 2) % 11 != 0)
    var frame = source(codes^, values^, keys^, valid^)
    # Repeated positive/negative updates exercise concurrent limb carries.
    verify(frame)
    var chunked = DataFrame(
        [
            Series._from_chunks(
                [
                    frame.column("key").slice(0, 9001),
                    frame.column("key").slice(9001, n - 9001),
                ]
            ),
            Series._from_chunks(
                [
                    frame.column("value").slice(0, 5003),
                    frame.column("value").slice(5003, n - 5003),
                ]
            ),
        ]
    )
    verify(chunked)
    verify(chunked.slice(3, n - 7))
    var cancellation = source(
        [0, 0, 0, 1, 1, 1],
        [Int64.MAX, 1, -1, Int64.MIN, -1, 1],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    )
    verify(cancellation)


# requires: metal
def test_mojo_gpu_coded_sum_overflow_budget_and_route_boundaries() raises:
    _ = require_metal()
    var frame = source([0, 0], [Int64.MAX, 1], [True, True], [True, True])
    var plan = frame.lazy().group_by("key").agg(col("value").sum())
    with assert_raises(contains="Int64 expression sum overflow"):
        _ = gpu_execute(plan, -1)
    frame = source([0, 0], [Int64.MIN, -1], [True, True], [True, True])
    plan = frame.lazy().group_by("key").agg(col("value").sum())
    with assert_raises(contains="Int64 expression sum overflow"):
        _ = gpu_execute(plan, -1)
    with assert_raises(contains="memory budget"):
        _ = gpu_execute(plan, 0)
    assert_true(
        not eligible(frame.lazy().group_by("key").agg(col("value").max()))
    )
    assert_true(
        not eligible(
            frame.lazy().group_by("key").agg(col("value").sum(min_count=1))
        )
    )
    assert_true(
        not eligible(frame.lazy().group_by("key").agg((col("value") + 1).sum()))
    )
    assert_true(
        not eligible(
            frame.lazy()
            .filter(col("value") > 0)
            .group_by("key")
            .agg(col("value").sum())
        )
    )
    var plain = DataFrame(
        [
            Series("key", StringColumn(["a"])),
            Series("value", Column[Int64]([1])),
        ]
    )
    assert_true(
        not eligible(plain.lazy().group_by("key").agg(col("value").sum()))
    )


# requires: metal
def test_mojo_gpu_coded_sum_dictionary_admission_boundaries() raises:
    _ = require_metal()
    var dictionary = CategoricalDictionary()
    for i in range(127):
        dictionary.append(String(i))
    var strings = StringColumn(["126"])
    strings._codes = ArcPointer(
        StringCodes(
            ArcPointer(List[UInt32](length=1, fill=126)),
            DataType.categorical(dictionary^),
        )
    )
    var frame = DataFrame(
        [Series("key", strings^), Series("value", Column[Int64]([9]))]
    )
    verify(frame)
    var large = CategoricalDictionary()
    for i in range(128):
        large.append(String(i))
    strings = StringColumn(["0"])
    strings._codes = ArcPointer(
        StringCodes(
            ArcPointer(List[UInt32](length=1, fill=0)),
            DataType.categorical(large^),
        )
    )
    frame = DataFrame(
        [Series("key", strings^), Series("value", Column[Int64]([9]))]
    )
    assert_true(
        not eligible(frame.lazy().group_by("key").agg(col("value").sum()))
    )
    var duplicated = CategoricalDictionary()
    duplicated.append("same")
    duplicated.append("same")
    strings = StringColumn(["same", "same"])
    var codes: List[UInt32] = [0, 1]
    strings._codes = ArcPointer(
        StringCodes(ArcPointer(codes^), DataType.categorical(duplicated^))
    )
    frame = DataFrame(
        [Series("key", strings^), Series("value", Column[Int64]([7, 8]))]
    )
    assert_true(eligible(frame.lazy().group_by("key").agg(col("value").sum())))
    var actual = (
        frame.lazy()
        .group_by("key")
        .agg(col("value").sum())
        .collect(accelerator=require_metal())
    )
    # A retained code is the group's identity: two codes naming "same" are two
    # groups, as on the CPU and on the general Metal grouping route (#740).
    assert_equal(actual.height(), 2)
    assert_equal(actual.item(0, "key").string(), "same")
    assert_equal(actual.item(1, "key").string(), "same")
    assert_equal(actual.item(0, "value").int64(), Int64(7))
    assert_equal(actual.item(1, "value").int64(), Int64(8))
