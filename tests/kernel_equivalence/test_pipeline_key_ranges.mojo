"""A pipeline narrows its morsels by each join build's key range before
any step runs, deciding whole
morsels from the zone maps of the probe key's buffer. Joins against a
scalar model: sorted keys (morsels wholly outside, wholly inside, or
straddling the range) and scattered ones, nullable probe keys, a build
with no rows, a build whose keys are all null, Int64, Int32 and Date
keys, inner and semi joins chained after one another, on one and eight
workers."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    lit,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 40_000
comptime KEYS = 1_000


def sorted_key(row: Int) -> Int:
    """Ascending: each key holds a run of rows, so morsels hold few keys."""
    return row * KEYS // ROWS


def scattered_key(row: Int) -> Int:
    return (row * 7919) % KEYS


def key_valid(row: Int) -> Bool:
    return row % 89 != 4


def probe(dtype: DataType) raises -> DataFrame:
    """Row ids, a sorted key and a scattered key, both nullable."""
    var ids = List[Int64]()
    var sorted = List[Int64]()
    var scattered = List[Int64]()
    var valid = List[Bool]()
    for row in range(ROWS):
        ids.append(Int64(row))
        sorted.append(Int64(sorted_key(row)))
        scattered.append(Int64(scattered_key(row)))
        valid.append(key_valid(row))
    var frame = DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("sk", Column[Int64](sorted^, valid.copy())),
            Series("xk", Column[Int64](scattered^, valid)),
        ]
    )
    if dtype != DataType.INT64:
        frame = frame.with_columns(
            [
                col("sk").cast(dtype).alias("sk"),
                col("xk").cast(dtype).alias("xk"),
            ]
        )
    return frame^


def build(
    name: String, low: Int, high: Int, step: Int, dtype: DataType
) raises -> DataFrame:
    """Keys low, low + step, ... below high, with a payload."""
    var keys = List[Int64]()
    var payload = List[Int64]()
    var k = low
    while k < high:
        keys.append(Int64(k))
        payload.append(Int64(3 * k))
        k += step
    var frame = DataFrame(
        [
            Series(name, Column[Int64](keys^)),
            Series(name + "_v", Column[Int64](payload^)),
        ]
    )
    if dtype != DataType.INT64:
        frame = frame.with_columns([col(name).cast(dtype).alias(name)])
    return frame^


def all_null_build(name: String, dtype: DataType) raises -> DataFrame:
    var frame = DataFrame(
        [
            Series(
                name,
                Column[Int64]([Int64(1), Int64(2)], [False, False]),
            ),
            Series(name + "_v", Column[Int64]([Int64(0), Int64(0)])),
        ]
    )
    if dtype != DataType.INT64:
        frame = frame.with_columns([col(name).cast(dtype).alias(name)])
    return frame^


def in_build(key: Int, low: Int, high: Int, step: Int) -> Bool:
    return key >= low and key < high and (key - low) % step == 0


def ids(frame: DataFrame) raises -> List[Int]:
    var sorted = frame.sort(["id"]).column("id")
    var out = List[Int](capacity=len(sorted))
    for r in range(len(sorted)):
        out.append(Int(sorted.get(r).int64()))
    return out^


def check(
    dtype: DataType,
    sorted_low: Int,
    sorted_high: Int,
    sorted_step: Int,
    scattered_low: Int,
    scattered_high: Int,
    semi: Bool,
) raises:
    """Probe ⋈ sorted-key build ⋈ scattered-key build (the second a semi
    join when `semi`), against the rows the model keeps."""
    var source = probe(dtype)
    var plan = (
        source.lazy()
        .join(
            build("b", sorted_low, sorted_high, sorted_step, dtype).lazy(),
            left_on=["sk"],
            right_on=["b"],
        )
        .join(
            build("c", scattered_low, scattered_high, 1, dtype).lazy(),
            left_on=["xk"],
            right_on=["c"],
            how="semi" if semi else "inner",
        )
    )
    var expected = List[Int]()
    for row in range(ROWS):
        if not key_valid(row):
            continue
        if not in_build(sorted_key(row), sorted_low, sorted_high, sorted_step):
            continue
        if not in_build(scattered_key(row), scattered_low, scattered_high, 1):
            continue
        expected.append(row)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var result = plan.collect()
    assert_true(
        "dataframe-path: pipeline.key_range\n" in observed.text(),
        "no key range narrowed the pipeline",
    )
    _ = observed^
    _ = trace^
    assert_equal(ids(result), expected)
    if not semi and len(expected) > 0:
        # The build payloads line up with the keys that joined.
        var row = result.sort(["id"])
        for r in [0, len(expected) // 2, len(expected) - 1]:
            var source_row = expected[r]
            assert_equal(
                row.item(r, "b_v").int64(),
                Int64(3 * sorted_key(source_row)),
            )


def test_key_ranges_skip_keep_and_narrow_morsels() raises:
    # Eight workers cut the rows into about 32 morsels, mostly wholly
    # outside or inside a narrow run of the sorted key; one worker takes
    # one morsel, which straddles it and is narrowed row by row.
    for threads in [1, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        for dtype in [DataType.INT64, DataType.INT32, DataType.DATE]:
            # A narrow run of the sorted key; a scattered build covering
            # half the keys straddles every morsel.
            check(dtype, 400, 430, 1, 0, 500, False)
        # Every other key in a wide run, as a semi join.
        check(DataType.INT64, 100, 900, 2, 250, 750, True)
        # The whole key span: every morsel is wholly inside.
        check(DataType.INT64, 0, KEYS, 1, 0, KEYS, False)
        # Keys beyond every probe key: nothing joins.
        check(DataType.INT64, KEYS + 5, KEYS + 50, 1, 0, KEYS, False)
        _ = workers^


def test_empty_and_all_null_builds_join_nothing() raises:
    for dtype in [DataType.INT64, DataType.INT32]:
        var source = probe(dtype)
        var empty = build("b", 0, 0, 1, dtype)
        assert_equal(
            source.lazy()
            .join(empty.lazy(), left_on=["sk"], right_on=["b"])
            .collect()
            .height(),
            0,
        )
        assert_equal(
            source.lazy()
            .join(
                all_null_build("b", dtype).lazy(),
                left_on=["sk"],
                right_on=["b"],
                how="semi",
            )
            .collect()
            .height(),
            0,
        )
