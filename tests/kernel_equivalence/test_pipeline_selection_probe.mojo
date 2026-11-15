"""Rows a scan-side join filter keeps travel through the pipeline as one
selection over the unchanged source columns, and every later join probes its
keys through that selection (or through the rows an earlier join kept) where
they are, with no gather. Joins against a scalar model: a bloom-filtered
Int64 key, then a join on a nullable string key and an Int32 key read
through the first join's kept rows, then a semi join on an Int64 key, with
and without a row-local filter between the joins, on one and eight
workers."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
)
from dataframe.storage.string import StringColumn
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 300_000
comptime SPAN = 10_000_000


def spread(row: Int, salt: Int) -> Int:
    """A key scattered over the whole span."""
    return Int((UInt64(row) * 2654435761 + UInt64(salt) * 40503) % UInt64(SPAN))


def name_of(m: Int) -> String:
    """A string key longer than a view's inline bytes."""
    return "a-string-key-number-" + String(m)


def key_valid(row: Int) -> Bool:
    return row % 97 != 5


def name_valid(row: Int) -> Bool:
    return row % 89 != 7


def probe() raises -> DataFrame:
    var ids = List[Int64]()
    var a = List[Int64]()
    var names = List[String]()
    var small = List[Int32]()
    var c = List[Int64]()
    var valid = List[Bool]()
    var named = List[Bool]()
    for row in range(ROWS):
        ids.append(Int64(row))
        a.append(Int64(spread(row % 5000, 1)))
        names.append(name_of(row % 700))
        small.append(Int32((row % 700) % 13))
        c.append(Int64(row % 300))
        valid.append(key_valid(row))
        named.append(name_valid(row))
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("ak", Column[Int64](a^, valid^)),
            Series("sk", StringColumn(names^, named^)),
            Series("dk", Column[Int32](small^)),
            Series("ck", Column[Int64](c^)),
        ]
    )


def build_a() raises -> DataFrame:
    """Every 100th of the probe's 5,000 `ak` values: a bloom filter."""
    var keys = List[Int64]()
    var payload = List[Int64]()
    for m in range(0, 5000, 100):
        keys.append(Int64(spread(m, 1)))
        payload.append(Int64(m))
    return DataFrame(
        [
            Series("a", Column[Int64](keys^)),
            Series("a_v", Column[Int64](payload^)),
        ]
    )


def build_b() raises -> DataFrame:
    """Every other of the probe's 700 (`sk`, `dk`) pairs."""
    var names = List[String]()
    var small = List[Int32]()
    var payload = List[Int64]()
    for m in range(0, 700, 2):
        names.append(name_of(m))
        small.append(Int32(m % 13))
        payload.append(Int64(m))
    return DataFrame(
        [
            Series("sk", StringColumn(names^)),
            Series("dk", Column[Int32](small^)),
            Series("b_v", Column[Int64](payload^)),
        ]
    )


def build_c() raises -> DataFrame:
    """Every `ck` value but the multiples of 7."""
    var keys = List[Int64]()
    for m in range(300):
        if m % 7 != 0:
            keys.append(Int64(m))
    return DataFrame([Series("c", Column[Int64](keys^))])


def check(between: Bool) raises:
    """`between` puts a row-local filter on `id` between the first two
    joins."""
    var plan = (
        probe().lazy().join(build_a().lazy(), left_on=["ak"], right_on=["a"])
    )
    if between:
        plan = plan.filter(col("id") % 3 != 0)
    var result = (
        plan.join(build_b().lazy(), on=["sk", "dk"])
        .join(build_c().lazy(), left_on=["ck"], right_on=["c"], how="semi")
        .sort(["id"])
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var frame = result.collect()
    var text = observed.text()
    _ = observed^
    _ = trace^
    assert_true(
        "dataframe-path: pipeline.join_filter_selection\n" in text,
        "the join filter's rows did not travel as a selection",
    )
    assert_true(
        "dataframe-path: pipeline.join_probe_selection\n" in text,
        "no join probed its keys through a selection",
    )
    var ids = frame.column("id")
    var a_v = frame.column("a_v")
    var b_v = frame.column("b_v")
    var at = 0
    for row in range(ROWS):
        if not key_valid(row) or not name_valid(row):
            continue
        if (row % 5000) % 100 != 0 or (row % 700) % 2 != 0:
            continue
        if (row % 300) % 7 == 0:
            continue
        if between and row % 3 == 0:
            continue
        assert_true(at < len(ids), "a joining row is missing")
        assert_equal(ids.get(at).int64(), Int64(row))
        assert_equal(a_v.get(at).int64(), Int64(row % 5000))
        assert_equal(b_v.get(at).int64(), Int64(row % 700))
        at += 1
    assert_equal(at, len(ids))


def test_filtered_rows_are_probed_through_their_selection() raises:
    for threads in [1, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        check(False)
        check(True)
        _ = workers^
