"""A streamed group-by with many groups collects its batches and runs one
eager group-by (#485), within a byte budget beyond which it reduces what it
holds into mergeable states."""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, LazyFrame, Series, col, lit


def set_env(name: String, value: String):
    var key = List[UInt8](name.as_bytes())
    key.append(0)
    var text = List[UInt8](value.as_bytes())
    text.append(0)
    _ = external_call["setenv", Int32](
        Int(key.unsafe_ptr()), Int(text.unsafe_ptr()), Int32(1)
    )
    _ = key^
    _ = text^


def unset_env(name: String):
    var key = List[UInt8](name.as_bytes())
    key.append(0)
    _ = external_call["unsetenv", Int32](Int(key.unsafe_ptr()))
    _ = key^


def facts(rows: Int, groups: Int) raises -> DataFrame:
    """Rows whose keys are mostly distinct early and repeat late, with a
    string key, nulls and NaN in the values."""
    var ints = List[Int64](capacity=rows)
    var words = List[String](capacity=rows)
    var floats = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var dims = List[Int64](capacity=rows)
    var state = UInt64(3)
    var nan = Float64(0) / Float64(0)
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var key = Int64((state >> 33) % UInt64(groups))
        ints.append(key)
        words.append("w" + String(key % 97))
        floats.append(nan if i % 101 == 0 else Float64(i % 1000) / 7)
        valid.append(i % 13 != 0)
        dims.append(Int64(i % 50))
    return DataFrame(
        [
            Series("k", Column[Int64](ints^)),
            Series("s", Column[String](words^)),
            Series("x", Column[Float64](floats^, valid.copy())),
            Series("d", Column[Int64](dims^)),
        ]
    )


def query(frame: DataFrame, keys: List[String]) raises -> LazyFrame:
    """Joined first, so the group-by streams rather than running eagerly
    on the scanned frame."""
    var dim = DataFrame(
        [
            Series("d", Column[Int64](List[Int64](length=50, fill=0))),
            Series("w", Column[Int64](List[Int64](length=50, fill=2))),
        ]
    )
    var ids = List[Int64](capacity=50)
    for i in range(50):
        ids.append(Int64(i))
    dim = dim.with_column(Series("d", Column[Int64](ids^)))
    return (
        frame.lazy()
        .join(dim.lazy(), left_on=["d"], right_on=["d"])
        .group_by(keys, maintain_order=True)
        .agg(
            [
                col("x").sum().alias("sum"),
                col("x").mean().alias("mean"),
                col("x").median().alias("median"),
                col("x").n_unique().alias("distinct"),
                col("w").min().alias("w"),
                col("x").len().alias("rows"),
            ]
        )
    )


def test_many_groups_collect_and_match_the_eager_result() raises:
    var frame = facts(200_000, 60_000)
    for keyset in range(2):
        var keys: List[String] = ["k"]
        if keyset == 1:
            keys = ["s", "k"]
        var plan = query(frame, keys)
        var expected = plan.collect(streaming=False)
        var streamed = plan.collect(batch_size=4096)
        assert_true(streamed.equals(expected), "collected group-by differs")
        assert_true(streamed.height() > 4096)
        # The report shows the group-by ran once over every row.
        var profiled = plan.profile(batch_size=4096)
        var report = profiled[1].copy()
        for r in range(report.height()):
            if report.item(r, "operator").string().startswith("GROUP_BY"):
                assert_equal(report.item(r, "executions").int64(), 1)
                assert_equal(
                    report.item(r, "output_rows").int64(),
                    Int64(expected.height()),
                )


def test_budget_reduces_collected_batches_into_states() raises:
    """With a tiny budget the stream reduces the collected batches into a
    state several times and merges them at the end; the result and its
    order are unchanged."""
    var frame = facts(120_000, 30_000)
    var plan = query(frame, ["k"])
    var expected = plan.collect(streaming=False)
    for budget in ["1", "200000", "3000000"]:
        set_env("DATAFRAME_STREAM_COLLECT_BYTES", budget)
        assert_true(
            plan.collect(batch_size=2048).equals(expected),
            "bounded collect differs at budget " + budget,
        )
    unset_env("DATAFRAME_STREAM_COLLECT_BYTES")


def test_few_groups_keep_the_batch_states() raises:
    var frame = facts(50_000, 300)
    var plan = query(frame, ["s"])
    var expected = plan.collect(streaming=False)
    assert_true(plan.collect(batch_size=4096).equals(expected))
    assert_equal(expected.height(), 97)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
