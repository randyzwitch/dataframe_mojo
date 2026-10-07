"""`LazyFrame.profile`: observed execution counters per plan node (#439)."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, LazyFrame, Series, col, lit


def ints(name: String, n: Int, modulus: Int) -> Series:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64((i * 7919) % modulus))
    return Series(name, Column[Int64](values^))


def rows_of(report: DataFrame, label: String) raises -> List[Int64]:
    """The input_rows, build_rows, output_rows, builds, executions of the row
    whose operator starts with `label`; raises when absent."""
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith(label):
            return [
                report.item(r, "input_rows").int64(),
                report.item(r, "build_rows").int64(),
                report.item(r, "output_rows").int64(),
                report.item(r, "builds").int64(),
                report.item(r, "executions").int64(),
            ]
    raise Error("no operator " + label + " in\n" + String(report))


def test_streaming_join_builds_once_and_counts_rows() raises:
    """A filtered scan streamed through two joins: each join builds its
    index once, every batch's rows are summed, the filter's input and
    output rows are the scan's and the kept rows, and the result is the
    one collect returns."""
    var facts = DataFrame([ints("k", 300_000, 1000), ints("v", 300_000, 97)])
    var dim = DataFrame([ints("k", 1000, 1000), ints("w", 1000, 5)])
    var other = DataFrame([ints("j", 97, 97), ints("z", 97, 3)])
    var plan = (
        facts.lazy()
        .filter(col("v") < lit(Int64(50)))
        .join(dim.lazy(), left_on=["k"], right_on=["k"])
        .join(other.lazy(), left_on=["v"], right_on=["j"])
        .select_exprs([col("w").sum().alias("w"), col("z").sum().alias("z")])
    )
    var profiled = plan.profile(batch_size=4096)
    var result = profiled[0].copy()
    var report = profiled[1].copy()
    assert_true(result.equals(plan.collect(batch_size=4096)))
    var kept = facts.filter(col("v") < lit(Int64(50))).height()
    var scanned = List[Int64]()
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith("SCAN frame"):
            scanned.append(report.item(r, "output_rows").int64())
    sort(scanned)
    assert_equal(scanned, [Int64(97), Int64(1000), Int64(300_000)])
    var filtered = rows_of(report, "FILTER")
    assert_equal(filtered[0], Int64(300_000))
    assert_equal(filtered[2], Int64(kept))
    var joins = 0
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith("JOIN"):
            joins += 1
            assert_equal(report.item(r, "builds").int64(), 1)
            assert_equal(report.item(r, "executions").int64(), 1)
            assert_equal(report.item(r, "executor").string(), "streaming")
            assert_equal(report.item(r, "build_side").string(), "right")
            assert_true(
                report.item(r, "algorithm").string()
                in ["hash_index", "progression"]
            )
            # Both joins keep every row: the keys cover the modulus.
            assert_equal(report.item(r, "input_rows").int64(), Int64(kept))
            assert_equal(report.item(r, "output_rows").int64(), Int64(kept))
    assert_equal(joins, 2)
    var reduced = rows_of(report, "SELECT w, z")
    assert_equal(reduced[0], Int64(kept))
    assert_equal(reduced[2], 1)
    # Node order: inputs before the operators that read them.
    for r in range(1, report.height()):
        assert_true(
            report.item(r, "node").int64() > report.item(r - 1, "node").int64()
        )


def test_eager_executor_records_each_input_once() raises:
    """Without streaming, every node records one execution and a join
    records its build rows and the side the build rule chose. The filter
    on the right input's column is pushed below the join, so the join's
    build rows are the rows it keeps."""
    var left = DataFrame([ints("k", 5000, 50), ints("v", 5000, 7)])
    var right = DataFrame([ints("k", 50, 50), ints("w", 50, 3)])
    var plan = (
        left.lazy()
        .join(right.lazy(), left_on=["k"], right_on=["k"])
        .filter(col("w") == lit(Int64(1)))
        .sort("v")
    )
    var profiled = plan.profile(streaming=False)
    var report = profiled[1].copy()
    assert_true(profiled[0].equals(plan.collect(streaming=False)))
    var scans = 0
    for r in range(report.height()):
        assert_equal(report.item(r, "executor").string(), "eager")
        assert_equal(report.item(r, "executions").int64(), 1)
        if report.item(r, "operator").string().startswith("SCAN frame"):
            scans += 1
    assert_equal(scans, 2)
    var right_kept = right.filter(col("w") == lit(Int64(1))).height()
    var filtered = rows_of(report, "FILTER")
    assert_equal(filtered[0], 50)
    assert_equal(filtered[2], Int64(right_kept))
    var joined = rows_of(report, "JOIN")
    assert_equal(joined[0], 5000)
    assert_equal(joined[1], Int64(right_kept))
    assert_equal(joined[2], Int64(profiled[0].height()))
    assert_equal(joined[3], 1)
    var sorted = rows_of(report, "SORT")
    assert_equal(sorted[0], Int64(profiled[0].height()))
    assert_equal(sorted[2], Int64(profiled[0].height()))


def test_frame_scan_under_an_eager_step_is_not_streamed() raises:
    """A plan whose terminal step runs eagerly over an in-memory frame
    (a many-group aggregation, a full sort) reads the frame as it is: the
    scan is not streamed into batches that would only be concatenated
    and rechunked again (ClickBench q16, 275 -> 181 ms)."""
    var frame = DataFrame([ints("k", 200_000, 150_000), ints("v", 200_000, 7)])
    var grouped = frame.lazy().group_by(["k"]).agg([col("v").sum().alias("s")])
    var profiled = grouped.profile()
    var report = profiled[1].copy()
    assert_true(profiled[0].equals(grouped.collect(streaming=False)))
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith("SCAN frame"):
            assert_equal(report.item(r, "executor").string(), "eager")
            assert_equal(report.item(r, "output_rows").int64(), 200_000)
    var sorted = frame.lazy().sort("v").profile()
    report = sorted[1].copy()
    assert_true(sorted[0].equals(frame.sort("v")))
    for r in range(report.height()):
        assert_equal(report.item(r, "executor").string(), "eager")
    # A streamed reduction over the frame still streams.
    var summed = frame.lazy().select_exprs([col("v").sum().alias("s")])
    report = summed.profile()[1].copy()
    assert_equal(rows_of(report, "SCAN frame")[2], 200_000)
    for r in range(report.height()):
        assert_equal(report.item(r, "executor").string(), "streaming")


def test_collect_without_profile_records_nothing() raises:
    var frame = DataFrame([ints("k", 100, 10)])
    var plan = frame.lazy().filter(col("k") > lit(Int64(3)))
    assert_true(not plan._report)
    _ = plan.collect()
    assert_true(not plan._report)
    var profiled = plan.profile()
    assert_true(not plan._report)
    assert_equal(profiled[1].height(), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
