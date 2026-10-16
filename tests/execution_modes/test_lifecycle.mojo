"""Termination drains actual jobs, propagates late failures, and permits retry."""
# contract-group: kernel.reduce
# contract-group: execution.modes
# contract-group: execution.lifecycle
# contract-group: io.csv
from std.ffi import external_call
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, Expr, col
from dataframe.parallel import Pool, _ProducedJobs, run_jobs, _helpers
from dataframe.frame import _StreamReduction
from tests.support.lifecycle_jobs import (
    CountedJob,
    JobCounts,
    WorkerSetting,
    assert_job_counts,
    jobs_for,
)


def test_worker_rounds_complete_each_job_once_before_releasing_resources() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for claim in [False, True]:
            var counts = ArcPointer(JobCounts(32))
            var jobs = jobs_for(counts, 32)
            var caller = external_call["pthread_self", UInt64]().cast[
                DType.int64
            ]()
            var pool = Pool(workers)
            pool.run(jobs, claim=claim)
            assert_job_counts(counts, 32, False)
            var helper_entries = 0
            for id in range(len(jobs)):
                assert_equal(jobs[id].id, id)
                assert_equal(jobs[id].result, id * id)
                helper_entries += Int(counts[].thread[id].load() != caller)
            if workers > 1 and not claim:
                assert_true(
                    helper_entries > 0,
                    "worker lifecycle test must execute on actual helper threads",
                )
            _ = jobs^
            assert_job_counts(counts, 32, True)
            pool.release()
            pool.release()
            assert_equal(pool.address, 0)
        _ = setting^


def test_worker_error_drains_all_jobs_and_pool_can_run_again() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for claim in [False, True]:
            var counts = ArcPointer(JobCounts(16))
            var jobs = jobs_for(counts, 16, 0)
            var pool = Pool(workers)
            var failed = False
            try:
                pool.run(jobs, claim=claim)
            except error:
                failed = True
                assert_true(
                    String(error).find("deliberate lifecycle worker failure 0")
                    >= 0
                )
            assert_true(failed)
            assert_job_counts(counts, 16, True)
            if workers > 1 and not claim:
                var caller = external_call["pthread_self", UInt64]().cast[
                    DType.int64
                ]()
                assert_true(
                    counts[].thread[0].load() != caller,
                    "failure must originate on an actual helper thread",
                )
            var retry_counts = ArcPointer(JobCounts(16))
            var retry = jobs_for(retry_counts, 16)
            pool.run(retry, claim=claim)
            assert_job_counts(retry_counts, 16, False)
            _ = retry^
            assert_job_counts(retry_counts, 16, True)
            pool.release()
        var standalone = ArcPointer(JobCounts(16))
        var jobs = jobs_for(standalone, 16, 3)
        with assert_raises():
            run_jobs(jobs)
        assert_job_counts(standalone, 16, True)
        _ = setting^


def interrupted_producer(
    mut pool: Pool, counts: ArcPointer[JobCounts], limit: Int
) raises:
    var producer = _ProducedJobs[CountedJob](32)
    pool.run_produced(producer)
    for id in range(limit):
        producer.submit(CountedJob(counts, id))
    raise Error("deliberate producer cancellation before EOF")


def test_producer_cancellation_drains_owned_jobs_before_unwind_and_retry() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        var pool = Pool(workers)
        for limit in [0, 1, 7, 32]:
            var counts = ArcPointer(JobCounts(32))
            with assert_raises():
                interrupted_producer(pool, counts, limit)
            assert_job_counts(counts, limit, True)
            var retry_counts = ArcPointer(JobCounts(8))
            var retry = jobs_for(retry_counts, 8)
            pool.run(retry)
            _ = retry^
            assert_job_counts(retry_counts, 8, True)
        pool.release()
        _ = setting^


def test_produced_eof_and_late_worker_failure_have_exact_cleanup() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for fail in [False, True]:
            var counts = ArcPointer(JobCounts(16))
            var pool = Pool(workers)
            var producer = _ProducedJobs[CountedJob](16)
            pool.run_produced(producer)
            for id in range(16):
                producer.submit(CountedJob(counts, id, fail and id == 15))
            if fail:
                with assert_raises():
                    _ = producer.finish()
                assert_job_counts(counts, 16, False)
                _ = producer^
                assert_job_counts(counts, 16, True)
            else:
                var jobs = producer.finish()
                assert_job_counts(counts, 16, False)
                for id in range(len(jobs)):
                    assert_equal(jobs[id].id, id)
                    assert_equal(jobs[id].result, id * id)
                _ = jobs^
                assert_job_counts(counts, 16, True)
            pool.release()
        _ = setting^


def test_public_streaming_reductions_allow_partial_overflow_cancellation() raises:
    from std.os import getenv
    from dataframe import CsvField, CsvSchema, scan_csv

    var path = getenv("COVERAGE_TEST_TMP") + "/lifecycle-wide-partials.csv"
    with open(path, "w") as file:
        file.write("x,g\n")
        for value in [Int64.MAX, Int64.MAX, -Int64.MAX, -Int64.MAX, 7]:
            file.write(String(value) + ",a\n")
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("g")])
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Int64](
                        [Int64.MAX, Int64.MAX, -Int64.MAX, -Int64.MAX, 7]
                    ),
                ),
                Series("g", Column[String](["a", "a", "a", "a", "a"])),
            ]
        )
        for batch in [1, 2, 3, 64]:
            for streaming in [False, True]:
                var sum = (
                    frame.lazy()
                    .select(col("x").sum())
                    .collect(
                        streaming=streaming, batch_size=batch, engine="cpu"
                    )
                )
                assert_equal(sum.item(0, "x").int64(), Int64(7))
                var mean = (
                    frame.lazy()
                    .select(col("x").mean())
                    .collect(
                        streaming=streaming, batch_size=batch, engine="cpu"
                    )
                )
                assert_equal(mean.item(0, "x").float64(), Float64(1.4))
                var grouped = (
                    frame.lazy()
                    .group_by("g")
                    .agg(col("x").sum())
                    .collect(
                        streaming=streaming, batch_size=batch, engine="cpu"
                    )
                )
                assert_equal(grouped.height(), 1)
                assert_equal(grouped.item(0, "g").string(), "a")
                assert_equal(grouped.item(0, "x").int64(), Int64(7))
            var from_csv = (
                scan_csv(path, schema)
                .group_by("g")
                .agg(col("x").sum())
                .collect(streaming=True, batch_size=batch, engine="cpu")
            )
            assert_equal(from_csv.item(0, "x").int64(), Int64(7))
        # Exercise the production partial-state seam explicitly: the public
        # executor may materialize a tiny grouped input despite streaming=True.
        # Each two-row state is outside Int64; only the full merged result fits.
        for grouped in [False, True]:
            var keys = List[String]()
            if grouped:
                keys.append("g")
            var expressions: List[Expr] = [
                col("x").sum().alias("sum"),
                col("x").mean().alias("mean"),
            ]
            var positive = _StreamReduction(frame.head(2), expressions, keys)
            var cannot_finalize = False
            try:
                _ = positive.finish()
            except error:
                if String(error).find("expression sum overflow") < 0:
                    raise error^
                cannot_finalize = True
            assert_true(
                cannot_finalize, "positive partial sum must exceed Int64"
            )
            var negative = _StreamReduction(
                frame.slice(2, 2), expressions, keys
            )
            var tail = _StreamReduction(frame.tail(1), expressions, keys)
            positive.merge(negative)
            positive.merge(tail)
            var merged = positive.finish()
            assert_equal(merged.item(0, "sum").int64(), Int64(7))
            assert_equal(merged.item(0, "mean").float64(), Float64(1.4))
        _ = setting^


def test_public_query_late_arithmetic_overflow_raises_and_retry_succeeds() raises:
    var setting = WorkerSetting(4)
    var values = List[Int64](length=131073, fill=1)
    values[len(values) - 1] = Int64.MAX
    var source = DataFrame([Series("x", Column[Int64](values^))])
    for streaming in [False, True]:
        for batch in [1, 65536, 131073]:
            var rejected = False
            try:
                _ = (
                    source.lazy()
                    .select((col("x") + 1).alias("value"))
                    .collect(
                        streaming=streaming, batch_size=batch, engine="cpu"
                    )
                )
            except error:
                if String(error).find("Int64 sum overflow") < 0:
                    raise error^
                rejected = True
            assert_true(
                rejected, "late arithmetic overflow must reach the caller"
            )
            assert_equal(_helpers().load(), Int64(0))
            var retry = (
                source.head(3)
                .lazy()
                .select((col("x") + 1).alias("value"))
                .collect(streaming=streaming, batch_size=batch, engine="cpu")
            )
            assert_equal(retry.height(), 3)
            for row in range(3):
                assert_equal(retry.item(row, "value").int64(), Int64(2))
    _ = setting^


def write_late_csv(path: String, malformed: Bool, final_newline: Bool) raises:
    with open(path, "w") as file:
        file.write("x,s\n")
        for row in range(33):
            file.write(
                String(row)
                + ',"part '
                + String(row)
                + ' ""quoted""\ncontinued"\n'
            )
        file.write(
            ("not-an-integer" if malformed else "33")
            + ',"final ""quoted""\ncontinued"'
        )
        if final_newline:
            file.write("\n")


def test_public_csv_late_error_is_not_hidden_by_quotes_eof_or_early_consumer_stop() raises:
    from std.os import getenv
    from dataframe import CsvField, CsvSchema, scan_csv

    var path = getenv("COVERAGE_TEST_TMP") + "/lifecycle-late-error.csv"
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("s")])
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        for final_newline in [False, True]:
            for batch in [1, 7, 32]:
                write_late_csv(path, True, final_newline)
                var query = scan_csv(path, schema)
                var first = query.head(3).collect(
                    streaming=True, batch_size=batch, engine="cpu"
                )
                assert_equal(first.height(), 3)
                for row in range(3):
                    assert_equal(first.item(row, "x").int64(), Int64(row))
                    assert_equal(
                        first.item(row, "s").string(),
                        "part " + String(row) + ' "quoted"\ncontinued',
                    )
                var rejected = False
                try:
                    _ = query.collect(
                        streaming=True, batch_size=batch, engine="cpu"
                    )
                except error:
                    var message = String(error)
                    if (
                        message.find("CSV record") < 0
                        or message.find("field 'x'") < 0
                    ):
                        raise error^
                    rejected = True
                assert_true(
                    rejected,
                    "late CSV field error must propagate after earlier batches",
                )
                assert_equal(_helpers().load(), Int64(0))
                write_late_csv(path, False, final_newline)
                var retry = query.collect(
                    streaming=True, batch_size=batch, engine="cpu"
                )
                assert_equal(retry.height(), 34)
                for row in range(34):
                    assert_equal(retry.item(row, "x").int64(), Int64(row))
                    var expected = "part " + String(
                        row
                    ) + ' "quoted"\ncontinued' if row < 33 else String(
                        'final "quoted"\ncontinued'
                    )
                    assert_equal(retry.item(row, "s").string(), expected)
        _ = setting^


def assert_reported_reduction_route(
    report: DataFrame, prefix: String, routes: List[String]
) raises:
    var matched = False
    var observed = String()
    for row in range(report.height()):
        var operation = report.item(row, "operator").string()
        var executor = report.item(row, "executor").string()
        observed += operation + ": " + executor + "; "
        if operation.startswith(prefix):
            for allowed in routes:
                matched = matched or executor == allowed
    assert_true(
        matched,
        "dispatch-sized reduction must use its actual streamed route: "
        + observed,
    )


def test_public_dispatch_sized_partial_overflow_streams_and_merges_exactly() raises:
    from std.os import getenv
    from dataframe import CsvField, CsvSchema, scan_csv

    comptime ROWS = 196609
    var values = List[Int64](length=ROWS, fill=0)
    values[0] = Int64.MAX
    values[1] = Int64.MAX
    values[65536] = -Int64.MAX
    values[65537] = -Int64.MAX
    values[ROWS - 1] = 7
    var path = getenv("COVERAGE_TEST_TMP") + "/lifecycle-dispatch-partials.csv"
    with open(path, "w") as file:
        file.write("x,g\n")
        for value in values:
            file.write(String(value) + ",0\n")
    var frame = DataFrame([Series("x", Column[Int64](values^))])
    var schema = CsvSchema([CsvField.int64("x"), CsvField.int64("g")])
    var expressions: List[Expr] = [
        col("x").sum().alias("sum"),
        col("x").mean().alias("mean"),
    ]
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        for batch in [65536, 65537]:
            # The first full batch contains +2*MAX; the next -2*MAX.
            # Both partial sums lie outside Int64, but their merged sum is seven.
            var grouped = (
                scan_csv(path, schema)
                .group_by("g")
                .agg(expressions)
                .profile(streaming=True, batch_size=batch, engine="cpu")
            )
            assert_equal(grouped[0].height(), 1)
            assert_equal(grouped[0].item(0, "g").int64(), Int64(0))
            assert_equal(grouped[0].item(0, "sum").int64(), Int64(7))
            assert_equal(
                grouped[0].item(0, "mean").float64(), Float64(7) / Float64(ROWS)
            )
            assert_reported_reduction_route(
                grouped[1], "GROUP_BY", ["streaming", "pipeline"]
            )
            var global_result = (
                frame.lazy()
                .select_exprs(expressions)
                .profile(streaming=True, batch_size=batch, engine="cpu")
            )
            assert_equal(global_result[0].item(0, "sum").int64(), Int64(7))
            assert_equal(
                global_result[0].item(0, "mean").float64(),
                Float64(7) / Float64(ROWS),
            )
            assert_reported_reduction_route(
                global_result[1], "SELECT", ["pipeline", "streaming"]
            )
            assert_equal(_helpers().load(), Int64(0))
        _ = setting^
