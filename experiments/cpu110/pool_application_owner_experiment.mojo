"""Explicit-owner thread-pool experiment for #103.

This is deliberately not a process-global pool. `ApplicationPool` is owned by
an application/pipeline driver and must be closed before that driver returns;
its destructor is a second safety net. This chosen lifetime model avoids
leaving workers parked in JIT code during `mojo run` teardown. A hidden
process-wide implementation and its teardown hooks are not evaluated here.

Nested work is intentionally inline. A Job cannot borrow this mutable owner,
so it calls `run_nested_inline`, which executes all child jobs in input order
and raises the first child error only after the children have drained. This
avoids recursively mutating the pool's single active round or oversubscribing
it. The owner itself preserves Pool's outer-round error/reuse contract.

The benchmark compares the existing per-stage `run_jobs` path at the legacy
65,536-row grain with a candidate 8,192-row grain, then the same candidate
with one explicit owner reused across pipelines. It covers 4K, 16K, and 100K
frames. Output-frame validation and setup occur outside the timed interval.

Usage: pool_application_owner_experiment WORKERS REPETITIONS
       pool_application_owner_experiment check WORKERS
"""
from std.memory import ArcPointer
from std.sys import argv
from std.time import monotonic

from dataframe.parallel import Job, Pool, run_jobs


comptime LEGACY_GRAIN = 65_536
comptime CANDIDATE_GRAIN = 8_192


struct ApplicationPool(Movable):
    """An explicit caller-owned pool, never a hidden process-global service."""

    var pool: Pool
    var closed: Bool

    def __init__(out self, workers: Int):
        self.pool = Pool(workers)
        self.closed = False

    def run[J: Job](mut self, mut jobs: List[J]) raises:
        if self.closed:
            raise Error("application pool is closed")
        self.pool.run(jobs)

    def close(mut self):
        if not self.closed:
            self.pool.release()
            self.closed = True

    def __deinit__(deinit self):
        # Scoped destruction joins workers even when an application raises.
        self.close()


def run_nested_inline[J: Job](mut jobs: List[J]) raises:
    """Nested contract: drain children in order, then raise first error.

    A nested Job must use this helper instead of trying to submit into the
    application owner. The helper does not touch the active pool round, so it
    cannot deadlock or start a second worker set from within a worker.
    """
    var first_error = String()
    for i in range(len(jobs)):
        try:
            jobs[i].run()
        except e:
            if first_error == "":
                first_error = String(e)
    if first_error != "":
        raise Error(first_error)


struct NestedChild(Job):
    var value: Int64
    var fail: Bool
    var output: Int64

    def __init__(out self, value: Int64, fail: Bool = False):
        self.value = value
        self.fail = fail
        self.output = 0

    def run(mut self) raises:
        self.output = self.value
        if self.fail:
            raise Error("nested child " + String(self.value) + " failed")


struct NestedParent(Job):
    var children: List[NestedChild]
    var total: Int64

    def __init__(out self, count: Int, fail_at: Int = -1):
        self.children = List[NestedChild](capacity=count)
        for i in range(count):
            self.children.append(NestedChild(Int64(i + 1), i == fail_at))
        self.total = 0

    def run(mut self) raises:
        # This is safe regardless of whether this parent runs on the caller
        # or one of ApplicationPool's workers.
        run_nested_inline(self.children)
        for i in range(len(self.children)):
            self.total += self.children[i].output


struct Marker(Job):
    var output: Int

    def __init__(out self):
        self.output = 0

    def run(mut self) raises:
        self.output = 1


def check_nested_and_error_contract(mut owner: ApplicationPool) raises:
    # Two parent jobs force the owner path instead of a one-job serial round.
    var nested = List[NestedParent]()
    nested.append(NestedParent(4))
    nested.append(NestedParent(7))
    owner.run(nested)
    if nested[0].total != 10 or nested[1].total != 28:
        raise Error("nested inline result contract changed")

    # The failing parent drains every child inline before its outer slot
    # records the error; Pool then drains the other outer parent and re-raises.
    var failing = List[NestedParent]()
    failing.append(NestedParent(4, fail_at=1))
    failing.append(NestedParent(3))
    var saw_expected_error = False
    try:
        owner.run(failing)
    except e:
        saw_expected_error = String(e) == "nested child 2 failed"
    if not saw_expected_error:
        raise Error("nested inline error contract changed")

    # An error cannot leave the application owner busy for its next pipeline.
    var markers = List[Marker]()
    markers.append(Marker())
    markers.append(Marker())
    owner.run(markers)
    if markers[0].output != 1 or markers[1].output != 1:
        raise Error("pool did not recover after nested error")


struct FilterJob(Job):
    var input: ArcPointer[List[Int64]]
    var start: Int
    var end: Int
    var output: List[Int64]

    def __init__(
        out self, input: ArcPointer[List[Int64]], start: Int, end: Int
    ):
        self.input = input.copy()
        self.start = start
        self.end = end
        self.output = List[Int64]()

    def run(mut self) raises:
        self.output.reserve((self.end - self.start) // 2 + 1)
        for i in range(self.start, self.end):
            var value = self.input[][i]
            if value > 0:
                self.output.append(value)

    def take(deinit self) -> List[Int64]:
        return self.output^


struct SumJob(Job):
    var input: List[Int64]
    var total: Int64

    def __init__(out self, var input: List[Int64]):
        self.input = input^
        self.total = 0

    def run(mut self) raises:
        for value in self.input:
            self.total += value


def stage_workers(rows: Int, cap: Int, grain: Int) -> Int:
    return max(1, min(cap, rows // grain))


def filter_jobs(
    input: ArcPointer[List[Int64]], workers: Int
) -> List[FilterJob]:
    var jobs = List[FilterJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            FilterJob(
                input,
                len(input[]) * worker // workers,
                len(input[]) * (worker + 1) // workers,
            )
        )
    return jobs^


def sum_jobs(var filters: List[FilterJob]) -> List[SumJob]:
    var sums = List[SumJob](capacity=len(filters))
    while len(filters) > 0:
        sums.append(SumJob(filters.pop().take()))
    return sums^


def combine(sums: List[SumJob]) -> Int64:
    var total = Int64(0)
    for i in range(len(sums)):
        total += sums[i].total
    return total


def pipeline_fresh(
    input: ArcPointer[List[Int64]], workers: Int
) raises -> Int64:
    var filters = filter_jobs(input, workers)
    run_jobs(filters)
    var sums = sum_jobs(filters^)
    run_jobs(sums)
    return combine(sums)


def pipeline_owned(
    input: ArcPointer[List[Int64]], workers: Int, mut owner: ApplicationPool
) raises -> Int64:
    var filters = filter_jobs(input, workers)
    owner.run(filters)
    var sums = sum_jobs(filters^)
    owner.run(sums)
    return combine(sums)


def input_for(rows: Int) -> Tuple[ArcPointer[List[Int64]], Int64]:
    var values = List[Int64](capacity=rows)
    var expected = Int64(0)
    for i in range(rows):
        var value = Int64(i % 1000) - 500
        values.append(value)
        if value > 0:
            expected += value
    return (ArcPointer(values^), expected)


def check_filter_sum_contract(mut owner: ApplicationPool, cap: Int) raises:
    """Correctness-only pass; it deliberately does not read the clock."""
    for rows in [4_096, 16_384, 100_000]:
        var input = input_for(rows)
        var legacy = pipeline_fresh(
            input[0], stage_workers(rows, cap, LEGACY_GRAIN)
        )
        var candidate_fresh = pipeline_fresh(
            input[0], stage_workers(rows, cap, CANDIDATE_GRAIN)
        )
        var candidate_owned = pipeline_owned(
            input[0], stage_workers(rows, cap, CANDIDATE_GRAIN), owner
        )
        if (
            legacy != input[1]
            or candidate_fresh != input[1]
            or candidate_owned != input[1]
        ):
            raise Error("filter/sum correctness contract changed")


def report(
    rows: Int,
    grain: Int,
    stage_count: Int,
    mode: String,
    best_ns: Int,
    mean_ns: Int,
):
    print(
        "rows,grain,stage_workers,mode,best_ns,mean_ns=",
        rows,
        ",",
        grain,
        ",",
        stage_count,
        ",",
        mode,
        ",",
        best_ns,
        ",",
        mean_ns,
        sep="",
    )


def benchmark(
    rows: Int,
    grain: Int,
    cap: Int,
    repetitions: Int,
    mode: String,
    mut owner: ApplicationPool,
) raises:
    var input = input_for(rows)
    var workers = stage_workers(rows, cap, grain)
    # Warm and validate outside the reported interval.
    var checked = pipeline_owned(
        input[0], workers, owner
    ) if mode == "owned" else pipeline_fresh(input[0], workers)
    if checked != input[1]:
        raise Error("filter/sum differs from reference")
    var best = Int.MAX
    var total = 0
    for _ in range(repetitions):
        var started = monotonic()
        var actual = pipeline_owned(
            input[0], workers, owner
        ) if mode == "owned" else pipeline_fresh(input[0], workers)
        var elapsed = monotonic() - started
        if actual != input[1]:
            raise Error("filter/sum differs from reference")
        best = min(best, elapsed)
        total += elapsed
    report(rows, grain, workers, mode, best, total // repetitions)


def main() raises:
    var args = argv()
    if len(args) == 3 and String(args[1]) == "check":
        var check_workers = Int(String(args[2]))
        if check_workers < 1:
            raise Error("workers must be positive")
        var checked_owner = ApplicationPool(check_workers)
        check_nested_and_error_contract(checked_owner)
        check_filter_sum_contract(checked_owner, check_workers)
        checked_owner.close()
        print("pool_owner_check=pass")
        return
    if len(args) != 3:
        raise Error(
            "usage: pool_application_owner_experiment WORKERS REPETITIONS"
        )
    var workers = Int(String(args[1]))
    var repetitions = Int(String(args[2]))
    if workers < 1 or repetitions < 1:
        raise Error("workers and repetitions must be positive")

    var started = monotonic()
    var owner = ApplicationPool(workers)
    print("owner_startup_ns=", monotonic() - started, sep="")
    check_nested_and_error_contract(owner)
    for rows in [4_096, 16_384, 100_000]:
        benchmark(rows, LEGACY_GRAIN, workers, repetitions, "fresh", owner)
        benchmark(rows, CANDIDATE_GRAIN, workers, repetitions, "fresh", owner)
        benchmark(rows, CANDIDATE_GRAIN, workers, repetitions, "owned", owner)
    owner.close()
