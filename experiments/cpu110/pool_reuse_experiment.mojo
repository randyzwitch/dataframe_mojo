"""Two-stage filter/sum with per-stage threads vs an explicitly scoped pool.

Usage: pool_reuse_experiment ROWS WORKERS. Pool startup is amortized across
10 pipelines; report it separately. This does not create a process-global
pool or change the dataframe API.
"""
from std.memory import ArcPointer
from std.sys import argv
from std.time import monotonic
from dataframe.parallel import Job, Pool, run_jobs


struct FilterJob(Job):
    var input: ArcPointer[List[Int64]]
    var start: Int
    var end: Int
    var output: List[Int64]

    def __init__(
        out self, input: ArcPointer[List[Int64]], start: Int, end: Int
    ):
        self.input = input
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


def pipeline(
    input: ArcPointer[List[Int64]], workers: Int, pooled: Bool, mut pool: Pool
) raises -> Int64:
    var jobs = List[FilterJob]()
    for w in range(workers):
        jobs.append(
            FilterJob(
                input,
                len(input[]) * w // workers,
                len(input[]) * (w + 1) // workers,
            )
        )
    if pooled:
        pool.run(jobs)
    else:
        run_jobs(jobs)
    var sums = List[SumJob]()
    while len(jobs) > 0:
        sums.append(SumJob(jobs.pop().take()))
    if pooled:
        pool.run(sums)
    else:
        run_jobs(sums)
    var total = Int64(0)
    for i in range(len(sums)):
        total += sums[i].total
    return total


def main() raises:
    var args = argv()
    var rows = Int(String(args[1]))
    var workers = Int(String(args[2]))
    var values = List[Int64](capacity=rows)
    var expected = Int64(0)
    for i in range(rows):
        var value = Int64(i % 1000) - 500
        values.append(value)
        if value > 0:
            expected += value
    var input = ArcPointer(values^)
    var started = monotonic()
    var pool = Pool(workers)
    print("startup_ns", monotonic() - started, sep=",")
    for pooled in [False, True]:
        var best = Int.MAX
        var elapsed_total = 0
        for iteration in range(11):
            started = monotonic()
            var actual = pipeline(input, workers, pooled, pool)
            var elapsed = monotonic() - started
            if actual != expected:
                raise Error("filter/sum differs from reference")
            if iteration > 0:
                best = min(best, elapsed)
                elapsed_total += elapsed
        print(
            "pooled",
            pooled,
            "best_ns",
            best,
            "mean_ns",
            elapsed_total // 10,
            sep=",",
        )
    pool.release()
