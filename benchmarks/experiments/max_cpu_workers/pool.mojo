from std.time import perf_counter_ns
from std.testing import assert_equal
from dataframe.parallel import Job, Pool, Crew
from tests.support.lifecycle_jobs import WorkerSetting


struct Work(Job):
    var seed: UInt64
    var loops: Int
    var result: UInt64

    def __init__(out self, id: Int, loops: Int):
        self.seed = UInt64(id + 1)
        self.loops = loops
        self.result = 0

    def run(mut self) raises:
        var value = self.seed
        for _ in range(self.loops):
            value = value * 6364136223846793005 + 1442695040888963407
            value ^= value >> 19
        self.result = value


def main() raises:
    for workers in [1, 2, 4, 8]:
        var setting = WorkerSetting(workers)
        for claim in [False, True]:
            for loops in [0, 32, 4096]:
                for count in [workers, workers * 8]:
                    var jobs = List[Work]()
                    var expected = List[UInt64]()
                    for i in range(count):
                        var job = Work(i, loops)
                        job.run()
                        expected.append(job.result)
                        jobs.append(job^)
                    for lifetime in [False, True]:
                        var pool = Pool(workers)
                        pool.run(jobs, claim=claim)
                        for repetition in range(5):
                            var start = perf_counter_ns()
                            for _ in range(100):
                                if lifetime:
                                    var fresh = Pool(workers)
                                    fresh.run(jobs, claim=claim)
                                    fresh.release()
                                else:
                                    pool.run(jobs, claim=claim)
                            var elapsed = perf_counter_ns() - start
                            for i in range(count):
                                assert_equal(jobs[i].result, expected[i])
                            print(
                                workers,
                                Int(claim),
                                loops,
                                count,
                                Int(lifetime),
                                repetition,
                                elapsed // 100,
                            )
                        pool.release()
        _ = setting^
    Crew.shutdown()
