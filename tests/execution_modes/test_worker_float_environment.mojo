"""Worker dispatch must preserve subnormal inputs and results.

Changing the thread runtime must not turn small nonzero values into zero by
enabling flush-to-zero arithmetic inside worker callbacks.
"""
# contract-group: execution.lifecycle
from std.memory import bitcast
from std.testing import assert_equal
from dataframe.parallel import Job, Crew, Pool, run_jobs
from tests.support.lifecycle_jobs import WorkerSetting


struct SubnormalJob(Job):
    var input32: Float32
    var input64: Float64
    var output32: Float32
    var output64: Float64

    def __init__(out self, negative: Bool):
        self.input32 = bitcast[DType.float32](
            UInt32(1) | (UInt32(negative) << 31)
        )
        self.input64 = bitcast[DType.float64](
            UInt64(1) | (UInt64(negative) << 63)
        )
        self.output32 = 0
        self.output64 = 0

    def run(mut self) raises:
        # Inputs reside in the job, so this exercises worker arithmetic
        # rather than constant folding on the calling thread.
        self.output32 = self.input32 + self.input32
        self.output64 = self.input64 + self.input64


def test_worker_paths_preserve_signed_float32_and_float64_subnormals() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for with_crew in [False, True]:
            var crew = Crew.start() if with_crew else Crew()
            var pool = Pool(workers)
            for mode in range(3):
                for _ in range(2):
                    var jobs = List[SubnormalJob]()
                    for row in range(17):
                        jobs.append(SubnormalJob(row % 2 != 0))
                    if mode == 0:
                        run_jobs(jobs)
                    else:
                        pool.run(jobs, claim=mode == 2)
                    for row in range(len(jobs)):
                        var negative = row % 2 != 0
                        assert_equal(
                            bitcast[DType.uint32](jobs[row].output32),
                            UInt32(2) | (UInt32(negative) << 31),
                        )
                        assert_equal(
                            bitcast[DType.uint64](jobs[row].output64),
                            UInt64(2) | (UInt64(negative) << 63),
                        )
            pool.release()
            crew.release()
        _ = setting^
