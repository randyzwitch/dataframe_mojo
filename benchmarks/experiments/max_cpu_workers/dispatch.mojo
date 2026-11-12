from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.testing import assert_equal, assert_true, assert_raises
from std.time import perf_counter_ns
from dataframe.parallel import (
    Job,
    Crew,
    Pool,
    run_jobs,
    _Slot,
    _reserve_helpers,
    _release_helpers,
    _helpers,
)
from tests.support.lifecycle_jobs import (
    WorkerSetting,
    JobCounts,
    jobs_for,
    assert_job_counts,
)


def candidate[J: Job](mut jobs: List[J], mode: Int) raises:
    if mode == 0:
        run_jobs(jobs)
        return
    if not jobs:
        return
    var slots = List[_Slot[J]](capacity=len(jobs))
    while jobs:
        slots.append(_Slot[J](jobs.pop(0)))
    var helpers = _reserve_helpers(len(slots) - 1)
    var pointer = slots.unsafe_ptr()
    var count = len(slots)
    var next_job = Atomic[Int64](0)
    var counter = Pointer(to=next_job)

    def body(worker: Int) {imm pointer, imm count, imm counter}:
        while True:
            var index = Int(counter[].fetch_add(1))
            if index >= count:
                return
            pointer[unsafe_offset=index].run()

    if helpers == 0:
        body(0)
    elif mode == 1:
        parallelize(body, helpers + 1, helpers + 1)
    else:
        var ctx = DeviceContext.cpu()
        ctx.enqueue_cpu_range(body, count=helpers + 1)
        ctx.synchronize()
    _release_helpers(helpers)
    for i in range(len(slots)):
        if slots[i].failed:
            raise Error(slots[i].message)
    while slots:
        jobs.append(slots.pop(0).into_job())


struct Gauge(Movable):
    var active: Atomic[Int64]
    var peak: Atomic[Int64]

    def __init__(out self):
        self.active = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)


struct Leaf(Job):
    var gauge: ArcPointer[Gauge]
    var result: Int

    def __init__(out self, gauge: ArcPointer[Gauge]):
        self.gauge = gauge.copy()
        self.result = 0

    def run(mut self) raises:
        var active = self.gauge[].active.fetch_add(1) + 1
        var peak = self.gauge[].peak.load()
        while active > peak:
            if self.gauge[].peak.compare_exchange(peak, active):
                break
        _ = external_call["usleep", Int32](UInt32(500))
        self.result = 1
        _ = self.gauge[].active.fetch_sub(1)


struct Nested(Job):
    var gauge: ArcPointer[Gauge]
    var mode: Int
    var result: Int

    def __init__(out self, gauge: ArcPointer[Gauge], mode: Int):
        self.gauge = gauge.copy()
        self.mode = mode
        self.result = 0

    def run(mut self) raises:
        var jobs = List[Leaf]()
        for _ in range(7):
            jobs.append(Leaf(self.gauge))
        candidate(jobs, self.mode)
        for i in range(len(jobs)):
            self.result += jobs[i].result


struct Numeric(Job):
    var input: Float64
    var output: Float64

    def __init__(out self, value: Float64):
        self.input = value
        self.output = 0

    def run(mut self) raises:
        self.output = self.input + self.input


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
        for mode in range(3):
            var gauge = ArcPointer(Gauge())
            for n in [0, 1, 3, 17]:
                var nested = List[Nested]()
                for _ in range(n):
                    nested.append(Nested(gauge, mode))
                candidate(nested, mode)
                for i in range(len(nested)):
                    assert_equal(nested[i].result, 7)
                assert_true(gauge[].peak.load() <= Int64(workers))
                assert_equal(_helpers().load(), 0)
            print("nested_budget", workers, mode, gauge[].peak.load())
            for failure in [-1, 1, 6]:
                var counts = ArcPointer(JobCounts(11))
                var jobs = jobs_for(counts, 11, failure)
                if failure >= 0:
                    with assert_raises(
                        contains="deliberate lifecycle worker failure "
                        + String(failure)
                    ):
                        candidate(jobs, mode)
                else:
                    candidate(jobs, mode)
                    for i in range(len(jobs)):
                        assert_equal(jobs[i].result, i * i)
                _ = jobs^
                assert_job_counts(counts, 11, True)
            var numeric = List[Numeric]()
            for _ in range(17):
                numeric.append(Numeric(bitcast[DType.float64](UInt64(1))))
            candidate(numeric, mode)
            var correct = 0
            for i in range(len(numeric)):
                correct += Int(bitcast[DType.uint64](numeric[i].output) == 2)
            print("subnormals", workers, mode, correct, "of 17")
            if mode != 1:
                assert_equal(correct, 17)
        _ = setting^
    for workers in [1, 2, 4, 8]:
        var setting = WorkerSetting(workers)
        for loops in [0, 32, 4096]:
            for count in [workers, workers * 8]:
                var expected = List[UInt64]()
                for id in range(count):
                    var serial = Work(id, loops)
                    serial.run()
                    expected.append(serial.result)
                for repeat in range(4):
                    for reverse_mode in range(3):
                        var mode = (
                            reverse_mode if repeat % 2
                            == 0 else 2 - reverse_mode
                        )
                        var jobs = List[Work]()
                        for id in range(count):
                            jobs.append(Work(id, loops))
                        candidate(jobs, mode)
                        var start = perf_counter_ns()
                        for _ in range(100):
                            candidate(jobs, mode)
                        var elapsed = perf_counter_ns() - start
                        for id in range(count):
                            assert_equal(jobs[id].result, expected[id])
                        print(
                            "timing_ns",
                            workers,
                            loops,
                            count,
                            repeat,
                            mode,
                            elapsed // 100,
                        )
        _ = setting^
    Crew.shutdown()
    print("CPU candidate checks complete")
