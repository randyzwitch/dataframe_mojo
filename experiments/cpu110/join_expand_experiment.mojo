"""Isolated #105 experiment: ordered count-prefix-fill join expansion.

The existing join already encodes keys and gathers output in parallel. This
benchmarks only its serial inner-join expansion from left ids and right CSR
groups. Contiguous left ranges count first, then fill disjoint output spans,
so the result remains left-major with each left row's right matches ordered.
"""
from std.memory import ArcPointer, Pointer
from std.time import monotonic

from dataframe.parallel import Job, configured_workers, partitions, run_jobs


comptime LEFT_ROWS = 1_000_000
comptime RIGHT_ROWS = 500_000
comptime KEYS = 500_003
comptime REPETITIONS = 3


struct _CountJob(Job):
    var left_ids: ArcPointer[List[Int]]
    var starts: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var count: Int

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        starts: ArcPointer[List[Int]],
        start: Int,
        end: Int,
    ):
        self.left_ids = left_ids.copy()
        self.starts = starts.copy()
        self.start = start
        self.end = end
        self.count = 0

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            if key >= 0:
                self.count += self.starts[][key + 1] - self.starts[][key]


struct _FillJob(Job):
    var left_ids: ArcPointer[List[Int]]
    var starts: ArcPointer[List[Int]]
    var right_flat: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var output: Int
    var left_output: Int
    var right_output: Int

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        starts: ArcPointer[List[Int]],
        right_flat: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        output: Int,
        left_output: Int,
        right_output: Int,
    ):
        self.left_ids = left_ids.copy()
        self.starts = starts.copy()
        self.right_flat = right_flat.copy()
        self.start = start
        self.end = end
        self.output = output
        self.left_output = left_output
        self.right_output = right_output

    def run(mut self) raises:
        ref left_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.left_output
        )[]
        ref right_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_output
        )[]
        var output = self.output
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            if key >= 0:
                for offset in range(self.starts[][key], self.starts[][key + 1]):
                    left_rows[output] = row
                    right_rows[output] = self.right_flat[][offset]
                    output += 1


def _serial(
    left_ids: List[Int], starts: List[Int], right_flat: List[Int]
) -> Tuple[List[Int], List[Int]]:
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    for row in range(len(left_ids)):
        var key = left_ids[row]
        if key >= 0:
            for offset in range(starts[key], starts[key + 1]):
                left_rows.append(row)
                right_rows.append(right_flat[offset])
    return (left_rows^, right_rows^)


def _parallel(
    left_ids: List[Int], starts: List[Int], right_flat: List[Int], workers: Int
) raises -> Tuple[List[Int], List[Int]]:
    var shared_left = ArcPointer(left_ids.copy())
    var shared_starts = ArcPointer(starts.copy())
    var shared_flat = ArcPointer(right_flat.copy())
    var bounds = partitions(len(left_ids), workers, 1)
    var counts = List[_CountJob](capacity=workers)
    for worker in range(workers):
        counts.append(
            _CountJob(
                shared_left,
                shared_starts,
                bounds[worker],
                bounds[worker + 1],
            )
        )
    run_jobs(counts)
    var outputs = List[Int](capacity=workers)
    var total = 0
    for worker in range(workers):
        outputs.append(total)
        total += counts[worker].count
    var left_rows = List[Int](length=total, fill=0)
    var right_rows = List[Int](length=total, fill=0)
    var fills = List[_FillJob](capacity=workers)
    for worker in range(workers):
        fills.append(
            _FillJob(
                shared_left,
                shared_starts,
                shared_flat,
                bounds[worker],
                bounds[worker + 1],
                outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
            )
        )
    run_jobs(fills)
    return (left_rows^, right_rows^)


def _input() -> Tuple[List[Int], List[Int], List[Int]]:
    var left = List[Int](capacity=LEFT_ROWS)
    var right = List[Int](capacity=RIGHT_ROWS)
    var state = UInt64(17)
    for _ in range(LEFT_ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        left.append(Int((state >> 24) % KEYS))
    for _ in range(RIGHT_ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        right.append(Int((state >> 24) % KEYS))
    var starts = List[Int](length=KEYS + 1, fill=0)
    for key in right:
        starts[key + 1] += 1
    for key in range(KEYS):
        starts[key + 1] += starts[key]
    var flat = List[Int](length=RIGHT_ROWS, fill=0)
    var next = starts.copy()
    for row in range(RIGHT_ROWS):
        var key = right[row]
        flat[next[key]] = row
        next[key] += 1
    return (left^, starts^, flat^)


def _best_serial(
    left: List[Int], starts: List[Int], flat: List[Int]
) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = _serial(left, starts, flat)
        best = min(best, monotonic() - start)
        if len(result[0]) != len(result[1]):
            raise Error("serial output lengths differ")
    return best


def _best_parallel(
    left: List[Int], starts: List[Int], flat: List[Int], workers: Int
) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = _parallel(left, starts, flat, workers)
        best = min(best, monotonic() - start)
        if len(result[0]) != len(result[1]):
            raise Error("parallel output lengths differ")
    return best


def main() raises:
    var input = _input()
    var left = input[0].copy()
    var starts = input[1].copy()
    var flat = input[2].copy()
    var serial = _serial(left, starts, flat)
    var workers = max(2, configured_workers())
    var parallel = _parallel(left, starts, flat, workers)
    if parallel[0] != serial[0] or parallel[1] != serial[1]:
        raise Error("parallel expansion differs from left-major serial order")
    print(
        "workload,left_rows,right_rows,output_rows,workers,serial_ns,parallel_ns"
    )
    print(
        "inner_expand,",
        LEFT_ROWS,
        ",",
        RIGHT_ROWS,
        ",",
        len(serial[0]),
        ",",
        workers,
        ",",
        _best_serial(left, starts, flat),
        ",",
        _best_parallel(left, starts, flat, workers),
        sep="",
    )
