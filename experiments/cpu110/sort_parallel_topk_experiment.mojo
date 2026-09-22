"""#108 experiment: per-worker top-k then a k*workers final selection.

This measures selection only, after rank encoding. A rank plus original row
is a total order, so keeping k rows from each contiguous worker range cannot
discard a member of the global stable k. The final selection runs over at
most k * workers rows.
"""
from std.memory import ArcPointer
from std.time import monotonic

from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import smallest_indices, sort_indices


comptime ROWS = 1_000_000
comptime K = 10
comptime REPETITIONS = 3


def _less(keys: List[Int], a: Int, b: Int) -> Bool:
    return keys[a] < keys[b] or (keys[a] == keys[b] and a < b)


struct _PartialTopKJob(Job):
    var keys: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var k: Int
    var rows: List[Int]

    def __init__(
        out self,
        keys: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        k: Int,
    ):
        self.keys = keys.copy()
        self.start = start
        self.end = end
        self.k = k
        self.rows = List[Int]()

    def run(mut self) raises:
        var heap = List[Int](capacity=self.k)
        for row in range(self.start, self.end):
            if len(heap) < self.k:
                heap.append(row)
                var child = len(heap) - 1
                while child > 0:
                    var parent = (child - 1) // 2
                    if not _less(self.keys[], heap[parent], heap[child]):
                        break
                    var old = heap[parent]
                    heap[parent] = heap[child]
                    heap[child] = old
                    child = parent
            elif self.k > 0 and _less(self.keys[], row, heap[0]):
                heap[0] = row
                var parent = 0
                while True:
                    var largest = parent
                    var left = 2 * parent + 1
                    var right = left + 1
                    if left < len(heap) and _less(
                        self.keys[], heap[largest], heap[left]
                    ):
                        largest = left
                    if right < len(heap) and _less(
                        self.keys[], heap[largest], heap[right]
                    ):
                        largest = right
                    if largest == parent:
                        break
                    var old = heap[parent]
                    heap[parent] = heap[largest]
                    heap[largest] = old
                    parent = largest
        self.rows = heap^


def parallel_top_k(keys: List[Int], k: Int, workers: Int) raises -> List[Int]:
    if k <= 0:
        return List[Int]()
    var shared = ArcPointer(keys.copy())
    var bounds = partitions(len(keys), workers, 1)
    var jobs = List[_PartialTopKJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _PartialTopKJob(shared, bounds[worker], bounds[worker + 1], k)
        )
    run_jobs(jobs)
    var ranks = List[Int]()
    var rows = List[Int]()
    for worker in range(workers):
        for row in jobs[worker].rows:
            ranks.append(keys[row])
            rows.append(row)
    var local = sort_indices([ranks.copy(), rows.copy()])
    var result = List[Int](capacity=min(k, len(rows)))
    for index in range(min(k, len(local))):
        result.append(rows[local[index]])
    return result^


def _keys() -> List[Int]:
    var keys = List[Int](capacity=ROWS)
    var state = UInt64(0xA24BAED4963EE407)
    for _ in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        # Repeated ranks force the original-row stability tie break.
        keys.append(Int((state >> 23) % 20_003))
    return keys^


def _best_serial(keys: List[Int]) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = smallest_indices([keys.copy()], K)
        best = min(best, monotonic() - start)
        if len(result) != K:
            raise Error("serial top-k returned wrong length")
    return best


def _best_parallel(keys: List[Int], workers: Int) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = parallel_top_k(keys, K, workers)
        best = min(best, monotonic() - start)
        if len(result) != K:
            raise Error("parallel top-k returned wrong length")
    return best


def main() raises:
    var keys = _keys()
    var workers = max(2, configured_workers())
    var expected = smallest_indices([keys.copy()], K)
    var actual = parallel_top_k(keys, K, workers)
    if actual != expected:
        raise Error("parallel top-k differs from stable serial selection")
    print("workload,rows,k,workers,serial_ns,parallel_ns")
    print(
        "encoded_rank_only_repeated_int64,",
        ROWS,
        ",",
        K,
        ",",
        workers,
        ",",
        _best_serial(keys),
        ",",
        _best_parallel(keys, workers),
        sep="",
    )
