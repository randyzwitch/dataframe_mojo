"""#108 experiment: stable parallel radix sort of one signed order word.

`sort_indices` compares signed Int rank words. Flipping the sign bit maps that
order to unsigned radix order. Four LSD passes retain stability: every worker
scatters a contiguous source range, and its cursor for each bucket follows all
earlier workers' counts. Rank encoding (including null/direction words) is out
of scope; this tests only a single already-encoded word.
"""
from std.memory import ArcPointer, Pointer, bitcast
from std.time import monotonic

from dataframe.parallel import Job, Pool, configured_workers, partitions
from dataframe.series import sort_indices


comptime ROWS = 1_000_000
comptime REPETITIONS = 3
comptime RADIX = 1 << 16
comptime MASK = UInt64(RADIX - 1)


def _bucket(keys: List[Int], row: Int, shift: Int) -> Int:
    var word = bitcast[DType.uint64](Int64(keys[row]) ^ Int64.MIN)
    return Int((word >> UInt64(shift)) & MASK)


struct _RadixCountJob(Job):
    var keys: ArcPointer[List[Int]]
    var source: Int
    var start: Int
    var end: Int
    var shift: Int
    var counts: List[Int]

    def __init__(
        out self,
        keys: ArcPointer[List[Int]],
        source: Int,
        start: Int,
        end: Int,
        shift: Int,
    ):
        self.keys = keys.copy()
        self.source = source
        self.start = start
        self.end = end
        self.shift = shift
        self.counts = List[Int]()

    def run(mut self) raises:
        ref source = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.source
        )[]
        self.counts = List[Int](length=RADIX, fill=0)
        for index in range(self.start, self.end):
            self.counts[_bucket(self.keys[], source[index], self.shift)] += 1


struct _RadixScatterJob(Job):
    var keys: ArcPointer[List[Int]]
    var source: Int
    var target: Int
    var start: Int
    var end: Int
    var shift: Int
    var cursor: List[Int]

    def __init__(
        out self,
        keys: ArcPointer[List[Int]],
        source: Int,
        target: Int,
        start: Int,
        end: Int,
        shift: Int,
        var cursor: List[Int],
    ):
        self.keys = keys.copy()
        self.source = source
        self.target = target
        self.start = start
        self.end = end
        self.shift = shift
        self.cursor = cursor^

    def run(mut self) raises:
        ref source = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.source
        )[]
        ref target = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.target
        )[]
        for index in range(self.start, self.end):
            var row = source[index]
            var bucket = _bucket(self.keys[], row, self.shift)
            target[self.cursor[bucket]] = row
            self.cursor[bucket] += 1


def parallel_radix_indices(keys: List[Int], workers: Int) raises -> List[Int]:
    var shared = ArcPointer(keys.copy())
    var source = List[Int](capacity=len(keys))
    for row in range(len(keys)):
        source.append(row)
    var target = List[Int](length=len(keys), fill=0)
    var bounds = partitions(len(keys), workers, 1)
    var pool = Pool(workers)
    for shift in [0, 16, 32, 48]:
        var counts = List[_RadixCountJob](capacity=workers)
        for worker in range(workers):
            counts.append(
                _RadixCountJob(
                    shared,
                    Int(Pointer(to=source)),
                    bounds[worker],
                    bounds[worker + 1],
                    shift,
                )
            )
        pool.run(counts)
        var bucket_start = List[Int](length=RADIX, fill=0)
        for bucket in range(RADIX):
            for worker in range(workers):
                bucket_start[bucket] += counts[worker].counts[bucket]
        var total = 0
        for bucket in range(RADIX):
            var count = bucket_start[bucket]
            bucket_start[bucket] = total
            total += count
        var cursor = bucket_start.copy()
        var scatters = List[_RadixScatterJob](capacity=workers)
        for worker in range(workers):
            var worker_cursor = cursor.copy()
            for bucket in range(RADIX):
                cursor[bucket] += counts[worker].counts[bucket]
            scatters.append(
                _RadixScatterJob(
                    shared,
                    Int(Pointer(to=source)),
                    Int(Pointer(to=target)),
                    bounds[worker],
                    bounds[worker + 1],
                    shift,
                    worker_cursor^,
                )
            )
        pool.run(scatters)
        var old = source^
        source = target^
        target = old^
    pool.release()
    return source^


def _keys() -> List[Int]:
    var keys = List[Int](capacity=ROWS)
    var state = UInt64(0x8CB92BA72F3D8DD7)
    for _ in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        # Full signed range, with repeats, exercises sign handling/stability.
        keys.append(Int(bitcast[DType.int64](state & ~UInt64(0xF))))
    return keys^


def _best_merge(keys: List[Int]) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = sort_indices([keys.copy()])
        best = min(best, monotonic() - start)
        if len(result) != len(keys):
            raise Error("merge sort returned wrong length")
    return best


def _best_radix(keys: List[Int], workers: Int) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = parallel_radix_indices(keys, workers)
        best = min(best, monotonic() - start)
        if len(result) != len(keys):
            raise Error("radix sort returned wrong length")
    return best


def main() raises:
    var keys = _keys()
    var workers = max(2, configured_workers())
    var expected = sort_indices([keys.copy()])
    var actual = parallel_radix_indices(keys, workers)
    if actual != expected:
        raise Error("radix order differs from stable signed merge order")
    print("workload,rows,workers,merge_ns,parallel_radix_ns")
    print(
        "encoded_signed_rank_only_repeated_int64,",
        ROWS,
        ",",
        workers,
        ",",
        _best_merge(keys),
        ",",
        _best_radix(keys, workers),
        sep="",
    )
