"""Right-side hash index for high-cardinality inner and left joins.

The index stores row positions, rather than dense key codes. A slot holds a
chain of right rows in input order. Hash equality narrows candidate rows, and
column equality resolves collisions exactly. Workers probe disjoint left row
ranges, so concatenating their results preserves the join's left-major order.
"""
from std.memory import ArcPointer

from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .parallel import Job, partitions, run_jobs, worker_count
from .partition import Partitioner
from .series import Series
from .string_column import StringColumn


def _key_equal(left: Series, right: Series, i: Int, j: Int) -> Bool:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if left._data.isa[Column[Scalar[D]]]():
            ref a = left._data[Column[Scalar[D]]]
            ref b = right._data[Column[Scalar[D]]]
            if not a._valid(i) or not b._valid(j):
                return False
            comptime if D.is_floating_point():
                return float_key(Float64(a._get(i))) == float_key(
                    Float64(b._get(j))
                )
            else:
                return a._get(i) == b._get(j)
    if left._data.isa[BoolColumn]():
        ref a = left._data[BoolColumn]
        ref b = right._data[BoolColumn]
        return a._valid(i) and b._valid(j) and a._get(i) == b._get(j)
    ref a = left._data[StringColumn]
    ref b = right._data[StringColumn]
    return a._valid(i) and b._valid(j) and a._get(i) == b._get(j)


def _row_equal(left: List[Series], right: List[Series], i: Int, j: Int) -> Bool:
    for k in range(len(left)):
        if not _key_equal(left[k], right[k], i, j):
            return False
    return True


@fieldwise_init
struct _HashBucket(Copyable):
    var first: Int
    var heads: List[Int]
    var next_rows: List[Int]

    def mask(self) -> Int:
        return len(self.heads) - 1


struct _HashBuildJob(Job):
    """Build one hash bucket from stable right-row positions."""

    var hashes: ArcPointer[List[UInt64]]
    var order: ArcPointer[List[Int]]
    var first: Int
    var last: Int
    var result: _HashBucket

    def __init__(
        out self,
        hashes: ArcPointer[List[UInt64]],
        order: ArcPointer[List[Int]],
        first: Int,
        last: Int,
    ):
        self.hashes = hashes.copy()
        self.order = order.copy()
        self.first = first
        self.last = last
        self.result = _HashBucket(first, List[Int](), List[Int]())

    def run(mut self) raises:
        var size = 2
        while size < 2 * (self.last - self.first):
            size *= 2
        var heads = List[Int](length=size, fill=-1)
        var next_rows = List[Int](length=self.last - self.first, fill=-1)
        for position in range(self.last - 1, self.first - 1, -1):
            var row = self.order[][position]
            var slot = Int(self.hashes[][row] & UInt64(size - 1))
            next_rows[position - self.first] = heads[slot]
            heads[slot] = position
        self.result = _HashBucket(self.first, heads^, next_rows^)

    def into_result(deinit self) -> _HashBucket:
        return self.result^


struct _HashProbeJob(Job):
    var left_keys: List[Series]
    var right_keys: List[Series]
    var left_hashes: ArcPointer[List[UInt64]]
    var right_hashes: ArcPointer[List[UInt64]]
    var order: ArcPointer[List[Int]]
    var buckets: ArcPointer[List[_HashBucket]]
    var fold: Int
    var start: Int
    var end: Int
    var include_unmatched: Bool
    var left_rows: List[Int]
    var right_rows: List[Int]

    def __init__(
        out self,
        left_keys: List[Series],
        right_keys: List[Series],
        left_hashes: ArcPointer[List[UInt64]],
        right_hashes: ArcPointer[List[UInt64]],
        order: ArcPointer[List[Int]],
        buckets: ArcPointer[List[_HashBucket]],
        fold: Int,
        start: Int,
        end: Int,
        include_unmatched: Bool,
    ):
        self.left_keys = left_keys.copy()
        self.right_keys = right_keys.copy()
        self.left_hashes = left_hashes.copy()
        self.right_hashes = right_hashes.copy()
        self.order = order.copy()
        self.buckets = buckets.copy()
        self.fold = fold
        self.start = start
        self.end = end
        self.include_unmatched = include_unmatched
        self.left_rows = List[Int](capacity=end - start)
        self.right_rows = List[Int](capacity=end - start)

    def run(mut self) raises:
        for i in range(self.start, self.end):
            var hash = self.left_hashes[][i]
            var bucket = Int(hash >> 56) >> self.fold
            ref index = self.buckets[][bucket]
            var position = index.heads[Int(hash & UInt64(index.mask()))]
            var matched = False
            while position >= 0:
                var j = self.order[][position]
                if hash == self.right_hashes[][j] and _row_equal(
                    self.left_keys, self.right_keys, i, j
                ):
                    self.left_rows.append(i)
                    self.right_rows.append(j)
                    matched = True
                position = index.next_rows[position - index.first]
            if not matched and self.include_unmatched:
                self.left_rows.append(i)
                self.right_rows.append(-1)


def direct_hash_join_rows(
    left_keys: List[Series],
    right_keys: List[Series],
    include_unmatched: Bool,
) raises -> Tuple[List[Int], List[Int]]:
    """Exact left-major matches through a read-only right-row hash index."""
    var left = List[Series](capacity=len(left_keys))
    var right = List[Series](capacity=len(right_keys))
    for key in left_keys:
        left.append(key.rechunk() if key.is_chunked() else key.copy())
    for key in right_keys:
        right.append(key.rechunk() if key.is_chunked() else key.copy())
    var workers = worker_count(len(left[0]))
    var left_hashes = Partitioner(left, workers)
    var right_hashes = Partitioner(right, worker_count(len(right[0])))
    var right_parts = right_hashes.scatter(worker_count(len(right[0])))
    var shared_left_hashes = ArcPointer(left_hashes.hashes.copy())
    var shared_right_hashes = ArcPointer(right_hashes.hashes.copy())
    var shared_order = ArcPointer(right_parts.order.copy())
    var builders = List[_HashBuildJob](capacity=right_parts.buckets())
    for bucket in range(right_parts.buckets()):
        builders.append(
            _HashBuildJob(
                shared_right_hashes,
                shared_order,
                right_parts.bounds[bucket],
                right_parts.bounds[bucket + 1],
            )
        )
    run_jobs(builders)
    var indexes = List[_HashBucket](capacity=len(builders))
    while len(builders) > 0:
        indexes.append(builders.pop(0).into_result())
    var shared_indexes = ArcPointer(indexes^)
    var fold = 8
    var count = 1
    while count < right_parts.buckets():
        count *= 2
        fold -= 1
    var bounds = partitions(len(left[0]), workers, 1)
    var jobs = List[_HashProbeJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _HashProbeJob(
                left,
                right,
                shared_left_hashes,
                shared_right_hashes,
                shared_order,
                shared_indexes,
                fold,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
            )
        )
    run_jobs(jobs)
    var total = 0
    for worker in range(len(jobs)):
        total += len(jobs[worker].left_rows)
    var left_rows = List[Int](capacity=total)
    var right_rows = List[Int](capacity=total)
    for worker in range(len(jobs)):
        for i in range(len(jobs[worker].left_rows)):
            left_rows.append(jobs[worker].left_rows[i])
            right_rows.append(jobs[worker].right_rows[i])
    return (left_rows^, right_rows^)
