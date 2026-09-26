"""Right-side hash index for high-cardinality joins.

Each open-addressed slot holds its first matching right row. Extra rows for
the same key use a compact duplicate chain; distinct keys with the same hash
continue to the next slot. Exact column equality resolves hash collisions.
Building in reverse row order and probing disjoint left ranges preserves the
join's documented left-major, right-input match order.
"""
from std.memory import ArcPointer, bitcast

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
struct _HashSlot(Copyable):
    var row: Int32
    var next_position: Int32
    var key: UInt64


@fieldwise_init
struct _DuplicateEntry(Copyable):
    var row: Int32
    var next_position: Int32


@fieldwise_init
struct _HashBucket(Copyable):
    var slots: List[_HashSlot]
    var duplicates: List[_DuplicateEntry]

    def mask(self) -> Int:
        return len(self.slots) - 1


struct _HashBuildJob(Job):
    """Build one hash bucket from stable right-row positions."""

    var hashes: ArcPointer[List[UInt64]]
    var order: ArcPointer[List[Int]]
    var right_keys: List[Series]
    var typed_int: Bool
    var first: Int
    var last: Int
    var result: _HashBucket

    def __init__(
        out self,
        hashes: ArcPointer[List[UInt64]],
        order: ArcPointer[List[Int]],
        right_keys: List[Series],
        typed_int: Bool,
        first: Int,
        last: Int,
    ):
        self.hashes = hashes.copy()
        self.order = order.copy()
        self.right_keys = right_keys.copy()
        self.typed_int = typed_int
        self.first = first
        self.last = last
        self.result = _HashBucket(List[_HashSlot](), List[_DuplicateEntry]())

    def run(mut self) raises:
        var size = 2
        # Keep at least one-third of slots empty while avoiding a second
        # power-of-two jump for buckets with many duplicate rows.
        while size * 2 < 3 * (self.last - self.first):
            size *= 2
        var slots = List[_HashSlot](length=size, fill=_HashSlot(-1, -1, 0))
        var duplicates = List[_DuplicateEntry]()
        for position in range(self.last - 1, self.first - 1, -1):
            var row = self.order[][position]
            var hash = self.hashes[][row]
            var key = hash
            if self.typed_int:
                ref values = self.right_keys[0]._data[Column[Int64]]
                if not values._valid(row):
                    continue
                key = bitcast[DType.uint64](values._get(row))
            var slot = Int(hash & UInt64(size - 1))
            while slots[slot].row >= 0:
                var same = slots[slot].key == key
                if same and not self.typed_int:
                    same = _row_equal(
                        self.right_keys,
                        self.right_keys,
                        row,
                        Int(slots[slot].row),
                    )
                if same:
                    duplicates.append(
                        _DuplicateEntry(
                            slots[slot].row, slots[slot].next_position
                        )
                    )
                    slots[slot].row = Int32(row)
                    slots[slot].next_position = Int32(len(duplicates) - 1)
                    break
                slot = (slot + 1) & (size - 1)
            if slots[slot].row < 0:
                slots[slot] = _HashSlot(Int32(row), -1, key)
        self.result = _HashBucket(slots^, duplicates^)

    def into_result(deinit self) -> _HashBucket:
        return self.result^


struct _HashProbeJob(Job):
    var left_keys: List[Series]
    var right_keys: List[Series]
    var left_hashes: ArcPointer[List[UInt64]]
    var buckets: ArcPointer[List[_HashBucket]]
    var fold: Int
    var start: Int
    var end: Int
    var include_unmatched: Bool
    var omit_identity: Bool
    var identity: Bool
    # -1 emits (left, right) pairs; 1 or 0 emits each left row once when
    # it has a match (semi) or none (anti). Null left keys never match.
    var membership: Int
    var left_rows: List[Int]
    var right_rows: List[Int]

    def __init__(
        out self,
        left_keys: List[Series],
        right_keys: List[Series],
        left_hashes: ArcPointer[List[UInt64]],
        buckets: ArcPointer[List[_HashBucket]],
        fold: Int,
        start: Int,
        end: Int,
        include_unmatched: Bool,
        omit_identity: Bool,
        membership: Int = -1,
    ):
        self.left_keys = left_keys.copy()
        self.right_keys = right_keys.copy()
        self.left_hashes = left_hashes.copy()
        self.buckets = buckets.copy()
        self.fold = fold
        self.start = start
        self.end = end
        self.include_unmatched = include_unmatched
        self.omit_identity = (
            omit_identity
            and membership < 0
            and len(left_keys) == 1
            and left_keys[0]._data.isa[StringColumn]()
        )
        self.identity = False
        self.membership = membership
        self.left_rows = List[Int](
            capacity=0 if self.omit_identity else end - start
        )
        self.right_rows = List[Int](
            capacity=0 if membership >= 0 else end - start
        )

    def _matches(self, i: Int) -> Bool:
        """Whether probe row i has at least one exact match in the index.
        A null key never matches: `_key_equal` requires both sides valid,
        and the Int64 path checks validity itself."""
        var hash = self.left_hashes[][i]
        var bucket = Int(hash >> 56) >> self.fold
        ref index = self.buckets[][bucket]
        var position = Int(hash & UInt64(index.mask()))
        if (
            len(self.left_keys) == 1
            and self.left_keys[0]._data.isa[Column[Int64]]()
        ):
            ref left = self.left_keys[0]._data[Column[Int64]]
            if not left._valid(i):
                return False
            var key = bitcast[DType.uint64](left._get(i))
            while index.slots[position].row >= 0:
                if key == index.slots[position].key:
                    return True
                position = (position + 1) & index.mask()
            return False
        while index.slots[position].row >= 0:
            ref slot = index.slots[position]
            if hash == slot.key and _row_equal(
                self.left_keys, self.right_keys, i, Int(slot.row)
            ):
                return True
            position = (position + 1) & index.mask()
        return False

    def run_membership(mut self) raises:
        """Semi (membership == 1) or anti (0): each left row at most once,
        in row order; duplicate right keys do not repeat it."""
        var keep = self.membership == 1
        for i in range(self.start, self.end):
            if self._matches(i) == keep:
                self.left_rows.append(i)

    def run(mut self) raises:
        if self.membership >= 0:
            self.run_membership()
            return
        if (
            len(self.left_keys) == 1
            and self.left_keys[0]._data.isa[Column[Int64]]()
        ):
            ref left = self.left_keys[0]._data[Column[Int64]]
            var left_all_valid = len(left._bits[]) == 0
            for i in range(self.start, self.end):
                if not (left_all_valid or left._valid(i)):
                    if self.include_unmatched:
                        self.left_rows.append(i)
                        self.right_rows.append(-1)
                    continue
                var hash = self.left_hashes[][i]
                var bucket = Int(hash >> 56) >> self.fold
                ref index = self.buckets[][bucket]
                var position = Int(hash & UInt64(index.mask()))
                var matched = False
                var key = bitcast[DType.uint64](left._get(i))
                while index.slots[position].row >= 0:
                    ref slot = index.slots[position]
                    if key == slot.key:
                        self.left_rows.append(i)
                        self.right_rows.append(Int(slot.row))
                        var next = Int(slot.next_position)
                        while next >= 0:
                            ref entry = index.duplicates[next]
                            self.left_rows.append(i)
                            self.right_rows.append(Int(entry.row))
                            next = Int(entry.next_position)
                        matched = True
                        break
                    position = (position + 1) & index.mask()
                if not matched and self.include_unmatched:
                    self.left_rows.append(i)
                    self.right_rows.append(-1)
            return
        if (
            len(self.left_keys) == 1
            and self.left_keys[0]._data.isa[StringColumn]()
        ):
            ref left = self.left_keys[0]._data[StringColumn]
            ref right = self.right_keys[0]._data[StringColumn]
            var all_valid = len(left._bits[]) == 0 and len(right._bits[]) == 0
            var generic_start = self.start
            # Emit only right positions while every input row has one output.
            # Missing inner matches or duplicate right keys end this prefix.
            if self.omit_identity:
                var row = self.start
                while row < self.end:
                    if not left._valid(row):
                        if self.include_unmatched:
                            self.right_rows.append(-1)
                            row += 1
                            continue
                        break
                    var hash = self.left_hashes[][row]
                    var bucket = Int(hash >> 56) >> self.fold
                    ref index = self.buckets[][bucket]
                    var position = Int(hash & UInt64(index.mask()))
                    var matched_row = -1
                    var duplicate = False
                    while index.slots[position].row >= 0:
                        ref slot = index.slots[position]
                        var j = Int(slot.row)
                        if hash == slot.key and (
                            left._equal_at_valid(
                                right, row, j
                            ) if all_valid else left._equal_at(right, row, j)
                        ):
                            matched_row = j
                            duplicate = slot.next_position >= 0
                            break
                        position = (position + 1) & index.mask()
                    if matched_row >= 0 and not duplicate:
                        self.right_rows.append(matched_row)
                        row += 1
                        continue
                    if matched_row < 0 and self.include_unmatched:
                        self.right_rows.append(-1)
                        row += 1
                        continue
                    break
                if row == self.end:
                    self.identity = True
                    return
                # Materialize the prefix once, then use the ordinary path.
                self.left_rows = List[Int](capacity=self.end - self.start)
                for previous in range(self.start, row):
                    self.left_rows.append(previous)
                generic_start = row
            for i in range(generic_start, self.end):
                var hash = self.left_hashes[][i]
                var bucket = Int(hash >> 56) >> self.fold
                ref index = self.buckets[][bucket]
                var position = Int(hash & UInt64(index.mask()))
                var matched = False
                while index.slots[position].row >= 0:
                    ref slot = index.slots[position]
                    var j = Int(slot.row)
                    if hash == slot.key and (
                        left._equal_at_valid(
                            right, i, j
                        ) if all_valid else left._equal_at(right, i, j)
                    ):
                        self.left_rows.append(i)
                        self.right_rows.append(j)
                        var next = Int(slot.next_position)
                        while next >= 0:
                            ref entry = index.duplicates[next]
                            self.left_rows.append(i)
                            self.right_rows.append(Int(entry.row))
                            next = Int(entry.next_position)
                        matched = True
                        break
                    position = (position + 1) & index.mask()
                if not matched and self.include_unmatched:
                    self.left_rows.append(i)
                    self.right_rows.append(-1)
            return
        if (
            len(self.left_keys) == 2
            and self.left_keys[0]._data.isa[Column[Int64]]()
            and self.left_keys[1]._data.isa[Column[Int64]]()
        ):
            ref left_first = self.left_keys[0]._data[Column[Int64]]
            ref left_second = self.left_keys[1]._data[Column[Int64]]
            ref right_first = self.right_keys[0]._data[Column[Int64]]
            ref right_second = self.right_keys[1]._data[Column[Int64]]
            if (
                len(left_first._bits[]) == 0
                and len(left_second._bits[]) == 0
                and len(right_first._bits[]) == 0
                and len(right_second._bits[]) == 0
            ):
                for i in range(self.start, self.end):
                    var first = left_first._get(i)
                    var second = left_second._get(i)
                    var hash = self.left_hashes[][i]
                    var bucket = Int(hash >> 56) >> self.fold
                    ref index = self.buckets[][bucket]
                    var position = Int(hash & UInt64(index.mask()))
                    var matched = False
                    while index.slots[position].row >= 0:
                        ref slot = index.slots[position]
                        var j = Int(slot.row)
                        if (
                            hash == slot.key
                            and first == right_first._get(j)
                            and second == right_second._get(j)
                        ):
                            self.left_rows.append(i)
                            self.right_rows.append(j)
                            var next = Int(slot.next_position)
                            while next >= 0:
                                ref entry = index.duplicates[next]
                                self.left_rows.append(i)
                                self.right_rows.append(Int(entry.row))
                                next = Int(entry.next_position)
                            matched = True
                            break
                        position = (position + 1) & index.mask()
                    if not matched and self.include_unmatched:
                        self.left_rows.append(i)
                        self.right_rows.append(-1)
                return
        for i in range(self.start, self.end):
            var hash = self.left_hashes[][i]
            var bucket = Int(hash >> 56) >> self.fold
            ref index = self.buckets[][bucket]
            var position = Int(hash & UInt64(index.mask()))
            var matched = False
            while index.slots[position].row >= 0:
                ref slot = index.slots[position]
                var j = Int(slot.row)
                if hash == slot.key and _row_equal(
                    self.left_keys, self.right_keys, i, j
                ):
                    self.left_rows.append(i)
                    self.right_rows.append(j)
                    var next = Int(slot.next_position)
                    while next >= 0:
                        ref entry = index.duplicates[next]
                        self.left_rows.append(i)
                        self.right_rows.append(Int(entry.row))
                        next = Int(entry.next_position)
                    matched = True
                    break
                position = (position + 1) & index.mask()
            if not matched and self.include_unmatched:
                self.left_rows.append(i)
                self.right_rows.append(-1)


@fieldwise_init
struct _HashIndex(Movable):
    """A built right-row index plus the probe-side hashes it was keyed with."""

    var left: List[Series]
    var right: List[Series]
    var left_hashes: ArcPointer[List[UInt64]]
    var indexes: ArcPointer[List[_HashBucket]]
    var fold: Int
    var workers: Int


def _build_hash_index(
    left_keys: List[Series], right_keys: List[Series]
) raises -> _HashIndex:
    if len(right_keys[0]) > Int(Int32.MAX):
        raise Error("Direct hash join exceeds 32-bit row index capacity")
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
    var typed_int = len(right) == 1 and right[0]._data.isa[Column[Int64]]()
    var builders = List[_HashBuildJob](capacity=right_parts.buckets())
    for bucket in range(right_parts.buckets()):
        builders.append(
            _HashBuildJob(
                shared_right_hashes,
                shared_order,
                right,
                typed_int,
                right_parts.bounds[bucket],
                right_parts.bounds[bucket + 1],
            )
        )
    run_jobs(builders)
    var indexes = List[_HashBucket](capacity=len(builders))
    while len(builders) > 0:
        indexes.append(builders.pop(0).into_result())
    var fold = 8
    var count = 1
    while count < right_parts.buckets():
        count *= 2
        fold -= 1
    return _HashIndex(
        left^, right^, shared_left_hashes, ArcPointer(indexes^), fold, workers
    )


def direct_hash_semi_anti_rows(
    left_keys: List[Series], right_keys: List[Series], keep_matches: Bool
) raises -> List[Int]:
    """Left rows with (semi) or without (anti) an exact match, in row order.

    Same index as `direct_hash_join_rows`, probed for membership only: a
    left row appears once however many right rows share its key, and a
    null left key never matches, so semi drops it and anti keeps it.
    """
    var index = _build_hash_index(left_keys, right_keys)
    var bounds = partitions(len(index.left[0]), index.workers, 1)
    var jobs = List[_HashProbeJob](capacity=index.workers)
    for worker in range(index.workers):
        jobs.append(
            _HashProbeJob(
                index.left,
                index.right,
                index.left_hashes,
                index.indexes,
                index.fold,
                bounds[worker],
                bounds[worker + 1],
                False,
                False,
                membership=1 if keep_matches else 0,
            )
        )
    run_jobs(jobs)
    var total = 0
    for worker in range(len(jobs)):
        total += len(jobs[worker].left_rows)
    var rows = List[Int](capacity=total)
    for worker in range(len(jobs)):
        for row in jobs[worker].left_rows:
            rows.append(row)
    return rows^


def direct_hash_join_rows(
    left_keys: List[Series],
    right_keys: List[Series],
    include_unmatched: Bool,
    omit_identity: Bool = False,
) raises -> Tuple[List[Int], List[Int], Bool]:
    """Exact left-major matches through a read-only right-row hash index.

    The third result means every probe row appears exactly once in order;
    when requested, the first list is then omitted as an implicit identity.
    """
    var index = _build_hash_index(left_keys, right_keys)
    var workers = index.workers
    var bounds = partitions(len(index.left[0]), workers, 1)
    var jobs = List[_HashProbeJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _HashProbeJob(
                index.left,
                index.right,
                index.left_hashes,
                index.indexes,
                index.fold,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
                omit_identity,
            )
        )
    run_jobs(jobs)
    var total = 0
    var identity = omit_identity
    for worker in range(len(jobs)):
        total += len(jobs[worker].right_rows)
        identity = identity and jobs[worker].identity
    var left_rows = List[Int](capacity=0 if identity else total)
    var right_rows = List[Int](capacity=total)
    for worker in range(len(jobs)):
        if not identity:
            if jobs[worker].identity:
                for row in range(jobs[worker].start, jobs[worker].end):
                    left_rows.append(row)
            else:
                for row in jobs[worker].left_rows:
                    left_rows.append(row)
        for row in jobs[worker].right_rows:
            right_rows.append(row)
    return (left_rows^, right_rows^, identity)
