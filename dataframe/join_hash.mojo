"""Right-side hash index for high-cardinality joins.

Each open-addressed slot holds its first matching right row. Extra rows for
the same key use a compact duplicate chain; distinct keys with the same hash
continue to the next slot. Exact column equality resolves hash collisions.
Building in reverse row order and probing disjoint left ranges preserves the
join's documented left-major, right-input match order.
"""
from std.bit import count_trailing_zeros
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy
from std.sys import size_of

from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .huge_pages import huge_list, huge_uninit
from .parallel import Job, partitions, run_jobs, worker_count
from .partition import Partitioner, _mix
from .series import Series
from .string_column import StringColumn
from .trace import trace_path


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
    if left._data.isa[Column[Int128]]():
        ref a = left._data[Column[Int128]]
        ref b = right._data[Column[Int128]]
        return a._valid(i) and b._valid(j) and a._get(i) == b._get(j)
    if left._data.isa[BoolColumn]():
        ref a = left._data[BoolColumn]
        ref b = right._data[BoolColumn]
        return a._valid(i) and b._valid(j) and a._get(i) == b._get(j)
    ref a = left._data[StringColumn]
    ref b = right._data[StringColumn]
    return a._valid(i) and b._valid(j) and a._get(i) == b._get(j)


def _row_valid(keys: List[Series], row: Int) -> Bool:
    """Null in any join component makes the build row unmatchable."""
    for key in keys:
        var valid = True
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if key._data.isa[Column[Scalar[D]]]():
                valid = key._data[Column[Scalar[D]]]._valid(row)
        if key._data.isa[Column[Int128]]():
            valid = key._data[Column[Int128]]._valid(row)
        if key._data.isa[BoolColumn]():
            valid = key._data[BoolColumn]._valid(row)
        elif key._data.isa[StringColumn]():
            valid = key._data[StringColumn]._valid(row)
        if not valid:
            return False
    return True


def _row_equal(left: List[Series], right: List[Series], i: Int, j: Int) -> Bool:
    for k in range(len(left)):
        if not _key_equal(left[k], right[k], i, j):
            return False
    return True


@always_inline
def _int_word[D: DType](value: Scalar[D]) -> UInt64:
    """An integer key as the 64-bit word `_hash_column` mixes: unsigned
    values zero-extended, signed values sign-extended. Equal values of one
    type give equal words and unequal values unequal words, so the word is
    both the hash input and the exact key."""
    comptime if D.is_unsigned():
        return UInt64(value)
    else:
        return bitcast[DType.uint64](Int64(value))


def _is_int_key(key: Series) -> Bool:
    """Whether a key is stored as fixed-width integers of any width: the
    integer types, the Int64-backed dates, times, datetimes and durations,
    narrow decimals and categorical codes."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                return True
    return False


def _key_words(key: Series) -> List[UInt64]:
    """Every row's word (`_int_word`) of an integer key with no nulls."""
    var words = huge_uninit[UInt64](len(key))
    var out = words.unsafe_ptr()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                var values = key._data[Column[Scalar[D]]].unsafe_values()
                for i in range(len(key)):
                    out.unsafe_offset(i)[] = _int_word[D](
                        values.unsafe_offset(i)[]
                    )
    return words^


def _int_key_at(key: Series, row: Int) -> Tuple[Bool, UInt64]:
    """Whether an integer key's row is valid, and its word."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                ref values = key._data[Column[Scalar[D]]]
                return (values._valid(row), _int_word[D](values._get(row)))
    return (False, UInt64(0))


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
    # Heavy duplication uses [count, row, row, ...] groups instead of
    # pointer chains. Slot.next_position addresses the group's count.
    var groups: List[Int32]
    # One byte per slot: 0 for empty, otherwise a tag from the hash with
    # its top bit set. The table's last _TAG_LANES - 1 entries are
    # mirrored after the end, so a probe reads _TAG_LANES tags from any
    # position with one load and compares them with one SIMD op, as a
    # Swiss table does: a probe that finds no tag match walks no chain
    # and compares no key, so the loads of consecutive probes overlap
    # instead of each waiting on a data-dependent branch.
    var tags: List[UInt8]

    def mask(self) -> Int:
        return len(self.slots) - 1


comptime _TAG_LANES = 16
# Slot bytes per bucket up to which tags are built. A bucket's slots within
# a few MB stay in the last-level cache, where the tag line is cheap and
# saves the chain's branches; past that every probe misses to memory for
# its slot already, and the tag line is a second miss on every match (H2O
# join q4, 10M rows hashed and every probe matching, was 17% slower with
# tags).
comptime _TAG_BUCKET_BYTES = 2 << 20


@always_inline
def _tag_of(hash: UInt64) -> UInt8:
    """A slot tag: seven high bits of the hash below the bucket bits,
    with the top bit set so an empty slot (0) never matches."""
    return UInt8((hash >> 48) & 0x7F) | UInt8(0x80)


@always_inline
def _tagged_probe(index: _HashBucket, hash: UInt64, key: UInt64) -> Int:
    """The slot holding `key`, or -1: sixteen tags at a time, the key
    compared only where a tag matches, the search over once an empty
    tag is seen."""
    var mask = len(index.slots) - 1
    var tags = index.tags.unsafe_ptr()
    var slots = index.slots.unsafe_ptr()
    var wanted = SIMD[DType.uint8, 16](_tag_of(hash))
    var position = Int(hash & UInt64(mask))
    if len(index.tags) == 0:
        # A bucket past _TAG_BUCKET_BYTES: the plain slot walk.
        while slots.unsafe_offset(position)[].row >= 0:
            if key == slots.unsafe_offset(position)[].key:
                return position
            position = (position + 1) & mask
        return -1
    while True:
        var group = tags.unsafe_offset(position).unsafe_load[width=16]()
        # Lane masks: the candidates are the matching tags before the
        # first empty lane, walked by their set bits instead of a scalar
        # pass over sixteen lanes (q2's 800K probes spent a third of the
        # probe there).
        var hits = _lane_bits(group.eq(wanted))
        var empties = _lane_bits(group.eq(SIMD[DType.uint8, 16](0)))
        var candidates = hits
        if empties != 0:
            candidates &= (empties & (0 - empties)) - 1
        while candidates != 0:
            var lane = Int(count_trailing_zeros(candidates))
            candidates &= candidates - 1
            var at = (position + lane) & mask
            if slots.unsafe_offset(at)[].key == key:
                return at
        if empties != 0:
            return -1
        position = (position + _TAG_LANES) & mask


@always_inline
def _lane_bits(lanes: SIMD[DType.bool, 16]) -> UInt16:
    """Bit `l` set where lane `l` is true."""
    comptime weights = SIMD[DType.uint16, 16](
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    )
    return (lanes.cast[DType.uint16]() * weights).reduce_add()


struct _TagCursor:
    """Walks the slots whose tag matches a hash, in probe order, stopping
    at the first empty slot: `next()` yields each candidate position or
    -1 when the search is over. Sixteen tags are read per load."""

    # The tag table's address, as an Int: a struct field cannot hold an
    # unsafe-origin pointer. The bucket outlives every probe of it.
    var tags: Int
    var mask: Int
    var position: Int
    var lane: Int
    var hits: SIMD[DType.bool, 16]
    var empties: SIMD[DType.bool, 16]
    var wanted: SIMD[DType.uint8, 16]
    var done: Bool
    # Without tags (a bucket past _TAG_BUCKET_BYTES) every occupied slot
    # in probe order is a candidate.
    var slots: Int
    var plain: Bool

    def __init__(out self, index: _HashBucket, hash: UInt64):
        self.tags = Int(index.tags.unsafe_ptr())
        self.slots = Int(index.slots.unsafe_ptr())
        self.plain = len(index.tags) == 0
        self.mask = len(index.slots) - 1
        self.position = Int(hash & UInt64(self.mask))
        self.lane = 0
        self.wanted = SIMD[DType.uint8, 16](_tag_of(hash))
        self.hits = SIMD[DType.bool, 16](fill=False)
        self.empties = SIMD[DType.bool, 16](fill=False)
        self.done = False
        if not self.plain:
            self._load()

    def _load(mut self):
        """Load groups until one holds a candidate or an empty slot."""
        while True:
            var group = (
                Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.tags)
                .unsafe_offset(self.position)
                .unsafe_load[width=16]()
            )
            self.hits = group.eq(self.wanted)
            self.empties = group.eq(SIMD[DType.uint8, 16](0))
            self.lane = 0
            if self.hits.reduce_or():
                return
            # No candidate but an empty slot: the search is over. No
            # candidate and no empty slot: the next group.
            if self.empties.reduce_or():
                self.done = True
                return
            self.position = (self.position + _TAG_LANES) & self.mask

    def next(mut self) -> Int:
        if self.plain:
            if self.done:
                return -1
            var slots = Pointer[_HashSlot, MutAnyOrigin](
                unsafe_from_address=self.slots
            )
            if slots.unsafe_offset(self.position)[].row < 0:
                self.done = True
                return -1
            var at = self.position
            self.position = (self.position + 1) & self.mask
            return at
        while not self.done:
            while self.lane < _TAG_LANES:
                var lane = self.lane
                self.lane += 1
                if self.empties[lane]:
                    self.done = True
                    return -1
                if self.hits[lane]:
                    return (self.position + lane) & self.mask
            self.position = (self.position + _TAG_LANES) & self.mask
            self._load()
        return -1


@always_inline
def _int64_probe_slot(index: _HashBucket, hash: UInt64, key: UInt64) -> Int:
    return _tagged_probe(index, hash, key)


@always_inline
def _string_probe_slot(
    index: _HashBucket,
    hash: UInt64,
    probe: StringColumn,
    build: StringColumn,
    row: Int,
) -> Int:
    # The caller proves probe validity; null build rows never enter the index.
    # Equal hashes still require exact bytes, including embedded NULs.
    var at = Int(hash & UInt64(index.mask()))
    while index.slots[at].row >= 0:
        ref slot = index.slots[at]
        if hash == slot.key and probe._get(row) == build._get(Int(slot.row)):
            return at
        at = (at + 1) & index.mask()
    return -1


def _append_duplicate_rows(
    index: _HashBucket,
    first: Int,
    left_row: Int,
    mut left_rows: List[Int],
    mut right_rows: List[Int],
):
    if first < 0:
        return
    if len(index.groups):
        var count = Int(index.groups[first])
        for at in range(first + 1, first + 1 + count):
            left_rows.append(left_row)
            right_rows.append(Int(index.groups[at]))
    else:
        var next = first
        while next >= 0:
            ref entry = index.duplicates[next]
            left_rows.append(left_row)
            right_rows.append(Int(entry.row))
            next = Int(entry.next_position)


struct _HashBuildJob(Job):
    """Build one hash bucket from stable right-row positions."""

    var hashes: ArcPointer[List[UInt64]]
    var order: ArcPointer[List[Int]]
    # The hashes, and for an integer key without nulls its words, in
    # bucket order: read in sequence instead of at each row (#378).
    var ordered_hashes: ArcPointer[List[UInt64]]
    var ordered_words: ArcPointer[List[UInt64]]
    var right_keys: List[Series]
    var typed_int: Bool
    var skip_nulls: Bool
    var first: Int
    var last: Int
    var result: _HashBucket

    def __init__(
        out self,
        hashes: ArcPointer[List[UInt64]],
        order: ArcPointer[List[Int]],
        ordered_hashes: ArcPointer[List[UInt64]],
        ordered_words: ArcPointer[List[UInt64]],
        right_keys: List[Series],
        typed_int: Bool,
        first: Int,
        last: Int,
        skip_nulls: Bool,
    ):
        self.hashes = hashes.copy()
        self.order = order.copy()
        self.ordered_hashes = ordered_hashes.copy()
        self.ordered_words = ordered_words.copy()
        self.right_keys = right_keys.copy()
        self.typed_int = typed_int
        self.skip_nulls = skip_nulls and not typed_int
        self.first = first
        self.last = last
        self.result = _HashBucket(
            List[_HashSlot](),
            List[_DuplicateEntry](),
            List[Int32](),
            List[UInt8](),
        )

    def run(mut self) raises:
        var size = 2
        # Keep at least one-third of slots empty while avoiding a second
        # power-of-two jump for buckets with many duplicate rows.
        while size * 2 < 3 * (self.last - self.first):
            size *= 2
        var slots = huge_list(size, _HashSlot(-1, -1, 0))
        var tagged = size * size_of[_HashSlot]() <= _TAG_BUCKET_BYTES
        var tags = List[UInt8](
            length=size + _TAG_LANES if tagged else 0, fill=0
        )
        var duplicates = List[_DuplicateEntry]()
        var unique = 0
        var order = self.order[].unsafe_ptr()
        var ordered_hashes = self.ordered_hashes[].unsafe_ptr()
        var words = self.ordered_words[].unsafe_ptr()
        var have_words = len(self.ordered_words[]) > 0
        for position in range(self.last - 1, self.first - 1, -1):
            var row = order.unsafe_offset(position)[]
            if self.skip_nulls and not _row_valid(self.right_keys, row):
                continue
            var hash = ordered_hashes.unsafe_offset(position)[]
            var key = hash
            if have_words:
                key = words.unsafe_offset(position)[]
            elif self.typed_int:
                var word = _int_key_at(self.right_keys[0], row)
                if not word[0]:
                    continue
                key = word[1]
            var slot = Int(hash & UInt64(size - 1))
            var table = slots.unsafe_ptr()
            while table.unsafe_offset(slot)[].row >= 0:
                ref entry = table.unsafe_offset(slot)[]
                var same = entry.key == key
                if same and not self.typed_int:
                    same = _row_equal(
                        self.right_keys,
                        self.right_keys,
                        row,
                        Int(entry.row),
                    )
                if same:
                    duplicates.append(
                        _DuplicateEntry(entry.row, entry.next_position)
                    )
                    entry.row = Int32(row)
                    entry.next_position = Int32(len(duplicates) - 1)
                    break
                slot = (slot + 1) & (size - 1)
            if table.unsafe_offset(slot)[].row < 0:
                table.unsafe_offset(slot)[] = _HashSlot(Int32(row), -1, key)
                if tagged:
                    tags.unsafe_ptr().unsafe_offset(slot)[] = _tag_of(hash)
                unique += 1
        var groups = List[Int32]()
        if len(duplicates) > 4 * unique:
            # Retain compact groups, as the previous dictionary/CSR stream
            # path did. Long random duplicate chains waste cache bandwidth;
            # a table sized to rows also wastes space when keys repeat.
            var compact_size = 2
            while compact_size * 2 < 3 * unique:
                compact_size *= 2
            var compact = huge_list(compact_size, _HashSlot(-1, -1, 0))
            tagged = compact_size * size_of[_HashSlot]() <= _TAG_BUCKET_BYTES
            var compact_tags = List[UInt8](
                length=compact_size + _TAG_LANES if tagged else 0, fill=0
            )
            groups = List[Int32](capacity=unique + len(duplicates))
            for old in slots:
                if old.row < 0:
                    continue
                var first = -1
                var next = Int(old.next_position)
                if next >= 0:
                    first = len(groups)
                    groups.append(0)
                    var count = 0
                    while next >= 0:
                        ref entry = duplicates[next]
                        groups.append(entry.row)
                        count += 1
                        next = Int(entry.next_position)
                    groups[first] = Int32(count)
                var hash = self.hashes[][Int(old.row)]
                var at = Int(hash & UInt64(compact_size - 1))
                while compact[at].row >= 0:
                    at = (at + 1) & (compact_size - 1)
                compact[at] = _HashSlot(old.row, Int32(first), old.key)
                if tagged:
                    compact_tags[at] = _tag_of(hash)
            slots = compact^
            tags = compact_tags^
            duplicates = List[_DuplicateEntry]()
        # Mirror the first lanes after the end for unaligned tag loads.
        if tagged:
            for at in range(_TAG_LANES):
                tags[len(slots) + at] = tags[at]
        self.result = _HashBucket(slots^, duplicates^, groups^, tags^)

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
        A null key never matches: `_key_equal` requires both sides valid.
        Single integer keys never reach this; they have no stored hashes."""
        var hash = self.left_hashes[][i]
        var bucket = Int(hash >> 56) >> self.fold
        ref index = self.buckets[][bucket]
        var cursor = _TagCursor(index, hash)
        var position = cursor.next()
        while position >= 0:
            ref slot = index.slots[position]
            if hash == slot.key and _row_equal(
                self.left_keys, self.right_keys, i, Int(slot.row)
            ):
                return True
            position = cursor.next()
        return False

    def run_membership(mut self) raises:
        """Semi (membership == 1) or anti (0): each left row at most once,
        in row order; duplicate right keys do not repeat it."""
        var keep = self.membership == 1
        if len(self.left_keys) == 1:
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                comptime if not D.is_floating_point():
                    if self.left_keys[0]._data.isa[Column[Scalar[D]]]():
                        ref probe = self.left_keys[0]._data[Column[Scalar[D]]]
                        var all_valid = len(probe._bits[]) == 0
                        for i in range(self.start, self.end):
                            var matched = False
                            if all_valid or probe._valid(i):
                                var key = _int_word[D](probe._get(i))
                                var hash = _mix(key)
                                var bucket = Int(hash >> 56) >> self.fold
                                matched = (
                                    _int64_probe_slot(
                                        self.buckets[][bucket], hash, key
                                    )
                                    >= 0
                                )
                            if matched == keep:
                                self.left_rows.append(i)
                        return
        if (
            len(self.left_keys) == 1
            and self.left_keys[0]._data.isa[StringColumn]()
        ):
            ref probe = self.left_keys[0]._data[StringColumn]
            ref build = self.right_keys[0]._data[StringColumn]
            var all_valid = len(probe._bits[]) == 0
            for i in range(self.start, self.end):
                var matched = False
                if all_valid or probe._valid(i):
                    var hash = self.left_hashes[][i]
                    var bucket = Int(hash >> 56) >> self.fold
                    matched = (
                        _string_probe_slot(
                            self.buckets[][bucket], hash, probe, build, i
                        )
                        >= 0
                    )
                if matched == keep:
                    self.left_rows.append(i)
            return
        for i in range(self.start, self.end):
            if self._matches(i) == keep:
                self.left_rows.append(i)

    def run(mut self) raises:
        if self.membership >= 0:
            self.run_membership()
            return
        if len(self.left_keys) == 1:
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                comptime if not D.is_floating_point():
                    if self.left_keys[0]._data.isa[Column[Scalar[D]]]():
                        ref left = self.left_keys[0]._data[Column[Scalar[D]]]
                        var left_all_valid = len(left._bits[]) == 0
                        # Keys and buckets through pointers: no reference-count
                        # or bounds check per row (#378).
                        var keys = left._ptr()
                        var buckets = self.buckets[].unsafe_ptr()
                        for i in range(self.start, self.end):
                            if not (left_all_valid or left._valid(i)):
                                if self.include_unmatched:
                                    self.left_rows.append(i)
                                    self.right_rows.append(-1)
                                continue
                            var key = _int_word[D](keys.unsafe_offset(i)[])
                            var hash = _mix(key)
                            var bucket = Int(hash >> 56) >> self.fold
                            ref index = buckets.unsafe_offset(bucket)[]
                            var hit = _tagged_probe(index, hash, key)
                            if hit >= 0:
                                ref slot = (
                                    index.slots.unsafe_ptr().unsafe_offset(
                                        hit
                                    )[]
                                )
                                self.left_rows.append(i)
                                self.right_rows.append(Int(slot.row))
                                if slot.next_position >= 0:
                                    _append_duplicate_rows(
                                        index,
                                        Int(slot.next_position),
                                        i,
                                        self.left_rows,
                                        self.right_rows,
                                    )
                            elif self.include_unmatched:
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
                    # Strings keep the plain walk: their slots hold the full
                    # hash, so a collision is rejected without reading a
                    # string, and H2O join q4 (10M probes, every one
                    # matching its first slot) was 17% slower through the
                    # tags' extra work per probe.
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
                        _append_duplicate_rows(
                            index,
                            Int(slot.next_position),
                            i,
                            self.left_rows,
                            self.right_rows,
                        )
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
                    var cursor = _TagCursor(index, hash)
                    var matched = False
                    var position = cursor.next()
                    while position >= 0:
                        ref slot = index.slots[position]
                        var j = Int(slot.row)
                        if (
                            hash == slot.key
                            and first == right_first._get(j)
                            and second == right_second._get(j)
                        ):
                            self.left_rows.append(i)
                            self.right_rows.append(j)
                            _append_duplicate_rows(
                                index,
                                Int(slot.next_position),
                                i,
                                self.left_rows,
                                self.right_rows,
                            )
                            matched = True
                            break
                        position = cursor.next()
                    if not matched and self.include_unmatched:
                        self.left_rows.append(i)
                        self.right_rows.append(-1)
                return
        for i in range(self.start, self.end):
            var hash = self.left_hashes[][i]
            var bucket = Int(hash >> 56) >> self.fold
            ref index = self.buckets[][bucket]
            var cursor = _TagCursor(index, hash)
            var matched = False
            var position = cursor.next()
            while position >= 0:
                ref slot = index.slots[position]
                var j = Int(slot.row)
                if hash == slot.key and _row_equal(
                    self.left_keys, self.right_keys, i, j
                ):
                    self.left_rows.append(i)
                    self.right_rows.append(j)
                    _append_duplicate_rows(
                        index,
                        Int(slot.next_position),
                        i,
                        self.left_rows,
                        self.right_rows,
                    )
                    matched = True
                    break
                position = cursor.next()
            if not matched and self.include_unmatched:
                self.left_rows.append(i)
                self.right_rows.append(-1)


@fieldwise_init
struct _HashIndex(Movable):
    """A built right-row index; single integer probes hash from their keys."""

    var left: List[Series]
    var right: List[Series]
    var left_hashes: ArcPointer[List[UInt64]]
    var indexes: ArcPointer[List[_HashBucket]]
    var fold: Int
    var workers: Int


@fieldwise_init
struct PreparedHashIndex(Copyable):
    """Immutable build-side state, shared across independent probe batches."""

    var right: List[Series]
    var indexes: ArcPointer[List[_HashBucket]]
    var fold: Int

    var progression: Bool
    var base: Int64
    var stride: UInt64

    @always_inline
    def progression_row(self, value: Int64, right_count: Int) -> Int:
        """The build row holding value, or -1 (see int64_progression)."""
        if value < self.base:
            return -1
        var distance = UInt64(value) - UInt64(self.base)
        if self.stride != 1:
            if distance % self.stride != 0:
                return -1
            distance //= self.stride
        return Int(distance) if distance < UInt64(right_count) else -1

    def probe(self, left_keys: List[Series]) raises -> _HashIndex:
        var left = List[Series](capacity=len(left_keys))
        for key in left_keys:
            left.append(key.rechunk() if key.is_chunked() else key.copy())
        var workers = worker_count(len(left[0]))
        if len(left) == 1 and _is_int_key(left[0]):
            # Hash fixed-width probe keys as they are read by the workers.
            # No row-sized hash buffer or separate hashing/histogram pass.
            return _HashIndex(
                left^,
                self.right.copy(),
                ArcPointer(List[UInt64]()),
                self.indexes.copy(),
                self.fold,
                workers,
            )
        var hashes = Partitioner(left, workers)
        return _HashIndex(
            left^,
            self.right.copy(),
            ArcPointer(hashes^.into_hashes()),
            self.indexes.copy(),
            self.fold,
            workers,
        )


@always_inline
def int64_progression(key: Series) -> Tuple[Bool, Int64, UInt64]:
    """Base and step when the keys ascend from base by one positive step.

    The key must have physical Int64 storage. Every row must be valid and
    each key must exceed the previous one by the same amount, so row r holds
    base + r * step. Sorted sequential IDs and calendar grids have this
    shape. The scan stops at the first null or irregular step.
    """
    if len(key) == 0:
        return (False, Int64(0), UInt64(1))
    var base = Int64(0)
    var previous = Int64(0)
    var stride = UInt64(0)
    var row = 0
    for part in key.chunks():
        ref column = part._data[Column[Int64]]
        for i in range(len(column)):
            if not column._valid(i):
                return (False, Int64(0), UInt64(1))
            var value = column._get(i)
            if row == 0:
                base = value
            else:
                if value <= previous:
                    return (False, Int64(0), UInt64(1))
                # Unsigned subtraction is the exact distance of value > previous.
                var step = UInt64(value) - UInt64(previous)
                if stride == 0:
                    stride = step
                elif step != stride:
                    return (False, Int64(0), UInt64(1))
            previous = value
            row += 1
    return (True, base, max(stride, UInt64(1)))


def prepare_progression_index(
    right_keys: List[Series],
) raises -> Optional[PreparedHashIndex]:
    """Validate a compact progression without building a fallback hash table."""
    if (
        len(right_keys) != 1
        or len(right_keys[0]) > Int(Int32.MAX)
        or right_keys[0].dtype().physical() != DataType.INT64
    ):
        return None
    var progression = int64_progression(right_keys[0])
    if not progression[0]:
        return None
    var right = List[Series]()
    right.append(
        right_keys[0]
        .rechunk() if right_keys[0]
        .is_chunked() else right_keys[0]
        .copy()
    )
    return PreparedHashIndex(
        right^,
        ArcPointer(List[_HashBucket]()),
        8,
        True,
        progression[1],
        progression[2],
    )


def prepare_hash_index(
    right_keys: List[Series], allow_progression: Bool = True
) raises -> PreparedHashIndex:
    if len(right_keys[0]) > Int(Int32.MAX):
        raise Error("Direct hash join exceeds 32-bit row index capacity")
    var right = List[Series](capacity=len(right_keys))
    for key in right_keys:
        right.append(key.rechunk() if key.is_chunked() else key.copy())
    var progression = (False, Int64(0), UInt64(1))
    if (
        allow_progression
        and len(right) == 1
        and right[0].dtype().physical() == DataType.INT64
    ):
        progression = int64_progression(right[0])
    if progression[0]:
        return PreparedHashIndex(
            right^,
            ArcPointer(List[_HashBucket]()),
            8,
            True,
            progression[1],
            progression[2],
        )
    var right_hashes = Partitioner(right, worker_count(len(right[0])))
    # Enough buckets that each one's slot table fits a core's L2 cache:
    # a bucket is built by one thread inserting at random positions, and a
    # table past the cache misses to memory on nearly every insert (1.5M
    # rows in 16 buckets: 4 MB tables, about 100 ns a row). A table this
    # size also keeps its tags (_TAG_BUCKET_BYTES), so probes take the
    # branch-free tag path.
    var typed_int = len(right) == 1 and _is_int_key(right[0])
    var words = List[UInt64]()
    if typed_int and right[0].null_count() == 0:
        words = _key_words(right[0])
    var right_parts = right_hashes.scatter(
        worker_count(len(right[0])),
        with_hashes=True,
        words=Int(words.unsafe_ptr()) if len(words) > 0 else 0,
    )
    _ = words^
    # Moved, not copied: each is 8 bytes a row (#378).
    var order = List[Int]()
    swap(order, right_parts.order)
    var ordered_hashes = List[UInt64]()
    swap(ordered_hashes, right_parts.hashes)
    var ordered_words = List[UInt64]()
    swap(ordered_words, right_parts.words)
    var shared_right_hashes = ArcPointer(right_hashes^.into_hashes())
    var shared_order = ArcPointer(order^)
    var shared_hashes = ArcPointer(ordered_hashes^)
    var shared_words = ArcPointer(ordered_words^)
    var skip_nulls = False
    for key in right:
        skip_nulls = skip_nulls or key.null_count() > 0
    var builders = List[_HashBuildJob](capacity=right_parts.buckets())
    for bucket in range(right_parts.buckets()):
        builders.append(
            _HashBuildJob(
                shared_right_hashes,
                shared_order,
                shared_hashes,
                shared_words,
                right,
                typed_int,
                right_parts.bounds[bucket],
                right_parts.bounds[bucket + 1],
                skip_nulls,
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
    return PreparedHashIndex(right^, ArcPointer(indexes^), fold, False, 0, 1)


def direct_hash_semi_anti_rows(
    left_keys: List[Series], right_keys: List[Series], keep_matches: Bool
) raises -> List[Int]:
    """Left rows with (semi) or without (anti) an exact match, in row order.

    Same index as `direct_hash_join_rows`, probed for membership only: a
    left row appears once however many right rows share its key, and a
    null left key never matches, so semi drops it and anti keeps it.
    """
    return prepared_hash_semi_anti_rows(
        left_keys,
        prepare_hash_index(right_keys, allow_progression=False),
        keep_matches,
    )


def prepared_hash_semi_anti_rows(
    left_keys: List[Series], prepared: PreparedHashIndex, keep_matches: Bool
) raises -> List[Int]:
    if prepared.progression:
        trace_path("join.progression_membership")
    else:
        trace_path("join.hash_membership")
    if prepared.progression:
        var key = left_keys[0].rechunk()
        ref values = key._data[Column[Int64]]
        var rows = List[Int](capacity=len(values))
        var right_count = len(prepared.right[0])
        for i in range(len(values)):
            var matched = (
                values._valid(i)
                and prepared.progression_row(values._get(i), right_count) >= 0
            )
            if matched == keep_matches:
                rows.append(i)
        return rows^
    var index = prepared.probe(left_keys)
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


struct RowsCopyJob(Job):
    """Copy one probe job's matched rows to their offset in the output
    lists, so join row lists are concatenated on every worker (#335).
    A `left` address of 0 means the job's left rows are its own range
    [first, first + count) in order, which is written directly."""

    var left: Int
    var right: Int
    var first: Int
    var count: Int
    var out_left: Int
    var out_right: Int
    var at: Int

    def __init__(
        out self,
        left: Int,
        right: Int,
        first: Int,
        count: Int,
        out_left: Int,
        out_right: Int,
        at: Int,
    ):
        self.left = left
        self.right = right
        self.first = first
        self.count = count
        self.out_left = out_left
        self.out_right = out_right
        self.at = at

    def run(mut self) raises:
        if self.count == 0:
            return
        if self.out_left != 0:
            var target = (
                Pointer[List[Int], MutAnyOrigin](
                    unsafe_from_address=self.out_left
                )[]
                .unsafe_ptr()
                .unsafe_offset(self.at)
            )
            if self.left == 0:
                for k in range(self.count):
                    target.unsafe_offset(k)[] = self.first + k
            else:
                unsafe_memcpy(
                    dest=target,
                    src=Pointer[List[Int], MutAnyOrigin](
                        unsafe_from_address=self.left
                    )[].unsafe_ptr(),
                    count=self.count,
                )
        unsafe_memcpy(
            dest=Pointer[List[Int], MutAnyOrigin](
                unsafe_from_address=self.out_right
            )[]
            .unsafe_ptr()
            .unsafe_offset(self.at),
            src=Pointer[List[Int], MutAnyOrigin](
                unsafe_from_address=self.right
            )[].unsafe_ptr(),
            count=self.count,
        )


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
    return prepared_hash_join_rows(
        left_keys,
        prepare_hash_index(right_keys, allow_progression=False),
        include_unmatched,
        omit_identity,
    )


def prepared_hash_join_rows(
    left_keys: List[Series],
    prepared: PreparedHashIndex,
    include_unmatched: Bool,
    omit_identity: Bool = False,
) raises -> Tuple[List[Int], List[Int], Bool]:
    if prepared.progression:
        trace_path("join.progression_prepared")
    else:
        trace_path("join.hash_index")
    if prepared.progression:
        var key = left_keys[0].rechunk()
        ref values = key._data[Column[Int64]]
        var left_rows = List[Int](capacity=len(values))
        var right_rows = List[Int](capacity=len(values))
        var right_count = len(prepared.right[0])
        for i in range(len(values)):
            var first = -1
            if values._valid(i):
                first = prepared.progression_row(values._get(i), right_count)
            if first >= 0 or include_unmatched:
                left_rows.append(i)
                right_rows.append(first)
        return (left_rows^, right_rows^, False)
    var index = prepared.probe(left_keys)
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
    # Every worker copies its own matches to their offset in the output.
    var left_rows = List[Int](unsafe_uninit_length=0 if identity else total)
    var right_rows = List[Int](unsafe_uninit_length=total)
    var copies = List[RowsCopyJob](capacity=len(jobs))
    var at = 0
    for worker in range(len(jobs)):
        var count = len(jobs[worker].right_rows)
        copies.append(
            RowsCopyJob(
                0 if jobs[worker].identity else Int(
                    Pointer(to=jobs[worker].left_rows)
                ),
                Int(Pointer(to=jobs[worker].right_rows)),
                jobs[worker].start,
                count,
                0 if identity else Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                at,
            )
        )
        at += count
    run_jobs(copies)
    _ = jobs^
    return (left_rows^, right_rows^, identity)


def prefer_left_build(left_rows: Int, right_rows: Int) -> Bool:
    """Measured conservative crossover; see docs/join-performance-followups.md.

    Small probe tables lose on M1 despite a large ratio. Require both a
    512K-row probe table and a 32:1 imbalance before restoring logical order.
    """
    return (
        right_rows >= 524288
        and left_rows > 0
        and left_rows <= Int(Int32.MAX)
        and left_rows <= right_rows // 32
    )
