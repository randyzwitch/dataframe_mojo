"""Right-side hash index for high-cardinality joins.

Each open-addressed slot holds its first matching right row. Extra rows for
the same key use a compact duplicate chain; distinct keys with the same hash
continue to the next slot. Exact column equality resolves hash collisions.
Building in reverse row order and probing disjoint left ranges preserves the
join's documented left-major, right-input match order.
"""
from dataframe.storage.buffer import _Buffer
from std.bit import count_trailing_zeros
from std.memory import ArcPointer, Pointer, bitcast, pack_bits, unsafe_memcpy
from std.ffi import external_call
from std.sys import size_of
from std.sys.intrinsics import prefetch

from dataframe.aggregation.reducer import float_key
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column, _validity_at
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.storage.allocation import huge_capacity, huge_list, huge_uninit
from dataframe.kernels.join_fuse import (
    FusedJoin,
    MatchJob,
    fused_gather,
    probe_blocks,
)
from dataframe.parallel import Job, partitions, run_jobs, worker_count
from dataframe.kernels.partition import (
    Partitioner,
    _hash_bytes,
    _hash_column,
    _mix,
    _short_hash,
)
from dataframe.series import Series
from dataframe.text.bytes import _Bytes
from dataframe.storage.string import StringColumn
from dataframe.storage.string_view import StringView
from dataframe.trace import trace_path


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


def _bitmap_of(key: Series) -> Tuple[Int, Int]:
    """A key's validity bitmap address (0 when it has none, so every row is
    valid) and the bit of its row 0."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            ref column = key._data[Column[Scalar[D]]]
            if len(column._bits[]) == 0:
                return (0, 0)
            return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[Column[Int128]]():
        ref column = key._data[Column[Int128]]
        if len(column._bits[]) == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[BoolColumn]():
        ref column = key._data[BoolColumn]
        if len(column._bits[]) == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    ref column = key._data[StringColumn]
    if len(column._bits[]) == 0:
        return (0, 0)
    return (Int(column._bits[].unsafe_ptr()), column._offset)


def _row_equal(left: List[Series], right: List[Series], i: Int, j: Int) -> Bool:
    for k in range(len(left)):
        if not _key_equal(left[k], right[k], i, j):
            return False
    return True


# Probe rows whose candidates are matched together, as DuckDB's RowMatcher
# matches a vector of candidates (STANDARD_VECTOR_SIZE is 2048).
comptime _MATCH_BATCH = 1024


struct _Candidates(Movable):
    """Build slots whose stored hash equals a probe row's, for a batch of
    probe rows in row order: the probe row, the build row, and the bucket
    and duplicate chain that list the build row's equal keys.

    The keys are compared a column at a time over candidates that remain
    equal (`sel`). Resolve each column's type once per batch instead of once
    per key per candidate, which
    `_row_equal` did (two thirds of a 65K-row join on four string keys and
    an integer)."""

    var left: List[Int]
    var right: List[Int]
    var bucket: List[Int]
    var chain: List[Int]
    var sel: List[Int]
    var equal: List[Bool]

    def __init__(out self):
        self.left = List[Int](capacity=_MATCH_BATCH)
        self.right = List[Int](capacity=_MATCH_BATCH)
        self.bucket = List[Int](capacity=_MATCH_BATCH)
        self.chain = List[Int](capacity=_MATCH_BATCH)
        self.sel = List[Int](capacity=_MATCH_BATCH)
        self.equal = List[Bool](capacity=_MATCH_BATCH)

    def clear(mut self):
        self.left.clear()
        self.right.clear()
        self.bucket.clear()
        self.chain.clear()

    def add(mut self, left: Int, right: Int, bucket: Int, chain: Int):
        self.left.append(left)
        self.right.append(right)
        self.bucket.append(bucket)
        self.chain.append(chain)

    def match(mut self, left_keys: List[Series], right_keys: List[Series]):
        """Mark each candidate whose keys all equal its probe row's."""
        var n = len(self.left)
        self.sel.clear()
        for c in range(n):
            self.sel.append(c)
        var alive = n
        for k in range(len(left_keys)):
            if alive == 0:
                break
            alive = _match_key(
                left_keys[k],
                right_keys[k],
                self.left.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                self.right.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                self.sel.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                alive,
            )
        self.equal.clear()
        self.equal.resize(n, False)
        for s in range(alive):
            self.equal[self.sel[s]] = True


@always_inline
def _match_values[
    T: Copyable & Deinitable & Equatable
](
    a: Pointer[T, MutAnyOrigin],
    b: Pointer[T, MutAnyOrigin],
    left: Pointer[Int, MutAnyOrigin],
    right: Pointer[Int, MutAnyOrigin],
    sel: Pointer[Int, MutAnyOrigin],
    count: Int,
) -> Int:
    """Keep in `sel` the candidates whose values are equal; validity is
    checked after (`_keep_valid`)."""
    var kept = 0
    for s in range(count):
        var c = sel[unsafe_offset=s]
        var same = (
            a[unsafe_offset=left[unsafe_offset=c]]
            == b[unsafe_offset=right[unsafe_offset=c]]
        )
        sel[unsafe_offset=kept] = c
        kept += Int(same)
    return kept


def _keep_valid[
    T: Copyable & Deinitable
](
    a: Column[T],
    b: Column[T],
    left_rows: Pointer[Int, MutAnyOrigin],
    right_rows: Pointer[Int, MutAnyOrigin],
    sel: Pointer[Int, MutAnyOrigin],
    count: Int,
) -> Int:
    """Keep in `sel` the candidates valid on both sides: a null key never
    matches. Columns without a bitmap keep every candidate."""
    if len(a._bits[]) == 0 and len(b._bits[]) == 0:
        return count
    var kept = 0
    for s in range(count):
        var c = sel[unsafe_offset=s]
        var same = a._valid(left_rows[unsafe_offset=c]) and b._valid(
            right_rows[unsafe_offset=c]
        )
        sel[unsafe_offset=kept] = c
        kept += Int(same)
    return kept


def _match_key(
    left: Series,
    right: Series,
    left_rows: Pointer[Int, MutAnyOrigin],
    right_rows: Pointer[Int, MutAnyOrigin],
    sel: Pointer[Int, MutAnyOrigin],
    count: Int,
) -> Int:
    """Narrow `sel` to the candidates whose key column is equal on both
    sides, with `_key_equal`'s meaning: nulls never equal, floats by
    `float_key`."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if left._data.isa[Column[Scalar[D]]]():
            ref a = left._data[Column[Scalar[D]]]
            ref b = right._data[Column[Scalar[D]]]
            var kept = 0
            comptime if D.is_floating_point():
                var x = a._ptr()
                var y = b._ptr()
                for s in range(count):
                    var c = sel[unsafe_offset=s]
                    var same = float_key(
                        Float64(x[unsafe_offset=left_rows[unsafe_offset=c]])
                    ) == float_key(
                        Float64(y[unsafe_offset=right_rows[unsafe_offset=c]])
                    )
                    sel[unsafe_offset=kept] = c
                    kept += Int(same)
            else:
                kept = _match_values(
                    a._ptr(), b._ptr(), left_rows, right_rows, sel, count
                )
            return _keep_valid(a, b, left_rows, right_rows, sel, kept)
    if left._data.isa[Column[Int128]]():
        ref a = left._data[Column[Int128]]
        ref b = right._data[Column[Int128]]
        var kept = _match_values(
            a._ptr(), b._ptr(), left_rows, right_rows, sel, count
        )
        return _keep_valid(a, b, left_rows, right_rows, sel, kept)
    if left._data.isa[BoolColumn]():
        ref a = left._data[BoolColumn]
        ref b = right._data[BoolColumn]
        var kept = 0
        for s in range(count):
            var c = sel[unsafe_offset=s]
            var i = left_rows[unsafe_offset=c]
            var j = right_rows[unsafe_offset=c]
            var same = a._valid(i) and b._valid(j) and a._get(i) == b._get(j)
            sel[unsafe_offset=kept] = c
            kept += Int(same)
        return kept
    return _match_strings(
        left._data[StringColumn],
        right._data[StringColumn],
        left_rows,
        right_rows,
        sel,
        count,
    )


def _match_strings(
    a: StringColumn,
    b: StringColumn,
    left_rows: Pointer[Int, MutAnyOrigin],
    right_rows: Pointer[Int, MutAnyOrigin],
    sel: Pointer[Int, MutAnyOrigin],
    count: Int,
) -> Int:
    var nulls = len(a._bits[]) > 0 or len(b._bits[]) > 0
    var kept = 0
    if a._is_view_storage() and b._is_view_storage():
        # Arrow views: length and four-byte prefix in the first word,
        # the next eight bytes (up to twelve) or the bytes' location in
        # the second. Unequal first words settle most unequal pairs
        # without reading the bytes, as DuckDB's string_t compare does.
        var x = (
            a._view_storage.value()
            ._views[]
            .unsafe_ptr()
            .unsafe_offset(a._offset)
            .unsafe_bitcast[UInt64]()
        )
        var y = (
            b._view_storage.value()
            ._views[]
            .unsafe_ptr()
            .unsafe_offset(b._offset)
            .unsafe_bitcast[UInt64]()
        )
        var long_a = _StringBytes(a)
        var long_b = _StringBytes(b)
        for s in range(count):
            var c = sel[unsafe_offset=s]
            var i = left_rows[unsafe_offset=c]
            var j = right_rows[unsafe_offset=c]
            var head = x[unsafe_offset=2 * i]
            var length = Int(head & 0xFFFFFFFF)
            # Bytes past the string's end in an inline view are masked
            # off: Arrow does not promise they are zero.
            var shown = min(length, 4)
            var head_mask = UInt64(0xFFFFFFFF) | (
                ((UInt64(1) << UInt64(8 * shown)) - 1) << 32
            )
            var same = ((head ^ y[unsafe_offset=2 * j]) & head_mask) == 0
            if same and length > 4:
                if length <= 12:
                    var tail_mask = (
                        (UInt64(1) << UInt64(8 * (length - 4))) - 1
                    ) if length < 12 else UInt64.MAX
                    same = (
                        (
                            x[unsafe_offset=2 * i + 1]
                            ^ y[unsafe_offset=2 * j + 1]
                        )
                        & tail_mask
                    ) == 0
                else:
                    # Equal lengths and prefixes: the bytes after the
                    # prefix.
                    var u = long_a.at(i)
                    var v = long_b.at(j)
                    same = (
                        external_call["memcmp", Int32](
                            u[0] + 4, v[0] + 4, length - 4
                        )
                        == 0
                    )
            if nulls:
                same = same and a._valid(i) and b._valid(j)
            sel[unsafe_offset=kept] = c
            kept += Int(same)
        return kept
    # Either storage: the bytes and length read through pointers taken
    # once, lengths compared first, then the bytes.
    var x = _StringBytes(a)
    var y = _StringBytes(b)
    for s in range(count):
        var c = sel[unsafe_offset=s]
        var i = left_rows[unsafe_offset=c]
        var j = right_rows[unsafe_offset=c]
        var u = x.at(i)
        var v = y.at(j)
        var same = u[1] == v[1]
        if same and u[1] > 0:
            same = external_call["memcmp", Int32](u[0], v[0], u[1]) == 0
        if nulls:
            same = same and a._valid(i) and b._valid(j)
        sel[unsafe_offset=kept] = c
        kept += Int(same)
    return kept


struct _StringBytes:
    """Where a string column's rows start and how long they are, from
    pointers taken once: offsets and a byte buffer, or Arrow views (inline
    bytes, or a buffer and an offset)."""

    var views: Int
    var offsets: Int
    var base: Int
    var buffers: Int

    def __init__(out self, column: StringColumn):
        if column._is_view_storage():
            ref storage = column._view_storage.value()
            self.views = Int(
                storage._views[].unsafe_ptr().unsafe_offset(column._offset)
            )
            self.buffers = Int(storage._buffers[].unsafe_ptr())
            self.offsets = 0
            self.base = 0
        else:
            self.views = 0
            self.buffers = 0
            self.offsets = Int(
                column._offsets[].unsafe_ptr().unsafe_offset(column._offset)
            )
            self.base = Int(column._base())

    @always_inline
    def at(self, i: Int) -> Tuple[Int, Int]:
        """Row i's first byte's address and its length."""
        if self.views == 0:
            var offsets = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=self.offsets
            )
            var start = Int(offsets[unsafe_offset=i])
            return (
                self.base + start,
                Int(offsets[unsafe_offset=i + 1]) - start,
            )
        var view = Pointer[StringView, MutAnyOrigin](
            unsafe_from_address=self.views
        ).unsafe_offset(i)
        var length = Int(view[].length)
        if length <= 12:
            return (Int(view) + 4, length)
        var buffers = Pointer[_Buffer[UInt8], MutAnyOrigin](
            unsafe_from_address=self.buffers
        )
        return (
            Int(buffers[unsafe_offset=Int(view[].buffer_index)][].unsafe_ptr())
            + Int(view[].offset),
            length,
        )


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
    """Every row's word (`_int_word`) of an integer key; a null row's word
    is undefined."""
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
    """Bit `l` set where lane `l` is true: one move-mask instruction
    (`pmovmskb` on x86), where a weighted sum of the lanes took a dozen
    vector instructions twice a probe (PDS-H q5's 6M lineitem probes)."""
    return pack_bits[DType.uint16](lanes)


# Probe rows hashed, and their slots fetched, together (Polars 2.0's
# `BLOCK_SIZE`): the block's cache misses are in flight at once instead of
# one row's at a time, and the block's hashes stay in L1.
comptime _PREFETCH_BLOCK = 256
# Slots, over all buckets, from which a probe prefetches (Polars 2.0's
# `MIN_PREFETCH_BUCKETS`): a smaller table stays in cache.
comptime _PREFETCH_SLOTS = 1 << 15


@always_inline
def _prefetch_probe(index: _HashBucket, hash: UInt64):
    """Start loading the line a probe of `hash` reads first: its tags, or
    its slot in a bucket without tags."""
    var position = Int(hash & UInt64(len(index.slots) - 1))
    if len(index.tags) > 0:
        prefetch(index.tags.unsafe_ptr().unsafe_offset(position))
    else:
        prefetch(
            index.slots.unsafe_ptr()
            .unsafe_offset(position)
            .unsafe_bitcast[UInt8]()
        )


@always_inline
def _hash_candidates(
    index: _HashBucket,
    hash: UInt64,
    row: Int,
    bucket: Int,
    mut candidates: _Candidates,
):
    """Add every slot whose stored hash equals `hash` to `candidates`, in
    probe order: sixteen tags per load and their matching lanes walked by
    set bits, as `_tagged_probe` does, so a probe row whose tag matches no
    slot (most rows of a selective or outer join) reads one tag line and
    compares nothing. DuckDB's probe (`ProbeForPointers`) likewise skips
    entries whose salt differs and compares keys only where it matches."""
    var mask = len(index.slots) - 1
    var slots = index.slots.unsafe_ptr()
    var position = Int(hash & UInt64(mask))
    if len(index.tags) == 0:
        # A bucket past _TAG_BUCKET_BYTES: the plain slot walk.
        while slots.unsafe_offset(position)[].row >= 0:
            ref slot = slots.unsafe_offset(position)[]
            if slot.key == hash:
                candidates.add(
                    row, Int(slot.row), bucket, Int(slot.next_position)
                )
            position = (position + 1) & mask
        return
    var tags = index.tags.unsafe_ptr()
    var wanted = SIMD[DType.uint8, 16](_tag_of(hash))
    while True:
        var group = tags.unsafe_offset(position).unsafe_load[width=16]()
        var hits = _lane_bits(group.eq(wanted))
        var empties = _lane_bits(group.eq(SIMD[DType.uint8, 16](0)))
        if empties != 0:
            hits &= (empties & (0 - empties)) - 1
        while hits != 0:
            var lane = Int(count_trailing_zeros(hits))
            hits &= hits - 1
            var at = (position + lane) & mask
            ref slot = slots.unsafe_offset(at)[]
            if slot.key == hash:
                candidates.add(
                    row, Int(slot.row), bucket, Int(slot.next_position)
                )
        if empties != 0:
            return
        position = (position + _TAG_LANES) & mask


@always_inline
def _int64_probe_slot(index: _HashBucket, hash: UInt64, key: UInt64) -> Int:
    return _tagged_probe(index, hash, key)


@always_inline
def _string_probe_hash(bytes: _Bytes, row: Int) -> UInt64:
    """The build partitioner's exact string hash, read at the probe row.

    The caller proves validity. Short view keys have eight readable bytes
    inside their descriptor; offsets keys can load a word only when it
    stays inside the shared byte buffer. The span fallback handles tails
    and longer strings with the same hash primitives as `_hash_column`.
    """
    if bytes.is_views():
        var view = Pointer[StringView, ImmutAnyOrigin](
            unsafe_from_address=bytes.views
        ).unsafe_offset(row)
        var length = Int(view[].length)
        if length <= 8:
            var inline = view.unsafe_bitcast[UInt8]().unsafe_offset(4)
            var word = bitcast[DType.uint64, 1](inline.unsafe_load[width=8]())
            return _mix(_short_hash(word, length))
    else:
        var offsets = Pointer[Int64, ImmutAnyOrigin](
            unsafe_from_address=bytes.offsets
        )
        var start = Int(offsets[unsafe_offset=row])
        var length = Int(offsets[unsafe_offset=row + 1]) - start
        if length <= 8 and start <= len(bytes.column._bytes[]) - 8:
            var data = Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=bytes.data
            ).unsafe_offset(start)
            var word = bitcast[DType.uint64, 1](data.unsafe_load[width=8]())
            return _mix(_short_hash(word, length))
    return _mix(_hash_bytes(bytes.get(row)))


@always_inline
def _string_probe_slot(
    index: _HashBucket,
    hash: UInt64,
    probe: _Bytes,
    build: _Bytes,
    row: Int,
) -> Int:
    # The caller proves probe validity; null build rows never enter the index.
    # Equal hashes still require exact bytes, including embedded NULs.
    var mask = len(index.slots) - 1
    var slots = index.slots.unsafe_ptr()
    var at = Int(hash & UInt64(mask))
    while slots.unsafe_offset(at)[].row >= 0:
        ref slot = slots.unsafe_offset(at)[]
        if hash == slot.key and probe.equal(row, build, Int(slot.row)):
            return at
        at = (at + 1) & mask
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

    var order: ArcPointer[List[Int]]
    # The hashes, and for an integer key its words, in
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
        order: ArcPointer[List[Int]],
        ordered_hashes: ArcPointer[List[UInt64]],
        ordered_words: ArcPointer[List[UInt64]],
        right_keys: List[Series],
        typed_int: Bool,
        first: Int,
        last: Int,
        skip_nulls: Bool,
    ):
        self.order = order.copy()
        self.ordered_hashes = ordered_hashes.copy()
        self.ordered_words = ordered_words.copy()
        self.right_keys = right_keys.copy()
        self.typed_int = typed_int
        self.skip_nulls = skip_nulls
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
        # The bitmaps of the keys that have one, through addresses taken
        # once: a row with a null key component never enters the index,
        # tested with one bit read per bitmap instead of resolving every
        # key's type for every row.
        var bitmaps = List[Int]()
        var bit_offsets = List[Int]()
        if self.skip_nulls:
            for key in self.right_keys:
                var bitmap = _bitmap_of(key)
                if bitmap[0] != 0:
                    bitmaps.append(bitmap[0])
                    bit_offsets.append(bitmap[1])
        for position in range(self.last - 1, self.first - 1, -1):
            var row = order.unsafe_offset(position)[]
            var valid = True
            for b in range(len(bitmaps)):
                valid = valid and _validity_at(
                    Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=bitmaps[b]
                    ),
                    bit_offsets[b] + row,
                )
            if not valid:
                continue
            var hash = ordered_hashes.unsafe_offset(position)[]
            var key = hash
            if have_words:
                key = words.unsafe_offset(position)[]
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
                # Integer slots retain the exact key word; other slots
                # already retain the complete hash, including compound
                # keys. Reconstruct only while compacting duplicate groups
                # instead of retaining every input row's original hash.
                var hash = _mix(old.key) if self.typed_int else old.key
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


struct _HashProbeJob(MatchJob):
    var left_keys: List[Series]
    var right_keys: List[Series]
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
    # The build keys' bits (`PreparedHashIndex.filter`), empty for none.
    var filter: ArcPointer[List[UInt64]]
    var filter_base: Int64
    # One batch's probe hashes (`_candidates`), for keys hashed on read.
    var batch_hashes: List[UInt64]
    # Whether the table is large enough to prefetch (`_PREFETCH_SLOTS`).
    var prefetching: Bool

    def __init__(
        out self,
        left_keys: List[Series],
        right_keys: List[Series],
        buckets: ArcPointer[List[_HashBucket]],
        fold: Int,
        start: Int,
        end: Int,
        include_unmatched: Bool,
        omit_identity: Bool,
        membership: Int = -1,
        filter: ArcPointer[List[UInt64]] = ArcPointer(List[UInt64]()),
        filter_base: Int64 = 0,
    ):
        self.left_keys = left_keys.copy()
        self.right_keys = right_keys.copy()
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
        self.left_rows = huge_capacity[Int](
            0 if self.omit_identity else end - start
        )
        self.right_rows = huge_capacity[Int](
            0 if membership >= 0 else end - start
        )
        self.filter = filter.copy()
        self.filter_base = filter_base
        self.batch_hashes = List[UInt64]()
        var slots = 0
        for bucket in buckets[]:
            slots += len(bucket.slots)
        self.prefetching = slots >= _PREFETCH_SLOTS

    def take_matches(
        mut self, mut left: List[Int], mut right: List[Int]
    ) -> Bool:
        swap(left, self.left_rows)
        swap(right, self.right_rows)
        return self.identity

    def row_range(self) -> Tuple[Int, Int]:
        return (self.start, self.end)

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
                        var bits = Int(self.filter[].unsafe_ptr())
                        var words = len(self.filter[])
                        for i in range(self.start, self.end):
                            var matched = False
                            comptime if D == DType.int64:
                                if words > 0 and _key_bit_clear(
                                    bits,
                                    words,
                                    self.filter_base,
                                    rebind[Int64](probe._get(i)),
                                ):
                                    # No match (a null key has none either).
                                    if not keep:
                                        self.left_rows.append(i)
                                    continue
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
            var probe_bytes = _Bytes(probe)
            var build_bytes = _Bytes(self.right_keys[0]._data[StringColumn])
            var all_valid = len(probe._bits[]) == 0
            for i in range(self.start, self.end):
                var matched = False
                if all_valid or probe._valid(i):
                    var hash = _string_probe_hash(probe_bytes, i)
                    var bucket = Int(hash >> 56) >> self.fold
                    matched = (
                        _string_probe_slot(
                            self.buckets[][bucket],
                            hash,
                            probe_bytes,
                            build_bytes,
                            i,
                        )
                        >= 0
                    )
                if matched == keep:
                    self.left_rows.append(i)
            return
        var candidates = _Candidates()
        self.batch_hashes = List[UInt64](length=_MATCH_BATCH, fill=0)
        var row = self.start
        while row < self.end:
            var stop = min(self.end, row + _MATCH_BATCH)
            self._candidates(row, stop, candidates)
            var n = len(candidates.left)
            if keep:
                for c in range(n):
                    var i = candidates.left[c]
                    if candidates.equal[c] and (
                        len(self.left_rows) == 0
                        or self.left_rows[len(self.left_rows) - 1] != i
                    ):
                        self.left_rows.append(i)
                row = stop
                continue
            var c = 0
            for i in range(row, stop):
                var matched = False
                while c < n and candidates.left[c] == i:
                    matched = matched or candidates.equal[c]
                    c += 1
                if not matched:
                    self.left_rows.append(i)
            row = stop

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
                        # Keys, validity and buckets through pointers: no
                        # reference-count or bounds check per row (#378).
                        var keys = left._ptr()
                        var valid = left._bits[].unsafe_ptr()
                        var buckets = self.buckets[].unsafe_ptr()
                        var bits = Int(self.filter[].unsafe_ptr())
                        var words = len(self.filter[])
                        var bit_offset = left._offset
                        # The block's hashes, made once in the prefetch pass.
                        var hashed = List[UInt64](
                            length=_PREFETCH_BLOCK if self.prefetching else 0,
                            fill=0,
                        )
                        var block_hashes = hashed.unsafe_ptr()
                        for block in range(
                            self.start, self.end, _PREFETCH_BLOCK
                        ):
                            var block_end = min(
                                self.end, block + _PREFETCH_BLOCK
                            )
                            # The block's slots, fetched before any is
                            # probed (a null row's line too, harmlessly).
                            if self.prefetching:
                                for i in range(block, block_end):
                                    var hash = _mix(
                                        _int_word[D](keys.unsafe_offset(i)[])
                                    )
                                    block_hashes[unsafe_offset=i - block] = hash
                                    _prefetch_probe(
                                        buckets.unsafe_offset(
                                            Int(hash >> 56) >> self.fold
                                        )[],
                                        hash,
                                    )
                            for i in range(block, block_end):
                                if not (
                                    left_all_valid
                                    or _validity_at(valid, bit_offset + i)
                                ):
                                    if self.include_unmatched:
                                        self.left_rows.append(i)
                                        self.right_rows.append(-1)
                                    continue
                                comptime if D == DType.int64:
                                    if words > 0 and _key_bit_clear(
                                        bits,
                                        words,
                                        self.filter_base,
                                        rebind[Int64](keys.unsafe_offset(i)[]),
                                    ):
                                        if self.include_unmatched:
                                            self.left_rows.append(i)
                                            self.right_rows.append(-1)
                                        continue
                                var key = _int_word[D](keys.unsafe_offset(i)[])
                                var hash = block_hashes[
                                    unsafe_offset=i - block
                                ] if self.prefetching else _mix(key)
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
                        _ = hashed^
                        return
        if (
            len(self.left_keys) == 1
            and self.left_keys[0]._data.isa[StringColumn]()
        ):
            ref left = self.left_keys[0]._data[StringColumn]
            ref right = self.right_keys[0]._data[StringColumn]
            # Null build rows never enter the index: only the probe's
            # validity is read.
            var all_valid = len(left._bits[]) == 0
            var left_bytes = _Bytes(left)
            var right_bytes = _Bytes(right)
            # Buckets outlive this job, and the fold maps each row's hash to
            # an existing bucket. Hash on read, as for integer probes, so
            # preparing a probe needs no row-sized hash array or histogram.
            var buckets = self.buckets[].unsafe_ptr()
            var generic_start = self.start
            # Emit only right positions while every input row has one output.
            # Missing inner matches or duplicate right keys end this prefix.
            if self.omit_identity:
                var row = self.start
                while row < self.end:
                    if not (all_valid or left._valid(row)):
                        if self.include_unmatched:
                            self.right_rows.append(-1)
                            row += 1
                            continue
                        break
                    var hash = _string_probe_hash(left_bytes, row)
                    var bucket = Int(hash >> 56) >> self.fold
                    ref index = buckets.unsafe_offset(bucket)[]
                    # Strings keep the plain walk: their slots hold the full
                    # hash, so a collision is rejected without reading a
                    # string, and H2O join q4 (10M probes, every one
                    # matching its first slot) was 17% slower through the
                    # tags' extra work per probe.
                    var position = _string_probe_slot(
                        index, hash, left_bytes, right_bytes, row
                    )
                    var matched_row = -1
                    var duplicate = False
                    if position >= 0:
                        ref slot = index.slots.unsafe_ptr().unsafe_offset(
                            position
                        )[]
                        matched_row = Int(slot.row)
                        duplicate = slot.next_position >= 0
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
                self.left_rows = huge_capacity[Int](self.end - self.start)
                for previous in range(self.start, row):
                    self.left_rows.append(previous)
                generic_start = row
            for i in range(generic_start, self.end):
                if not (all_valid or left._valid(i)):
                    if self.include_unmatched:
                        self.left_rows.append(i)
                        self.right_rows.append(-1)
                    continue
                var hash = _string_probe_hash(left_bytes, i)
                var bucket = Int(hash >> 56) >> self.fold
                ref index = buckets.unsafe_offset(bucket)[]
                var position = _string_probe_slot(
                    index, hash, left_bytes, right_bytes, i
                )
                if position >= 0:
                    ref slot = index.slots.unsafe_ptr().unsafe_offset(
                        position
                    )[]
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
        # Two Int64 keys took a path of their own here, comparing raw
        # values; the general comparison below measured faster (TPC-DS q93
        # 17.9 -> 8.4 ms, q40 22.8 -> 17.4) once Parquet columns lost their
        # all-valid bitmaps and reached it.
        var candidates = _Candidates()
        self.batch_hashes = List[UInt64](length=_MATCH_BATCH, fill=0)
        var row = self.start
        while row < self.end:
            var stop = min(self.end, row + _MATCH_BATCH)
            self._candidates(row, stop, candidates)
            # Distinct build keys hold distinct slots, so at most one
            # candidate of a probe row is equal; its duplicate chain
            # follows it, as before.
            var n = len(candidates.left)
            if not self.include_unmatched:
                # Candidates are in probe-row order: walk them, not the
                # rows, most of which find none in a selective join.
                for c in range(n):
                    if candidates.equal[c]:
                        self._emit(candidates, c)
                row = stop
                continue
            var c = 0
            for i in range(row, stop):
                var matched = False
                while c < n and candidates.left[c] == i:
                    if candidates.equal[c]:
                        self._emit(candidates, c)
                        matched = True
                    c += 1
                if not matched:
                    self.left_rows.append(i)
                    self.right_rows.append(-1)
            row = stop

    @always_inline
    def _emit(mut self, candidates: _Candidates, c: Int):
        """A matched candidate's pair, then its build row's duplicates."""
        var i = candidates.left[c]
        self.left_rows.append(i)
        self.right_rows.append(candidates.right[c])
        _append_duplicate_rows(
            self.buckets[][candidates.bucket[c]],
            candidates.chain[c],
            i,
            self.left_rows,
            self.right_rows,
        )

    def _candidates(
        mut self, start: Int, stop: Int, mut candidates: _Candidates
    ) raises:
        """The build slots of probe rows [start, stop) whose stored hash
        equals the row's, matched on the keys. The rows' hashes are made
        here, one batch at a time into a buffer that stays in cache, as
        DuckDB hashes each probe vector (`JoinHashTable::Hash`) before
        looking it up, rather than in a pass over the whole probe first."""
        candidates.clear()
        var hashes = self.batch_hashes.unsafe_ptr()
        for k in range(len(self.left_keys)):
            _hash_column(
                self.left_keys[k],
                start,
                stop,
                Int(hashes),
                k == 0,
                output_offset=start,
            )
        var buckets = self.buckets[].unsafe_ptr()
        var block = start
        while block < stop:
            var block_end = min(stop, block + _PREFETCH_BLOCK)
            if self.prefetching:
                for i in range(block, block_end):
                    var hash = hashes[unsafe_offset=i - start]
                    _prefetch_probe(
                        buckets.unsafe_offset(Int(hash >> 56) >> self.fold)[],
                        hash,
                    )
            for i in range(block, block_end):
                var hash = hashes[unsafe_offset=i - start]
                var bucket = Int(hash >> 56) >> self.fold
                _hash_candidates(
                    buckets.unsafe_offset(bucket)[], hash, i, bucket, candidates
                )
            block = block_end
        candidates.match(self.left_keys, self.right_keys)


@fieldwise_init
struct _HashIndex(Movable):
    """A built right-row index and the probe keys, hashed as they are read."""

    var left: List[Series]
    var right: List[Series]
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
    # DuckDB's perfect hash join: unique integer build keys within a span
    # of `_PERFECT_SPAN` values, each key's build row at `key - base`
    # (-1 for none). Empty for a progression or a hash table; with it,
    # `progression` is set so probes take the same direct loop.
    var slots: ArcPointer[List[Int32]]
    # A hash table's integer build keys as bits over their span, from
    # `filter_base` (empty when the span's bits would not stay in cache): a
    # probe key whose bit is clear has no match, found with one bit test
    # instead of a hash and a slot search (an exact Bloom filter, as
    # DuckDB pushes one into a join's probe side).
    var filter: ArcPointer[List[UInt64]]
    var filter_base: Int64

    @always_inline
    def progression_row(self, value: Int64, right_count: Int) -> Int:
        """The build row holding value, or -1 (see int64_progression)."""
        if value < self.base:
            return -1
        if len(self.slots[]) > 0:
            var offset = UInt64(value) - UInt64(self.base)
            if offset >= UInt64(len(self.slots[])):
                return -1
            return Int(self.slots[].unsafe_ptr()[unsafe_offset=Int(offset)])
        var distance = UInt64(value) - UInt64(self.base)
        if self.stride != 1:
            if distance % self.stride != 0:
                return -1
            distance //= self.stride
        return Int(distance) if distance < UInt64(right_count) else -1

    def probe(
        self, left_keys: List[Series], workers: Int = 0
    ) raises -> _HashIndex:
        """The probe of `left_keys`, split over `workers` ranges (0: by
        the row count; 1 on a pipeline worker, where the crew is busy)."""
        var left = List[Series](capacity=len(left_keys))
        for key in left_keys:
            left.append(key.rechunk() if key.is_chunked() else key.copy())
        var lanes = workers if workers > 0 else worker_count(len(left[0]))
        # Every key is hashed as the workers read it: integer and string
        # single keys row by row, compound keys a batch at a time
        # (`_HashProbeJob._candidates`). No row-sized hash buffer, and no
        # histogram pass, which only a partitioning build needs.
        return _HashIndex(
            left^,
            self.right.copy(),
            self.indexes.copy(),
            self.fold,
            lanes,
        )


@always_inline
def int64_progression(key: Series) -> Tuple[Bool, Int64, UInt64]:
    """Base and step when the keys ascend from base by one positive step.

    The key must have physical Int64 storage. Every row must be valid and
    each key must exceed the previous one by the same amount, so row r holds
    base + r * step. Sorted sequential IDs and calendar grids have this
    shape. The scan stops at the first null or irregular step.

    Whether a whole buffer is one is found once and kept with it
    (`_BufferFacts.progression`): a window onto a known progression is
    checked only where it meets the chunk before.
    """
    if len(key) == 0:
        return (False, Int64(0), UInt64(1))
    var base = Int64(0)
    var previous = Int64(0)
    var stride = UInt64(0)
    var row = 0
    for part in key.chunks():
        ref column = part._data[Column[Int64]]
        var n = len(column)
        if n == 0:
            continue
        var known = column._facts[].progression_state()
        if (
            known == 0
            and column._offset == 0
            and column._length == len(column._data[])
        ):
            # The window is the whole buffer: what it finds holds for
            # every later window onto it.
            known = 1 if _progression_scan(column)[0] else 2
            column._facts[].publish_progression(known == 1)
            if known == 2:
                return (False, Int64(0), UInt64(1))
        if known != 1 and len(column._bits[]) > 0 and column.null_count() > 0:
            # Every row must be valid (a Parquet column carries a bitmap
            # even when none is null; it is counted, not read per row).
            return (False, Int64(0), UInt64(1))
        var values = column.unsafe_values()
        var start = 0
        if row == 0:
            base = values.unsafe_load(0)
            previous = base
            start = 1
        if stride == 0 and start < n:
            var first = values.unsafe_load(start)
            if first <= previous:
                return (False, Int64(0), UInt64(1))
            stride = UInt64(first) - UInt64(previous)
        if known == 1:
            # A window onto a progression: one step throughout, so only
            # where it meets the chunk before and its own step are read.
            var first = values.unsafe_load(0)
            if start == 0 and (
                first <= previous or UInt64(first) - UInt64(previous) != stride
            ):
                return (False, Int64(0), UInt64(1))
            if n >= 2 and (
                UInt64(values.unsafe_load(1)) - UInt64(first) != stride
            ):
                return (False, Int64(0), UInt64(1))
            previous = values.unsafe_load(n - 1)
            row += n
            continue
        # The values read straight from storage, checked a block at a
        # time: the per-row checked reads took 8.4 ms over TPC-DS's 1.9M
        # customer_demographics keys, the whole cost of q84's join on them.
        var i = start
        while i < n:
            var stop = min(n, i + 1024)
            var bad = False
            for k in range(i, stop):
                var value = values.unsafe_load(k)
                # Unsigned subtraction is the exact distance of value >
                # previous; `|` keeps the loop free of branches.
                bad = (
                    bad
                    | (value <= previous)
                    | (UInt64(value) - UInt64(previous) != stride)
                )
                previous = value
            if bad:
                return (False, Int64(0), UInt64(1))
            i = stop
        row += n
    return (True, base, max(stride, UInt64(1)))


def _progression_scan(column: Column[Int64]) -> Tuple[Bool, UInt64]:
    """Whether the column's rows are valid and ascend by one positive step,
    and the step (1 for a single row)."""
    var n = len(column)
    if len(column._bits[]) > 0 and column.null_count() > 0:
        return (False, UInt64(1))
    if n < 2:
        return (n == 1, UInt64(1))
    var values = column.unsafe_values()
    var previous = values.unsafe_load(0)
    var first = values.unsafe_load(1)
    if first <= previous:
        return (False, UInt64(1))
    var stride = UInt64(first) - UInt64(previous)
    var i = 1
    while i < n:
        var stop = min(n, i + 1024)
        var bad = False
        for k in range(i, stop):
            var value = values.unsafe_load(k)
            bad = (
                bad
                | (value <= previous)
                | (UInt64(value) - UInt64(previous) != stride)
            )
            previous = value
        if bad:
            return (False, UInt64(1))
        i = stop
    return (True, stride)


def _l2_cache_bytes() -> Int:
    """A core's L2 cache size, from Linux's sysfs; 1 MiB where that is not
    readable."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index2/size", "r") as f:
            var text = f.read()
            var value = 0
            var scale = 1
            for b in text.as_bytes():
                var byte = Int(b)
                if byte >= 48 and byte <= 57:
                    value = 10 * value + byte - 48
                elif byte == 75:
                    scale = 1024
                elif byte == 77:
                    scale = 1024 * 1024
            if value > 0:
                return value * scale
    except:
        pass
    return 1 << 20


# DuckDB's perfect hash join span (`CanDoPerfectHashJoin`).
comptime _PERFECT_SPAN = 1_048_576


def _key_bits(key: Series) raises -> Optional[Tuple[Int64, List[UInt64]]]:
    """An Int64 build key's valid values as bits over their span, from the
    least: None when the span's bits would not stay within half a core's
    L2 cache, the bound the perfect hash join's slots keep."""
    var flat = key.rechunk() if key.is_chunked() else key.copy()
    ref column = flat._data[Column[Int64]]
    var values = column._ptr()
    var n = len(column)
    var all_valid = column.null_count() == 0
    var least = Int64.MAX
    var most = Int64.MIN
    for i in range(n):
        if all_valid or column._valid(i):
            var v = values[unsafe_offset=i]
            least = min(least, v)
            most = max(most, v)
    if least > most:
        return None
    if UInt64(most) - UInt64(least) >= UInt64(4 * _l2_cache_bytes()):
        return None
    var words = List[UInt64](
        length=Int((UInt64(most) - UInt64(least)) // 64) + 1, fill=0
    )
    var bits = words.unsafe_ptr()
    for i in range(n):
        if all_valid or column._valid(i):
            var at = UInt64(values[unsafe_offset=i]) - UInt64(least)
            bits[unsafe_offset=Int(at >> 6)] |= UInt64(1) << (at & 63)
    return (least, words^)


@always_inline
def _key_bit_clear(bits: Int, words: Int, base: Int64, key: Int64) -> Bool:
    """Whether `key` is certainly not a build key (`_key_bits`, at address
    `bits`)."""
    var at = UInt64(key) - UInt64(base)
    if at >= UInt64(words) * 64:
        return True
    var word = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=bits)[
        unsafe_offset=Int(at >> 6)
    ]
    return (word >> (at & 63)) & 1 == 0


def _perfect_slots(key: Series) -> Optional[Tuple[Int64, List[Int32]]]:
    """A build key's rows by `key - least`, when its values are integers,
    never null, unique, and span at most `_PERFECT_SPAN` in slots that stay
    in cache; None otherwise."""
    if key.null_count() > 0 or len(key) == 0 or len(key) > Int(Int32.MAX):
        return None
    ref column = key._data[Column[Int64]]
    var values = column._ptr()
    var n = len(column)
    var least = Int64.MAX
    var most = Int64.MIN
    for i in range(n):
        var v = values[unsafe_offset=i]
        least = min(least, v)
        most = max(most, v)
    if most - least >= Int64(_PERFECT_SPAN):
        return None
    # The slots must stay in cache while every probe row reads one at
    # random: no larger than the hash table they replace (about 16 bytes
    # a build row), or within half a core's L2 cache. A few hundred keys
    # spread over a wide span would otherwise miss on nearly every probe.
    var slot_bytes = 4 * (Int(most - least) + 1)
    if slot_bytes > max(16 * n, _l2_cache_bytes() // 2):
        return None
    var slots = List[Int32](length=Int(most - least) + 1, fill=-1)
    var table = slots.unsafe_ptr()
    for i in range(n):
        var at = Int(values[unsafe_offset=i] - least)
        if table[unsafe_offset=at] >= 0:
            return None
        table[unsafe_offset=at] = Int32(i)
    return (least, slots^)


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
        ArcPointer(List[Int32]()),
        ArcPointer(List[UInt64]()),
        0,
    )


def prepare_hash_index(
    right_keys: List[Series],
    allow_progression: Bool = True,
    probe_rows: Int = -1,
) raises -> PreparedHashIndex:
    """The build side's index. `probe_rows`, when known, bounds the rows
    that will probe it: build an integer-key bit filter only when the build
    has no more rows than the probe. A larger build pays more to set its
    bits than a small number of probes can save (PDS-H q16's 800K
    partsupp keys for 29.5K parts)."""
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
            ArcPointer(List[Int32]()),
            ArcPointer(List[UInt64]()),
            0,
        )
    if (
        allow_progression
        and len(right) == 1
        and right[0].dtype().physical() == DataType.INT64
    ):
        var perfect = _perfect_slots(right[0])
        if perfect:
            trace_path("join.perfect_hash")
            var made = perfect.take()
            return PreparedHashIndex(
                right^,
                ArcPointer(List[_HashBucket]()),
                8,
                True,
                made[0],
                1,
                ArcPointer(made[1].copy()),
                ArcPointer(List[UInt64]()),
                0,
            )
    var right_hashes = Partitioner(right, worker_count(len(right[0])))
    # Enough buckets that each one's slot table fits a core's L2 cache:
    # a bucket is built by one thread inserting at random positions, and a
    # table past the cache misses to memory on nearly every insert (1.5M
    # rows in 16 buckets: 4 MB tables, about 100 ns a row). A table this
    # size also keeps its tags (_TAG_BUCKET_BYTES), so probes take the
    # branch-free tag path.
    var typed_int = len(right) == 1 and _is_int_key(right[0])
    # An integer key's words, null rows' included: those rows are skipped
    # by their validity bit, so no row reads its key through the column.
    var words = List[UInt64]()
    if typed_int:
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
    # Scatter owns its reordered hashes. The input-order hash array and
    # histograms are dead before bucket tables and duplicate chains grow.
    _ = right_hashes^
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
    var filter = List[UInt64]()
    var filter_base = Int64(0)
    if (
        len(right) == 1
        and right[0].dtype().physical() == DataType.INT64
        and len(right[0]) <= probe_rows
    ):
        var bits = _key_bits(right[0])
        if bits:
            filter_base = bits.value()[0]
            filter = bits.value()[1].copy()
    return PreparedHashIndex(
        right^,
        ArcPointer(indexes^),
        fold,
        False,
        0,
        1,
        ArcPointer(List[Int32]()),
        ArcPointer(filter^),
        filter_base,
    )


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
        prepare_hash_index(
            right_keys, allow_progression=False, probe_rows=len(left_keys[0])
        ),
        keep_matches,
    )


def prepared_hash_semi_anti_rows(
    left_keys: List[Series],
    prepared: PreparedHashIndex,
    keep_matches: Bool,
    workers: Int = 0,
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
    var index = prepared.probe(left_keys, workers)
    var bounds = partitions(len(index.left[0]), index.workers, 1)
    var jobs = List[_HashProbeJob](capacity=index.workers)
    for worker in range(index.workers):
        jobs.append(
            _HashProbeJob(
                index.left,
                index.right,
                index.indexes,
                index.fold,
                bounds[worker],
                bounds[worker + 1],
                False,
                False,
                membership=1 if keep_matches else 0,
                filter=prepared.filter,
                filter_base=prepared.filter_base,
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


def prepared_hash_join_gather(
    left_keys: List[Series],
    prepared: PreparedHashIndex,
    include_unmatched: Bool,
    omit_identity: Bool,
    mut request: FusedJoin,
) raises:
    """`prepared_hash_join_rows` and the gather of the joined columns, one
    block of probe rows at a time (see `fused_gather`): the output is left
    in `request`, and no row list as long as it exists."""
    if prepared.progression:
        trace_path("join.progression_prepared")
    else:
        trace_path("join.hash_index")
    if prepared.progression:
        var key = left_keys[0].rechunk()
        var n = len(key)
        var blocks = probe_blocks(n, request)
        var bounds = partitions(n, blocks, 1)
        var probes = List[_DirectProbeJob](capacity=blocks)
        for block in range(blocks):
            probes.append(
                _DirectProbeJob(
                    key,
                    prepared,
                    bounds[block],
                    bounds[block + 1],
                    include_unmatched,
                )
            )
        fused_gather(probes, request, include_unmatched)
        return
    var index = prepared.probe(left_keys, 0)
    var n = len(index.left[0])
    var blocks = probe_blocks(n, request)
    var bounds = partitions(n, blocks, 1)
    var probes = List[_HashProbeJob](capacity=blocks)
    for block in range(blocks):
        probes.append(
            _HashProbeJob(
                index.left,
                index.right,
                index.indexes,
                index.fold,
                bounds[block],
                bounds[block + 1],
                include_unmatched,
                omit_identity,
                filter=prepared.filter,
                filter_base=prepared.filter_base,
            )
        )
    fused_gather(probes, request, include_unmatched)


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
        prepare_hash_index(
            right_keys, allow_progression=False, probe_rows=len(left_keys[0])
        ),
        include_unmatched,
        omit_identity,
    )


@always_inline
def _direct_matches[
    slotted: Bool, nullable: Bool
](
    keys: Pointer[Int64, MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    offset: Int,
    base: UInt64,
    slots: Pointer[Int32, MutAnyOrigin],
    span: Int,
    start: Int,
    stop: Int,
    lefts: Pointer[Int, MutAnyOrigin],
    rights: Pointer[Int, MutAnyOrigin],
) -> Int:
    """The rows of [start, stop) that match a perfect hash's slots
    (`slotted`) or a progression of step one, written to `lefts` and
    `rights` from the front; their count. A key below the base wraps to a
    large unsigned offset, so one compare bounds it on both sides."""
    var count = 0
    for i in range(start, stop):
        var at = UInt64(keys[unsafe_offset=i]) - base
        var inside = at < UInt64(span)
        var row: Int
        var hit: Bool
        comptime if slotted:
            row = Int(
                slots[unsafe_offset=Int(at if inside else UInt64(span - 1))]
            )
            hit = inside and row >= 0
        else:
            row = Int(at)
            hit = inside
        comptime if nullable:
            var bit = offset + i
            hit = (
                hit
                and ((bits[unsafe_offset=bit >> 3] >> UInt8(bit & 7)) & 1) == 1
            )
        lefts[unsafe_offset=count] = i
        rights[unsafe_offset=count] = row
        count += Int(hit)
    return count


struct _DirectProbeJob(MatchJob):
    """Probe rows [start, stop) of an Int64 key against a progression or
    perfect hash index (`PreparedHashIndex.progression_row`)."""

    var key: Series
    var prepared: PreparedHashIndex
    var start: Int
    var stop: Int
    var include_unmatched: Bool
    var left_rows: List[Int]
    var right_rows: List[Int]
    # Rows that found a build row.
    var matched: Int

    def __init__(
        out self,
        key: Series,
        prepared: PreparedHashIndex,
        start: Int,
        stop: Int,
        include_unmatched: Bool,
    ):
        self.key = key.copy()
        self.prepared = prepared.copy()
        self.start = start
        self.stop = stop
        self.include_unmatched = include_unmatched
        self.left_rows = List[Int]()
        self.right_rows = List[Int]()
        self.matched = 0

    def take_matches(
        mut self, mut left: List[Int], mut right: List[Int]
    ) -> Bool:
        swap(left, self.left_rows)
        swap(right, self.right_rows)
        return False

    def row_range(self) -> Tuple[Int, Int]:
        return (self.start, self.stop)

    def run(mut self) raises:
        ref values = self.key._data[Column[Int64]]
        var right_count = len(self.prepared.right[0])
        var n = self.stop - self.start
        # Keys and validity read through pointers, rows written into
        # buffers sized once and the counts kept in locals, as DuckDB's
        # perfect hash probe fills its selection vectors: a bounds check on
        # every key and validity read and a capacity check on every append
        # made this loop most of TPC-DS q23's store sales joins.
        var left_rows = List[Int](unsafe_uninit_length=n)
        var right_rows = List[Int](unsafe_uninit_length=n)
        var lefts = left_rows.unsafe_ptr()
        var rights = right_rows.unsafe_ptr()
        var keys = values._ptr()
        var nulls = len(values._bits[]) > 0
        var bits = values._bits[].unsafe_ptr()
        var include = self.include_unmatched
        var count = 0
        var matched = 0
        var slot_count = len(self.prepared.slots[])
        if not include and (slot_count > 0 or self.prepared.stride == 1):
            # An inner probe fills its selection without a branch on the
            # outcome, as DuckDB's perfect hash probe does: every row is
            # written at the next free place, which only a matching, valid
            # row then claims. A key below the base wraps to a large
            # unsigned offset, so one compare bounds it on both sides.
            var base = UInt64(self.prepared.base)
            var slots = (
                self.prepared.slots[]
                .unsafe_ptr()
                .unsafe_origin_cast[MutAnyOrigin]()
            )
            var to_left = lefts.unsafe_origin_cast[MutAnyOrigin]()
            var to_right = rights.unsafe_origin_cast[MutAnyOrigin]()
            var span = slot_count if slot_count > 0 else right_count
            var offset = values._offset
            if slot_count > 0 and nulls:
                count = _direct_matches[True, True](
                    keys,
                    bits,
                    offset,
                    base,
                    slots,
                    span,
                    self.start,
                    self.stop,
                    to_left,
                    to_right,
                )
            elif slot_count > 0:
                count = _direct_matches[True, False](
                    keys,
                    bits,
                    offset,
                    base,
                    slots,
                    span,
                    self.start,
                    self.stop,
                    to_left,
                    to_right,
                )
            elif nulls:
                count = _direct_matches[False, True](
                    keys,
                    bits,
                    offset,
                    base,
                    slots,
                    span,
                    self.start,
                    self.stop,
                    to_left,
                    to_right,
                )
            else:
                count = _direct_matches[False, False](
                    keys,
                    bits,
                    offset,
                    base,
                    slots,
                    span,
                    self.start,
                    self.stop,
                    to_left,
                    to_right,
                )
            matched = count
            left_rows.shrink(count)
            right_rows.shrink(count)
            self.left_rows = left_rows^
            self.right_rows = right_rows^
            self.matched = matched
            return
        for i in range(self.start, self.stop):
            var first = -1
            if not nulls or _validity_at(bits, values._offset + i):
                first = self.prepared.progression_row(
                    keys[unsafe_offset=i], right_count
                )
            if first >= 0:
                matched += 1
            if first >= 0 or include:
                lefts[unsafe_offset=count] = i
                rights[unsafe_offset=count] = first
                count += 1
        left_rows.shrink(count)
        right_rows.shrink(count)
        self.left_rows = left_rows^
        self.right_rows = right_rows^
        self.matched = matched


@always_inline
def _direct_matches_at[
    slotted: Bool, nullable: Bool
](
    keys: Pointer[Int64, MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    offset: Int,
    base: UInt64,
    slots: Pointer[Int32, MutAnyOrigin],
    span: Int,
    at: Pointer[Int, MutAnyOrigin],
    count: Int,
    keep_matches: Bool,
    lefts: Pointer[Int, MutAnyOrigin],
    rights: Pointer[Int, MutAnyOrigin],
) -> Int:
    """`_direct_matches` over the key rows `at[0:count]`: the positions in
    `at` whose row matches (or, without `keep_matches`, does not), and the
    build rows they match."""
    var kept = 0
    for k in range(count):
        var i = at[unsafe_offset=k]
        var distance = UInt64(keys[unsafe_offset=i]) - base
        var inside = distance < UInt64(span)
        var row: Int
        var hit: Bool
        comptime if slotted:
            row = Int(
                slots[
                    unsafe_offset=Int(distance if inside else UInt64(span - 1))
                ]
            )
            hit = inside and row >= 0
        else:
            row = Int(distance)
            hit = inside
        comptime if nullable:
            var bit = offset + i
            hit = (
                hit
                and ((bits[unsafe_offset=bit >> 3] >> UInt8(bit & 7)) & 1) == 1
            )
        lefts[unsafe_offset=kept] = k
        rights[unsafe_offset=kept] = row
        kept += Int(hit == keep_matches)
    return kept


def prepared_direct_rows_at(
    key: Series,
    at: List[Int],
    prepared: PreparedHashIndex,
    keep_matches: Bool,
) raises -> Optional[Tuple[List[Int], List[Int], Bool]]:
    """A progression or perfect hash index probed with the rows `at` of
    `key`, read where they are rather than gathered first, as DuckDB
    probes a dictionary vector through its selection: the positions in
    `at` that match (or, without `keep_matches`, that do not) and the
    build rows they match, both ascending with the positions, as
    `prepared_hash_join_rows` gives them (never the identity). None when
    the index is a hash table, its progression steps by more than one,
    or the key is chunked or not stored as Int64."""
    if not prepared.progression or key.is_chunked():
        return None
    if key.dtype().physical() != DataType.INT64:
        return None
    var slot_count = len(prepared.slots[])
    if slot_count == 0 and prepared.stride != 1:
        return None
    trace_path("join.direct_rows_at")
    ref values = key._data[Column[Int64]]
    var count = len(at)
    var left_rows = List[Int](unsafe_uninit_length=count)
    var right_rows = List[Int](unsafe_uninit_length=count)
    var to_left = left_rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var to_right = right_rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var rows = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(at.unsafe_ptr())
    )
    var keys = values._ptr()
    var bits = values._bits[].unsafe_ptr()
    var nulls = len(values._bits[]) > 0
    var base = UInt64(prepared.base)
    var slots = prepared.slots[].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var span = slot_count if slot_count > 0 else len(prepared.right[0])
    var offset = values._offset
    var kept: Int
    if slot_count > 0 and nulls:
        kept = _direct_matches_at[True, True](
            keys,
            bits,
            offset,
            base,
            slots,
            span,
            rows,
            count,
            keep_matches,
            to_left,
            to_right,
        )
    elif slot_count > 0:
        kept = _direct_matches_at[True, False](
            keys,
            bits,
            offset,
            base,
            slots,
            span,
            rows,
            count,
            keep_matches,
            to_left,
            to_right,
        )
    elif nulls:
        kept = _direct_matches_at[False, True](
            keys,
            bits,
            offset,
            base,
            slots,
            span,
            rows,
            count,
            keep_matches,
            to_left,
            to_right,
        )
    else:
        kept = _direct_matches_at[False, False](
            keys,
            bits,
            offset,
            base,
            slots,
            span,
            rows,
            count,
            keep_matches,
            to_left,
            to_right,
        )
    left_rows.shrink(kept)
    right_rows.shrink(kept)
    return (left_rows^, right_rows^, False)


def prepared_hash_join_rows(
    left_keys: List[Series],
    prepared: PreparedHashIndex,
    include_unmatched: Bool,
    omit_identity: Bool = False,
    workers: Int = 0,
) raises -> Tuple[List[Int], List[Int], Bool]:
    if prepared.progression:
        trace_path("join.progression_prepared")
    else:
        trace_path("join.hash_index")
    if prepared.progression:
        var key = left_keys[0].rechunk()
        var n = len(key)
        var lanes = workers if workers > 0 else worker_count(n)
        var bounds = partitions(n, lanes, 1)
        var jobs = List[_DirectProbeJob](capacity=lanes)
        for worker in range(lanes):
            jobs.append(
                _DirectProbeJob(
                    key,
                    prepared,
                    bounds[worker],
                    bounds[worker + 1],
                    include_unmatched,
                )
            )
        if len(jobs) == 1:
            jobs[0].run()
        else:
            run_jobs(jobs)
        var total = 0
        var matched = True
        for k in range(len(jobs)):
            total += len(jobs[k].left_rows)
            matched = matched and jobs[k].matched == len(jobs[k].left_rows)
        # Every probe row matched once (the build keys are unique), in
        # order: as the hash route does, the left rows are then implicit.
        var identity = omit_identity and matched and total == n
        if len(jobs) == 1:
            var left_only = List[Int]()
            var right_only = List[Int]()
            if not identity:
                swap(left_only, jobs[0].left_rows)
            swap(right_only, jobs[0].right_rows)
            return (left_only^, right_only^, identity)
        var left_rows = huge_capacity[Int](0 if identity else total)
        var right_rows = huge_capacity[Int](total)
        for k in range(len(jobs)):
            if not identity:
                left_rows.extend(Span(jobs[k].left_rows))
            right_rows.extend(Span(jobs[k].right_rows))
        return (left_rows^, right_rows^, identity)
    var index = prepared.probe(left_keys, workers)
    var lanes = index.workers
    var bounds = partitions(len(index.left[0]), lanes, 1)
    var jobs = List[_HashProbeJob](capacity=lanes)
    for worker in range(lanes):
        jobs.append(
            _HashProbeJob(
                index.left,
                index.right,
                index.indexes,
                index.fold,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
                omit_identity,
                filter=prepared.filter,
                filter_base=prepared.filter_base,
            )
        )
    run_jobs(jobs)
    var total = 0
    var identity = omit_identity
    for worker in range(len(jobs)):
        total += len(jobs[worker].right_rows)
        identity = identity and jobs[worker].identity
    # Every worker copies its own matches to their offset in the output.
    var left_rows = huge_uninit[Int](0 if identity else total)
    var right_rows = huge_uninit[Int](total)
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
