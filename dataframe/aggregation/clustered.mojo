"""DuckDB's clustered aggregate update (`ClusteredAggr`,
`common/clustered_aggregate.cpp`, and `RowOperations::UpdateStatesClustered`).

A chunk's rows are clustered by group into runs: each run is one group's
rows of the chunk, in order. When at least 80% of the chunk's rows fall in
runs of six or more, each reduction updates its state once per run, the
run's values accumulated in registers, instead of a load, add and store on
the group's state for every row. With few groups (or repeating keys) the
per-row updates hit the same state words back to back and wait on each
other; a run's accumulation does not.

The clustering is DuckDB's: a short dual-cursor sample hashes the chunk's
first group ids to find hot keys and to see whether the input is already
sorted. Sorted input is cut into runs at each change of group (no
hashing); otherwise hot keys keep collecting through a small hash table
while every other group forms sequential runs. A chunk that does not
cluster backs off: the next attempts are skipped, twice as many after each
failure.
"""
from std.math import isnan
from std.memory import Pointer, bitcast

from dataframe.aggregation.function import (
    AggregateFunction,
    _BATCH,
    _K_COUNT,
    _K_DECIMAL_SUM,
    _K_FLOAT_SUM,
    _K_INT_SUM,
    _K_LEN,
    _K_MAX,
    _K_MIN,
    _K_MOMENTS,
    _K_NARROW_SUM,
    _K_NULLS,
    _batch_nulls,
    _load_floats,
    _words_at,
)
from dataframe.aggregation.selected_rows import _canonical_int
from dataframe.decimal import check_limit, precision_limit
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column, _validity_at
from dataframe.storage.string import StringColumn

# ClusteredAggr's constants (clustered_aggregate.hpp).
comptime RUNLENGTH_THRESHOLD = 6
comptime HOTKEYS_LOG2 = 5
comptime MAX_HOTKEYS = 1 << HOTKEYS_LOG2
comptime SAMPLE_SIZE = 128
comptime HASHTAB_LOG2 = 11
comptime HASHTAB_SZ = 1 << HASHTAB_LOG2
comptime SLOT_GRP_BITS = 13
comptime SLOT_CURSOR_BITS = (HOTKEYS_LOG2 + 1) + SLOT_GRP_BITS
comptime SLOT_GID_BITS = 64 - (SLOT_GRP_BITS + SLOT_CURSOR_BITS)
comptime MAX_GID_COUNT = 1 << SLOT_GID_BITS
comptime FREE_SLOT = UInt64.MAX
comptime _HALF_VEC = _BATCH // 2
comptime _MAX_HOT_PER_CURSOR = MAX_HOTKEYS // 2
comptime _SLICE_HALF_VECS = _MAX_HOT_PER_CURSOR + 1
comptime _GID_MASK = UInt64(MAX_GID_COUNT - 1)


# A slot packs a run's cursor (its next position in the arena slice), the
# run's index and its group id, as DuckDB's `Slot` bitfields do; adding one
# advances the cursor.
@always_inline
def _slot(cursor: Int, idx: Int, gid: UInt64) -> UInt64:
    return (
        UInt64(cursor)
        | (UInt64(idx) << UInt64(SLOT_CURSOR_BITS))
        | (gid << UInt64(SLOT_CURSOR_BITS + SLOT_GRP_BITS))
    )


@always_inline
def _cursor(s: UInt64) -> Int:
    return Int(s & UInt64((1 << SLOT_CURSOR_BITS) - 1))


@always_inline
def _idx(s: UInt64) -> Int:
    return Int(
        (s >> UInt64(SLOT_CURSOR_BITS)) & UInt64((1 << SLOT_GRP_BITS) - 1)
    )


@always_inline
def _gid(s: UInt64) -> UInt64:
    return s >> UInt64(SLOT_CURSOR_BITS + SLOT_GRP_BITS)


@always_inline
def _slot_hash(gid: UInt64) -> Int:
    return Int(
        ((gid * 3217161767) ^ (gid >> UInt64(HASHTAB_LOG2)))
        & UInt64(HASHTAB_SZ - 1)
    )


struct _SlotTab(Copyable, Movable):
    """One cursor's hash table, arena slice and run range (`SlotTab`):
    offsets into the shared arrays."""

    var tab: Int
    var slice: Int
    var seq: Int
    var group_base: Int
    var n_runs: Int
    var hot_n_runs: Int

    def __init__(out self, tab: Int, slice: Int, seq: Int, group_base: Int):
        self.tab = tab
        self.slice = slice
        self.seq = seq
        self.group_base = group_base
        self.n_runs = 0
        self.hot_n_runs = 0


struct ClusteredRuns(Movable):
    """A worker's clustering scratch and the last chunk's runs
    (`ClusteredAggrState` with its `ClusteredAggr`): run r is group
    `gids[r]`'s `counts[r]` rows, at chunk positions `arena[starts[r]:]`,
    in order."""

    var arena: List[UInt32]
    var slots: List[UInt64]
    var starts: List[Int]
    var gids: List[UInt64]
    var counts: List[Int]
    var n_runs: Int
    var tuples_in_large: Int
    var skipped: Int
    var backoff: Int

    def __init__(out self):
        self.arena = List[UInt32](
            unsafe_uninit_length=(MAX_HOTKEYS // 2 + 1) * _BATCH
        )
        self.slots = List[UInt64](length=2 * HASHTAB_SZ, fill=FREE_SLOT)
        self.starts = List[Int](length=_BATCH, fill=0)
        self.gids = List[UInt64](length=_BATCH, fill=0)
        self.counts = List[Int](length=_BATCH, fill=0)
        self.n_runs = 0
        self.tuples_in_large = 0
        self.skipped = 0
        self.backoff = 1

    def try_build(mut self, groups: Pointer[Int, _], count: Int) -> Bool:
        """Cluster `count` (at most a vector of) rows by their group ids,
        or back off (`ClusteredAggrState::TryBuild`)."""
        if self.skipped > 0:
            self.skipped -= 1
            return False
        if count >= SAMPLE_SIZE and self._try_clustered(groups, count):
            self.skipped = 0
            self.backoff = 1
            return True
        self.skipped = self.backoff
        self.backoff *= 2
        return False

    # --- ClusteredAggr::TryClustered ------------------------------------

    @always_inline
    def _finish_run(mut self, mut st: _SlotTab, s: UInt64) -> Int:
        var idx = _idx(s)
        var cnt = (st.slice + _cursor(s)) - self.starts[idx]
        self.counts[idx] = cnt
        if idx >= st.group_base + st.hot_n_runs:
            # Sequential runs allocate from the cursor's seq region.
            st.seq += cnt
        if cnt >= RUNLENGTH_THRESHOLD:
            self.tuples_in_large += cnt
        return _cursor(s)

    @always_inline
    def _new_run(mut self, mut st: _SlotTab, gid: UInt64, start: Int) -> UInt64:
        var idx = st.group_base + st.n_runs
        st.n_runs += 1
        self.starts[idx] = st.slice + start
        self.gids[idx] = gid
        return _slot(start, idx, gid)

    @always_inline
    def _flush(mut self, mut st: _SlotTab, s: UInt64):
        if _idx(s) < st.group_base + st.hot_n_runs:
            # A hot key's slot goes back into the table.
            self.slots[st.tab + _slot_hash(_gid(s))] = s
        else:
            _ = self._finish_run(st, s)

    @always_inline
    def _hash_store(
        mut self, mut st: _SlotTab, gid: UInt64, pos: Int
    ) -> UInt64:
        var hash = _slot_hash(gid)
        var s = self.slots[st.tab + hash]
        if s == FREE_SLOT:
            s = self._new_run(st, gid, st.n_runs * _HALF_VEC)
        elif _gid(s) != gid:
            # A collision: the old run closes, the new one continues in
            # the same region.
            s = self._new_run(st, gid, self._finish_run(st, s))
        self.arena[st.slice + _cursor(s)] = UInt32(pos)
        s += 1
        self.slots[st.tab + hash] = s
        return s

    def _mixed_loop(
        mut self,
        mut st: _SlotTab,
        start: UInt64,
        groups: Pointer[Int, _],
        lo: Int,
        hi: Int,
    ):
        """Hot keys through the table, every other group a sequential
        run."""
        var cur = start
        for pos in range(lo, hi):
            var gid = UInt64(groups[unsafe_offset=pos]) & _GID_MASK
            if _gid(cur) != gid:
                self._flush(st, cur)
                var s = self.slots[st.tab + _slot_hash(gid)]
                if s != FREE_SLOT and _gid(s) == gid:
                    cur = s
                else:
                    cur = self._new_run(st, gid, st.seq - st.slice)
            self.arena[st.slice + _cursor(cur)] = UInt32(pos)
            cur += 1
        if lo < hi:
            self._flush(st, cur)

    def _merge_loop(
        mut self,
        mut st: _SlotTab,
        start: UInt64,
        groups: Pointer[Int, _],
        lo: Int,
        hi: Int,
    ):
        """Sorted input: a new run at each change of group, no table."""
        var cur = start
        for pos in range(lo, hi):
            var gid = UInt64(groups[unsafe_offset=pos]) & _GID_MASK
            if _gid(cur) != gid:
                self._flush(st, cur)
                cur = self._new_run(st, gid, st.seq - st.slice)
            self.arena[st.slice + _cursor(cur)] = UInt32(pos)
            cur += 1
        if lo < hi:
            self._flush(st, cur)

    def _free_table(mut self, mut st: _SlotTab):
        for i in range(st.hot_n_runs):
            var hash = _slot_hash(self.gids[st.group_base + i])
            var s = self.slots[st.tab + hash]
            if s != FREE_SLOT:
                _ = self._finish_run(st, s)
                self.slots[st.tab + hash] = FREE_SLOT

    def _try_clustered(mut self, groups: Pointer[Int, _], count: Int) -> Bool:
        self.tuples_in_large = 0
        var slice2 = _SLICE_HALF_VECS * _HALF_VEC
        var st1 = _SlotTab(0, 0, _MAX_HOT_PER_CURSOR * _HALF_VEC, 0)
        var st2 = _SlotTab(
            HASHTAB_SZ,
            slice2,
            slice2 + _MAX_HOT_PER_CURSOR * _HALF_VEC,
            _HALF_VEC,
        )
        var sample = SAMPLE_SIZE // 2
        var half = count // 2
        var cur1 = FREE_SLOT
        var cur2 = FREE_SLOT
        var miss1 = 0
        var miss2 = 0
        var pos = 0
        while (
            pos < sample
            and st1.n_runs < _MAX_HOT_PER_CURSOR
            and st2.n_runs < _MAX_HOT_PER_CURSOR
        ):
            var gid1 = UInt64(groups[unsafe_offset=pos]) & _GID_MASK
            var gid2 = UInt64(groups[unsafe_offset=pos + half]) & _GID_MASK
            miss1 += Int(_gid(cur1) != gid1)
            miss2 += Int(_gid(cur2) != gid2)
            cur1 = self._hash_store(st1, gid1, pos)
            cur2 = self._hash_store(st2, gid2, pos + half)
            pos += 1
        # Two misses at the first position, then one per new run.
        var fully_clustered = miss1 + miss2 <= st1.n_runs + st2.n_runs + 2
        if not fully_clustered:
            # Hash until the hot-key budget is spent.
            while (
                pos < half
                and st1.n_runs < _MAX_HOT_PER_CURSOR
                and st2.n_runs < _MAX_HOT_PER_CURSOR
            ):
                cur1 = self._hash_store(
                    st1, UInt64(groups[unsafe_offset=pos]) & _GID_MASK, pos
                )
                cur2 = self._hash_store(
                    st2,
                    UInt64(groups[unsafe_offset=pos + half]) & _GID_MASK,
                    pos + half,
                )
                pos += 1
        st1.hot_n_runs = st1.n_runs
        st2.hot_n_runs = st2.n_runs
        if fully_clustered:
            self._merge_loop(st1, cur1, groups, pos, half)
            self._merge_loop(st2, cur2, groups, pos + half, count)
        else:
            self._mixed_loop(st1, cur1, groups, pos, half)
            self._mixed_loop(st2, cur2, groups, pos + half, count)
        self._free_table(st1)
        self._free_table(st2)
        if st2.n_runs > 0 and st1.n_runs < _HALF_VEC:
            # The second cursor's runs move down after the first's.
            for r in range(st2.n_runs):
                self.starts[st1.n_runs + r] = self.starts[_HALF_VEC + r]
                self.gids[st1.n_runs + r] = self.gids[_HALF_VEC + r]
                self.counts[st1.n_runs + r] = self.counts[_HALF_VEC + r]
        self.n_runs = st1.n_runs + st2.n_runs
        # At least 80% of the rows in runs long enough to pay.
        return 5 * self.tuples_in_large >= 4 * count


def update_clustered(
    slot: AggregateFunction,
    input: Series,
    start: Int,
    count: Int,
    runs: ClusteredRuns,
    table: Int,
    row_bytes: Int,
    base: Int,
) raises -> Bool:
    """Reduction `slot`'s update over a clustered chunk (its rows from
    `start` in `input`): each run's values accumulated in registers, its
    group's state (at `table + gid * row_bytes + base`) loaded and stored once
    (DuckDB's clustered `simple_update` kernels). False for a state or
    input this does not cover, which the caller updates row by row."""
    var kind = slot.kind
    if kind == _K_LEN:
        for r in range(runs.n_runs):
            _words_at(
                (table + Int(runs.gids[r]) * row_bytes) + base
            )[] += UInt64(runs.counts[r])
        return True
    if kind == _K_COUNT or kind == _K_NULLS:
        return _clustered_counts(
            kind, input, start, count, runs, table, row_bytes, base
        )
    if kind == _K_DECIMAL_SUM:
        if input._data.isa[Column[Int64]]():
            _clustered_decimal[DType.int64](
                slot,
                input._data[Column[Int64]],
                start,
                count,
                runs,
                table,
                row_bytes,
                base,
            )
            return True
        if input._data.isa[Column[Int32]]():
            _clustered_decimal[DType.int32](
                slot,
                input._data[Column[Int32]],
                start,
                count,
                runs,
                table,
                row_bytes,
                base,
            )
            return True
        if input._data.isa[Column[Int128]]():
            _clustered_decimal[DType.int128](
                slot,
                input._data[Column[Int128]],
                start,
                count,
                runs,
                table,
                row_bytes,
                base,
            )
            return True
        return False
    if not (
        kind == _K_INT_SUM
        or kind == _K_NARROW_SUM
        or kind == _K_FLOAT_SUM
        or kind == _K_MIN
        or kind == _K_MAX
        or kind == _K_MOMENTS
    ):
        return False
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if input._data.isa[Column[Scalar[D]]]():
            _clustered_numeric[D](
                kind,
                input._data[Column[Scalar[D]]],
                start,
                count,
                runs,
                table,
                row_bytes,
                base,
            )
            return True
    return False


def _clustered_counts(
    kind: Int,
    input: Series,
    start: Int,
    count: Int,
    runs: ClusteredRuns,
    table: Int,
    row_bytes: Int,
    base: Int,
) raises -> Bool:
    var bit = 0
    var nulls = False
    var valid_at = 0
    if input._data.isa[StringColumn]():
        ref strings = input._data[StringColumn]
        bit = strings._offset
        nulls = _batch_nulls(strings._bits[], bit, start, count)
        valid_at = Int(strings._bits[].unsafe_ptr())
    elif input._data.isa[BoolColumn]():
        ref bools = input._data[BoolColumn]
        bit = bools._offset
        nulls = _batch_nulls(bools._bits[], bit, start, count)
        valid_at = Int(bools._bits[].unsafe_ptr())
    else:
        var found = False
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if not found and input._data.isa[Column[Scalar[D]]]():
                ref column = input._data[Column[Scalar[D]]]
                bit = column._offset
                nulls = _batch_nulls(column._bits[], bit, start, count)
                valid_at = Int(column._bits[].unsafe_ptr())
                found = True
        if not found and input._data.isa[Column[Int128]]():
            ref wide = input._data[Column[Int128]]
            bit = wide._offset
            nulls = _batch_nulls(wide._bits[], bit, start, count)
            valid_at = Int(wide._bits[].unsafe_ptr())
            found = True
        if not found:
            return False
    var valid = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid_at)
    var arena = runs.arena.unsafe_ptr()
    for r in range(runs.n_runs):
        var n = runs.counts[r]
        var kept = n
        if nulls:
            kept = 0
            var at = runs.starts[r]
            for k in range(n):
                var row = start + Int(arena[unsafe_offset=at + k])
                kept += Int(_validity_at(valid, bit + row))
        var add = kept if kind == _K_COUNT else n - kept
        _words_at((table + Int(runs.gids[r]) * row_bytes) + base)[] += UInt64(
            add
        )
    return True


def _clustered_decimal[
    D: DType
](
    slot: AggregateFunction,
    column: Column[Scalar[D]],
    start: Int,
    count: Int,
    runs: ClusteredRuns,
    table: Int,
    row_bytes: Int,
    base: Int,
) raises:
    """Unscaled decimals into each run's group total and count, the total
    held to the result's precision on every add (as `_update_decimal`)."""
    var values = column._ptr()
    var nulls = _batch_nulls(column._bits[], column._offset, start, count)
    var valid = column._bits[].unsafe_ptr()
    var limit = precision_limit(slot.dtype.precision())
    var arena = runs.arena.unsafe_ptr()
    for r in range(runs.n_runs):
        var at = _words_at((table + Int(runs.gids[r]) * row_bytes) + base)
        var total = bitcast[DType.int128](
            (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[])
        )
        var seen = 0
        var first = runs.starts[r]
        for k in range(runs.counts[r]):
            var i = start + Int(arena[unsafe_offset=first + k])
            if nulls and not _validity_at(valid, column._offset + i):
                continue
            total = check_limit(
                total + Int128(values[unsafe_offset=i]), limit, slot.dtype
            )
            seen += 1
        var word = bitcast[DType.uint128](total)
        at[] = UInt64(word)
        at[unsafe_offset=1] = UInt64(word >> 64)
        at[unsafe_offset=2] += UInt64(seen)


def _clustered_numeric[
    D: DType
](
    kind: Int,
    column: Column[Scalar[D]],
    start: Int,
    count: Int,
    runs: ClusteredRuns,
    table: Int,
    row_bytes: Int,
    base: Int,
):
    """Numeric values into each run's group sum, extreme or moments, the
    state's words held in registers over the run (the same updates
    `_update_slot` makes row by row)."""
    var values = column._ptr()
    var nulls = _batch_nulls(column._bits[], column._offset, start, count)
    var valid = column._bits[].unsafe_ptr()
    var bit = column._offset
    var arena = runs.arena.unsafe_ptr()
    for r in range(runs.n_runs):
        var at = _words_at((table + Int(runs.gids[r]) * row_bytes) + base)
        var first = runs.starts[r]
        var n = runs.counts[r]
        if kind == _K_INT_SUM or kind == _K_NARROW_SUM:
            var total = Int128(0)
            var seen = 0
            for k in range(n):
                var i = start + Int(arena[unsafe_offset=first + k])
                if nulls and not _validity_at(valid, bit + i):
                    continue
                total += values[unsafe_offset=i].cast[DType.int128]()
                seen += 1
            if kind == _K_NARROW_SUM:
                at[] += bitcast[DType.uint64](total.cast[DType.int64]())
                at[unsafe_offset=1] += UInt64(seen)
            else:
                var sum = (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[])
                sum += bitcast[DType.uint128](total)
                at[] = UInt64(sum)
                at[unsafe_offset=1] = UInt64(sum >> 64)
                at[unsafe_offset=2] += UInt64(seen)
        elif kind == _K_FLOAT_SUM:
            var total = bitcast[DType.float64](at[])
            var seen = 0
            for k in range(n):
                var i = start + Int(arena[unsafe_offset=first + k])
                if nulls and not _validity_at(valid, bit + i):
                    continue
                total += Float64(values[unsafe_offset=i])
                seen += 1
            at[] = bitcast[DType.uint64](total)
            at[unsafe_offset=1] += UInt64(seen)
        elif kind == _K_MOMENTS:
            var seen_n = at[]
            var mean = bitcast[DType.float64](at[unsafe_offset=1])
            var m2 = bitcast[DType.float64](at[unsafe_offset=2])
            for k in range(n):
                var i = start + Int(arena[unsafe_offset=first + k])
                if nulls and not _validity_at(valid, bit + i):
                    continue
                var x = values[unsafe_offset=i].cast[DType.float64]()
                seen_n += 1
                var delta = x - mean
                mean += delta / Float64(seen_n)
                m2 += delta * (x - mean)
            at[] = seen_n
            at[unsafe_offset=1] = bitcast[DType.uint64](mean)
            at[unsafe_offset=2] = bitcast[DType.uint64](m2)
        else:
            var is_max = kind == _K_MAX
            var flags = at[unsafe_offset=1]
            comptime if D.is_floating_point():
                var best = bitcast[DType.float64](at[])
                for k in range(n):
                    var i = start + Int(arena[unsafe_offset=first + k])
                    if nulls and not _validity_at(valid, bit + i):
                        continue
                    var x = Float64(values[unsafe_offset=i])
                    if isnan(x):
                        flags |= 2
                        continue
                    if (
                        flags & 1 == 0
                        or (is_max and x > best)
                        or (not is_max and x < best)
                    ):
                        best = x
                        flags |= 1
                at[] = bitcast[DType.uint64](best)
            else:
                var best = bitcast[DType.int64](at[])
                for k in range(n):
                    var i = start + Int(arena[unsafe_offset=first + k])
                    if nulls and not _validity_at(valid, bit + i):
                        continue
                    var x = _canonical_int[D](values[unsafe_offset=i])
                    if (
                        flags & 1 == 0
                        or (is_max and x > best)
                        or (not is_max and x < best)
                    ):
                        best = x
                        flags |= 1
                at[] = bitcast[DType.uint64](best)
            at[unsafe_offset=1] = flags


def update_clustered_pair(
    x: Series,
    y: Series,
    start: Int,
    count: Int,
    runs: ClusteredRuns,
    table: Int,
    row_bytes: Int,
    base: Int,
    scratch: Int,
) raises:
    """A co-moment's (x, y) pairs over a clustered chunk: the pairs loaded
    into `scratch` as `_update_comoment` loads them, then each run's
    group state held in registers over the run's pairs (the same updates,
    in the same order); a pair with a null on either side is dropped."""
    var xs = Pointer[Float64, MutAnyOrigin](unsafe_from_address=scratch)
    var ys = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=scratch + 8 * _BATCH
    )
    var xv = Pointer[Bool, MutAnyOrigin](
        unsafe_from_address=scratch + 16 * _BATCH
    )
    var yv = Pointer[Bool, MutAnyOrigin](
        unsafe_from_address=scratch + 17 * _BATCH
    )
    _load_floats(x, start, count, Int(xs), Int(xv))
    _load_floats(y, start, count, Int(ys), Int(yv))
    var arena = runs.arena.unsafe_ptr()
    for r in range(runs.n_runs):
        var at = _words_at(table + Int(runs.gids[r]) * row_bytes + base)
        var n = at[]
        var mean_x = bitcast[DType.float64](at[unsafe_offset=1])
        var mean_y = bitcast[DType.float64](at[unsafe_offset=2])
        var cxy = bitcast[DType.float64](at[unsafe_offset=3])
        var mxx = bitcast[DType.float64](at[unsafe_offset=4])
        var myy = bitcast[DType.float64](at[unsafe_offset=5])
        var first = runs.starts[r]
        for k in range(runs.counts[r]):
            var p = Int(arena[unsafe_offset=first + k])
            if not (xv[unsafe_offset=p] and yv[unsafe_offset=p]):
                continue
            n += 1
            var dx = xs[unsafe_offset=p] - mean_x
            mean_x += dx / Float64(n)
            var dy = ys[unsafe_offset=p] - mean_y
            mean_y += dy / Float64(n)
            cxy += dx * (ys[unsafe_offset=p] - mean_y)
            mxx += dx * (xs[unsafe_offset=p] - mean_x)
            myy += dy * (ys[unsafe_offset=p] - mean_y)
        at[] = n
        at[unsafe_offset=1] = bitcast[DType.uint64](mean_x)
        at[unsafe_offset=2] = bitcast[DType.uint64](mean_y)
        at[unsafe_offset=3] = bitcast[DType.uint64](cxy)
        at[unsafe_offset=4] = bitcast[DType.uint64](mxx)
        at[unsafe_offset=5] = bitcast[DType.uint64](myy)
