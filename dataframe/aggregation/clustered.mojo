"""Aggregate updates over consecutive runs of rows in the same group.

A short sample estimates how often group ids change. Input already in runs
is cut at each change; otherwise rows are clustered by group id. A run can
reuse one state address and a local state copy instead of scattering every
row's update through a separate address.

Only reductions with a compatible cluster update use this route. Decimal
sums retain their per-value overflow checks, and unsupported state kinds
keep the ordinary scatter path. Clustering state is allocated only when
multiple reductions can share the work of finding runs.
"""
from std.math import isnan
from std.memory import Pointer, bitcast

from dataframe.aggregation.function import (
    AggregateFunction,
    _BATCH,
    _K_COUNT,
    _K_DECIMAL_SUM,
    _K_FIRST,
    _K_FLOAT_SUM,
    _K_INT_SUM,
    _K_LAST,
    _K_LEN,
    _K_MAX,
    _K_MIN,
    _K_NARROW_SUM,
    _K_WIDE_MAX,
    _K_WIDE_MIN,
    _format_nulls,
    _value_words,
    _words_at,
)
from dataframe.aggregation.selected_rows import _canonical_int
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.expr import COUNT, FIRST, LAST, LEN, MAX, MIN, SUM
from dataframe.vector.types import (
    SelectionVector,
    UnifiedVectorFormat,
    physical_of,
)

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
comptime GID_HIGH_MASK = ~UInt64(MAX_GID_COUNT - 1)
comptime FREE_SLOT = UInt64.MAX
comptime _HALF_VEC = _BATCH // 2
comptime _MAX_HOT_PER_CURSOR = MAX_HOTKEYS // 2
comptime _SLICE_HALF_VECS = _MAX_HOT_PER_CURSOR + 1
comptime _GID_MASK = UInt64(MAX_GID_COUNT - 1)
comptime _CURSOR_MASK = UInt64((1 << SLOT_CURSOR_BITS) - 1)
comptime _GRP_MASK = UInt64((1 << SLOT_GRP_BITS) - 1)


def cluster_update(op: Int, kind: Int) -> Bool:
    """Whether reduction `op` with state `kind` supports run updates: count_star and
    count, sum (`IntegerSumOperation`, `SumToHugeintOperation`,
    `NumericSumOperation`, the decimal sums over them), numeric min and max
    (`NumericMinMaxBase`), first and last. avg, the variances, quantiles
    and string min/max have none."""
    if op == LEN:
        return kind == _K_LEN
    if op == COUNT:
        return kind == _K_COUNT
    if op == SUM:
        return (
            kind == _K_INT_SUM
            or kind == _K_NARROW_SUM
            or kind == _K_FLOAT_SUM
            or kind == _K_DECIMAL_SUM
        )
    if op == MIN or op == MAX:
        return (
            kind == _K_MIN
            or kind == _K_MAX
            or kind == _K_WIDE_MIN
            or kind == _K_WIDE_MAX
        )
    if op == FIRST or op == LAST:
        return kind == _K_FIRST or kind == _K_LAST
    return False


def cluster_kernel(op: Int, kind: Int) -> Bool:
    """Whether `update_clustered` folds runs into the state: the cluster
    updates of counts, integer and float sums, and numeric min and max.
    first and last.
    Decimal sums (whose overflow is checked as each value is added) and
    128-bit min and max keep their scatter, through the rows' addresses,
    when their state has no compatible cluster update."""
    return cluster_update(op, kind) and (
        kind == _K_LEN
        or kind == _K_FIRST
        or kind == _K_LAST
        or kind == _K_COUNT
        or kind == _K_INT_SUM
        or kind == _K_NARROW_SUM
        or kind == _K_FLOAT_SUM
        or kind == _K_MIN
        or kind == _K_MAX
    )


# A slot packs a run's cursor (its next position in its arena slice), the
# run's index and its group id; adding one
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
    return Int(s & _CURSOR_MASK)


@always_inline
def _idx(s: UInt64) -> Int:
    return Int((s >> UInt64(SLOT_CURSOR_BITS)) & _GRP_MASK)


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
    """One cursor's hash table, arena slice and run range (`SlotTab`), as
    offsets into the state's arrays: the table's first slot, the slice's
    first position, the bump position of its sequential region, the
    cursor's first run, its runs so far and those made while hashing."""

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


struct ClusteredAggrState(Movable):
    """A worker's clustering scratch (`ClusteredAggrState`) and the last
    chunk's runs (its `ClusteredAggr`): run r is `counts[r]` rows of group
    `gids[r]`, at the chunk positions `arena[starts[r]:]`, in order, and
    `rows[r]` is its group row's address (`InitializeStates`). Empty until
    `initialize`, when no reduction would use it."""

    var arena: List[UInt32]
    var slots: List[UInt64]
    var starts: List[Int]
    var gids: List[UInt64]
    var counts: List[Int]
    var rows: List[Int]
    # A first or last row per run: its vector index, and its position.
    var picks: List[UInt32]
    var positions: List[Int]
    var n_runs: Int
    var tuples_in_large: Int
    var skipped: Int
    var backoff: Int
    var ready: Bool

    def __init__(out self):
        self.arena = List[UInt32]()
        self.slots = List[UInt64]()
        self.starts = List[Int]()
        self.gids = List[UInt64]()
        self.counts = List[Int]()
        self.rows = List[Int]()
        self.picks = List[UInt32]()
        self.positions = List[Int]()
        self.n_runs = 0
        self.tuples_in_large = 0
        self.skipped = 0
        self.backoff = 1
        self.ready = False

    def initialize(mut self):
        """`ClusteredAggrState::Initialize`: two cursor slices of
        MAX_HOTKEYS / 2 hot regions and one sequential region of half a
        vector each, two hash tables, a vector of runs."""
        self.arena = List[UInt32](
            unsafe_uninit_length=(MAX_HOTKEYS // 2 + 1) * _BATCH
        )
        self.slots = List[UInt64](length=2 * HASHTAB_SZ, fill=FREE_SLOT)
        self.starts = List[Int](length=_BATCH, fill=0)
        self.gids = List[UInt64](length=_BATCH, fill=0)
        self.counts = List[Int](length=_BATCH, fill=0)
        self.rows = List[Int](length=_BATCH, fill=0)
        self.picks = List[UInt32](length=_BATCH, fill=0)
        self.positions = List[Int](length=_BATCH, fill=0)
        self.skipped = 0
        self.backoff = 1
        self.ready = True

    def try_build(mut self, group_ids: Int, count: Int) -> Bool:
        """Cluster `count` rows (at most a vector) by their group ids, the
        UInt64 words at address `group_ids` (only their low SLOT_GID_BITS
        bits read), or back off (`ClusteredAggrState::TryBuild`)."""
        if not self.ready:
            return False
        if self.skipped > 0:
            self.skipped -= 1
            return False
        if count >= SAMPLE_SIZE and self._try_clustered(
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=group_ids),
            count,
        ):
            self.skipped = 0
            self.backoff = 1
            return True
        self.skipped = self.backoff
        self.backoff = min(Int.MAX // 2, self.backoff) * 2
        return False

    # ClusteredAggr::TryClustered's lambdas.

    @always_inline
    def _finish_run(mut self, mut st: _SlotTab, s: UInt64) -> Int:
        var idx = _idx(s)
        var cnt = (st.slice + _cursor(s)) - self.starts.unsafe_ptr()[
            unsafe_offset=idx
        ]
        self.counts.unsafe_ptr()[unsafe_offset=idx] = cnt
        if idx >= st.group_base + st.hot_n_runs:
            # Sequential runs allocate from the cursor's sequential region.
            st.seq += cnt
        if cnt >= RUNLENGTH_THRESHOLD:
            self.tuples_in_large += cnt
        return _cursor(s)

    @always_inline
    def _new_run(
        mut self, mut st: _SlotTab, gid: UInt64, list_start: Int
    ) -> UInt64:
        var idx = st.group_base + st.n_runs
        st.n_runs += 1
        self.starts.unsafe_ptr()[unsafe_offset=idx] = st.slice + list_start
        self.gids.unsafe_ptr()[unsafe_offset=idx] = gid
        return _slot(list_start, idx, gid)

    @always_inline
    def _flush(mut self, mut st: _SlotTab, s: UInt64):
        if _idx(s) < st.group_base + st.hot_n_runs:
            # A hot key's slot goes back into the table.
            self.slots.unsafe_ptr()[
                unsafe_offset=st.tab + _slot_hash(_gid(s))
            ] = s
        else:
            _ = self._finish_run(st, s)

    @always_inline
    def _hash_store(
        mut self, mut st: _SlotTab, gid: UInt64, pos: Int
    ) -> UInt64:
        var hash = _slot_hash(gid)
        var table = self.slots.unsafe_ptr()
        var s = table[unsafe_offset=st.tab + hash]
        if s == FREE_SLOT:
            s = self._new_run(st, gid, st.n_runs * _HALF_VEC)
        elif _gid(s) != gid:
            # A collision: the old run closes, the new one continues in the
            # same region.
            s = self._new_run(st, gid, self._finish_run(st, s))
        self.arena.unsafe_ptr()[unsafe_offset=st.slice + _cursor(s)] = UInt32(
            pos
        )
        s += 1
        table[unsafe_offset=st.tab + hash] = s
        return s

    def _mixed_loop(
        mut self,
        mut st: _SlotTab,
        start: UInt64,
        group_ids: Pointer[UInt64, MutAnyOrigin],
        lo: Int,
        hi: Int,
    ):
        """Hot keys found through the table; every other group's rows a
        sequential run."""
        var cur = start
        var arena = self.arena.unsafe_ptr()
        for pos in range(lo, hi):
            var gid = group_ids[unsafe_offset=pos] & _GID_MASK
            if _gid(cur) != gid:
                self._flush(st, cur)
                var s = self.slots.unsafe_ptr()[
                    unsafe_offset=st.tab + _slot_hash(gid)
                ]
                if s != FREE_SLOT and _gid(s) == gid:
                    cur = s
                else:
                    cur = self._new_run(st, gid, st.seq - st.slice)
            arena[unsafe_offset=st.slice + _cursor(cur)] = UInt32(pos)
            cur += 1
        if lo < hi:
            self._flush(st, cur)

    def _merge_loop(
        mut self,
        mut st: _SlotTab,
        start: UInt64,
        group_ids: Pointer[UInt64, MutAnyOrigin],
        lo: Int,
        hi: Int,
    ):
        """Input in runs: a new sequential run at each change of group, no
        table lookups."""
        var cur = start
        var arena = self.arena.unsafe_ptr()
        for pos in range(lo, hi):
            var gid = group_ids[unsafe_offset=pos] & _GID_MASK
            if _gid(cur) != gid:
                self._flush(st, cur)
                cur = self._new_run(st, gid, st.seq - st.slice)
            arena[unsafe_offset=st.slice + _cursor(cur)] = UInt32(pos)
            cur += 1
        if lo < hi:
            self._flush(st, cur)

    def _free_table(mut self, mut st: _SlotTab):
        for i in range(st.hot_n_runs):
            var hash = _slot_hash(
                self.gids.unsafe_ptr()[unsafe_offset=st.group_base + i]
            )
            var s = self.slots.unsafe_ptr()[unsafe_offset=st.tab + hash]
            if s != FREE_SLOT:
                _ = self._finish_run(st, s)
                self.slots.unsafe_ptr()[unsafe_offset=st.tab + hash] = FREE_SLOT

    def _try_clustered(
        mut self, group_ids: Pointer[UInt64, MutAnyOrigin], count: Int
    ) -> Bool:
        """`ClusteredAggr::TryClustered`: the chunk's runs, and whether at
        least 80% of its rows are in runs of RUNLENGTH_THRESHOLD or more."""
        self.tuples_in_large = 0
        var slice2 = _SLICE_HALF_VECS * _HALF_VEC
        var st1 = _SlotTab(0, 0, _MAX_HOT_PER_CURSOR * _HALF_VEC, 0)
        var st2 = _SlotTab(
            HASHTAB_SZ,
            slice2,
            slice2 + _MAX_HOT_PER_CURSOR * _HALF_VEC,
            _HALF_VEC,
        )
        # Dual cursors: half as many iterations as positions.
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
            var gid1 = group_ids[unsafe_offset=pos] & _GID_MASK
            var gid2 = group_ids[unsafe_offset=pos + half] & _GID_MASK
            miss1 += Int(_gid(cur1) != gid1)
            miss2 += Int(_gid(cur2) != gid2)
            cur1 = self._hash_store(st1, gid1, pos)
            cur2 = self._hash_store(st2, gid2, pos + half)
            pos += 1
        # Both cursors miss at the first position, then once per new run.
        var fully_clustered = miss1 + miss2 <= st1.n_runs + st2.n_runs + 2
        if not fully_clustered:
            # Hash until too many distinct hot keys are found.
            while (
                pos < half
                and st1.n_runs < _MAX_HOT_PER_CURSOR
                and st2.n_runs < _MAX_HOT_PER_CURSOR
            ):
                cur1 = self._hash_store(
                    st1, group_ids[unsafe_offset=pos] & _GID_MASK, pos
                )
                cur2 = self._hash_store(
                    st2,
                    group_ids[unsafe_offset=pos + half] & _GID_MASK,
                    pos + half,
                )
                pos += 1
        # Every run made from here on is sequential.
        st1.hot_n_runs = st1.n_runs
        st2.hot_n_runs = st2.n_runs
        if fully_clustered:
            self._merge_loop(st1, cur1, group_ids, pos, half)
            self._merge_loop(st2, cur2, group_ids, pos + half, count)
        else:
            self._mixed_loop(st1, cur1, group_ids, pos, half)
            self._mixed_loop(st2, cur2, group_ids, pos + half, count)
        self._free_table(st1)
        self._free_table(st2)
        if st2.n_runs > 0 and st1.n_runs < _HALF_VEC:
            # The second cursor's runs move down next to the first's.
            var starts = self.starts.unsafe_ptr()
            var gids = self.gids.unsafe_ptr()
            var counts = self.counts.unsafe_ptr()
            for r in range(st2.n_runs):
                starts[unsafe_offset=st1.n_runs + r] = starts[
                    unsafe_offset=_HALF_VEC + r
                ]
                gids[unsafe_offset=st1.n_runs + r] = gids[
                    unsafe_offset=_HALF_VEC + r
                ]
                counts[unsafe_offset=st1.n_runs + r] = counts[
                    unsafe_offset=_HALF_VEC + r
                ]
        self.n_runs = st1.n_runs + st2.n_runs
        return 5 * self.tuples_in_large >= 4 * count

    @always_inline
    def first_position(self, r: Int) -> Int:
        """The chunk position of run r's first row."""
        return Int(
            self.arena.unsafe_ptr()[
                unsafe_offset=self.starts.unsafe_ptr()[unsafe_offset=r]
            ]
        )


def update_clustered(
    slots: List[AggregateFunction],
    kernels: List[Bool],
    inputs: List[UnifiedVectorFormat],
    mut runs: ClusteredAggrState,
    states_at: Int,
    count: Int,
    first: Int,
    scratch: Int,
) raises:
    """`RowOperations::UpdateStatesClustered` for the reductions with a
    run kernel (`kernels`): each run of the chunk's `count` rows folded
    into its group's state (at `runs.rows[r]`, states from word
    `states_at`) once, from a local copy of the state. `first` is the source row of the chunk's first, and `scratch`
    the value states' three vectors of words."""
    for s in range(len(slots)):
        if not kernels[s]:
            continue
        ref slot = slots[s]
        var base = 8 * (states_at + slot.offset)
        var kind = slot.kind
        if kind == _K_LEN:
            # count_star's `NullaryClustUpdate`.
            var rows = runs.rows.unsafe_ptr()
            var counts = runs.counts.unsafe_ptr()
            for r in range(runs.n_runs):
                _words_at(rows[unsafe_offset=r] + base)[] += UInt64(
                    counts[unsafe_offset=r]
                )
            continue
        var format = inputs[s]
        if kind == _K_FIRST or kind == _K_LAST:
            _clustered_first_last(slot, format, runs, base, first, scratch)
            continue
        var nulls = _format_nulls(format, 0, count)
        if kind == _K_COUNT:
            _clustered_counts(format, runs, base, nulls)
            continue
        var physical = format.physical
        var selected = format.sel.is_set()
        var done = False
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if not done and physical == physical_of[D]():
                if selected:
                    if nulls:
                        _clustered_kind[D, True, True](kind, format, runs, base)
                    else:
                        _clustered_kind[D, True, False](
                            kind, format, runs, base
                        )
                else:
                    if nulls:
                        _clustered_kind[D, False, True](
                            kind, format, runs, base
                        )
                    else:
                        _clustered_kind[D, False, False](
                            kind, format, runs, base
                        )
                done = True
        if not done:
            raise Error("aggregate table: unsupported clustered input")


@always_inline
def _row[
    selected: Bool
](arena: Pointer[UInt32, _], at: Int, sel: SelectionVector) -> Int:
    var i = Int(arena[unsafe_offset=at])
    comptime if selected:
        i = sel.get_index_unsafe(i)
    return i


def _clustered_first_last(
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    mut runs: ClusteredAggrState,
    base: Int,
    first: Int,
    scratch: Int,
) raises:
    """first and last's cluster update (`ClusteredOp`'s early exit): only
    a run's first (last) row can be its group's first (last) of the chunk,
    so each run's state takes that row, null or not, as the scatter would
    (a set first state is kept; a last state is replaced). A group's runs
    come in row order."""
    var last = slot.kind == _K_LAST
    var n = runs.n_runs
    var picks = runs.picks.unsafe_ptr()
    var positions = runs.positions.unsafe_ptr()
    var starts = runs.starts.unsafe_ptr()
    var counts = runs.counts.unsafe_ptr()
    var arena = runs.arena.unsafe_ptr()
    var sel = format.sel
    for r in range(n):
        var at = starts[unsafe_offset=r]
        if last:
            at += counts[unsafe_offset=r] - 1
        var p = Int(arena[unsafe_offset=at])
        positions[unsafe_offset=r] = p
        picks[unsafe_offset=r] = UInt32(sel.get_index(p))
    # The picked rows, through a selection over the vector's data.
    var picked = UnifiedVectorFormat(
        SelectionVector(Int(picks)),
        format.data,
        format.validity,
        format.physical,
    )
    var first_words = _words_at(scratch)
    var second_words = _words_at(scratch + 8 * _BATCH)
    var nulls = _words_at(scratch + 16 * _BATCH)
    for r in range(n):
        nulls[unsafe_offset=r] = 0
    _value_words(
        picked, 0, n, scratch, scratch + 8 * _BATCH, scratch + 16 * _BATCH
    )
    var width = slot.width
    var rows = runs.rows.unsafe_ptr()
    for r in range(n):
        var state = _words_at(rows[unsafe_offset=r] + base)
        if not last and state[unsafe_offset=width + 1] != 0:
            continue
        state[] = first_words[unsafe_offset=r]
        if width == 2:
            state[unsafe_offset=1] = second_words[unsafe_offset=r]
        state[unsafe_offset=width] = UInt64(first + positions[unsafe_offset=r])
        state[unsafe_offset=width + 1] = UInt64(1) if nulls[
            unsafe_offset=r
        ] != 0 else UInt64(3)


def _clustered_counts(
    format: UnifiedVectorFormat,
    runs: ClusteredAggrState,
    base: Int,
    nulls: Bool,
):
    """count's cluster update (`CountClusteredRuns`): each run's valid
    rows."""
    var rows = runs.rows.unsafe_ptr()
    var counts = runs.counts.unsafe_ptr()
    if not nulls:
        for r in range(runs.n_runs):
            _words_at(rows[unsafe_offset=r] + base)[] += UInt64(
                counts[unsafe_offset=r]
            )
        return
    var arena = Pointer[UInt32, MutAnyOrigin](
        unsafe_from_address=Int(runs.arena.unsafe_ptr())
    )
    var starts = runs.starts.unsafe_ptr()
    var sel = format.sel
    var validity = format.validity
    for r in range(runs.n_runs):
        var p = starts[unsafe_offset=r]
        var valid = 0
        for j in range(counts[unsafe_offset=r]):
            var i = sel.get_index(Int(arena[unsafe_offset=p + j]))
            valid += Int(validity.row_is_valid_unsafe(i))
        _words_at(rows[unsafe_offset=r] + base)[] += UInt64(valid)


def _clustered_kind[
    D: DType, selected: Bool, nulls: Bool
](
    kind: Int,
    format: UnifiedVectorFormat,
    runs: ClusteredAggrState,
    base: Int,
) raises:
    if kind == _K_INT_SUM:
        _clustered[D, selected, nulls, _K_INT_SUM](format, runs, base)
    elif kind == _K_NARROW_SUM:
        _clustered[D, selected, nulls, _K_NARROW_SUM](format, runs, base)
    elif kind == _K_FLOAT_SUM:
        _clustered[D, selected, nulls, _K_FLOAT_SUM](format, runs, base)
    elif kind == _K_MIN:
        _clustered[D, selected, nulls, _K_MIN](format, runs, base)
    elif kind == _K_MAX:
        _clustered[D, selected, nulls, _K_MAX](format, runs, base)
    else:
        raise Error("aggregate table: no cluster update for the state")


def _clustered[
    D: DType, selected: Bool, nulls: Bool, kind: Int
](format: UnifiedVectorFormat, runs: ClusteredAggrState, base: Int):
    """One state kind's cluster update over every run
    (`ExecuteUnaryClusteredOpt`): the group's state copied into locals,
    the run's rows folded in (a null skipped only when the vector may hold
    one), the locals stored back."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var arena = Pointer[UInt32, MutAnyOrigin](
        unsafe_from_address=Int(runs.arena.unsafe_ptr())
    )
    var starts = runs.starts.unsafe_ptr()
    var counts = runs.counts.unsafe_ptr()
    var rows = runs.rows.unsafe_ptr()
    for r in range(runs.n_runs):
        var at = _words_at(rows[unsafe_offset=r] + base)
        var p = starts[unsafe_offset=r]
        var n = counts[unsafe_offset=r]
        comptime if kind == _K_INT_SUM:
            var total = (UInt128(at[unsafe_offset=1]) << 64) | UInt128(
                at[unsafe_offset=0]
            )
            var valid = 0
            for j in range(n):
                var i = _row[selected](arena, p + j, sel)
                comptime if nulls:
                    if not validity.row_is_valid_unsafe(i):
                        continue
                total += bitcast[DType.uint128](
                    values[unsafe_offset=i].cast[DType.int128]()
                )
                valid += 1
            at[unsafe_offset=0] = UInt64(total)
            at[unsafe_offset=1] = UInt64(total >> 64)
            at[unsafe_offset=2] += UInt64(valid)
        elif kind == _K_NARROW_SUM:
            var total = at[unsafe_offset=0]
            var valid = 0
            for j in range(n):
                var i = _row[selected](arena, p + j, sel)
                comptime if nulls:
                    if not validity.row_is_valid_unsafe(i):
                        continue
                total += bitcast[DType.uint64](
                    values[unsafe_offset=i].cast[DType.int64]()
                )
                valid += 1
            at[unsafe_offset=0] = total
            at[unsafe_offset=1] += UInt64(valid)
        elif kind == _K_FLOAT_SUM:
            var total = bitcast[DType.float64](at[unsafe_offset=0])
            var valid = 0
            for j in range(n):
                var i = _row[selected](arena, p + j, sel)
                comptime if nulls:
                    if not validity.row_is_valid_unsafe(i):
                        continue
                total += Float64(values[unsafe_offset=i])
                valid += 1
            at[unsafe_offset=0] = bitcast[DType.uint64](total)
            at[unsafe_offset=1] += UInt64(valid)
        else:
            comptime is_max = kind == _K_MAX
            var best = at[unsafe_offset=0]
            var flags = at[unsafe_offset=1]
            for j in range(n):
                var i = _row[selected](arena, p + j, sel)
                comptime if nulls:
                    if not validity.row_is_valid_unsafe(i):
                        continue
                var value = values[unsafe_offset=i]
                var seen = flags & 1 != 0
                comptime if D.is_floating_point():
                    var x = Float64(value)
                    if isnan(x):
                        flags |= 2
                        continue
                    var held = bitcast[DType.float64](best)
                    var better: Bool
                    comptime if is_max:
                        better = x > held
                    else:
                        better = x < held
                    if not seen or better:
                        best = bitcast[DType.uint64](x)
                        flags |= 1
                else:
                    # Ties keep the earlier value, as the scatter does.
                    var x = _canonical_int[D](value)
                    var held = bitcast[DType.int64](best)
                    var better: Bool
                    comptime if is_max:
                        better = x > held
                    else:
                        better = x < held
                    if not seen or better:
                        best = bitcast[DType.uint64](x)
                        flags |= 1
            at[unsafe_offset=0] = best
            at[unsafe_offset=1] = flags
