"""DuckDB 2.0's WindowSegmentTree (function/window/window_segment_tree.cpp)
for framed window aggregates: sum, mean, min and max over each row's frame.

The tree is built over the rows of whole partitions laid out one after
another: level 0 holds one state per row, and each node of a level above
holds the combined state of TREE_FANOUT nodes below it. A frame
[begin, end) inside one partition is then combined from at most
2 * (TREE_FANOUT - 1) nodes per level (WindowSegmentTreePart::
EvaluateUpperLevels): the partial runs at the frame's edges on each level,
the whole nodes in between from the level above, left parts first and
right parts after in order.

The states are simple ones of their own (step 4 moves them onto the
aggregate state interface): an exact 128-bit integer sum, a Float64 sum,
or the best dense rank (min or max) with the row holding it, each with the
count of valid values; min and max also keep a NaN row, since a frame
holding NaN is NaN (as Polars' rolling_min / rolling_max). Frames are
evaluated in parallel over ranges of rows, the tree being read-only once
built (the GETDATA tasks of one hash group).
"""
from std.memory import Pointer

from dataframe.parallel import Job, configured_workers, partitions, run_jobs

# window_segment_tree.hpp: TREE_FANOUT needs to cleanly divide
# STANDARD_VECTOR_SIZE.
comptime TREE_FANOUT = 16
comptime STANDARD_VECTOR_SIZE = 2048

comptime INT_SUM = 0
comptime FLOAT_SUM = 1
comptime RANK_MIN = 2
comptime RANK_MAX = 3


@fieldwise_init
struct FrameState(ImplicitlyCopyable, Movable):
    """A frame's combined state."""

    var wide: Int128
    var real: Float64
    var best: Int
    var row: Int
    var nan: Int
    var count: Int

    @staticmethod
    def empty() -> Self:
        return Self(0, 0, -1, -1, -1, 0)


struct SegmentTree[kind: Int](Movable):
    """Levels of combined states over `n` rows (level 0: the rows)."""

    var level_starts: List[Int]
    var level_sizes: List[Int]
    var wide: List[Int128]
    var real: List[Float64]
    var best: List[Int]
    var row: List[Int]
    var nan: List[Int]
    var count: List[Int32]

    def __init__(
        out self,
        n: Int,
        valid_address: Int,
        ints_address: Int,
        floats_address: Int,
        ranks_address: Int,
        nans_address: Int,
    ):
        """Build over `n` rows from the addresses of their values: INT_SUM
        reads Int64 `ints`, FLOAT_SUM Float64 `floats`, RANK_MIN and
        RANK_MAX Int `ranks` and Bool `nans` (0 when there are none); all
        read Bool `valid`."""
        var valid = Pointer[Bool, MutAnyOrigin](
            unsafe_from_address=valid_address
        )
        var ints = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=ints_address
        )
        var floats = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=floats_address
        )
        var ranks = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=ranks_address
        )
        var nans = Pointer[Bool, MutAnyOrigin](unsafe_from_address=nans_address)
        self.level_starts = List[Int]()
        self.level_sizes = List[Int]()
        var total = 0
        var size = n
        while True:
            self.level_starts.append(total)
            self.level_sizes.append(size)
            total += size
            if size <= 1:
                break
            size = (size + TREE_FANOUT - 1) // TREE_FANOUT
        comptime if Self.kind == INT_SUM:
            self.wide = List[Int128](unsafe_uninit_length=total)
        else:
            self.wide = List[Int128]()
        comptime if Self.kind == FLOAT_SUM:
            self.real = List[Float64](unsafe_uninit_length=total)
        else:
            self.real = List[Float64]()
        comptime if Self.kind == RANK_MIN or Self.kind == RANK_MAX:
            self.best = List[Int](unsafe_uninit_length=total)
            self.row = List[Int](unsafe_uninit_length=total)
            self.nan = List[Int](unsafe_uninit_length=total)
        else:
            self.best = List[Int]()
            self.row = List[Int]()
            self.nan = List[Int]()
        self.count = List[Int32](unsafe_uninit_length=total)
        # Level 0: one state per row.
        for i in range(n):
            var ok = valid[unsafe_offset=i]
            self.count[i] = Int32(1) if ok else Int32(0)
            comptime if Self.kind == INT_SUM:
                self.wide[i] = Int128(ints[unsafe_offset=i]) if ok else Int128(
                    0
                )
            elif Self.kind == FLOAT_SUM:
                self.real[i] = floats[unsafe_offset=i] if ok else Float64(0)
            else:
                var is_nan = ok and nans_address != 0 and nans[unsafe_offset=i]
                self.best[i] = (
                    ranks[unsafe_offset=i] if ok and not is_nan else -1
                )
                self.row[i] = i if ok and not is_nan else -1
                self.nan[i] = i if is_nan else -1
        # Upper levels (WindowSegmentTree::ConstructTree).
        for level in range(1, len(self.level_starts)):
            var below = self.level_starts[level - 1]
            var below_size = self.level_sizes[level - 1]
            var here = self.level_starts[level]
            for node in range(self.level_sizes[level]):
                var state = FrameState.empty()
                var lo = node * TREE_FANOUT
                var hi = min(lo + TREE_FANOUT, below_size)
                for child in range(lo, hi):
                    self._combine(state, below + child)
                self._store(here + node, state)

    @always_inline
    def _combine(self, mut state: FrameState, at: Int):
        state.count += Int(self.count[at])
        comptime if Self.kind == INT_SUM:
            state.wide += self.wide[at]
        elif Self.kind == FLOAT_SUM:
            state.real += self.real[at]
        else:
            var best = self.best[at]
            if best >= 0:
                comptime if Self.kind == RANK_MIN:
                    if state.best < 0 or best < state.best:
                        state.best = best
                        state.row = self.row[at]
                else:
                    if state.best < 0 or best > state.best:
                        state.best = best
                        state.row = self.row[at]
            if state.nan < 0:
                state.nan = self.nan[at]

    @always_inline
    def _store(mut self, at: Int, state: FrameState):
        self.count[at] = Int32(state.count)
        comptime if Self.kind == INT_SUM:
            self.wide[at] = state.wide
        elif Self.kind == FLOAT_SUM:
            self.real[at] = state.real
        else:
            self.best[at] = state.best
            self.row[at] = state.row
            self.nan[at] = state.nan

    def frame(self, begin_: Int, end_: Int, mut right: List[Int]) -> FrameState:
        """The combined state of rows [begin_, end_), left to right.
        `right` is scratch space of two entries per level."""
        var state = FrameState.empty()
        var begin = begin_
        var end = end_
        if begin >= end:
            return state
        # Right-edge runs, combined after the left parts in reverse level
        # order so the result stays in row order.
        var rights = 0
        var level = 0
        while level < len(self.level_starts):
            var base = self.level_starts[level]
            var parent_begin = begin // TREE_FANOUT
            var parent_end = end // TREE_FANOUT
            if parent_begin == parent_end or level + 1 == len(
                self.level_starts
            ):
                for k in range(begin, end):
                    self._combine(state, base + k)
                break
            var group_begin = parent_begin * TREE_FANOUT
            if begin != group_begin:
                for k in range(begin, group_begin + TREE_FANOUT):
                    self._combine(state, base + k)
                parent_begin += 1
            var group_end = parent_end * TREE_FANOUT
            if end != group_end:
                right[2 * rights] = base + group_end
                right[2 * rights + 1] = base + end
                rights += 1
            begin = parent_begin
            end = parent_end
            level += 1
        for r in range(rights - 1, -1, -1):
            for k in range(right[2 * r], right[2 * r + 1]):
                self._combine(state, k)
        return state


struct _FrameJob[kind: Int](Job):
    """Evaluate the frames of rows [lo, hi): each row's state is written
    out by `emit` as a FrameState into `out` (one per row)."""

    var tree: Int
    var begins: Int
    var ends: Int
    var out: Int
    var lo: Int
    var hi: Int

    def __init__(
        out self, tree: Int, begins: Int, ends: Int, out_: Int, lo: Int, hi: Int
    ):
        self.tree = tree
        self.begins = begins
        self.ends = ends
        self.out = out_
        self.lo = lo
        self.hi = hi

    def run(mut self) raises:
        var tree = Pointer[SegmentTree[Self.kind], MutAnyOrigin](
            unsafe_from_address=self.tree
        )
        var begins = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.begins)
        var ends = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ends)
        var out = Pointer[FrameState, MutAnyOrigin](
            unsafe_from_address=self.out
        )
        var right = List[Int](length=2 * len(tree[].level_starts) + 2, fill=0)
        for i in range(self.lo, self.hi):
            out[unsafe_offset=i] = tree[].frame(
                begins[unsafe_offset=i], ends[unsafe_offset=i], right
            )


def evaluate_frames[
    kind: Int
](tree: SegmentTree[kind], begins: List[Int], ends: List[Int]) raises -> List[
    FrameState
]:
    """Every row's frame state, ranges of rows on parallel workers."""
    var n = len(begins)
    var out = List[FrameState](unsafe_uninit_length=n)
    var workers = max(1, min(configured_workers(), n // STANDARD_VECTOR_SIZE))
    var bounds = partitions(n, workers, STANDARD_VECTOR_SIZE)
    var jobs = List[_FrameJob[kind]](capacity=workers)
    for w in range(workers):
        jobs.append(
            _FrameJob[kind](
                Int(Pointer(to=tree)),
                Int(begins.unsafe_ptr()),
                Int(ends.unsafe_ptr()),
                Int(out.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
            )
        )
    if len(jobs) == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    return out^
