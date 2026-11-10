"""Evaluate window reductions with a tree of partial aggregate states.

Build aggregate states for consecutive input ranges, then answer a frame
by combining states that cover it. Reusing interior ranges avoids
recomputing each overlapping window from every source row. Order-sensitive
reductions use their dedicated paths rather than unordered state merges.
"""
from std.ffi import external_call
from std.memory import Pointer, bitcast
from std.collections import Optional

from dataframe.aggregation.function import (
    AggregateFunction,
    ReducerSpec,
    _Finalizer,
    _Groups,
    _HeldBuffers,
    _K_COUNT,
    _K_DECIMAL_SUM,
    _K_MAX,
    _K_MIN,
    _VALUE_NONE,
    _ValueArena,
    _numeric_part,
    _slot_kind,
    _storage_kind,
    _update_states,
    combine,
)
from dataframe.aggregation.reducer import Reducer
from dataframe.dtype import DataType
from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import Series
from dataframe.storage.string import StringColumn
from dataframe.vector.data_chunk import ColumnArena, DataChunk, flat_over
from dataframe.vector.types import SelectionVector, UnifiedVectorFormat

# window_segment_tree.cpp: TREE_FANOUT needs to cleanly divide
# STANDARD_VECTOR_SIZE.
comptime TREE_FANOUT = 16
comptime STANDARD_VECTOR_SIZE = 2048


struct WindowSegmentTree(Movable):
    """WindowSegmentTreeGlobalState: the states of the tree's upper levels
    over `leaf_count` rows, and the slots they are made of."""

    var slots: List[AggregateFunction]
    var stride: Int
    var leaf_count: Int
    var levels_flat_start: List[Int]
    var levels_flat_native: List[UInt64]
    var partition_offsets: List[Int]
    var format: UnifiedVectorFormat

    def __init__(
        out self,
        var slots: List[AggregateFunction],
        stride: Int,
        leaf_count: Int,
        format: UnifiedVectorFormat,
        var partition_offsets: List[Int],
    ):
        self.slots = slots^
        self.stride = stride
        self.leaf_count = leaf_count
        self.format = format
        self.partition_offsets = partition_offsets^
        self.levels_flat_start = List[Int]()
        self.levels_flat_start.append(0)
        var levels_flat_offset = 0
        var level_current = 0
        while True:
            var level_size = leaf_count if level_current == 0 else (
                levels_flat_offset - self.levels_flat_start[level_current - 1]
            )
            if level_size <= 1:
                break
            levels_flat_offset += (level_size + TREE_FANOUT - 1) // TREE_FANOUT
            self.levels_flat_start.append(levels_flat_offset)
            level_current += 1
        # Corner case: a single row.
        self.levels_flat_native = List[UInt64](
            length=max(levels_flat_offset, 1) * stride, fill=0
        )

    @always_inline
    def state_at(self, entry: Int) -> Int:
        """The address of entry `entry` of the flat upper levels."""
        return Int(self.levels_flat_native.unsafe_ptr()) + 8 * entry * (
            self.stride
        )

    def level_size(self, level: Int) -> Int:
        if level == 0:
            return self.leaf_count
        return self.levels_flat_start[level] - self.levels_flat_start[level - 1]


struct _TreePart(Movable):
    """WindowSegmentTreePart: a worker's vectors of pending updates and
    combines, each flushed a STANDARD_VECTOR_SIZE at a time."""

    var slots: List[AggregateFunction]
    var inputs: List[UnifiedVectorFormat]
    var statep: List[Int]
    var statel: List[Int]
    var sel: List[UInt32]
    var flush_count: Int
    var arena: _ValueArena
    var discard: List[UInt64]
    var scratch: List[UInt64]

    def __init__(out self, tree: WindowSegmentTree):
        self.slots = tree.slots.copy()
        self.statep = List[Int](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        self.statel = List[Int](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        self.sel = List[UInt32](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        self.flush_count = 0
        self.arena = _ValueArena()
        self.discard = List[UInt64](length=tree.stride + 8, fill=0)
        self.scratch = List[UInt64](length=18 * STANDARD_VECTOR_SIZE, fill=0)
        # Every slot reads the one input, through this part's selection.
        var selected = UnifiedVectorFormat(
            SelectionVector(Int(self.sel.unsafe_ptr())),
            tree.format.data,
            tree.format.validity,
            tree.format.physical,
        )
        self.inputs = List[UnifiedVectorFormat](
            length=len(self.slots), fill=selected
        )

    def flush_states(mut self, combining: Bool) raises:
        if self.flush_count == 0:
            return
        if combining:
            var into = self.statep.unsafe_ptr()
            var other = self.statel.unsafe_ptr()
            for k in range(self.flush_count):
                combine(
                    self.slots,
                    into[unsafe_offset=k],
                    other[unsafe_offset=k],
                )
        else:
            _update_states(
                self.slots,
                self.inputs,
                0,
                self.flush_count,
                Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(self.statep.unsafe_ptr())
                ),
                0,
                0,
                Pointer[_ValueArena, MutAnyOrigin](
                    unsafe_from_address=Int(Pointer(to=self.arena))
                ),
                Int(self.discard.unsafe_ptr()),
                Int(self.scratch.unsafe_ptr()),
            )
        self.flush_count = 0

    def window_segment_value(
        mut self,
        tree: WindowSegmentTree,
        l_idx: Int,
        begin: Int,
        end: Int,
        state: Int,
    ) raises:
        """Entries [begin, end) of level `l_idx` into `state`."""
        if begin >= end:
            return
        var statep = self.statep.unsafe_ptr()
        if l_idx == 0:
            var sel = self.sel.unsafe_ptr()
            var row = begin
            while row < end:
                var at = self.flush_count
                var take = min(end - row, STANDARD_VECTOR_SIZE - at)
                for k in range(take):
                    statep[unsafe_offset=at + k] = state
                    sel[unsafe_offset=at + k] = UInt32(row + k)
                row += take
                self.flush_count = at + take
                if self.flush_count >= STANDARD_VECTOR_SIZE:
                    self.flush_states(False)
            return
        var statel = self.statel.unsafe_ptr()
        var entry = tree.state_at(begin + tree.levels_flat_start[l_idx - 1])
        var step = 8 * tree.stride
        for i in range(end - begin):
            statep[unsafe_offset=self.flush_count] = state
            statel[unsafe_offset=self.flush_count] = entry + i * step
            self.flush_count += 1
            if self.flush_count >= STANDARD_VECTOR_SIZE:
                self.flush_states(True)

    def evaluate_upper(
        mut self,
        tree: WindowSegmentTree,
        begins: Pointer[Int, MutAnyOrigin],
        ends: Pointer[Int, MutAnyOrigin],
        states: Int,
        step: Int,
        lo: Int,
        hi: Int,
    ) raises:
        """The tree nodes of rows [lo, hi)'s frames. Adjacent rows with
        the same frame on level 1 share the previous row's result."""
        var max_level = len(tree.levels_flat_start) + 1
        var prev_begin = 1
        var prev_end = 0
        var prev_state = 0
        for rid in range(lo, hi):
            var state = states + rid * step
            var begin = begins[unsafe_offset=rid]
            var end = ends[unsafe_offset=rid]
            if begin >= end:
                continue
            var l_idx = 0
            while l_idx < max_level:
                var parent_begin = begin // TREE_FANOUT
                var parent_end = end // TREE_FANOUT
                if (
                    prev_state != 0
                    and l_idx == 1
                    and begin == prev_begin
                    and end == prev_end
                ):
                    self.statel[self.flush_count] = prev_state
                    self.statep[self.flush_count] = state
                    self.flush_count += 1
                    if self.flush_count >= STANDARD_VECTOR_SIZE:
                        self.flush_states(True)
                    break
                if l_idx == 1:
                    prev_state = state
                    prev_begin = begin
                    prev_end = end
                if parent_begin == parent_end:
                    if l_idx > 0:
                        self.window_segment_value(
                            tree, l_idx, begin, end, state
                        )
                    break
                var group_begin = parent_begin * TREE_FANOUT
                if begin != group_begin:
                    if l_idx > 0:
                        self.window_segment_value(
                            tree,
                            l_idx,
                            begin,
                            group_begin + TREE_FANOUT,
                            state,
                        )
                    parent_begin += 1
                var group_end = parent_end * TREE_FANOUT
                if end != group_end and l_idx > 0:
                    self.window_segment_value(
                        tree, l_idx, group_end, end, state
                    )
                begin = parent_begin
                end = parent_end
                l_idx += 1
        self.flush_states(True)

    def evaluate_leaves(
        mut self,
        tree: WindowSegmentTree,
        begins: Pointer[Int, MutAnyOrigin],
        ends: Pointer[Int, MutAnyOrigin],
        states: Int,
        step: Int,
        lo: Int,
        hi: Int,
    ) raises:
        """The ragged input rows at the edges of rows [lo, hi)'s frames."""
        for rid in range(lo, hi):
            var state = states + rid * step
            var begin = begins[unsafe_offset=rid]
            var end = ends[unsafe_offset=rid]
            if begin >= end:
                continue
            var parent_begin = begin // TREE_FANOUT
            var parent_end = end // TREE_FANOUT
            if parent_begin == parent_end:
                self.window_segment_value(tree, 0, begin, end, state)
                continue
            var group_begin = parent_begin * TREE_FANOUT
            if begin != group_begin:
                self.window_segment_value(
                    tree, 0, begin, group_begin + TREE_FANOUT, state
                )
            var group_end = parent_end * TREE_FANOUT
            if end != group_end:
                self.window_segment_value(tree, 0, group_end, end, state)
        self.flush_states(False)


struct _BuildJob(Job):
    """Entries [build_begin, build_end) of level `level` (the level below
    the one written), skipping those containing a partition boundary (a
    state may start on one, but not contain one)."""

    var tree: Int
    var level: Int
    var build_begin: Int
    var build_end: Int

    def __init__(out self, tree: Int, level: Int, begin: Int, end: Int):
        self.tree = tree
        self.level = level
        self.build_begin = begin
        self.build_end = end

    def run(mut self) raises:
        ref tree = Pointer[WindowSegmentTree, MutAnyOrigin](
            unsafe_from_address=self.tree
        )[]
        var part = _TreePart(tree)
        var combine_width = TREE_FANOUT
        for _ in range(self.level):
            combine_width *= TREE_FANOUT
        var level_size = tree.level_size(self.level)
        var offsets = tree.partition_offsets.unsafe_ptr()
        var offset_count = len(tree.partition_offsets)
        # The first partition start after this job's first node's left edge.
        var cursor = 0
        while (
            cursor < offset_count
            and offsets[unsafe_offset=cursor]
            <= self.build_begin * combine_width
        ):
            cursor += 1
        for build_idx in range(self.build_begin, self.build_end):
            var combine_left = build_idx * combine_width
            var combine_right = min(
                combine_left + combine_width, tree.leaf_count
            )
            while (
                cursor < offset_count
                and offsets[unsafe_offset=cursor] <= combine_left
            ):
                cursor += 1
            var has_boundary = (
                cursor < offset_count
                and offsets[unsafe_offset=cursor] < combine_right
            )
            if not has_boundary:
                var pos = build_idx * TREE_FANOUT
                var state = tree.state_at(
                    tree.levels_flat_start[self.level] + build_idx
                )
                part.window_segment_value(
                    tree,
                    self.level,
                    pos,
                    min(level_size, pos + TREE_FANOUT),
                    state,
                )
        part.flush_states(self.level > 0)


struct _EvaluateJob(Job):
    """The frames of rows [lo, hi) into their states, a vector at a time
    (WindowSegmentTreeLocalState::Evaluate), each state zeroed first on
    this worker."""

    var tree: Int
    var begins: Int
    var ends: Int
    var states: Int
    var rows: Int
    var step: Int
    var lo: Int
    var hi: Int
    # WindowNaiveAggregator: each frame's rows folded into its state in
    # order, frames short of `needed` valid values (per `running_valid`,
    # the valid rows before each row) left empty.
    var naive: Bool
    var needed: Int
    var running_valid: Int

    def __init__(
        out self,
        tree: Int,
        begins: Int,
        ends: Int,
        states: Int,
        rows: Int,
        step: Int,
        lo: Int,
        hi: Int,
        naive: Bool,
        needed: Int,
        running_valid: Int,
    ):
        self.naive = naive
        self.needed = needed
        self.running_valid = running_valid
        self.tree = tree
        self.begins = begins
        self.ends = ends
        self.states = states
        self.rows = rows
        self.step = step
        self.lo = lo
        self.hi = hi

    def run(mut self) raises:
        ref tree = Pointer[WindowSegmentTree, MutAnyOrigin](
            unsafe_from_address=self.tree
        )[]
        var begins = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.begins)
        var ends = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ends)
        var part = _TreePart(tree)
        _ = external_call["memset", Int](
            self.rows + self.lo * self.step,
            Int32(0),
            (self.hi - self.lo) * self.step,
        )
        if self.naive:
            var counted = Pointer[Int, MutAnyOrigin](
                unsafe_from_address=self.running_valid
            )
            for rid in range(self.lo, self.hi):
                var begin = begins[unsafe_offset=rid]
                var end = ends[unsafe_offset=rid]
                if begin >= end or (
                    counted[unsafe_offset=end] - counted[unsafe_offset=begin]
                    < self.needed
                ):
                    continue
                part.window_segment_value(
                    tree, 0, begin, end, self.states + rid * self.step
                )
                part.flush_states(False)
            return
        var at = self.lo
        while at < self.hi:
            var stop = min(at + STANDARD_VECTOR_SIZE, self.hi)
            part.evaluate_upper(
                tree, begins, ends, self.states, self.step, at, stop
            )
            part.evaluate_leaves(
                tree, begins, ends, self.states, self.step, at, stop
            )
            at = stop


def _workers(items: Int) -> Int:
    return max(1, min(configured_workers(), items // STANDARD_VECTOR_SIZE))


def _build(mut tree: WindowSegmentTree) raises:
    """Every level, bottom up (WindowSegmentTreeLocalState::Finalize):
    the entries of a level in parallel ranges, all done before the next."""
    var address = Int(Pointer(to=tree))
    for level in range(len(tree.levels_flat_start)):
        var level_size = tree.level_size(level)
        if level_size <= 1:
            break
        var build_count = (level_size + TREE_FANOUT - 1) // TREE_FANOUT
        var workers = _workers(build_count * TREE_FANOUT)
        var bounds = partitions(build_count, workers, 1)
        var jobs = List[_BuildJob](capacity=workers)
        for w in range(workers):
            jobs.append(_BuildJob(address, level, bounds[w], bounds[w + 1]))
        if len(jobs) == 1:
            jobs[0].run()
        else:
            run_jobs(jobs)


struct FrameResults(Movable):
    """Each row's aggregate over its frame: `values` as the aggregate's
    finalize made them, `counts` the frame's valid inputs, and `nan`
    (empty unless a float minimum or maximum saw one) whether its frame
    held a NaN."""

    var values: Series
    var counts: List[Int64]
    var nan: List[Bool]

    def __init__(
        out self,
        var values: Series,
        var counts: List[Int64],
        var nan: List[Bool],
    ):
        self.values = values^
        self.counts = counts^
        self.nan = nan^


def frame_states_supported(spec: ReducerSpec, input: Series) -> Bool:
    """WindowSegmentTree::CanAggregate: whether `input`'s aggregate has a
    state held inline, combined without its rows."""
    return _slot_kind(spec, input) >= 0


def window_segment_tree(
    spec: ReducerSpec,
    input: Series,
    begins: List[Int],
    ends: List[Int],
    partition_offsets: List[Int],
    needed: Int,
) raises -> FrameResults:
    """The aggregate of each row's frame [begins[i], ends[i]) of `input`,
    whose frames lie inside the partitions starting at `partition_offsets`
   . A frame with fewer than `needed` valid values is left empty,
    so its finalize cannot raise. One contiguous column, whose lifetime
    the arena holds."""
    var n = len(begins)
    var kind = _slot_kind(spec, input)
    var floating = False
    if _numeric_part(input):
        floating = _numeric_part(input).value().dtype().is_float()
    var slots = List[AggregateFunction]()
    slots.append(
        AggregateFunction(
            kind,
            0,
            floating,
            spec.make(0).dtype,
            0,
            False,
            _storage_kind(input),
        )
    )
    var state_words = slots[0].state_size()
    var count_at = state_words
    slots.append(
        AggregateFunction(
            _K_COUNT,
            count_at,
            False,
            DataType.INT64,
            0,
            False,
            _storage_kind(input),
        )
    )
    state_words += slots[1].state_size()
    var arena = ColumnArena()
    var handle = arena.add(input)
    var chunk = DataChunk(1)
    chunk.data[0] = flat_over(arena, handle, 0)
    chunk.count = n
    var format = chunk.unified(0, arena)
    var tree = WindowSegmentTree(
        slots.copy(), state_words, n, format, partition_offsets.copy()
    )
    var naive = kind == _K_DECIMAL_SUM
    var running_valid = List[Int]()
    if naive:
        running_valid = List[Int](unsafe_uninit_length=n + 1)
        running_valid[0] = 0
        var mask = format.validity
        var every = mask.all_valid()
        for i in range(n):
            running_valid[i + 1] = running_valid[i] + Int(
                every or mask.row_is_valid_unsafe(i)
            )
    else:
        _build(tree)
    var groups = _Groups(0, List[Bool](), state_words, n)
    groups.n = n
    var step = 8 * groups.stride
    var states = groups.at(0) + 8 * groups.states_at
    var workers = _workers(n)
    var bounds = partitions(n, workers, STANDARD_VECTOR_SIZE)
    var jobs = List[_EvaluateJob](capacity=workers)
    for w in range(workers):
        jobs.append(
            _EvaluateJob(
                Int(Pointer(to=tree)),
                Int(begins.unsafe_ptr()),
                Int(ends.unsafe_ptr()),
                states,
                groups.at(0),
                step,
                bounds[w],
                bounds[w + 1],
                naive,
                needed,
                Int(running_valid.unsafe_ptr()),
            )
        )
    if len(jobs) == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    # A frame short of `needed` values is empty before its finalize, which
    # may raise (a sum's overflow) only for frames that have a result.
    var empty_after = groups.stride
    var head = groups.states_at
    var words = count_at
    var row = groups.rows.unsafe_ptr()
    if not naive:
        for g in range(n):
            var count = bitcast[DType.int64](
                row[unsafe_offset=g * empty_after + head + count_at]
            )
            if Int(count) < needed:
                for w in range(words):
                    row[unsafe_offset=g * empty_after + head + w] = 0
    var kept = List[List[Series]]()
    if slots[0].holds_strings():
        var one = List[Series]()
        one.append(arena.columns[handle].copy())
        kept.append(one^)
    var held = _HeldBuffers(kept)
    var finalizer = _Finalizer(slots[0], spec, n, held)
    finalizer.scan(groups, 0, n, held)
    var values = finalizer^.reducer().finish()
    var at = groups.rows.unsafe_ptr()
    var word = groups.states_at
    var stride = groups.stride
    var counts = List[Int64](unsafe_uninit_length=n)
    var nan = List[Bool]()
    var any_nan = False
    var nan_seen = floating and (kind == _K_MIN or kind == _K_MAX)
    for g in range(n):
        counts[g] = bitcast[DType.int64](
            at[unsafe_offset=g * stride + word + count_at]
        )
        if nan_seen and at[unsafe_offset=g * stride + word + 1] & 2 != 0:
            any_nan = True
    if any_nan:
        nan = List[Bool](unsafe_uninit_length=n)
        for g in range(n):
            nan[g] = at[unsafe_offset=g * stride + word + 1] & 2 != 0
    _ = running_valid^
    _ = arena^
    _ = tree^
    _ = groups^
    _ = held^
    return FrameResults(values^, counts^, nan^)
