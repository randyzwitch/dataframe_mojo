"""Mergeable reduction state shared by eager grouping and streaming."""
from dataframe.hash_bits import _splitmix64
from dataframe.aggregation.reducer import (
    Reducer,
)
from dataframe.aggregation.hash_table import (
    AggregateResult,
    ReducerSpec,
    key_supported,
    reads_second_input,
    table_reduction,
    table_states_supported,
)
from dataframe.aggregation.expressions import (
    _stream_reductions,
    aggregate_filter,
    filter_value,
    without_filters,
)
from dataframe.aggregation.keys import (
    _group_keys,
)
from dataframe.binding import (
    AGGREGATE,
    BoundExpr,
    bind,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.execution import _ColumnStats, _ReduceInputs, _ReduceJob
from dataframe.expr import (
    COL,
    Expr,
    col,
    is_reduction,
    subtree,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.kernels.hashing import (
    RowKeys,
    encode_rows,
)
from dataframe.storage.allocation import (
    huge_list,
    release_huge,
)
from dataframe.parallel import (
    Job,
    Rounds,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.kernels.partition import (
    Partitioner,
    encode_partitioned,
)
from dataframe.kernels.row_encode import (
    STRING_PREFIX_BYTES,
    encodable,
    encode_sort_keys,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
)


def _encode_in_order(keys: List[Series]) raises -> RowKeys:
    """Dense ids in first-occurrence order, as `encode_rows` gives, using
    the hash-partitioned parallel encoder for large high-cardinality keys.

    A streaming merge encodes up to every group seen so far, and on 10M
    distinct keys the single-dictionary `encode_rows` took 6 s where the
    partitioned encoder takes a fraction of that (#326). Its numbering is
    unspecified, so one pass over the ids renumbers them by first
    occurrence and records each group's first row.
    """
    var n = len(keys[0])
    var workers = worker_count(n)
    if workers <= 1:
        return encode_rows(keys, nulls_equal=True)
    var encoded = encode_partitioned(keys, workers, nulls_equal=True)
    var renumbered = List[Int](length=encoded.count(), fill=-1)
    var ids = List[Int](capacity=n)
    var representatives = List[Int](capacity=encoded.count())
    for row in range(n):
        var id = encoded.ids[row]
        var target = renumbered[id]
        if target < 0:
            target = len(representatives)
            renumbered[id] = target
            representatives.append(row)
        ids.append(target)
    return RowKeys(ids^, representatives^)


def _equality_words(columns: List[Series]) raises -> List[List[Int]]:
    """Words that are equal for two rows exactly when their keys are equal
    as grouping defines it (every NaN one value, -0.0 equal to 0.0, a null
    one value), word-major.

    The values come from the sort-key encoder (row_encode.mojo), which
    already canonicalises NaN and -0.0 and pads strings of up to
    STRING_PREFIX_BYTES. It adds a rank word only for columns with nulls or
    floats, which would give batches different layouts, so every column's
    rank (0 value, 1 NaN, 2 null) is packed into one leading word instead,
    two bits per column. Callers check `encodable` and at most 32 columns.
    """
    var rows = len(columns[0])
    var rank = List[Int](length=rows, fill=0)
    var words = List[List[Int]]()
    words.append(List[Int]())
    for k in range(len(columns)):
        var encoded = encode_sort_keys([columns[k].copy()], [False], [True])
        var expected = 2
        if columns[k].dtype().physical() == DataType.STRING:
            expected += STRING_PREFIX_BYTES // 8
        if len(encoded) == expected:
            ref ranks = encoded[0]
            for i in range(rows):
                rank[i] |= ranks[i] << (2 * k)
            _ = encoded.pop(0)
        while len(encoded) > 0:
            words.append(encoded.pop(0))
    words[0] = rank^
    return words^


def _word_hash(words: List[List[Int]], row: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15)
    for w in range(len(words)):
        h = _splitmix64(h ^ UInt64(words[w][row]))
    return h


struct _KeyIndex(Movable):
    """Group ids by key words, kept across streaming merges (#326).

    Each group's words and hash are stored once; an open-addressing table
    maps a hash to candidate ids. Merging a batch probes only that batch's
    groups, so no merge re-encodes the groups already accumulated.
    """

    var words: List[List[Int]]
    var hashes: List[UInt64]
    var slots: List[Int]

    def __init__(out self):
        self.words = List[List[Int]]()
        self.hashes = List[UInt64]()
        self.slots = List[Int]()

    def __init__(out self, distinct: List[List[Int]]):
        """Index rows already known to be distinct, as ids 0, 1, ..."""
        var n = len(distinct[0])
        self.words = List[List[Int]](capacity=len(distinct))
        for _ in range(len(distinct)):
            self.words.append(List[Int](capacity=n))
        self.hashes = List[UInt64](capacity=n)
        var capacity = 1024
        while capacity < 2 * n:
            capacity *= 2
        self.slots = huge_list(capacity, -1)
        for row in range(n):
            _ = self.find_or_insert(distinct, row)

    def __deinit__(deinit self):
        release_huge(self.slots^)

    def count(self) -> Int:
        return len(self.hashes)

    def _grow(mut self):
        var capacity = max(1024, 2 * len(self.slots))
        release_huge(self.slots^)
        self.slots = huge_list(capacity, -1)
        var mask = capacity - 1
        for id in range(len(self.hashes)):
            var slot = Int(self.hashes[id]) & mask
            while self.slots[slot] >= 0:
                slot = (slot + 1) & mask
            self.slots[slot] = id

    def find_or_insert(mut self, source: List[List[Int]], row: Int) -> Int:
        """The id of source's row, adding it as the next id if new."""
        if len(self.words) == 0:
            for _ in range(len(source)):
                self.words.append(List[Int]())
        if 2 * (len(self.hashes) + 1) > len(self.slots):
            self._grow()
        var h = _word_hash(source, row)
        var mask = len(self.slots) - 1
        var slot = Int(h) & mask
        while True:
            var id = self.slots[slot]
            if id < 0:
                id = len(self.hashes)
                self.slots[slot] = id
                self.hashes.append(h)
                for w in range(len(source)):
                    self.words[w].append(source[w][row])
                return id
            if self.hashes[id] == h:
                var same = True
                for w in range(len(source)):
                    if self.words[w][id] != source[w][row]:
                        same = False
                        break
                if same:
                    return id
            slot = (slot + 1) & mask


struct _StreamMergeJob(Job):
    """Merge some hash parts' pending batch states into their accumulated
    states. Parts never share a key, so they merge independently (#326)."""

    var states: List[_StreamReduction]
    var pending: List[List[_StreamReduction]]

    def __init__(out self):
        self.states = List[_StreamReduction]()
        self.pending = List[List[_StreamReduction]]()

    def run(mut self) raises:
        for i in range(len(self.states)):
            self.states[i].merge_all(self.pending[i], parallel=False)
        self.pending = List[List[_StreamReduction]]()


struct _FinishJob(Job):
    """Finish one hash part's state into its output frame; when `ordered`,
    its groups put in first-occurrence order first, their first rows then
    in `firsts`."""

    var state: _StreamReduction
    var frame: DataFrame
    var ordered: Bool
    var firsts: List[Int]

    def __init__(
        out self, var state: _StreamReduction, ordered: Bool = False
    ) raises:
        self.state = state^
        self.frame = DataFrame(List[Series](), height=0)
        self.ordered = ordered
        self.firsts = List[Int]()

    def run(mut self) raises:
        if self.ordered:
            self.state.order_by_firsts()
            self.firsts = self.state.firsts.copy()
        self.frame = self.state.finish()


def _merge_ascending_runs(values: List[Int], runs: List[Int]) -> List[Int]:
    """Indices that order `values` ascending, where `values[runs[r] :
    runs[r + 1]]` is ascending for every run r: bottom-up pairwise merges
    of neighbouring runs, log2(runs) linear passes, stable (an earlier run
    wins a tie). Sequential two-pointer scans, where a heap of the runs'
    heads was bound by its dependent loads (22 ns a value)."""
    var n = len(values)
    var bounds = List[Int]()
    for r in range(len(runs) - 1):
        if runs[r] < runs[r + 1] or len(bounds) == 0:
            if len(bounds) == 0:
                bounds.append(runs[r])
            bounds.append(runs[r + 1])
    var order = List[Int](unsafe_uninit_length=n)
    var scratch = List[Int](unsafe_uninit_length=n)
    var v = values.unsafe_ptr()
    var src = order.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var dst = scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for i in range(n):
        src[unsafe_offset=i] = i
    var base = bounds[0]
    while len(bounds) > 2:
        var merged = List[Int]()
        merged.append(bounds[0])
        var r = 0
        while r + 1 < len(bounds):
            var lo = bounds[r] - base
            var hi = (
                bounds[r + 2] - base if r + 2
                < len(bounds) else bounds[r + 1] - base
            )
            var mid = bounds[r + 1] - base
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var a = src[unsafe_offset=i]
                var b = src[unsafe_offset=j]
                if v[unsafe_offset=b] < v[unsafe_offset=a]:
                    dst[unsafe_offset=k] = b
                    j += 1
                else:
                    dst[unsafe_offset=k] = a
                    i += 1
                k += 1
            while i < mid:
                dst[unsafe_offset=k] = src[unsafe_offset=i]
                i += 1
                k += 1
            while j < hi:
                dst[unsafe_offset=k] = src[unsafe_offset=j]
                j += 1
                k += 1
            merged.append(hi + base)
            r += 2
        bounds = merged^
        var held = src
        src = dst
        dst = held
    if src != order.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]():
        for i in range(n):
            order[i] = scratch[i]
    _ = scratch^
    return order^


struct PartsFinish(Movable, Rounds):
    """Hash parts finished into one frame as a round of tasks
    (`next_round`, `round_task`): each part's states finished into its
    frame by the task that takes it. After the round, the frames are the
    chunks of one frame, or, when `ordered`, their groups interleaved by
    first occurrence, the order a single state would have produced: the
    task that finishes a part puts its groups in first-occurrence order
    first, so the interleave is a merge of ascending runs of `firsts`. `frame` is the result once `next_round` returns 0."""

    var jobs: List[_FinishJob]
    # Each part's first-occurrence rows, when `ordered`.
    var firsts: List[List[Int]]
    var ordered: Bool
    var width: Int
    # Whether the round has run, and whether the frame is made.
    var ran: Bool
    var done: Bool
    var frame: DataFrame

    def __init__(
        out self, var parts: List[_StreamReduction], ordered: Bool
    ) raises:
        self.jobs = List[_FinishJob](capacity=len(parts))
        self.firsts = List[List[Int]]()
        self.ordered = ordered
        self.width = 0
        self.ran = False
        self.done = False
        self.frame = DataFrame(List[Series](), height=0)
        while len(parts) > 0:
            self.jobs.append(_FinishJob(parts.pop(0), ordered))

    @staticmethod
    def of_table(
        template: _StreamReduction, var result: AggregateResult, ordered: Bool
    ) raises -> Self:
        """The aggregate table's partitions finished into a frame: each
        part's states loaded into a `_StreamReduction` like `template`,
        each output column keeping the parts as its chunks; in
        first-occurrence order when `ordered`."""
        var parts = table_parts(template, result^)
        if len(parts) == 0:
            var empty = Self(parts^, False)
            empty.frame = template.finish()
            empty.done = True
            return empty^
        return Self(parts^, ordered)

    def next_round(mut self) raises -> Int:
        if self.done:
            return 0
        if not self.ran:
            self.ran = True
            self.width = round_width(len(self.jobs))
            if self.width > 0:
                return self.width
        self.done = True
        if self.ordered:
            for j in range(len(self.jobs)):
                var firsts = List[Int]()
                swap(firsts, self.jobs[j].firsts)
                self.firsts.append(firsts^)
            self.frame = self._interleaved()
        elif len(self.jobs) == 1:
            self.frame = self.jobs[0].frame.copy()
        else:
            var frames = List[DataFrame](capacity=len(self.jobs))
            for j in range(len(self.jobs)):
                frames.append(self.jobs[j].frame.copy())
            self.frame = chunks_frame(frames)
        self.jobs = List[_FinishJob]()
        return 0

    def round_task(mut self, task: Int) raises:
        var j = task
        while j < len(self.jobs):
            self.jobs[j].run()
            j += self.width

    def _interleaved(self) raises -> DataFrame:
        """The parts' frames merged into first-occurrence order."""
        var total = 0
        for p in range(len(self.firsts)):
            total += len(self.firsts[p])
        ref firsts = self.firsts
        var frames = List[DataFrame](capacity=len(self.jobs))
        for j in range(len(self.jobs)):
            frames.append(self.jobs[j].frame.copy())
        if total == 0:
            return concat(frames)
        # Each part's first rows ascend: merged pairwise, run by run.
        var values = List[Int](capacity=total)
        var runs = List[Int](capacity=len(firsts) + 1)
        runs.append(0)
        for p in range(len(firsts)):
            values.extend(Span(firsts[p]))
            runs.append(len(values))
        var order = _merge_ascending_runs(values, runs)
        return concat(frames).rechunk().take(order)


def _finish_parts(var parts: List[_StreamReduction]) raises -> DataFrame:
    """Finish every hash part and interleave their groups by first
    occurrence, the order a single state would have produced; the
    parts finish on their own threads (`PartsFinish`)."""
    var finish = PartsFinish(parts^, True)
    run_rounds(finish)
    return finish.frame.copy()


def table_specs(template: _StreamReduction) -> List[ReducerSpec]:
    """Each reduction's state, as the aggregate table makes it."""
    var specs = List[ReducerSpec](capacity=len(template.states))
    for s in range(len(template.states)):
        specs.append(ReducerSpec.of(template.states[s]))
    return specs^


def table_inputs(
    frame: DataFrame, keys: List[Series], expressions: List[Expr]
) raises -> List[Series]:
    """Each reduction's input over every row of `frame`, in the order of
    the template's states: its column, or its input expression evaluated
    once (a row count reads the first key). The second input of each
    correlation or covariance follows all of those, in order."""
    var inputs = List[Series]()
    var seconds = List[Series]()
    for expression in expressions:
        for i in range(len(expression._nodes)):
            ref node = expression._nodes[i]
            if not is_reduction(node.op):
                continue
            inputs.append(_table_input(frame, keys, expression, node.left))
            if reads_second_input(node.op):
                seconds.append(
                    _table_input(frame, keys, expression, node.right)
                )
    for second in seconds:
        inputs.append(second.copy())
    return inputs^


def _table_input(
    frame: DataFrame, keys: List[Series], expression: Expr, at: Int
) raises -> Series:
    """The column of the expression's node `at` over every row of `frame`
    (a row count, with none, reads the first key). An aggregate filter's
    input is the values it filters."""
    if at < 0:
        return keys[0].copy()
    var value = filter_value(expression, at)
    if (
        expression._nodes[value].op == COL
        and expression._nodes[value].text in frame.columns()
    ):
        return frame.column(expression._nodes[value].text)
    var evaluated = frame._select_exprs(
        [subtree(expression, value).alias("__input")]
    )
    return evaluated._columns[0].copy()


def table_supported(
    frame: DataFrame, keys: List[String], expressions: List[Expr]
) raises -> Bool:
    """Whether the aggregate table groups `frame` (empty is enough: only
    types are read) by `keys` and holds every reduction's state of
    `expressions`, inline or by address, aggregate filters applied per
    reduction."""
    if len(keys) == 0 or not _stream_reductions(expressions, True, True, True):
        return False
    for expression in expressions:
        for node in expression._nodes:
            if is_reduction(node.op) and not table_reduction(node.op):
                return False
    var columns = List[Series](capacity=len(keys))
    for name in keys:
        if name not in frame.columns():
            return False
        var column = frame.column(name)
        if not key_supported(column):
            return False
        columns.append(column^)
    var template = _StreamReduction(
        frame.clear(), without_filters(expressions), keys
    )
    return table_states_supported(
        table_specs(template), table_inputs(frame, columns, expressions)
    )


def table_frame(
    template: _StreamReduction, var result: AggregateResult, ordered: Bool
) raises -> DataFrame:
    """The aggregate table's partitions finished into a frame
    (`PartsFinish.of_table`), each part on its own thread."""
    var finish = PartsFinish.of_table(template, result^, ordered)
    run_rounds(finish)
    return finish.frame.copy()


def chunks_frame(frames: List[DataFrame]) raises -> DataFrame:
    """Frames of the same columns as one frame whose columns keep each
    frame's rows as a chunk, in order: no copy of the values. Empty
    frames add no chunk; with none left, the first frame's empty shape."""
    var kept = List[Int]()
    for j in range(len(frames)):
        if frames[j].height() > 0:
            kept.append(j)
    if len(kept) == 0:
        return frames[0].copy()
    if len(kept) == 1:
        return frames[kept[0]].copy()
    var columns = List[Series](capacity=frames[0].width())
    var height = 0
    for j in kept:
        height += frames[j].height()
    for c in range(frames[0].width()):
        var pieces = List[Series](capacity=len(kept))
        for j in kept:
            pieces.append(frames[j]._columns[c].copy())
        columns.append(Series._from_chunks(pieces))
    return DataFrame(columns^, height=height)


def table_parts(
    template: _StreamReduction, var result: AggregateResult
) raises -> List[_StreamReduction]:
    """Each nonempty partition of the aggregate table's `result` as a
    `_StreamReduction` like `template`, its states loaded, unfinished."""
    var parts = List[_StreamReduction](capacity=len(result.firsts))
    for p in range(len(result.firsts)):
        var size = len(result.firsts[p])
        if size == 0:
            continue
        var columns = List[Series]()
        swap(columns, result.keys[p])
        var states = List[Reducer]()
        swap(states, result.states[p])
        # Moved, not copied: a near-unique group-by's first rows are
        # megabytes, copied here on one thread.
        var firsts = List[Int]()
        swap(firsts, result.firsts[p])
        parts.append(
            _StreamReduction(
                keys=DataFrame(columns^, height=size),
                states=states^,
                names=template.names,
                dtypes=template.dtypes,
                outputs=template.outputs,
                grouped=True,
                firsts=firsts^,
                rows=0,
            )
        )
    return parts^


def _reduction_batch() -> Int:
    """Rows a reduction evaluates at a time: one bounded execution vector,
    STANDARD_VECTOR_SIZE (2,048)."""
    return 2048


struct _ReductionPlan(Copyable, Movable):
    """The reductions inside a list of aggregate expressions, each bound to
    a frame's columns, and the expressions that finish the outputs from
    their states (named `__stream_reduction_<i>`)."""

    var bounds: List[BoundExpr]
    # The reduction's node within its bound expression.
    var nodes: List[Int]
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]

    def __init__(
        out self,
        frame: DataFrame,
        expressions: List[Expr],
        key_names: Dict[String, Bool],
    ) raises:
        self.bounds = List[BoundExpr]()
        self.nodes = List[Int]()
        self.names = List[String]()
        self.dtypes = List[DataType]()
        self.outputs = List[Expr]()
        for expression in expressions:
            var bound = bind(expression, frame._columns)
            if bound.shape() != AGGREGATE:
                raise Error(
                    "Streaming aggregate requires scalar aggregate expressions"
                )
            if expression._name in key_names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression._name
                )
            var rewritten = expression.copy()
            for i in range(len(expression._nodes)):
                if not is_reduction(expression._nodes[i].op):
                    continue
                var name = "__stream_reduction_" + String(len(self.names))
                self.bounds.append(bound.copy())
                self.nodes.append(i)
                self.names.append(name)
                self.dtypes.append(bound.dtypes[i])
                rewritten._nodes[i] = col(name)._nodes[0].copy()
            self.outputs.append(subtree(rewritten, len(rewritten._nodes) - 1))


struct _StreamReduction(Movable):
    var keys: DataFrame
    var states: List[Reducer]
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var grouped: Bool
    # Row of each group's first occurrence, relative to the reduced frame
    # until the stream shifts it to the whole input; restores group order
    # when the state is split by key hash (#326).
    var firsts: List[Int]
    # Rows the state was reduced from.
    var rows: Int
    # Persistent key index for merges: 0 not built yet, 1 in use, -1 given
    # up (a key the word encoding cannot hold, such as a long string).
    var index: _KeyIndex
    var indexing: Int

    def __init__(
        out self,
        *,
        var keys: DataFrame,
        var states: List[Reducer],
        names: List[String],
        dtypes: List[DataType],
        outputs: List[Expr],
        grouped: Bool,
        var firsts: List[Int],
        rows: Int,
    ):
        self.keys = keys^
        self.states = states^
        self.names = names.copy()
        self.dtypes = dtypes.copy()
        self.outputs = outputs.copy()
        self.grouped = grouped
        self.firsts = firsts^
        self.rows = rows
        self.index = _KeyIndex()
        self.indexing = 0

    def __init__(
        out self, frame: DataFrame, expressions: List[Expr], names: List[String]
    ) raises:
        var keys = List[Series]()
        if len(names) > 0:
            keys = _group_keys(frame._subset_keys(names))
        var key_names = Dict[String, Bool]()
        for name in names:
            key_names[name] = True
        self = Self(frame, expressions, keys, key_names)

    def __init__(
        out self,
        frame: DataFrame,
        expressions: List[Expr],
        keys: List[Series],
        key_names: Dict[String, Bool],
    ) raises:
        """Reduce every row of frame, grouped by keys (already expanded and
        row-aligned with frame; none means one global group)."""
        self.grouped = len(keys) > 0
        self.rows = frame.height()
        self.index = _KeyIndex()
        self.indexing = 0
        var ids = List[Int]()
        var count = 1
        if self.grouped:
            var groups = encode_rows(keys, nulls_equal=True)
            count = groups.count()
            var columns = List[Series]()
            for key in keys:
                columns.append(key.take(groups.representatives.copy()))
            self.keys = DataFrame(columns^, height=count)
            self.firsts = groups.representatives.copy()
            ids = groups^.into_ids()
        else:
            self.keys = DataFrame(List[Series](), height=1)
            self.firsts = [0]
        var plan = _ReductionPlan(frame, expressions, key_names)
        self.outputs = plan.outputs.copy()
        self.names = plan.names.copy()
        self.dtypes = plan.dtypes.copy()
        self.states = List[Reducer]()
        var shared = ArcPointer(ids^)
        # Column statistics the batch's expressions share (`_ColumnStats`).
        var column_stats = ArcPointer(List[_ColumnStats]())
        for r in range(len(plan.bounds)):
            var job = _ReduceJob[8](
                ArcPointer(
                    _ReduceInputs(
                        plan.bounds[r],
                        frame._columns,
                        List[Series](),
                        plan.bounds[r].expr._nodes[plan.nodes[r]],
                    )
                ),
                0,
                frame.height(),
                _reduction_batch(),
                self.grouped,
                shared,
                count,
                Optional(column_stats),
            )
            job.run()
            self.states.append(job^.into_reducer())

    def merge_all(mut self, others: List[Self], parallel: Bool = True) raises:
        """Merge several partial states in one pass.

        The accumulated keys and every other state's keys are encoded
        together once, so merging k batches costs time in proportion to
        their groups plus the accumulated ones, not k times the accumulated
        groups. Accumulated keys come first and are distinct, so existing
        groups keep their ids and new ones follow in first-occurrence order.
        """
        if len(others) == 0:
            return
        if self.grouped and self.indexing >= 0:
            if self._indexable(others):
                self._merge_indexed(others)
                return
            self.indexing = -1
            self.index = _KeyIndex()
        var count = 1
        var mappings = List[List[Int]](capacity=len(others))
        if self.grouped:
            var parts = List[DataFrame](capacity=len(others) + 1)
            parts.append(self.keys.copy())
            for j in range(len(others)):
                parts.append(others[j].keys.copy())
            var combined = concat(parts)
            var groups = _encode_in_order(
                combined._columns
            ) if parallel else encode_rows(combined._columns, nulls_equal=True)
            count = groups.count()
            var start = self.keys.height()
            for j in range(len(others)):
                var height = others[j].keys.height()
                var mapping = List[Int](capacity=height)
                for i in range(height):
                    mapping.append(groups.ids[start + i])
                start += height
                mappings.append(mapping^)
            var firsts = self.firsts.copy()
            for j in range(len(others)):
                firsts.extend(others[j].firsts.copy())
            var kept = List[Int](capacity=count)
            for r in groups.representatives:
                kept.append(firsts[r])
            self.firsts = kept^
            self.keys = combined.take(groups.representatives)
        else:
            for _ in range(len(others)):
                mappings.append(List[Int]())
        for i in range(len(self.states)):
            self.states[i].grow(count)
            for j in range(len(others)):
                self.states[i].merge(others[j].states[i], mappings[j])

    def _indexable(self, others: List[Self]) raises -> Bool:
        if self.keys.width() == 0 or self.keys.width() > 32:
            return False
        if self.indexing == 0:
            for column in self.keys._columns:
                if not encodable(column):
                    return False
        for j in range(len(others)):
            for column in others[j].keys._columns:
                if not encodable(column):
                    return False
        return True

    def _merge_indexed(mut self, others: List[Self]) raises:
        """Merge through the persistent key index: each other state's
        groups are probed once, new ones appended in first-occurrence
        order, and the accumulated keys are never encoded again."""
        if self.indexing == 0:
            if self.keys.height() > 0:
                self.index = _KeyIndex(_equality_words(self.keys._columns))
            self.indexing = 1
        var key_parts = List[DataFrame](capacity=len(others) + 1)
        key_parts.append(self.keys.copy())
        var mappings = List[List[Int]](capacity=len(others))
        for j in range(len(others)):
            ref other = others[j]
            var n = other.keys.height()
            var mapping = List[Int](capacity=n)
            if n == 0:
                mappings.append(mapping^)
                continue
            var words = _equality_words(other.keys._columns)
            var added = List[Int]()
            for g in range(n):
                var before = self.index.count()
                var id = self.index.find_or_insert(words, g)
                mapping.append(id)
                if id == before:
                    added.append(g)
                    self.firsts.append(other.firsts[g])
            if len(added) > 0:
                key_parts.append(other.keys.take(added))
            mappings.append(mapping^)
        if len(key_parts) > 1:
            self.keys = concat(key_parts)
            if self.keys._columns[0].n_chunks() > 64:
                self.keys = self.keys.rechunk()
        var count = self.index.count()
        for i in range(len(self.states)):
            self.states[i].grow(count)
            for j in range(len(others)):
                self.states[i].merge(others[j].states[i], mappings[j])

    def group_count(self) -> Int:
        return self.keys.height()

    def order_by_firsts(mut self) raises:
        """Put the groups in first-occurrence order (states merged from
        rows taken out of order hold them in merge order)."""
        var n = self.keys.height()
        if n < 2:
            return
        var sorted = True
        for g in range(1, n):
            if self.firsts[g] < self.firsts[g - 1]:
                sorted = False
                break
        if sorted:
            return
        # Every group's first row is its own: sorting the rows, each
        # packed above its group, orders the groups with no comparison of
        # anything but machine integers.
        var keyed = List[UInt128](capacity=n)
        for g in range(n):
            keyed.append((UInt128(self.firsts[g]) << 64) | UInt128(g))
        sort(keyed)
        var order = List[Int](capacity=n)
        for key in keyed:
            order.append(Int(UInt64(key & 0xFFFFFFFFFFFFFFFF)))
        var firsts = List[Int](capacity=n)
        for g in order:
            firsts.append(self.firsts[g])
        var states = List[Reducer](capacity=len(self.states))
        for i in range(len(self.states)):
            ref source = self.states[i]
            var state = Reducer(
                source.op,
                source.input,
                n,
                source.min_count,
                source.integer,
                source.floating,
                source.text,
                source.logical,
            )
            if source.finished:
                state.finished = source.finished.value().take(order)
            else:
                state.merge(source, sources=order)
            states.append(state^)
        self.keys = self.keys.take(order)
        self.firsts = firsts^
        self.states = states^
        self.index = _KeyIndex()
        self.indexing = 0

    def shift_firsts(mut self, offset: Int):
        for i in range(len(self.firsts)):
            self.firsts[i] += offset

    def split(self, bits: Int) raises -> List[Self]:
        """This state's groups divided into 2^bits parts by key hash.

        The hash is the partitioner's (partition.mojo), so equal keys land
        in the same part in every batch, and each part can be merged on its
        own. Each part keeps its groups' first-occurrence rows.
        """
        var count = 1 << bits
        var n = self.keys.height()
        var members = List[List[Int]](length=count, fill=List[Int]())
        if n > 0:
            var hashed = Partitioner(self.keys._columns, 1)
            var shift = UInt64(64 - bits)
            for g in range(n):
                members[Int(hashed.hashes[g] >> shift)].append(g)
        var parts = List[Self](capacity=count)
        for p in range(count):
            var states = List[Reducer](capacity=len(self.states))
            for i in range(len(self.states)):
                ref source = self.states[i]
                var state = Reducer(
                    source.op,
                    source.input,
                    len(members[p]),
                    source.min_count,
                    source.integer,
                    source.floating,
                    source.text,
                    source.logical,
                )
                if len(members[p]) > 0:
                    state.merge(source, sources=members[p])
                states.append(state^)
            var firsts = List[Int](capacity=len(members[p]))
            for g in members[p]:
                firsts.append(self.firsts[g])
            parts.append(
                Self(
                    keys=self.keys.take(members[p]),
                    states=states^,
                    names=self.names,
                    dtypes=self.dtypes,
                    outputs=self.outputs,
                    grouped=True,
                    firsts=firsts^,
                    rows=0,
                )
            )
        return parts^

    def merge(mut self, other: Self) raises:
        var mapping = List[Int]()
        var count = 1
        if self.grouped:
            var old = self.keys.height()
            var combined = concat([self.keys.copy(), other.keys.copy()])
            var groups = encode_rows(combined._columns, nulls_equal=True)
            count = groups.count()
            for i in range(other.keys.height()):
                mapping.append(groups.ids[old + i])
            var firsts = self.firsts.copy()
            for f in other.firsts:
                firsts.append(f + self.rows)
            var kept = List[Int](capacity=count)
            for r in groups.representatives:
                kept.append(firsts[r])
            self.firsts = kept^
            self.keys = combined.take(groups.representatives)
        self.rows += other.rows
        for i in range(len(self.states)):
            self.states[i].grow(count)
            self.states[i].merge(other.states[i], mapping)

    def finish(self) raises -> DataFrame:
        var reduced = List[Series]()
        for i in range(len(self.states)):
            reduced.append(
                self.states[i]
                .finish()
                .with_dtype(self.dtypes[i])
                .renamed(self.names[i])
            )
        var values = DataFrame(
            reduced^, height=self.keys.height()
        )._select_exprs(self.outputs)
        var columns = self.keys._columns.copy()
        for column in values._columns:
            columns.append(column.copy())
        return DataFrame(columns^, height=self.keys.height())
