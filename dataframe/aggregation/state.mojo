"""Mergeable reduction state shared by eager grouping and streaming."""
from dataframe.aggregation.reducer import (
    Reducer,
)
from dataframe.aggregation.hash_table import (
    AggregateResult,
    ReducerSpec,
    key_supported,
    reduction_supported,
    states_supported,
)
from dataframe.aggregation.expressions import (
    _stream_reductions,
)
from dataframe.aggregation.keys import (
    _group_keys,
    _pack_struct_keys,
)
from dataframe.binding import (
    AGGREGATE,
    BoundExpr,
    bind,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.execution import (
    _ColumnStats,
    _ReduceJob,
)
from dataframe.expr import (
    COL,
    CORR,
    COV,
    Expr,
    col,
    is_reduction,
    subtree,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.hashing import (
    RowKeys,
    encode_rows,
)
from dataframe.storage.allocation import (
    huge_list,
)
from dataframe.parallel import (
    Job,
    run_jobs,
    worker_count,
)
from dataframe.partition import (
    Partitioner,
    encode_partitioned,
    low_cardinality,
)
from dataframe.row_encode import (
    STRING_PREFIX_BYTES,
    encodable,
    encode_sort_keys,
)
from dataframe.series import (
    Series,
    sort_indices,
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
    if workers <= 1 or low_cardinality(keys):
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
        var z = h ^ UInt64(words[w][row])
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        h = z ^ (z >> 31)
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

    def count(self) -> Int:
        return len(self.hashes)

    def _grow(mut self):
        var capacity = max(1024, 2 * len(self.slots))
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
    """Finish one hash part's state into its output frame."""

    var state: _StreamReduction
    var frame: DataFrame

    def __init__(out self, var state: _StreamReduction) raises:
        self.state = state^
        self.frame = DataFrame(List[Series](), height=0)

    def run(mut self) raises:
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


def _finish_parts(var parts: List[_StreamReduction]) raises -> DataFrame:
    """Finish every hash part and interleave their groups by first
    occurrence, the order a single state would have produced.

    Parts finish on their own threads. Each part's groups are already in
    first-occurrence order, so the interleave is a k-way merge on `firsts`
    with a binary heap of part cursors.
    """
    var bases = List[Int](capacity=len(parts))
    var total = 0
    for p in range(len(parts)):
        bases.append(total)
        total += parts[p].group_count()
    var firsts = List[List[Int]](capacity=len(parts))
    var jobs = List[_FinishJob](capacity=len(parts))
    while len(parts) > 0:
        var part = parts.pop(0)
        firsts.append(part.firsts.copy())
        jobs.append(_FinishJob(part^))
    run_jobs(jobs)
    var frames = List[DataFrame](capacity=len(jobs))
    for j in range(len(jobs)):
        frames.append(jobs[j].frame.copy())
    var sizes = List[Int](capacity=len(firsts))
    for p in range(len(firsts)):
        sizes.append(len(firsts[p]))
    # Heap entries: the part and its next first-occurrence row, cached so a
    # comparison reads two flat lists.
    var heap_part = List[Int]()
    var heap_key = List[Int]()
    var cursor = List[Int](length=len(sizes), fill=0)
    for p in range(len(sizes)):
        if sizes[p] > 0:
            heap_part.append(p)
            heap_key.append(firsts[p][0])
    var size = len(heap_part)

    @always_inline
    def sift_down(
        mut part: List[Int], mut keys: List[Int], size: Int, start: Int
    ):
        var i = start
        while True:
            var smallest = i
            var left = 2 * i + 1
            var right = left + 1
            if left < size and keys[left] < keys[smallest]:
                smallest = left
            if right < size and keys[right] < keys[smallest]:
                smallest = right
            if smallest == i:
                return
            var swap_key = keys[i]
            keys[i] = keys[smallest]
            keys[smallest] = swap_key
            var swap_part = part[i]
            part[i] = part[smallest]
            part[smallest] = swap_part
            i = smallest

    for i in range(size // 2 - 1, -1, -1):
        sift_down(heap_part, heap_key, size, i)
    var order = List[Int](capacity=total)
    while size > 0:
        var part = heap_part[0]
        order.append(bases[part] + cursor[part])
        cursor[part] += 1
        if cursor[part] == sizes[part]:
            size -= 1
            heap_part[0] = heap_part[size]
            heap_key[0] = heap_key[size]
        else:
            heap_key[0] = firsts[part][cursor[part]]
        sift_down(heap_part, heap_key, size, 0)
    return concat(frames).rechunk().take(order)


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
            if node.op == CORR or node.op == COV:
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
    (a row count, with none, reads the first key)."""
    if at < 0:
        return keys[0].copy()
    if (
        expression._nodes[at].op == COL
        and expression._nodes[at].text in frame.columns()
    ):
        return frame.column(expression._nodes[at].text)
    var evaluated = frame.select_exprs(
        [subtree(expression, at).alias("__input")]
    )
    return evaluated._columns[0].copy()


def table_supported(
    frame: DataFrame, keys: List[String], expressions: List[Expr]
) raises -> Bool:
    """Whether the aggregate table groups `frame` (empty is enough: only
    types are read) by `keys` and keeps every reduction's state of
    `expressions` inline."""
    if len(keys) == 0 or not _stream_reductions(expressions, True):
        return False
    for expression in expressions:
        for node in expression._nodes:
            if is_reduction(node.op) and not reduction_supported(node.op):
                return False
    var columns = List[Series](capacity=len(keys))
    for name in keys:
        if name not in frame.columns():
            return False
        var column = frame.column(name)
        if not key_supported(column):
            return False
        columns.append(column^)
    var template = _StreamReduction(frame.clear(), expressions, keys)
    return states_supported(
        table_specs(template), table_inputs(frame, columns, expressions)
    )


def table_frame(
    template: _StreamReduction, var result: AggregateResult, ordered: Bool
) raises -> DataFrame:
    """The aggregate table's partitions finished into a frame: each part's
    states loaded into a `_StreamReduction` like `template`, finished on
    its own thread, each
    output column keeping the parts as its chunks; in first-occurrence
    order when `ordered`."""
    var parts = List[_StreamReduction](capacity=len(result.firsts))
    for p in range(len(result.firsts)):
        var size = len(result.firsts[p])
        if size == 0:
            continue
        var columns = List[Series]()
        swap(columns, result.keys[p])
        var states = List[Reducer]()
        swap(states, result.states[p])
        parts.append(
            _StreamReduction(
                keys=DataFrame(columns^, height=size),
                states=states^,
                names=template.names,
                dtypes=template.dtypes,
                outputs=template.outputs,
                grouped=True,
                firsts=result.firsts[p].copy(),
                rows=0,
            )
        )
    if len(parts) == 0:
        return template.finish()
    if ordered:
        for p in range(len(parts)):
            parts[p].order_by_firsts()
        return _finish_parts(parts^)
    var count = len(parts)
    var jobs = List[_FinishJob](capacity=count)
    while len(parts) > 0:
        jobs.append(_FinishJob(parts.pop(0)))
    run_jobs(jobs)
    if count == 1:
        return jobs[0].frame.copy()
    var columns = List[Series](capacity=jobs[0].frame.width())
    var height = 0
    for j in range(count):
        height += jobs[j].frame.height()
    for c in range(jobs[0].frame.width()):
        var pieces = List[Series](capacity=count)
        for j in range(count):
            pieces.append(jobs[j].frame._columns[c].copy())
        columns.append(Series._from_chunks(pieces))
    return DataFrame(columns^, height=height)


def _reduction_batch(grouped: Bool) -> Int:
    """Rows a reduction evaluates at a time: 16K ungrouped, 4K grouped.
    The evaluator's fixed cost per batch (a column slice and a validity
    window per node) is paid a few times per stream batch instead of 64.
    TPC-DS q9's fifteen conditional sums took 90 ms at 1,024 rows and 25
    ms at 16K. A grouped reduction also touches its group states per
    batch, and ClickBench q27 (string lengths by CounterID) was 12% slower
    at 16K and level at 4K, where q59 keeps most of its gain (178 ms at
    1,024, 150 at 4K, 137 at 8K)."""
    return 4096 if grouped else 16384


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
                plan.bounds[r],
                frame._columns,
                List[Series](),
                plan.bounds[r].expr._nodes[plan.nodes[r]],
                0,
                frame.height(),
                _reduction_batch(self.grouped),
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
        var order = sort_indices([self.firsts.copy()])
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
        ).select_exprs(self.outputs)
        var columns = self.keys._columns.copy()
        for column in values._columns:
            columns.append(column.copy())
        return _pack_struct_keys(DataFrame(columns^, height=self.keys.height()))
