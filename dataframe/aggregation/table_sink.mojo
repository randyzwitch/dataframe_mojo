"""DuckDB's grouped hash aggregate sink (`PhysicalHashAggregate::Sink` into
`RadixPartitionedHashTable::Sink`) over the aggregate table
(`hash_table.mojo`).

Every data chunk a pipeline's steps produce is sunk into the worker's
table as it is made: the chunk's group columns are the group chunk
(`PopulateGroupChunk`), each reduction's input is the chunk's column or
its compiled vector expression over the chunk (the aggregate input chunk),
and both go into the worker's table a chunk at a time (`AddChunk`). At the
end of its run a worker's table is combined into the global state, the
list of every worker's partitioned rows (`Combine`), which the table's
finish merges partition by partition (`Finalize`, `aggregate_finish`).
"""
from dataframe.aggregation.function import (
    AggregateFunction,
    ReducerSpec,
    _K_LEN,
    _K_TOP_ROWS,
)
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateResult,
    AggregateSink,
    KeyVector,
    _Cursor,
    aggregate_finish,
    aggregate_layout,
    layout_widths,
)
from dataframe.dtype import DataType
from dataframe.expr import COL, CORR, COV, Expr, LEN, is_reduction, subtree
from dataframe.frame import DataFrame
from dataframe.kernels.canonical_labels import canonical_code_map
from dataframe.series import Series
from dataframe.storage.string import StringColumn
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    flat_over,
    from_frame_window,
    storage_physical,
    to_series,
)
from dataframe.parallel import Job, run_jobs
from dataframe.vector.expression import VectorExpression
from dataframe.vector.sink import Sink
from std.memory import Pointer, bitcast
from dataframe.vector.types import (
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    PHYSICAL_BOOL,
    PHYSICAL_INT128,
    PHYSICAL_INT64,
    PHYSICAL_OPAQUE,
    PHYSICAL_VARCHAR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    UnifiedVectorFormat,
    ValidityMask,
    Vector,
    zero_selection,
)


struct TableSinkLayout(Copyable, Movable):
    """What every worker's table sink shares: the table's layout, the
    group columns of the sink's input, and each reduction input as a
    column of the input (-1 for a row count, which reads no column, or for
    a compiled expression) with its type and the physical type its state
    reads."""

    var layout: AggregateLayout
    var keys: List[Int]
    var key_types: List[DataType]
    var key_physical: List[Int]
    var columns: List[Int]
    var expressions: List[Optional[VectorExpression]]
    var types: List[DataType]
    var physical: List[Int]
    # Whether the input's state reads its vector (a row count does not).
    var reads: List[Bool]

    def __init__(out self, *, copy: Self):
        self.layout = copy.layout.copy()
        self.keys = copy.keys.copy()
        self.key_types = copy.key_types.copy()
        self.key_physical = copy.key_physical.copy()
        self.columns = copy.columns.copy()
        self.expressions = copy.expressions.copy()
        self.types = copy.types.copy()
        self.physical = copy.physical.copy()
        self.reads = copy.reads.copy()

    def __init__(
        out self,
        layout: AggregateLayout,
        empty: DataFrame,
        keys: List[String],
        expressions: List[Expr],
        inputs: List[Series],
    ) raises:
        """The sink of a table laid out as `layout`, over a sink input typed
        as `empty`, grouping by `keys` and reducing `expressions`, whose
        inputs (`table_inputs` over `empty`) are typed as `inputs`."""
        self.layout = layout.copy()
        self.keys = List[Int](capacity=len(keys))
        self.key_types = List[DataType](capacity=len(keys))
        self.key_physical = List[Int](capacity=len(keys))
        for name in keys:
            var at = empty._index(name)
            self.keys.append(at)
            self.key_types.append(empty._columns[at].dtype())
            self.key_physical.append(storage_physical(empty._columns[at]))
        # In `table_inputs`' order: each reduction's input, then the
        # second input of each correlation or covariance.
        var given = List[Optional[Expr]]()
        var seconds = List[Optional[Expr]]()
        for expression in expressions:
            for node in expression._nodes:
                if not is_reduction(node.op):
                    continue
                if node.left < 0:
                    given.append(None)
                else:
                    given.append(subtree(expression, node.left))
                if node.op == CORR or node.op == COV:
                    seconds.append(subtree(expression, node.right))
        for second in seconds:
            given.append(second.copy())
        if len(given) != len(inputs):
            raise Error("table sink: inputs differ from the table's")
        self.columns = List[Int](capacity=len(given))
        self.expressions = List[Optional[VectorExpression]](capacity=len(given))
        self.types = List[DataType](capacity=len(given))
        self.physical = List[Int](capacity=len(given))
        self.reads = List[Bool](capacity=len(given))
        for i in range(len(given)):
            self.types.append(inputs[i].dtype())
            self.physical.append(storage_physical(inputs[i]))
            self.reads.append(
                i >= len(layout.slots) or layout.slots[i].kind != _K_LEN
            )
            if not given[i]:
                self.columns.append(-1)
                self.expressions.append(None)
                continue
            var value = given[i].value().copy()
            if (
                len(value._nodes) == 1
                and value._nodes[0].op == COL
                and value._nodes[0].text in empty.columns()
            ):
                self.columns.append(empty._index(value._nodes[0].text))
                self.expressions.append(None)
                continue
            self.columns.append(-1)
            self.expressions.append(VectorExpression(value, empty._columns))

    def __init__(
        out self,
        layout: AggregateLayout,
        keys: List[Series],
        inputs: List[Series],
    ):
        """The sink of a table laid out as `layout` over chunks of columns
        (`sink_columns`): the group columns `keys`, then the reductions'
        inputs `inputs` (a row count reads none of its own)."""
        self.layout = layout.copy()
        self.keys = List[Int](capacity=len(keys))
        self.key_types = List[DataType](capacity=len(keys))
        self.key_physical = List[Int](capacity=len(keys))
        for k in range(len(keys)):
            self.keys.append(k)
            self.key_types.append(keys[k].dtype())
            self.key_physical.append(storage_physical(keys[k]))
        self.columns = List[Int](capacity=len(inputs))
        self.expressions = List[Optional[VectorExpression]](
            capacity=len(inputs)
        )
        self.types = List[DataType](capacity=len(inputs))
        self.physical = List[Int](capacity=len(inputs))
        self.reads = List[Bool](capacity=len(inputs))
        for i in range(len(inputs)):
            var reads = i >= len(layout.slots) or layout.slots[i].kind != _K_LEN
            self.columns.append(len(keys) + i if reads else -1)
            self.expressions.append(None)
            self.types.append(inputs[i].dtype())
            self.physical.append(storage_physical(inputs[i]))
            self.reads.append(reads)


struct TableSinkInputs(Movable):
    """A worker's reader of a chunk's group chunk and aggregate input
    chunk (`PopulateGroupChunk` and the inputs `Sink` evaluates): its own
    copy of the compiled input expressions and the chunk they write, and
    the buffers the group chunk's key vectors are read through."""

    var keys: List[Int]
    var key_types: List[DataType]
    var key_physical: List[Int]
    var columns: List[Int]
    var expressions: List[Optional[VectorExpression]]
    var types: List[DataType]
    var physical: List[Int]
    var reads: List[Bool]
    var output: DataChunk
    # Per key, the canonical code of each categorical code (equal labels
    # group together); empty when its labels do not repeat.
    var maps: List[List[UInt32]]
    # Per key, a chunk's dictionary codes when its strings carry them.
    var codes: List[List[UInt32]]
    # The arena columns this morsel's key and input vectors read that the
    # table holds already (`hold` names each once a morsel).
    var held: List[Int]

    def __init__(out self, shared: TableSinkLayout):
        self.keys = shared.keys.copy()
        self.key_types = shared.key_types.copy()
        self.key_physical = shared.key_physical.copy()
        self.columns = shared.columns.copy()
        self.expressions = shared.expressions.copy()
        self.types = shared.types.copy()
        self.physical = shared.physical.copy()
        self.reads = shared.reads.copy()
        self.output = DataChunk(len(shared.expressions))
        self.maps = List[List[UInt32]](capacity=len(self.keys))
        self.codes = List[List[UInt32]](capacity=len(self.keys))
        for k in range(len(self.keys)):
            var map = List[UInt32]()
            ref dtype = self.key_types[k]
            if dtype.is_categorical() and dtype.has_dictionary():
                var canonical = canonical_code_map(dtype.dictionary()[])
                if canonical:
                    map = canonical.value().copy()
            self.maps.append(map^)
            self.codes.append(List[UInt32]())
        self.held = List[Int]()

    def begin(mut self):
        """A new morsel: its arena columns are new."""
        self.held.clear()

    def read_keys(
        mut self,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        mut hold: List[Series],
    ) raises -> List[KeyVector]:
        """The chunk's group columns as the table reads them: each key
        vector's unified format (DuckDB's group chunk), its codes when its
        strings carry a small dictionary's, and its categorical labels'
        canonical codes. The columns long string keys point into join
        `hold`. A vector stored other than as the key (a column narrower
        than the schema says) is cast first, and computed strings are
        copied into a column of their own, so a key points only into
        columns the table holds."""
        var keys = List[KeyVector](capacity=len(self.keys))
        for k in range(len(self.keys)):
            var c = self.keys[k]
            var v = chunk.data[c]
            if v.physical != self.key_physical[k] or (
                v.physical == PHYSICAL_VARCHAR and v.source < 0
            ):
                var cast = to_series(chunk, c, arena, "", self.key_types[k])
                chunk.data[c] = flat_over(arena, arena.add(cast), 0)
                v = chunk.data[c]
            if v.physical == PHYSICAL_VARCHAR:
                self._hold(arena, v.source, hold)
                ref strings = arena.columns[v.source]._data[StringColumn]
                # DuckDB's TryAddDictionaryGroups: a stored dictionary's
                # entries are looked up once only while it holds fewer
                # than MAX_DICTIONARY_SIZE_THRESHOLD (20,000); a larger
                # one is hashed row by row like any string.
                if strings._codes:
                    ref stored = strings._codes.value()[]
                    ref dictionary = stored.dtype.dictionary()[]
                    if len(dictionary) < 20_000:
                        _chunk_codes(strings, v, chunk.count, self.codes[k])
                        var sel = SelectionVector()
                        if v.vector_type == DICTIONARY_VECTOR:
                            sel = v.sel
                        elif v.vector_type == CONSTANT_VECTOR:
                            sel = zero_selection()
                        keys.append(
                            KeyVector(
                                UnifiedVectorFormat(
                                    sel, 0, v.validity, PHYSICAL_VARCHAR
                                ),
                                Int(self.codes[k].unsafe_ptr()),
                                Int(Pointer(to=dictionary)),
                                0,
                                0,
                                UInt64.MAX,
                            )
                        )
                        continue
            var key = KeyVector.of(chunk.unified(c, arena))
            if len(self.maps[k]) > 0:
                key.map = Int(self.maps[k].unsafe_ptr())
                key.map_length = len(self.maps[k])
            keys.append(key)
        return keys^

    def _hold(
        mut self, arena: ColumnArena, handle: Int, mut hold: List[Series]
    ):
        for h in self.held:
            if h == handle:
                return
        self.held.append(handle)
        hold.append(arena.columns[handle].copy())

    def read_inputs(
        mut self,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        mut hold: List[Series],
    ) raises -> List[UnifiedVectorFormat]:
        """Each reduction's input over the chunk's rows as its vector's
        unified format (DuckDB's payload chunk, which `UpdateStates`
        reads through `ToUnifiedFormat`): the chunk's column, or what its
        compiled expression wrote; none for a row count. The columns
        string inputs point into join `hold` (`input_vector`)."""
        var inputs = List[UnifiedVectorFormat](capacity=len(self.columns))
        var computed = False
        for i in range(len(self.columns)):
            var at = self.columns[i]
            var physical = self.physical[i]
            var dtype = self.types[i]
            if not self.reads[i]:
                inputs.append(UnifiedVectorFormat())
            elif at >= 0:
                var v = input_vector(chunk, at, arena, physical, dtype)
                if v.physical == PHYSICAL_VARCHAR:
                    self._hold(arena, v.source, hold)
                inputs.append(chunk.unified(at, arena))
            elif self.expressions[i]:
                if not computed:
                    self.output.reset()
                    self.output.count = chunk.count
                    computed = True
                self.expressions[i].value().write(chunk, arena, self.output, i)
                var v = input_vector(self.output, i, arena, physical, dtype)
                if v.physical == PHYSICAL_VARCHAR:
                    self._hold(arena, v.source, hold)
                inputs.append(self.output.unified(i, arena))
            else:
                inputs.append(UnifiedVectorFormat())
        return inputs^

    def release(mut self):
        """Only sinking reads the expressions."""
        self.expressions = List[Optional[VectorExpression]]()
        self.output = DataChunk()


def _chunk_codes(
    strings: StringColumn, v: Vector, count: Int, mut codes: List[UInt32]
):
    """The dictionary code of each of a chunk's `count` rows of `v`, a
    vector over the coded string column `strings` (a null row's code is
    unspecified), into `codes`: read through the column's codes as
    `StringColumn._write_codes` reads them."""
    if len(codes) < count:
        codes = List[UInt32](length=count, fill=0)
    var out = codes.unsafe_ptr()
    ref stored = strings._codes.value()[]
    var source = stored.codes[].unsafe_ptr()
    var base = strings._offset + v.source_offset
    if not stored.rows:
        for i in range(count):
            out[unsafe_offset=i] = source[unsafe_offset=base + v.child_row(i)]
        return
    var rows = stored.rows.value()[].unsafe_ptr().unsafe_offset(stored.first)
    for i in range(count):
        var row = rows[unsafe_offset=base + v.child_row(i)]
        out[unsafe_offset=i] = source[
            unsafe_offset=row + stored.shift
        ] if row != -1 else UInt32(0)


def sink_columns(
    mut table: AggregateSink,
    mut reader: TableSinkInputs,
    keys: List[Series],
    inputs: List[Series],
    first: Int,
) raises:
    """A morsel's group columns `keys` and reduction inputs `inputs` (each
    unchunked or one chunk; `first` the source row of their first) into
    `table` a data chunk at a time, as a table scan's chunks of flat
    vectors over them reach the pipeline's sink (`reader` from a
    `TableSinkLayout` over such columns)."""
    var arena = ColumnArena()
    var handles = List[Int](capacity=len(keys) + len(inputs))
    for key in keys:
        handles.append(arena.add(key))
    for i in range(len(inputs)):
        # A row count's chunk column is never read.
        handles.append(arena.add(inputs[i]) if reader.reads[i] else handles[0])
    reader.begin()
    var chunk = DataChunk(len(handles))
    var length = len(keys[0])
    for lo in range(0, length, STANDARD_VECTOR_SIZE):
        var n = min(STANDARD_VECTOR_SIZE, length - lo)
        chunk.reset()
        from_frame_window(chunk, arena, handles, lo, n)
        var hold = List[Series]()
        var key_vectors = reader.read_keys(chunk, arena, hold)
        var input_vectors = reader.read_inputs(chunk, arena, hold)
        table.sink_vectors(key_vectors, input_vectors, hold, first + lo, n)
    _ = chunk^
    _ = arena^


struct _SinkJob(Job):
    """Phase 1 over a frame's columns: one worker's morsels (row ranges
    within chunks, from a shared cursor) into its own table, a data chunk
    at a time (`sink_columns`)."""

    var keys: List[Series]
    var inputs: List[Series]
    var shared: TableSinkLayout
    var ranges: List[Int]
    var cursor: _Cursor
    var sink: AggregateSink

    def __init__(
        out self,
        keys: List[Series],
        inputs: List[Series],
        layout: AggregateLayout,
        ranges: List[Int],
        cursor: _Cursor,
        workers: Int,
    ):
        self.keys = keys.copy()
        self.inputs = inputs.copy()
        self.shared = TableSinkLayout(layout, keys, inputs)
        self.ranges = ranges.copy()
        self.cursor = cursor.copy()
        self.sink = AggregateSink(layout, workers)

    def run(mut self) raises:
        var reader = TableSinkInputs(self.shared)
        var morsels = len(self.ranges) // 2
        while True:
            var m = self.cursor.take()
            if m >= morsels:
                break
            var offset = self.ranges[2 * m]
            var length = self.ranges[2 * m + 1]
            var keys = List[Series](capacity=len(self.keys))
            for key in self.keys:
                keys.append(key.slice(offset, length))
            var inputs = List[Series](capacity=len(self.inputs))
            for s in range(len(self.inputs)):
                # A row count reads no input.
                var size = length if reader.reads[s] else 0
                inputs.append(self.inputs[s].slice(offset, size))
            sink_columns(self.sink, reader, keys, inputs, offset)

    def take_sink(deinit self) -> AggregateSink:
        return self.sink^


def _sink_jobs(
    keys: List[Series],
    inputs: List[Series],
    layout: AggregateLayout,
    ranges: List[Int],
    workers: Int,
) raises -> List[AggregateSink]:
    """Every worker's table after sinking the (offset, length) morsels
    `ranges` of `keys` and `inputs` on `workers` threads."""
    var cursor = _Cursor.new()
    var jobs = List[_SinkJob](capacity=workers)
    for _ in range(max(1, workers)):
        jobs.append(_SinkJob(keys, inputs, layout, ranges, cursor, workers))
    try:
        run_jobs(jobs)
    except e:
        cursor.free()
        raise e^
    cursor.free()
    var sinks = List[AggregateSink](capacity=len(jobs))
    while len(jobs) > 0:
        var job = jobs.pop()
        sinks.append(job^.take_sink())
    return sinks^


def aggregate_table(
    keys: List[Series],
    inputs: List[Series],
    templates: List[ReducerSpec],
    ranges: List[Int],
    workers: Int,
    ordered: Bool,
) raises -> AggregateResult:
    """Group `keys` and reduce each of `inputs` with its `templates` state,
    over the (offset, length) morsels `ranges` (within chunks) on `workers`
    threads. Returns per finish partition its groups' keys, first rows and
    states (unfinished), partitions in hash order; each partition's groups
    in first-row order when `ordered`, otherwise in no particular order.
    Every reduction's state must fit inline (`states_supported`)."""
    var layout = aggregate_layout(keys, inputs, templates)
    var sinks = _sink_jobs(keys, inputs, layout, ranges, workers)
    return aggregate_finish(sinks, layout, ordered=ordered)


def aggregate_top_rows(
    keys: List[Series],
    value: Series,
    k: Int,
    descending: Bool,
    ranges: List[Int],
    workers: Int,
) raises -> List[Int]:
    """The rows of each group of `keys` whose `value` ranks among its k
    first by ordinal rank (ties: the earlier row), over the (offset,
    length) morsels `ranges` on `workers` threads, in no order: each
    group's row in the table holds its k best (value, row) pairs, merged
    across workers at the finish. `k` is at most `TOP_ROWS_INLINE`; nulls
    are never kept."""
    var count_spec = ReducerSpec(
        LEN, DataType.INT64, 0, 0, 0.0, String(), DataType.INT64
    )
    var layout = aggregate_layout(keys, [value.copy()], [count_spec^])
    var dtype = layout.slots[0].dtype
    layout.slots[0] = AggregateFunction(
        _K_TOP_ROWS, 0, value.dtype().is_float(), dtype, k, descending
    )
    layout.state_words = layout.slots[0].state_size()
    var sinks = _sink_jobs(keys, [value.copy()], layout, ranges, workers)
    var result = aggregate_finish(sinks, layout, ordered=False)
    var rows = List[Int]()
    swap(rows, result.top_rows)
    return rows^


struct TableSink(Sink):
    """One worker's table sink state (DuckDB's `HashAggregateLocalSinkState`
    over a `RadixHTLocalSinkState`): its table, its chunk reader, and the
    next row's number."""

    var table: AggregateSink
    var inputs: TableSinkInputs
    var sequence: Int

    def __init__(out self, shared: TableSinkLayout, workers: Int):
        self.table = AggregateSink(shared.layout, workers)
        self.inputs = TableSinkInputs(shared)
        self.sequence = 0

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32
        self.inputs.begin()

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk into the table: its group chunk's and payload chunk's
        vectors as they are."""
        var n = chunk.count
        if n == 0:
            return
        var hold = List[Series]()
        var keys = self.inputs.read_keys(chunk, arena, hold)
        var inputs = self.inputs.read_inputs(chunk, arena, hold)
        self.table.sink_vectors(keys, inputs, hold, self.sequence, n)
        self.sequence += n

    def release(mut self):
        """Only the table outlives the worker's run."""
        self.inputs.release()

    def combine(var self, mut combined: List[TableSink]) raises:
        """This worker's table into the global state, every worker's
        partitioned rows (`RadixPartitionedHashTable::Combine`): the finish
        merges them partition by partition."""
        combined.append(self^)

    def take_table(deinit self) -> AggregateSink:
        """The worker's table, for the finish."""
        return self.table^


# The constant key a grouping sets' grand total groups by (the aggregate
# table needs a key column); dropped from that set's result.
comptime TOTAL_KEY = "__grouping_total"


struct GroupingSetsLayout(Copyable, Movable):
    """What every worker's grouping sets sink shares: the reader of the
    group columns (every key of every set) and the reductions' inputs;
    per set, its table's layout, its key columns (indices into the keys;
    empty is the grand total) and their masks (a key packing several coded
    keys is grouped by the fields the set keeps, the others masked to
    zero; -1 keeps the whole value; empty: none); and per set derived
    from another's group rows rather than sunk from the chunks, its source
    set (-1 for none), its key columns among the source's and their masks,
    with the sets by size, largest first (a set is derived after its
    source)."""

    var inputs: TableSinkLayout
    var layouts: List[AggregateLayout]
    var sets: List[List[Int]]
    var set_masks: List[List[Int64]]
    var derived_from: List[Int]
    var derived_columns: List[List[Int]]
    var derived_masks: List[List[Int64]]
    var derive_order: List[Int]

    def __init__(out self, *, copy: Self):
        self.inputs = copy.inputs.copy()
        self.layouts = copy.layouts.copy()
        self.sets = copy.sets.copy()
        self.set_masks = copy.set_masks.copy()
        self.derived_from = copy.derived_from.copy()
        self.derived_columns = copy.derived_columns.copy()
        self.derived_masks = copy.derived_masks.copy()
        self.derive_order = copy.derive_order.copy()

    def __init__(
        out self,
        var inputs: TableSinkLayout,
        layouts: List[AggregateLayout],
        sets: List[List[Int]],
        set_masks: List[List[Int64]],
        derived_from: List[Int],
        derived_columns: List[List[Int]],
        derived_masks: List[List[Int64]],
        derive_order: List[Int],
    ):
        self.inputs = inputs^
        self.layouts = layouts.copy()
        self.sets = sets.copy()
        self.set_masks = set_masks.copy()
        self.derived_from = derived_from.copy()
        self.derived_columns = derived_columns.copy()
        self.derived_masks = derived_masks.copy()
        self.derive_order = derive_order.copy()


struct GroupingSetsSink(Sink):
    """One worker's grouping sets sink state: one table per set, every
    chunk's group columns and reduction inputs read once and sunk into the
    table of each set that is not derived (DuckDB's one sink feeding every
    grouping set's table, `PhysicalHashAggregate::Sink`). When the
    worker's run ends, each derived set's table takes its source set's
    group rows, so a coarser grouping reads groups, not input rows."""

    var shared: GroupingSetsLayout
    var tables: List[AggregateSink]
    var inputs: TableSinkInputs
    var sequence: Int
    # The grand total's constant key: one zero.
    var zero: List[Int64]

    def __init__(out self, shared: GroupingSetsLayout, workers: Int):
        self.shared = shared.copy()
        self.tables = List[AggregateSink](capacity=len(shared.sets))
        for s in range(len(shared.sets)):
            self.tables.append(
                AggregateSink(shared.layouts[s], workers, len(shared.sets))
            )
        self.inputs = TableSinkInputs(shared.inputs)
        self.sequence = 0
        self.zero = List[Int64](length=1, fill=0)

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32
        self.inputs.begin()

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk into every set's table that is not derived, each taking
        the key vectors it groups by: a packed key masked to the fields
        the set keeps, the grand total a constant zero."""
        var n = chunk.count
        if n == 0:
            return
        var hold = List[Series]()
        var all_keys = self.inputs.read_keys(chunk, arena, hold)
        var inputs = self.inputs.read_inputs(chunk, arena, hold)
        for s in range(len(self.shared.sets)):
            if self.shared.derived_from[s] >= 0:
                continue
            ref one = self.shared.sets[s]
            var keys = List[KeyVector](capacity=max(1, len(one)))
            if len(one) == 0:
                keys.append(
                    KeyVector.of(
                        UnifiedVectorFormat(
                            zero_selection(),
                            Int(self.zero.unsafe_ptr()),
                            ValidityMask(),
                            PHYSICAL_INT64,
                        )
                    )
                )
            else:
                for j in range(len(one)):
                    var key = all_keys[one[j]]
                    var mask = self.shared.set_masks[s][j] if len(
                        self.shared.set_masks
                    ) > 0 else Int64(-1)
                    if mask != -1:
                        if key.format.physical != PHYSICAL_INT64:
                            raise Error("grouping sets: a masked key is Int64")
                        key.mask = bitcast[DType.uint64](mask)
                    keys.append(key)
            self.tables[s].sink_vectors(keys, inputs, hold, self.sequence, n)
        self.sequence += n

    def release(mut self):
        self.inputs.release()

    def derive(mut self) raises:
        """The derived sets' tables from their sources' group rows, largest
        set first: run on the worker's own thread when its run ends."""
        for t in self.shared.derive_order:
            var from_set = self.shared.derived_from[t]
            if from_set < 0:
                continue
            # The target comes out of the list while the source is read.
            var target = self.tables.pop(t)
            target.sink_group_rows(
                self.tables[from_set if from_set < t else from_set - 1],
                self.shared.derived_columns[t],
                layout_widths(self.shared.layouts[from_set]),
                self.shared.derived_masks[t],
            )
            self.tables.insert(t, target^)

    def combine(var self, mut combined: List[GroupingSetsSink]) raises:
        """This worker's tables into the global state, every worker's: the
        finish merges each set's tables partition by partition."""
        combined.append(self^)


def input_vector(
    mut chunk: DataChunk,
    c: Int,
    mut arena: ColumnArena,
    physical: Int,
    dtype: DataType,
) raises -> Vector:
    """Column `c` of `chunk` as a state update reads it, typed `dtype`
    whose storage is `physical`. The update dispatches on the vector's
    physical type, so a narrower integer or float is read as it is; only
    a vector of another kind (128-bit against narrower, or strings,
    Booleans or nested values against anything else) is cast first, into
    a column of the arena. Computed strings are copied into a column of
    their own too, so a string state points only into arena columns,
    which the sink holds."""
    var v = chunk.data[c]
    var cast = v.physical == PHYSICAL_VARCHAR and v.source < 0
    if v.physical != physical:
        cast = cast or not (
            _numeric_physical(v.physical) and _numeric_physical(physical)
        )
    if cast:
        var column = to_series(chunk, c, arena, "", dtype)
        chunk.data[c] = flat_over(arena, arena.add(column), 0)
    return chunk.data[c]


def _numeric_physical(physical: Int) -> Bool:
    """Whether a physical type is a number of at most 64 bits."""
    return physical != PHYSICAL_VARCHAR and (
        physical != PHYSICAL_BOOL
        and physical != PHYSICAL_INT128
        and physical != PHYSICAL_OPAQUE
    )
