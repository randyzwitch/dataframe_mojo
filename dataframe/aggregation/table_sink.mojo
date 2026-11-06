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
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    layout_widths,
)
from dataframe.dtype import DataType
from dataframe.expr import COL, CORR, COV, Expr, is_reduction, subtree
from dataframe.frame import DataFrame
from dataframe.kernels.canonical_labels import canonical_labels
from dataframe.series import Series
from dataframe.storage.column import Column, _copy_validity
from dataframe.vector.data_chunk import ColumnArena, DataChunk, to_series
from dataframe.vector.expression import VectorExpression
from dataframe.vector.sink import Sink


struct TableSinkLayout(Copyable, Movable):
    """What every worker's table sink shares: the table's layout, the
    group columns of the sink's input, and each reduction input as a
    column of the input (-1 for a row count, which reads the first group
    column, or for a compiled expression) with its type."""

    var layout: AggregateLayout
    var keys: List[Int]
    var key_types: List[DataType]
    var columns: List[Int]
    var expressions: List[Optional[VectorExpression]]
    var types: List[DataType]

    def __init__(out self, *, copy: Self):
        self.layout = copy.layout.copy()
        self.keys = copy.keys.copy()
        self.key_types = copy.key_types.copy()
        self.columns = copy.columns.copy()
        self.expressions = copy.expressions.copy()
        self.types = copy.types.copy()

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
        for name in keys:
            var at = empty._index(name)
            self.keys.append(at)
            self.key_types.append(empty._columns[at].dtype())
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
        for i in range(len(given)):
            self.types.append(inputs[i].dtype())
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


struct TableSinkInputs(Movable):
    """A worker's reader of a chunk's group chunk and aggregate input
    chunk (`PopulateGroupChunk` and the inputs `Sink` evaluates): its own
    copy of the compiled input expressions and the chunk they write."""

    var keys: List[Int]
    var key_types: List[DataType]
    var columns: List[Int]
    var expressions: List[Optional[VectorExpression]]
    var types: List[DataType]
    var output: DataChunk

    def __init__(out self, shared: TableSinkLayout):
        self.keys = shared.keys.copy()
        self.key_types = shared.key_types.copy()
        self.columns = shared.columns.copy()
        self.expressions = shared.expressions.copy()
        self.types = shared.types.copy()
        self.output = DataChunk(len(shared.expressions))

    def read(
        mut self, mut chunk: DataChunk, mut arena: ColumnArena
    ) raises -> Tuple[List[Series], List[Series]]:
        """The chunk's group columns (as canonical labels) and each
        reduction's input over its rows: a column read is shared by every
        input and key reading it."""
        var n = chunk.count
        var read = List[Int]()
        var made = List[Series]()
        var keys = List[Series](capacity=len(self.keys))
        for k in range(len(self.keys)):
            var column = _one_buffer(
                to_series(chunk, self.keys[k], arena, "", self.key_types[k])
            )
            read.append(self.keys[k])
            made.append(column.copy())
            keys.append(canonical_labels(column))
        var computed = False
        var inputs = List[Series](capacity=len(self.columns))
        for i in range(len(self.columns)):
            var at = self.columns[i]
            if at >= 0:
                var found = -1
                for j in range(len(read)):
                    if read[j] == at:
                        found = j
                if found >= 0 and made[found].dtype() == self.types[i]:
                    inputs.append(made[found].copy())
                    continue
                var column = _one_buffer(
                    to_series(chunk, at, arena, "", self.types[i])
                )
                read.append(at)
                made.append(column.copy())
                inputs.append(column^)
            elif self.expressions[i]:
                if not computed:
                    self.output.reset()
                    self.output.count = n
                    computed = True
                self.expressions[i].value().write(chunk, arena, self.output, i)
                inputs.append(
                    _one_buffer(
                        to_series(self.output, i, arena, "", self.types[i])
                    )
                )
            else:
                # A row count reads the first group column.
                inputs.append(keys[0].copy())
        return (keys^, inputs^)

    def release(mut self):
        """Only sinking reads the expressions."""
        self.expressions = List[Optional[VectorExpression]]()
        self.output = DataChunk()


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

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk into the table."""
        var n = chunk.count
        if n == 0:
            return
        var read = self.inputs.read(chunk, arena)
        self.table.sink(read[0], read[1], self.sequence)
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

    def __init__(out self, shared: GroupingSetsLayout, workers: Int):
        self.shared = shared.copy()
        self.tables = List[AggregateSink](capacity=len(shared.sets))
        for s in range(len(shared.sets)):
            self.tables.append(
                AggregateSink(shared.layouts[s], workers, len(shared.sets))
            )
        self.inputs = TableSinkInputs(shared.inputs)
        self.sequence = 0

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk into every set's table that is not derived, each taking
        the key columns it groups by (the grand total a constant one)."""
        var n = chunk.count
        if n == 0:
            return
        var read = self.inputs.read(chunk, arena)
        ref all_keys = read[0]
        for s in range(len(self.shared.sets)):
            if self.shared.derived_from[s] >= 0:
                continue
            ref one = self.shared.sets[s]
            var keys = List[Series](capacity=max(1, len(one)))
            if len(one) == 0:
                keys.append(
                    Series(
                        TOTAL_KEY,
                        Column[Int64](List[Int64](length=n, fill=0)),
                    )
                )
            else:
                for j in range(len(one)):
                    var mask = self.shared.set_masks[s][j] if len(
                        self.shared.set_masks
                    ) > 0 else Int64(-1)
                    if mask == -1:
                        keys.append(all_keys[one[j]].copy())
                    else:
                        keys.append(masked_key(all_keys[one[j]], mask))
            self.tables[s].sink(keys, read[1], self.sequence)
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


def masked_key(key: Series, mask: Int64) raises -> Series:
    """An Int64 key column's values with only `mask`'s bits kept, its
    validity unchanged."""
    ref typed = key._data[Column[Int64]]
    var n = len(typed)
    var values = List[Int64](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var source = typed._data[].unsafe_ptr().unsafe_offset(typed._offset)
    for i in range(n):
        out[unsafe_offset=i] = source[unsafe_offset=i] & mask
    return Series(
        key.name(),
        Column[Int64](
            values=values^, bits=_copy_validity(typed._bits[], typed._offset, n)
        ),
    )


def _one_buffer(var column: Series) raises -> Series:
    """A chunk's column as one buffer: a window of one source chunk is
    that chunk's view."""
    if not column.is_chunked():
        return column^
    if column.n_chunks() == 1:
        return column.chunks()[0].copy()
    return column.rechunk()
