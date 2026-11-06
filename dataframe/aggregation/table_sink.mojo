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
from dataframe.aggregation.hash_table import AggregateLayout, AggregateSink
from dataframe.dtype import DataType
from dataframe.expr import COL, CORR, COV, Expr, is_reduction, subtree
from dataframe.frame import DataFrame
from dataframe.kernels.canonical_labels import canonical_labels
from dataframe.series import Series
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


struct TableSink(Sink):
    """One worker's table sink state (DuckDB's `HashAggregateLocalSinkState`
    over a `RadixHTLocalSinkState`): its table, its own copy of the
    compiled input expressions, the chunk they write, and the next row's
    number."""

    var table: AggregateSink
    var keys: List[Int]
    var key_types: List[DataType]
    var columns: List[Int]
    var expressions: List[Optional[VectorExpression]]
    var types: List[DataType]
    var output: DataChunk
    var sequence: Int

    def __init__(out self, shared: TableSinkLayout, workers: Int):
        self.table = AggregateSink(shared.layout, workers)
        self.keys = shared.keys.copy()
        self.key_types = shared.key_types.copy()
        self.columns = shared.columns.copy()
        self.expressions = shared.expressions.copy()
        self.types = shared.types.copy()
        self.output = DataChunk(len(shared.expressions))
        self.sequence = 0

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk into the table: its group columns and each reduction's
        input over its rows (a column read is shared by every input and key
        reading it)."""
        var n = chunk.count
        if n == 0:
            return
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
        self.table.sink(keys, inputs, self.sequence)
        self.sequence += n

    def release(mut self):
        """Only the table outlives the worker's run."""
        self.expressions = List[Optional[VectorExpression]]()
        self.output = DataChunk()

    def combine(var self, mut combined: List[TableSink]) raises:
        """This worker's table into the global state, every worker's
        partitioned rows (`RadixPartitionedHashTable::Combine`): the finish
        merges them partition by partition."""
        combined.append(self^)

    def take_table(deinit self) -> AggregateSink:
        """The worker's table, for the finish."""
        return self.table^


def _one_buffer(var column: Series) raises -> Series:
    """A chunk's column as one buffer: a window of one source chunk is
    that chunk's view."""
    if not column.is_chunked():
        return column^
    if column.n_chunks() == 1:
        return column.chunks()[0].copy()
    return column.rechunk()
