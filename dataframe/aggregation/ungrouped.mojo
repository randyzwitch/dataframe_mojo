"""DuckDB's ungrouped aggregate (`PhysicalUngroupedAggregate`) on the
aggregate table's states (`function.mojo`).

Each worker holds one row of `AggregateFunction` states. Every data chunk
a pipeline's steps produce is sunk into it directly (`Sink`): each
reduction's input is computed over the chunk by its compiled vector
expression (the `ExpressionExecutor` over DuckDB's payload chunk) and
folded into its state a vector at a time through the unified vector format
(`simple_update`), with no frame or column built in between. The workers'
rows are combined at the end (`Combine`) and finalized once (`Finalize`).

A distinct count (`n_unique`) is DuckDB's `DistinctAggregateData`: each
worker sinks every chunk's input vector, as a group key, into its own
aggregate table of the input's distinct values (one radix table per
distinct aggregate, `SinkDistinct`); at the end every worker's table of an
aggregate is merged into its distinct values (`FinalizeDistinct`), whose
number is the count. `distinct_count` is the same over a column.
"""
from std.math import isnan
from std.sys import size_of
from std.memory import Pointer, bitcast

from dataframe.aggregation.function import (
    AggregateFunction,
    _BATCH,
    _Finalizer,
    _Groups,
    _HeldBuffers,
    _K_COMOMENT,
    _K_COUNT,
    _K_DECIMAL_SUM,
    _K_FLOAT_SUM,
    _K_LEN,
    _K_MAX,
    _K_MIN,
    _K_NARROW_SUM,
    _K_NULLS,
    _K_INT_SUM,
    _K_DISTINCT,
    ReducerSpec,
    _ValueArena,
    _update_states,
    combine as combine_states,
    reduction_supported,
    states_supported,
)
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    KeyVector,
    aggregate_finish,
    aggregate_layout,
)
from dataframe.aggregation.table_sink import _sink_jobs, input_vector
from dataframe.parallel import configured_workers
from dataframe.aggregation.expressions import _stream_reductions
from dataframe.aggregation.reducer import Reducer
from dataframe.aggregation.selected_rows import _canonical_int
from dataframe.aggregation.state import (
    _StreamReduction,
    table_specs,
)
from dataframe.decimal import check_limit, precision_limit
from dataframe.expr import (
    COL,
    CORR,
    COUNT,
    COV,
    Expr,
    LEN,
    MAX,
    MEAN,
    MIN,
    SUM,
    WHEN,
    is_reduction,
    subtree,
)
from dataframe.dtype import DataType
from dataframe.execution import _empty
from dataframe.frame import DataFrame
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    _NUMERIC_STORAGE,
    storage_physical,
)
from dataframe.vector.expression import VectorExpression
from dataframe.vector.sink import Sink
from dataframe.vector.types import (
    SelectionVector,
    UnifiedVectorFormat,
    Vector,
    physical_of,
)


struct UngroupedLayout(Copyable, Movable):
    """What every worker's sink shares: the states' layout, each state's
    input expression compiled for the sink's input columns (none for a
    row count), the inputs' types, and the reductions."""

    var layout: AggregateLayout
    var expressions: List[Optional[VectorExpression]]
    # Per input, an aggregate filter (DuckDB's `FILTER (WHERE ...)`): an
    # input `when(c).then(x)` with no otherwise, into a reduction that
    # skips nulls, is `x` over the rows where `c` holds, so `expressions`
    # computes `x` and the filter selects the rows.
    var filters: List[Optional[VectorExpression]]
    # Per input and per filter, the sink input column it is when it is a
    # bare column (read in place, nothing compiled), else -1.
    var columns: List[Int]
    var filter_columns: List[Int]
    var types: List[Series]
    var reductions: List[Expr]
    var schema: DataFrame
    # The template's result names, types and output expressions.
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]

    def __init__(out self, *, copy: Self):
        self.layout = copy.layout.copy()
        self.expressions = copy.expressions.copy()
        self.filters = copy.filters.copy()
        self.columns = copy.columns.copy()
        self.filter_columns = copy.filter_columns.copy()
        self.types = copy.types.copy()
        self.reductions = copy.reductions.copy()
        self.schema = copy.schema.copy()
        self.names = copy.names.copy()
        self.dtypes = copy.dtypes.copy()
        self.outputs = copy.outputs.copy()

    def __init__(
        out self,
        var layout: AggregateLayout,
        var expressions: List[Optional[VectorExpression]],
        var filters: List[Optional[VectorExpression]],
        var columns: List[Int],
        var filter_columns: List[Int],
        var types: List[Series],
        reductions: List[Expr],
        var schema: DataFrame,
        template: _StreamReduction,
    ):
        self.layout = layout^
        self.expressions = expressions^
        self.filters = filters^
        self.columns = columns^
        self.filter_columns = filter_columns^
        self.types = types^
        self.reductions = reductions.copy()
        self.schema = schema^
        self.names = template.names.copy()
        self.dtypes = template.dtypes.copy()
        self.outputs = template.outputs.copy()

    def template(self) raises -> _StreamReduction:
        """The reductions over no rows: names, result types, outputs."""
        return _StreamReduction(self.schema, self.reductions, List[String]())


def ungrouped_layout(
    empty: DataFrame, expressions: List[Expr]
) raises -> Optional[UngroupedLayout]:
    """The ungrouped aggregate over a sink whose input is typed as `empty`
    (no rows), or None when some reduction has no inline state: a
    positional or holistic reduction the table does not hold. A distinct
    count's state is the count its distinct table finishes with."""
    if len(expressions) == 0 or not _stream_reductions(expressions, True):
        return None
    var inputs = List[Optional[Expr]]()
    var seconds = List[Optional[Expr]]()
    var ops = List[Int]()
    for expression in expressions:
        for node in expression._nodes:
            if not is_reduction(node.op):
                continue
            if not reduction_supported(node.op):
                return None
            ops.append(node.op)
            if node.left < 0:
                if node.op != LEN:
                    return None
                inputs.append(None)
            else:
                inputs.append(subtree(expression, node.left))
            if node.op == CORR or node.op == COV:
                seconds.append(subtree(expression, node.right))
    for second in seconds:
        inputs.append(second.copy())
    var template = _StreamReduction(empty, expressions, List[String]())
    var specs = table_specs(template)
    # Each input as a bare column of the sink's input, or compiled; an
    # aggregate filter's condition likewise. Its type is the column's, or
    # what the compiled expression computes.
    var compiled = List[Optional[VectorExpression]](capacity=len(inputs))
    var filters = List[Optional[VectorExpression]](capacity=len(inputs))
    var columns = List[Int](capacity=len(inputs))
    var filter_columns = List[Int](capacity=len(inputs))
    var typed = List[Series](capacity=len(inputs))
    for i in range(len(inputs)):
        compiled.append(None)
        filters.append(None)
        columns.append(-1)
        filter_columns.append(-1)
        if not inputs[i]:
            # A row count reads no input; an Int64 column stands in.
            typed.append(Series("__rows", Column[Int64](List[Int64]())))
            continue
        var value = inputs[i].value().copy()
        ref root = value._nodes[len(value._nodes) - 1]
        if (
            i < len(ops)
            and root.op == WHEN
            and root.extra < 0
            and ops[i] in [SUM, MEAN, COUNT, MIN, MAX]
        ):
            var condition = subtree(value, root.left)
            var at = _bare_column(condition, empty)
            if at >= 0:
                filter_columns[i] = at
            else:
                filters[i] = VectorExpression(condition, empty._columns)
            value = subtree(value, root.right)
        var at = _bare_column(value, empty)
        if at >= 0:
            columns[i] = at
            typed.append(empty._columns[at].copy())
        else:
            var one = VectorExpression(value, empty._columns)
            typed.append(_empty(one.root_dtype()))
            compiled[i] = one^
    if not states_supported(specs, typed):
        return None
    var layout = aggregate_layout(List[Series](), typed, specs)
    var simple: List[Int] = [
        _K_COUNT,
        _K_INT_SUM,
        _K_NARROW_SUM,
        _K_FLOAT_SUM,
        _K_DECIMAL_SUM,
        _K_MIN,
        _K_MAX,
    ]
    for i in range(len(inputs)):
        var filtered = filter_columns[i] >= 0 or Bool(filters[i])
        if filtered and layout.slots[i].kind not in simple:
            # Only the simple updates take a filter's selection: this
            # input is computed whole.
            return _unfiltered(empty, expressions, inputs, ops)
    return UngroupedLayout(
        layout^,
        compiled^,
        filters^,
        columns^,
        filter_columns^,
        typed^,
        expressions,
        empty.copy(),
        template,
    )


def _bare_column(expression: Expr, empty: DataFrame) -> Int:
    """The input column a bare column expression reads, else -1."""
    if len(expression._nodes) != 1 or expression._nodes[0].op != COL:
        return -1
    var name = expression._nodes[0].text
    for c in range(len(empty._columns)):
        if empty._columns[c].name() == name:
            return c
    return -1


def _unfiltered(
    empty: DataFrame,
    expressions: List[Expr],
    inputs: List[Optional[Expr]],
    ops: List[Int],
) raises -> Optional[UngroupedLayout]:
    """The layout with every input computed whole, no aggregate filters."""
    var template = _StreamReduction(empty, expressions, List[String]())
    var specs = table_specs(template)
    var compiled = List[Optional[VectorExpression]](capacity=len(inputs))
    var filters = List[Optional[VectorExpression]](capacity=len(inputs))
    var columns = List[Int](capacity=len(inputs))
    var filter_columns = List[Int](capacity=len(inputs))
    var typed = List[Series](capacity=len(inputs))
    for i in range(len(inputs)):
        compiled.append(None)
        filters.append(None)
        columns.append(-1)
        filter_columns.append(-1)
        if not inputs[i]:
            typed.append(Series("__rows", Column[Int64](List[Int64]())))
            continue
        var at = _bare_column(inputs[i].value(), empty)
        if at >= 0:
            columns[i] = at
            typed.append(empty._columns[at].copy())
        else:
            var one = VectorExpression(inputs[i].value(), empty._columns)
            typed.append(_empty(one.root_dtype()))
            compiled[i] = one^
    if not states_supported(specs, typed):
        return None
    var layout = aggregate_layout(List[Series](), typed, specs)
    return UngroupedLayout(
        layout^,
        compiled^,
        filters^,
        columns^,
        filter_columns^,
        typed^,
        expressions,
        empty.copy(),
        template,
    )


@always_inline
def _words(address: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)


struct UngroupedSink(Sink):
    """One worker's states of an ungrouped aggregate, and its own copy of
    the compiled input expressions (DuckDB's thread-local
    `UngroupedAggregateLocalSinkState`); the global state is the first
    worker's, which every other one combines into."""

    var slots: List[AggregateFunction]
    var groups: _Groups
    var expressions: List[Optional[VectorExpression]]
    var filters: List[Optional[VectorExpression]]
    var columns: List[Int]
    var filter_columns: List[Int]
    var types: List[Series]
    var output: DataChunk
    # A filter's kept rows, and those rows through the input's selection.
    var kept: List[UInt32]
    var through: List[UInt32]
    var arena: _ValueArena
    # Every row of a batch updates the one state row.
    var rows: List[Int]
    var scratch: List[UInt64]
    var discard: List[UInt64]
    # String inputs a value state points into, held until the finish, and
    # the arena columns of this morsel held already.
    var held: List[Series]
    var held_handles: List[Int]
    # Rows sunk, and the next row's number (from the morsel's, `begin`).
    var count: Int
    var sequence: Int
    # Per distinct count, this worker's table of its input's distinct
    # values (DuckDB's `radix_tables` of `DistinctAggregateData`), in
    # `AggregateLayout.distinct` order; the global state holds every
    # worker's after it, worker by worker (`combine`). Per state, its
    # distinct count's index there, or -1.
    var distinct: List[AggregateSink]
    var distinct_of: List[Int]

    def __init__(out self, shared: UngroupedLayout):
        self.slots = shared.layout.slots.copy()
        self.groups = _Groups(0, List[Bool](), shared.layout.state_words, 1)
        var at = _words(self.groups.at(0))
        for w in range(self.groups.stride):
            at[unsafe_offset=w] = 0
        self.groups.n = 1
        self.expressions = shared.expressions.copy()
        self.filters = shared.filters.copy()
        self.columns = shared.columns.copy()
        self.filter_columns = shared.filter_columns.copy()
        self.types = shared.types.copy()
        self.kept = List[UInt32](length=_BATCH, fill=0)
        self.through = List[UInt32](length=_BATCH, fill=0)
        self.output = DataChunk(len(shared.expressions))
        self.arena = _ValueArena()
        self.rows = List[Int](length=_BATCH, fill=self.groups.at(0))
        # Made by the first table update that needs it.
        self.scratch = List[UInt64]()
        self.discard = List[UInt64](length=self.groups.stride + 8, fill=0)
        self.held = List[Series]()
        self.held_handles = List[Int]()
        self.count = 0
        self.sequence = 0
        self.distinct = List[AggregateSink]()
        self.distinct_of = List[Int](length=len(self.slots), fill=-1)
        var workers = configured_workers()
        for d in range(len(shared.layout.distinct)):
            self.distinct_of[shared.layout.distinct[d]] = d
            self.distinct.append(
                AggregateSink(shared.layout.distinct_layout(d), workers)
            )

    def counts_only(self) -> Bool:
        """Whether every state is a row count, which reads no column."""
        for slot in self.slots:
            if slot.kind != _K_LEN:
                return False
        return True

    def sink_rows(mut self, rows: Int):
        """`rows` rows into row counts only (`counts_only`)."""
        var state = self.groups.at(0) + 8 * self.groups.states_at
        for slot in self.slots:
            _words(state + 8 * slot.offset)[] += UInt64(rows)
        self.count += rows
        self.sequence += rows

    def begin(mut self, morsel: Int):
        """Rows from here on are the morsel's: numbered from `morsel << 32`
        in its output order, so first and last are the source's whichever
        worker took which morsel."""
        self.sequence = morsel << 32
        self.held_handles.clear()

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """A chunk of the sink's input into the states (DuckDB's `Sink`):
        each input computed over the chunk's rows, then each state updated
        from its vector."""
        var n = chunk.count
        if n == 0:
            return
        self.output.reset()
        self.output.count = n
        for i in range(len(self.expressions)):
            if self.expressions[i]:
                self.expressions[i].value().write(chunk, arena, self.output, i)
        var state = self.groups.at(0) + 8 * self.groups.states_at
        for s in range(len(self.slots)):
            var kind = self.slots[s].kind
            var offset = self.slots[s].offset
            if kind == _K_LEN:
                _words(state + 8 * offset)[] += UInt64(n)
                continue
            var format = self._format(s, chunk, arena)
            if kind == _K_DISTINCT:
                # The input's vector, as a group key, into the distinct
                # table (`SinkDistinct`); its strings are held with this
                # sink's (`_format`).
                self.distinct[self.distinct_of[s]].sink_vectors(
                    [KeyVector.of(format)],
                    List[UnifiedVectorFormat](),
                    List[Series](),
                    self.sequence,
                    n,
                )
                continue
            if self.filter_columns[s] >= 0 or self.filters[s]:
                self._filtered(s, chunk, arena, format, state + 8 * offset)
                continue
            if _simple_update(self.slots[s], format, n, state + 8 * offset):
                continue
            self._update(s, chunk, arena, format)
        self.count += n
        self.sequence += n

    def _format(
        mut self, i: Int, mut chunk: DataChunk, mut arena: ColumnArena
    ) raises -> UnifiedVectorFormat:
        """Input i's vector: the chunk's column, or what its expression
        wrote (`input_vector`). The column a string value state points
        into is held until the finish."""
        var physical = storage_physical(self.types[i])
        var dtype = self.types[i].dtype()
        var c = self.columns[i]
        var v: Vector
        if c >= 0:
            v = input_vector(chunk, c, arena, physical, dtype)
        else:
            v = input_vector(self.output, i, arena, physical, dtype)
        if i < len(self.slots) and (
            self.slots[i].holds_strings() or self.slots[i].kind == _K_DISTINCT
        ):
            var known = False
            for h in self.held_handles:
                known = known or h == v.source
            if not known:
                self.held_handles.append(v.source)
                self.held.append(arena.columns[v.source].copy())
        if c >= 0:
            return chunk.unified(c, arena)
        return self.output.unified(i, arena)

    def _scratch(mut self) -> Int:
        """The address of the table update's scratch vectors (values,
        validity and a second input's, `_update_states`)."""
        if len(self.scratch) == 0:
            self.scratch = List[UInt64](length=18 * _BATCH, fill=0)
        return Int(self.scratch.unsafe_ptr())

    def _filtered(
        mut self,
        s: Int,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        format: UnifiedVectorFormat,
        state: Int,
    ) raises:
        """State s updated from its input's rows its filter keeps
        (DuckDB's `AggregateFilterData::ApplyFilter` before the update)."""
        var through = self.through.unsafe_ptr()
        var kept = 0
        if self.filter_columns[s] >= 0:
            var condition = chunk.unified(self.filter_columns[s], arena)
            if (
                not condition.sel.is_set()
                and condition.validity.all_valid()
                and _flat_update(
                    self.slots[s], format, condition.data, chunk.count, state
                )
            ):
                return
            # A Boolean column: its vector read in place, each row kept
            # when true and valid, no lanes loaded.
            var flags = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=condition.data
            )
            var whole = condition.validity.all_valid()
            for k in range(chunk.count):
                var r = condition.sel.get_index(k)
                var keep = flags[unsafe_offset=r] != 0
                if not whole:
                    keep = keep and condition.validity.row_is_valid_unsafe(r)
                through[unsafe_offset=kept] = UInt32(format.sel.get_index(k))
                kept += Int(keep)
        else:
            kept = (
                self.filters[s]
                .value()
                .select(
                    chunk,
                    arena,
                    SelectionVector(),
                    chunk.count,
                    SelectionVector(Int(self.kept.unsafe_ptr())),
                )
            )
            var rows = self.kept.unsafe_ptr()
            for k in range(kept):
                through[unsafe_offset=k] = UInt32(
                    format.sel.get_index(Int(rows[unsafe_offset=k]))
                )
        if kept == 0:
            return
        var selected = UnifiedVectorFormat(
            SelectionVector(Int(through)),
            format.data,
            format.validity,
            format.physical,
        )
        if _simple_update(self.slots[s], selected, kept, state):
            return
        # A state the simple updates do not keep: the kept rows' vector
        # through the table's update.
        var one = List[AggregateFunction]()
        one.append(self.slots[s].copy())
        var lone = List[UnifiedVectorFormat]()
        lone.append(selected)
        _update_states(
            one,
            lone,
            0,
            kept,
            Pointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(self.rows.unsafe_ptr())
            ),
            self.groups.states_at,
            self.sequence,
            Pointer[_ValueArena, MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=self.arena))
            ),
            Int(self.discard.unsafe_ptr()),
            self._scratch(),
        )

    def _update(
        mut self,
        s: Int,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        format: UnifiedVectorFormat,
    ) raises:
        """State s through the aggregate table's update from its input's
        vector `format` (and a two-input reduction's second), every row to
        the one state row."""
        var n = chunk.count
        var slot = self.slots[s].copy()
        # The slot alone, its input first and a pair's second after it.
        var lone = List[UnifiedVectorFormat]()
        lone.append(format)
        if slot.kind == _K_COMOMENT:
            lone.append(self._format(slot.k, chunk, arena))
            slot.k = 1
        var one = List[AggregateFunction]()
        one.append(slot^)
        for lo in range(0, n, _BATCH):
            _update_states(
                one,
                lone,
                lo,
                min(_BATCH, n - lo),
                Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(self.rows.unsafe_ptr())
                ),
                self.groups.states_at,
                self.sequence,
                Pointer[_ValueArena, MutAnyOrigin](
                    unsafe_from_address=Int(Pointer(to=self.arena))
                ),
                Int(self.discard.unsafe_ptr()),
                self._scratch(),
            )

    def release(mut self):
        """Free everything but the states and the inputs they point into,
        on the worker's own thread when its run ends."""
        self.expressions = List[Optional[VectorExpression]]()
        self.filters = List[Optional[VectorExpression]]()
        self.columns = List[Int]()
        self.filter_columns = List[Int]()
        self.output = DataChunk()
        self.scratch = List[UInt64]()
        self.kept = List[UInt32]()
        self.through = List[UInt32]()
        self.rows = List[Int]()

    def combine_into(self, mut other: UngroupedSink) raises:
        """These states folded into `other`'s (DuckDB's `Combine`)."""
        combine_states(
            self.slots,
            other.groups.at(0) + 8 * other.groups.states_at,
            self.groups.at(0) + 8 * self.groups.states_at,
        )
        other.count += self.count
        for column in self.held:
            other.held.append(column.copy())

    def combine(var self, mut combined: List[UngroupedSink]) raises:
        """These states into the global ones (DuckDB's `Combine`): the
        first worker's states are the global states, which take this
        worker's distinct tables too."""
        if len(combined) == 0:
            combined.append(self^)
            return
        self.combine_into(combined[0])
        while len(self.distinct) > 0:
            combined[0].distinct.append(self.distinct.pop(0))

    def finalize_distinct(mut self, layout: AggregateLayout) raises:
        """DuckDB's `FinalizeDistinct` for the global state: each distinct
        count's tables (every worker's) merged into the input's distinct
        values, whose number (less a null's, when nulls are ignored) is
        written into the count's state."""
        var count = len(layout.distinct)
        var state = self.groups.at(0) + 8 * self.groups.states_at
        for d in range(count):
            # Every worker's tables are in worker order, `count - d` of
            # them each: this count's is each worker's first.
            var tables = List[AggregateSink]()
            var rest = List[AggregateSink]()
            var i = 0
            while len(self.distinct) > 0:
                if i % (count - d) == 0:
                    tables.append(self.distinct.pop(0))
                else:
                    rest.append(self.distinct.pop(0))
                i += 1
            swap(self.distinct, rest)
            for column in self.held:
                tables[0].held.append(column.copy())
            var slot = layout.distinct[d]
            var values = distinct_values(
                tables,
                layout.distinct_layout(d),
                layout.templates[slot].integer != 0,
            )
            _words(state + 8 * self.slots[slot].offset)[] = UInt64(values)
            _ = tables^


def distinct_values(
    mut tables: List[AggregateSink],
    layout: AggregateLayout,
    ignore_nulls: Bool,
) raises -> Int:
    """The distinct values every worker's distinct table (`layout`, the
    value its one key) holds together: their groups merged, a null value
    left out when `ignore_nulls`."""
    var groups = aggregate_finish(
        tables, layout, rows_only=True, ordered=False
    )
    var values = 0
    for p in range(len(groups.rows)):
        ref part = groups.rows[p]
        values += part.count()
        if ignore_nulls:
            for g in range(part.count()):
                if part.null(g) != 0:
                    values -= 1
    return values


# Rows a worker takes at a time: DuckDB's DEFAULT_ROW_GROUP_SIZE, the unit
# its table scans hand to threads.
comptime _MORSEL_ROWS = 122_880


def distinct_count(values: Series, ignore_nulls: Bool) raises -> Int:
    """The distinct values of a column (`n_unique` over no groups): every
    row sunk, a morsel per worker at a time, into the worker's aggregate
    table of the values (the ungrouped aggregate's distinct table), whose
    groups are then merged (`distinct_values`). Every NaN is one value and
    -0.0 equals 0.0, as the table's key words make them; a null is one
    value unless `ignore_nulls`. The column's type must be one the table
    groups by (`key_supported`)."""
    var keys: List[Series] = [values.copy()]
    var layout = aggregate_layout(keys, List[Series](), List[ReducerSpec]())
    var ranges = List[Int]()
    var start = 0
    for part in values.chunks():
        var at = 0
        var length = len(part)
        while at < length:
            ranges.append(start + at)
            ranges.append(min(_MORSEL_ROWS, length - at))
            at += _MORSEL_ROWS
        start += length
    var workers = max(1, min(configured_workers(), len(ranges) // 2))
    var tables = _sink_jobs(keys, List[Series](), layout, ranges, workers)
    return distinct_values(tables, layout, ignore_nulls)


def ungrouped_finish(
    shared: UngroupedLayout, mut sinks: List[UngroupedSink]
) raises -> DataFrame:
    """Every worker's states combined and finalized into one row (DuckDB's
    `Combine` and `Finalize`)."""
    var combined = List[UngroupedSink]()
    while len(sinks) > 0:
        sinks.pop(0).combine(combined)
    if len(combined) == 0:
        return shared.template().finish()
    sinks = combined^
    sinks[0].finalize_distinct(shared.layout)
    if sinks[0].count == 0:
        # No row survived: the reductions of an empty frame.
        return shared.template().finish()
    var all_held = List[List[Series]]()
    all_held.append(sinks[0].held.copy())
    var held = _HeldBuffers(all_held)
    var states = List[Reducer](capacity=len(sinks[0].slots))
    for s in range(len(sinks[0].slots)):
        var one = _Finalizer(
            sinks[0].slots[s], shared.layout.templates[s], 1, held
        )
        one.scan(sinks[0].groups, 0, 1, held)
        states.append(one^.reducer())
    var reduction = _StreamReduction(
        keys=DataFrame(List[Series](), height=1),
        states=states^,
        names=shared.names,
        dtypes=shared.dtypes,
        outputs=shared.outputs,
        grouped=False,
        firsts=[0],
        rows=sinks[0].count,
    )
    var result = reduction.finish()
    _ = held^
    return result^


def _flat_update(
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    flags: Int,
    count: Int,
    state: Int,
) -> Bool:
    """A flat vector with every row valid folded into a count or sum
    state a SIMD block at a time, each row kept where its byte at `flags`
    is set (an aggregate filter's flat Boolean vector; 0 keeps every row).
    False for any other vector or state. Integer sums take inputs of at
    most 32 bits, whose 2048-row totals fit 64-bit lanes."""
    if format.sel.is_set() or not format.validity.all_valid():
        return False
    var kind = slot.kind
    var at = _words(state)
    if kind == _K_COUNT:
        if flags == 0:
            at[] += UInt64(count)
        else:
            at[] += UInt64(_kept_rows(flags, count))
        return True
    if kind != _K_FLOAT_SUM and kind != _K_INT_SUM and kind != _K_NARROW_SUM:
        return False
    comptime for k in range(len(_NUMERIC_STORAGE)):
        comptime D = _NUMERIC_STORAGE[k]
        comptime if D.is_floating_point() or (
            D.is_integral() and size_of[Scalar[D]]() <= 4
        ):
            if format.physical == physical_of[D]():
                var seen = count if flags == 0 else _kept_rows(flags, count)
                comptime if D.is_floating_point():
                    var total = _block_sum[D, DType.float64](
                        format.data, flags, count
                    )
                    at[] = bitcast[DType.uint64](
                        bitcast[DType.float64](at[]) + total
                    )
                    at[unsafe_offset=1] += UInt64(seen)
                else:
                    var total = _block_sum[D, DType.int64](
                        format.data, flags, count
                    )
                    if kind == _K_NARROW_SUM:
                        at[] += bitcast[DType.uint64](total)
                        at[unsafe_offset=1] += UInt64(seen)
                    elif kind == _K_INT_SUM:
                        var sum = (
                            UInt128(at[unsafe_offset=1]) << 64
                        ) | UInt128(at[])
                        sum += bitcast[DType.uint128](
                            total.cast[DType.int128]()
                        )
                        at[] = UInt64(sum)
                        at[unsafe_offset=1] = UInt64(sum >> 64)
                        at[unsafe_offset=2] += UInt64(seen)
                    else:
                        return False
                return True
    return False


def _kept_rows(flags: Int, count: Int) -> Int:
    """How many of `count` bytes at `flags` are set."""
    comptime W = 32
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags)
    var total = SIMD[DType.uint32, W](0)
    var k = 0
    while k + W <= count:
        total += (
            bytes.unsafe_offset(k)
            .unsafe_load[width=W]()
            .ne(0)
            .cast[DType.uint32]()
        )
        k += W
    var kept = Int(total.reduce_add())
    while k < count:
        kept += Int(bytes[unsafe_offset=k] != 0)
        k += 1
    return kept


def _block_sum[
    D: DType, A: DType
](data: Int, flags: Int, count: Int) -> Scalar[A]:
    """The sum, in type A, of `count` values at `data`, only those whose
    byte at `flags` is set when `flags` is not 0."""
    comptime W = 8
    var values = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=data)
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags)
    var total = SIMD[A, W](0)
    var k = 0
    if flags == 0:
        while k + W <= count:
            total += values.unsafe_offset(k).unsafe_load[width=W]().cast[A]()
            k += W
    else:
        while k + W <= count:
            var x = values.unsafe_offset(k).unsafe_load[width=W]().cast[A]()
            var keep = bytes.unsafe_offset(k).unsafe_load[width=W]().ne(0)
            total += keep.select(x, SIMD[A, W](0))
            k += W
    var result = total.reduce_add()
    while k < count:
        if flags == 0 or bytes[unsafe_offset=k] != 0:
            result += values[unsafe_offset=k].cast[A]()
        k += 1
    return result


def _simple_update(
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    count: Int,
    state: Int,
) raises -> Bool:
    """A vector's `count` rows folded into one state (DuckDB's
    `simple_update`): counts, sums, decimal sums and numeric extremes,
    accumulated in registers and stored once. False for any other state
    or input, which the caller updates through the table's update."""
    if _flat_update(slot, format, 0, count, state):
        return True
    var kind = slot.kind
    var at = _words(state)
    var sel = format.sel
    var validity = format.validity
    if kind == _K_COUNT or kind == _K_NULLS:
        var valid = count
        if not validity.all_valid():
            valid = 0
            for i in range(count):
                if validity.row_is_valid_unsafe(sel.get_index(i)):
                    valid += 1
        at[] += UInt64(valid if kind == _K_COUNT else count - valid)
        return True
    if kind == _K_DECIMAL_SUM:
        comptime for k in range(len(_NUMERIC_STORAGE)):
            comptime D = _NUMERIC_STORAGE[k]
            comptime if D.is_integral() and D.is_signed():
                if format.physical == physical_of[D]():
                    _decimal_update[D](slot, format, count, at)
                    return True
        return False
    if (
        kind == _K_INT_SUM
        or kind == _K_NARROW_SUM
        or kind == _K_FLOAT_SUM
        or kind == _K_MIN
        or kind == _K_MAX
    ):
        comptime for k in range(len(_NUMERIC_STORAGE)):
            comptime D = _NUMERIC_STORAGE[k]
            comptime if D != DType.int128:
                if format.physical == physical_of[D]():
                    _numeric_update[D](slot, format, count, at)
                    return True
    return False


def _decimal_update[
    D: DType
](
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    count: Int,
    at: Pointer[UInt64, MutAnyOrigin],
) raises:
    """Unscaled decimal values into a 128-bit total and count, the total
    held to the result's precision on every add (as `_update_decimal`)."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var all_valid = validity.all_valid()
    var limit = precision_limit(slot.dtype.precision())
    var total = bitcast[DType.int128](
        (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[])
    )
    var seen = 0
    for i in range(count):
        var r = sel.get_index(i)
        if not all_valid and not validity.row_is_valid_unsafe(r):
            continue
        total = check_limit(
            total + values[unsafe_offset=r].cast[DType.int128](),
            limit,
            slot.dtype,
        )
        seen += 1
    var word = bitcast[DType.uint128](total)
    at[] = UInt64(word)
    at[unsafe_offset=1] = UInt64(word >> 64)
    at[unsafe_offset=2] += UInt64(seen)


def _numeric_update[
    D: DType
](
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    count: Int,
    at: Pointer[UInt64, MutAnyOrigin],
):
    """Numeric values into a sum (128-bit, narrow or float) or an extreme
    (as `_update_slot` keeps them: best, then flags 1 seen, 2 NaN seen)."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var all_valid = validity.all_valid()
    var kind = slot.kind
    if kind == _K_MIN or kind == _K_MAX:
        var is_max = kind == _K_MAX
        var flags = at[unsafe_offset=1]
        comptime if D.is_floating_point():
            var best = bitcast[DType.float64](at[])
            for i in range(count):
                var r = sel.get_index(i)
                if not all_valid and not validity.row_is_valid_unsafe(r):
                    continue
                var x = Float64(values[unsafe_offset=r])
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
            for i in range(count):
                var r = sel.get_index(i)
                if not all_valid and not validity.row_is_valid_unsafe(r):
                    continue
                var x = _canonical_int[D](values[unsafe_offset=r])
                if (
                    flags & 1 == 0
                    or (is_max and x > best)
                    or (not is_max and x < best)
                ):
                    best = x
                    flags |= 1
            at[] = bitcast[DType.uint64](best)
        at[unsafe_offset=1] = flags
        return
    var seen = 0
    if kind == _K_FLOAT_SUM:
        var total = bitcast[DType.float64](at[])
        for i in range(count):
            var r = sel.get_index(i)
            if not all_valid and not validity.row_is_valid_unsafe(r):
                continue
            total += Float64(values[unsafe_offset=r])
            seen += 1
        at[] = bitcast[DType.uint64](total)
        at[unsafe_offset=1] += UInt64(seen)
        return
    if kind == _K_NARROW_SUM:
        var total = Int64(0)
        for i in range(count):
            var r = sel.get_index(i)
            if not all_valid and not validity.row_is_valid_unsafe(r):
                continue
            total += values[unsafe_offset=r].cast[DType.int64]()
            seen += 1
        at[] += bitcast[DType.uint64](total)
        at[unsafe_offset=1] += UInt64(seen)
        return
    var total = Int128(0)
    for i in range(count):
        var r = sel.get_index(i)
        if not all_valid and not validity.row_is_valid_unsafe(r):
            continue
        total += values[unsafe_offset=r].cast[DType.int128]()
        seen += 1
    var sum = (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[])
    sum += bitcast[DType.uint128](total)
    at[] = UInt64(sum)
    at[unsafe_offset=1] = UInt64(sum >> 64)
    at[unsafe_offset=2] += UInt64(seen)
