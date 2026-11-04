"""DuckDB's ungrouped aggregate (`PhysicalUngroupedAggregate`) on the
aggregate table's states (`function.mojo`).

Each worker holds one row of `AggregateFunction` states. Every data chunk
a pipeline's steps produce is sunk into it directly (`Sink`): each
reduction's input is computed over the chunk by its compiled vector
expression (the `ExpressionExecutor` over DuckDB's payload chunk) and
folded into its state a vector at a time through the unified vector format
(`simple_update`), with no frame or column built in between. The workers'
rows are combined at the end (`Combine`) and finalized once (`Finalize`).
"""
from std.math import isnan
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
    _ValueArena,
    _update_states,
    combine,
    reduction_supported,
    states_supported,
)
from dataframe.aggregation.hash_table import AggregateLayout, aggregate_layout
from dataframe.aggregation.expressions import _stream_reductions
from dataframe.aggregation.reducer import Reducer
from dataframe.aggregation.selected_rows import _canonical_int
from dataframe.aggregation.state import (
    _StreamReduction,
    table_inputs,
    table_specs,
)
from dataframe.decimal import check_limit, precision_limit
from dataframe.expr import (
    CORR,
    COV,
    Expr,
    LEN,
    N_UNIQUE,
    is_reduction,
    subtree,
)
from dataframe.frame import DataFrame
from dataframe.series import Series, Storage
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    _NUMERIC_STORAGE,
    to_series,
)
from dataframe.vector.expression import VectorExpression
from dataframe.vector.types import UnifiedVectorFormat, physical_of


struct UngroupedLayout(Copyable, Movable):
    """What every worker's sink shares: the states' layout, each state's
    input expression compiled for the sink's input columns (none for a
    row count), the inputs' types, and the reductions."""

    var layout: AggregateLayout
    var expressions: List[Optional[VectorExpression]]
    var types: List[Series]
    var reductions: List[Expr]
    var schema: DataFrame

    def __init__(out self, *, copy: Self):
        self.layout = copy.layout.copy()
        self.expressions = copy.expressions.copy()
        self.types = copy.types.copy()
        self.reductions = copy.reductions.copy()
        self.schema = copy.schema.copy()

    def __init__(
        out self,
        var layout: AggregateLayout,
        var expressions: List[Optional[VectorExpression]],
        var types: List[Series],
        reductions: List[Expr],
        var schema: DataFrame,
    ):
        self.layout = layout^
        self.expressions = expressions^
        self.types = types^
        self.reductions = reductions.copy()
        self.schema = schema^

    def template(self) raises -> _StreamReduction:
        """The reductions over no rows: names, result types, outputs."""
        return _StreamReduction(self.schema, self.reductions, List[String]())


def ungrouped_layout(
    empty: DataFrame, expressions: List[Expr]
) raises -> Optional[UngroupedLayout]:
    """The ungrouped aggregate over a sink whose input is typed as `empty`
    (no rows), or None when some reduction has no inline state: a distinct
    count (its distinct table needs group keys), or a positional or
    holistic reduction the table does not hold."""
    if len(expressions) == 0 or not _stream_reductions(expressions, True):
        return None
    var inputs = List[Optional[Expr]]()
    var seconds = List[Optional[Expr]]()
    for expression in expressions:
        for node in expression._nodes:
            if not is_reduction(node.op):
                continue
            if not reduction_supported(node.op) or node.op == N_UNIQUE:
                return None
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
    # A row count reads no input; an Int64 column stands in for its type.
    var count_key = Series("__rows", Column[Int64](List[Int64]()))
    var typed = table_inputs(empty, [count_key^], expressions)
    if len(typed) != len(inputs) or not states_supported(specs, typed):
        return None
    var layout = aggregate_layout(List[Series](), typed, specs)
    var compiled = List[Optional[VectorExpression]](capacity=len(inputs))
    for input in inputs:
        if input:
            compiled.append(VectorExpression(input.value(), empty._columns))
        else:
            compiled.append(None)
    return UngroupedLayout(
        layout^, compiled^, typed^, expressions, empty.copy()
    )


@always_inline
def _words(address: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)


struct UngroupedSink(Movable):
    """One worker's states of an ungrouped aggregate, and its own copy of
    the compiled input expressions (DuckDB's thread-local
    `UngroupedAggregateLocalSinkState`)."""

    var slots: List[AggregateFunction]
    var groups: _Groups
    var expressions: List[Optional[VectorExpression]]
    var types: List[Series]
    var output: DataChunk
    var arena: _ValueArena
    # Every row of a batch updates the one state row.
    var rows: List[Int]
    var scratch: List[UInt64]
    var discard: List[UInt64]
    # String inputs a value state points into, held until the finish.
    var held: List[Series]
    # Rows sunk, and the next row's number (from the morsel's, `begin`).
    var count: Int
    var sequence: Int

    def __init__(out self, shared: UngroupedLayout):
        self.slots = shared.layout.slots.copy()
        self.groups = _Groups(0, List[Bool](), shared.layout.state_words, 1)
        var at = _words(self.groups.at(0))
        for w in range(self.groups.stride):
            at[unsafe_offset=w] = 0
        self.groups.n = 1
        self.expressions = shared.expressions.copy()
        self.types = shared.types.copy()
        self.output = DataChunk(len(shared.expressions))
        self.arena = _ValueArena()
        self.rows = List[Int](length=_BATCH, fill=self.groups.at(0))
        self.scratch = List[UInt64](length=18 * _BATCH, fill=0)
        self.discard = List[UInt64](length=self.groups.stride + 8, fill=0)
        self.held = List[Series]()
        self.count = 0
        self.sequence = 0

    def begin(mut self, morsel: Int):
        """Rows from here on are the morsel's: numbered from `morsel << 32`
        in its output order, so first and last are the source's whichever
        worker took which morsel."""
        self.sequence = morsel << 32

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
            if _simple_update(
                self.slots[s],
                self.output.unified(s, arena),
                n,
                state + 8 * offset,
            ):
                continue
            self._update(s, n, arena)
        self.count += n
        self.sequence += n

    def _update(mut self, s: Int, n: Int, mut arena: ColumnArena) raises:
        """State s through the aggregate table's update, its input (and a
        two-input reduction's second) as a column, every row to the one
        state row."""
        var slot = self.slots[s].copy()
        var pair = slot.kind == _K_COMOMENT
        # The slot alone, its input first and a pair's second after it.
        var lone = List[Series]()
        lone.append(to_series(self.output, s, arena, "", self.types[s].dtype()))
        if pair:
            lone.append(
                to_series(
                    self.output, slot.k, arena, "", self.types[slot.k].dtype()
                )
            )
            slot.k = 1
        if slot.holds_strings():
            ref strings = lone[0]._data[StringColumn]
            if strings._is_view_storage():
                if strings._view_storage.value()._selects():
                    var flat = lone[0].copy()
                    flat._data = Storage(strings._flat_views())
                    lone[0] = flat^
            self.held.append(lone[0].copy())
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
                Int(self.scratch.unsafe_ptr()),
            )

    def combine_into(self, mut other: UngroupedSink) raises:
        """These states folded into `other`'s (DuckDB's `Combine`)."""
        combine(
            self.slots,
            other.groups.at(0) + 8 * other.groups.states_at,
            self.groups.at(0) + 8 * self.groups.states_at,
        )
        other.count += self.count
        for column in self.held:
            other.held.append(column.copy())


def ungrouped_finish(
    shared: UngroupedLayout, mut sinks: List[UngroupedSink]
) raises -> DataFrame:
    """Every worker's states combined and finalized into one row (DuckDB's
    `Combine` and `Finalize`)."""
    var template = shared.template()
    if len(sinks) == 0:
        return template.finish()
    while len(sinks) > 1:
        var other = sinks.pop()
        other.combine_into(sinks[0])
    if sinks[0].count == 0:
        # No row survived: the reductions of an empty frame.
        return template.finish()
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
        names=template.names,
        dtypes=template.dtypes,
        outputs=template.outputs,
        grouped=False,
        firsts=[0],
        rows=sinks[0].count,
    )
    var result = reduction.finish()
    _ = held^
    return result^


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
