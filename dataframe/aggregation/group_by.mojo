"""Eager grouping requests, strategy dispatch, and worker jobs."""
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    _Cursor,
    aggregate_finish,
    aggregate_layout,
    ReducerSpec,
    key_supported,
    table_reduction,
    table_states_supported,
)
from dataframe.aggregation.table_sink import (
    TableSinkInputs,
    TableSinkLayout,
    aggregate_table,
    sink_frame,
)
from dataframe.aggregation.expressions import (
    count_rows_by,
    _bind_all,
    _stream_reductions,
)
from dataframe.aggregation.keys import (
    _key_prefix,
    _pack_struct_keys,
)
from dataframe.aggregation.state import (
    _StreamReduction,
    _merge_ascending_runs,
    table_frame,
    table_inputs,
    table_specs,
)
from dataframe.binding import (
    AGGREGATE,
    BoundExpr,
    bind,
)
from dataframe.kernels.canonical_labels import (
    canonical_labels,
)
from dataframe.categorical import (
    decode,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.execution import (
    evaluate,
)
from dataframe.expr import (
    COL,
    Expr,
    SELECTOR,
    group_sequence_rows,
    col,
    is_reduction,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.kernels.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    take_sorted_chunked,
)
from dataframe.kernels.hashing import (
    RowKeys,
    encode_rows,
    encode_rows_parallel,
    encode_string_rows_parallel,
)
from dataframe.aggregation.selected_rows import (
    indexed_reductions,
    reduce_indexed,
    reduce_indexed_batches,
)
from dataframe.lazy import (
    LazyGroupingSets,
)
from dataframe.parallel import (
    Job,
    configured_workers,
    morsel_ranges,
    morsel_size,
    partitions,
    run_jobs,
    worker_count,
)
from dataframe.kernels.partition import (
    Partitioner,
    encode_bucket,
)
from dataframe.series import (
    Series,
)
from dataframe.storage.string import (
    StringColumn,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Dict,
    Optional,
)


struct _EncodeJob(Job):
    """Encode one row range of a heavy bucket with a private dictionary."""

    var keys: List[Series]
    var ids: List[Int]
    var representatives: List[Int]

    def __init__(out self, var keys: List[Series]):
        self.keys = keys^
        self.ids = List[Int]()
        self.representatives = List[Int]()

    def run(mut self) raises:
        var local = encode_rows(self.keys, nulls_equal=True)
        self.ids = local.ids.copy()
        self.representatives = local.representatives.copy()


def _encode_parallel(keys: List[Series], workers: Int) raises -> RowKeys:
    """Dense ids in first-occurrence order, encoded per row range and then
    reconciled through the representative rows.

    Only worth it when distinct keys are few relative to rows, which is
    what makes a bucket heavy: the reconciliation re-encodes one row per
    (worker, distinct key), so its cost is workers x distinct keys.
    """
    var n = len(keys[0])
    var bounds = partitions(n, workers, 64)
    var jobs = List[_EncodeJob](capacity=workers)
    for w in range(workers):
        var slices = List[Series](capacity=len(keys))
        for key in keys:
            slices.append(key.slice(bounds[w], bounds[w + 1] - bounds[w]))
        jobs.append(_EncodeJob(slices^))
    run_jobs(jobs)

    # One row per (worker, local id), in worker order, so encoding those
    # rows yields each local id's global id at a known position.
    var rows = List[Int]()
    var offsets = List[Int](capacity=workers + 1)
    for w in range(workers):
        offsets.append(len(rows))
        for r in jobs[w].representatives:
            rows.append(bounds[w] + r)
    offsets.append(len(rows))
    var samples = List[Series](capacity=len(keys))
    for key in keys:
        samples.append(key.take(rows))
    var merged = encode_rows(samples, nulls_equal=True)

    var ids = List[Int](length=n, fill=0)
    for w in range(workers):
        var base = offsets[w]
        var start = bounds[w]
        for i in range(len(jobs[w].ids)):
            ids[start + i] = merged.ids[base + jobs[w].ids[i]]
    var representatives = List[Int](capacity=merged.count())
    for r in merged.representatives:
        representatives.append(rows[r])
    return RowKeys(ids^, representatives^)


def _plain_inputs(expressions: List[Expr], frame: DataFrame) -> Bool:
    """Whether every reduction reads a column of `frame` as it is (or no
    input), so nothing is evaluated before its states update."""
    for expression in expressions:
        ref nodes = expression._nodes
        for node in nodes:
            if not is_reduction(node.op):
                continue
            for child in [node.left, node.right]:
                if child < 0:
                    continue
                if nodes[child].op != COL or (
                    nodes[child].text not in frame.columns()
                ):
                    return False
    return True


struct _BatchSinkJob(Job):
    """One worker's morsels into its aggregate table sink, a data chunk of
    at most `batch` rows at a time, each chunk's reduction inputs computed
    by their compiled vector expressions."""

    var columns: List[Series]
    var layout: AggregateLayout
    var shared: TableSinkLayout
    var ranges: List[Int]
    var cursor: _Cursor
    var workers: Int
    var batch: Int
    var sinks: List[AggregateSink]
    var failed_morsel: Int
    var failure: String

    def __init__(
        out self,
        columns: List[Series],
        layout: AggregateLayout,
        shared: TableSinkLayout,
        ranges: List[Int],
        cursor: _Cursor,
        workers: Int,
        batch: Int,
    ):
        self.columns = columns.copy()
        self.layout = layout.copy()
        self.shared = shared.copy()
        self.ranges = ranges.copy()
        self.cursor = cursor.copy()
        self.workers = workers
        self.batch = max(1, batch)
        self.sinks = List[AggregateSink]()
        self.failed_morsel = -1
        self.failure = String()

    def run(mut self) raises:
        var sink = AggregateSink(self.layout, self.workers)
        var reader = TableSinkInputs(self.shared)
        var morsels = len(self.ranges) // 2
        while True:
            var m = self.cursor.take()
            if m >= morsels:
                break
            var offset = self.ranges[2 * m]
            var length = self.ranges[2 * m + 1]
            var columns = List[Series](capacity=len(self.columns))
            for column in self.columns:
                columns.append(column.slice(offset, length))
            try:
                sink_frame(sink, reader, columns, offset, self.batch)
            except e:
                # Kept for the caller to raise the earliest morsel's error;
                # morsels are claimed in order, so every earlier one runs.
                self.failed_morsel = m
                self.failure = String(e)
                break
        reader.release()
        self.sinks.append(sink^)


@fieldwise_init
struct GroupingSets(Copyable):
    """A pending `group_by_rollup`, `group_by_cube` or
    `group_by_grouping_sets`; finish it with agg."""

    var _plan: LazyGroupingSets

    def agg(self, exprs: List[Expr]) raises -> DataFrame:
        """The rows of every grouping set, in set order: the keys (null in
        the sets that do not group by them), then the aggregations. Use
        `grouping` / `grouping_id` in `exprs` to tell subtotal rows from
        null key values. See LazyGroupingSets.agg."""
        return self._plan.agg(exprs).collect()

    def agg(self, expr: Expr) raises -> DataFrame:
        return self.agg([expr.copy()])


@fieldwise_init
struct GroupBy(Copyable):
    """An eager grouping request over a snapshot and evaluated key columns."""

    var _frame: DataFrame
    var _keys: List[Series]
    var _maintain_order: Bool

    def head(self, n: Int = 5) raises -> DataFrame:
        """First n rows of each group, retaining every original column."""
        return self._edge_rows(n, False)

    def tail(self, n: Int = 5) raises -> DataFrame:
        """Last n rows of each group, retaining every original column."""
        return self._edge_rows(n, True)

    def _edge_rows(self, n: Int, tail: Bool) raises -> DataFrame:
        if n < 0:
            raise Error("GroupBy.head/tail count must be nonnegative")
        var encoded = encode_rows(self._keys, nulls_equal=True)
        var rows = List[List[Int]](length=encoded.count(), fill=List[Int]())
        for row in range(self._frame.height()):
            rows[encoded.ids[row]].append(row)
        var selected = List[Int]()
        for group in rows:
            var start = max(0, len(group) - n) if tail else 0
            var end = len(group) if tail else min(len(group), n)
            for k in range(start, end):
                selected.append(group[k])
        return self._frame.take(selected)

    def _coded_keys(self) raises -> Optional[GroupBy]:
        """This grouping with each String key that carries dictionary codes
        replaced by those codes as a categorical; None when no key does."""
        var keys = List[Series](capacity=len(self._keys))
        var any = False
        for key in self._keys:
            var codes = key._dictionary_codes()
            if codes:
                keys.append(canonical_labels(codes.take()))
                any = True
            else:
                keys.append(key.copy())
        if not any:
            return None
        return GroupBy(self._frame.copy(), keys^, self._maintain_order)

    def _key_names(self) -> Dict[String, Bool]:
        var names = Dict[String, Bool]()
        for key in self._keys:
            var prefix = _key_prefix(key.name())
            names[prefix if prefix != "" else key.name()] = True
        return names^

    def agg(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> DataFrame:
        return self.agg([expression.copy()], batch_size=batch_size)

    def agg(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> DataFrame:
        var prepared = List[Expr](capacity=len(expressions))
        var changed = False
        var counted = len(self._keys) > 0 and (
            self._keys[0].name() in self._frame.columns()
        )
        for expression in expressions:
            if group_sequence_rows(expression):
                prepared.append(expression.implode())
                changed = True
            elif counted:
                var rows = count_rows_by(expression, self._keys[0].name())
                changed = changed or len(rows._nodes) != len(expression._nodes)
                prepared.append(rows^)
            else:
                prepared.append(expression.copy())
        if changed:
            return self.agg(prepared, batch_size=batch_size)
        # String keys that carry their source's dictionary codes (a Parquet
        # scan) group by the codes, as categorical keys do, and come back as
        # strings: integer codes hash and compare far faster than the strings,
        # and the strings are read once per group, not per row.
        var coded = self._coded_keys()
        if coded:
            var result = coded.value().agg(expressions, batch_size=batch_size)
            var columns = List[Series](capacity=result.width())
            for column in result._columns:
                if column.dtype().is_categorical() and column.name() in (
                    self._key_names()
                ):
                    columns.append(decode(column))
                else:
                    columns.append(column.copy())
            return DataFrame(columns^)
        var bound = _bind_all(expressions, self._frame._columns)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var key_names = self._key_names()
        for expression in bound:
            if expression.shape() != AGGREGATE:
                raise Error(
                    "Group aggregation requires scalar aggregate expressions"
                )
            if expression.expr._name in key_names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression.expr._name
                )
        var height = self._frame.height()
        var workers = group_by_workers(height)
        # Every grouping goes through the aggregate table whatever its
        # number of groups, whose per-thread tables adapt to however many
        # there are; keys whose statistics bound them to a few values make
        # it a perfect hash table, using the key ranges. An expression the
        # table has no state for is evaluated over each group's row list.
        if self._table_supported(expressions):
            var table = self._agg_table(expressions, workers, batch_size)
            if table:
                return _pack_struct_keys(table.value())
        return _pack_struct_keys(self._agg_groups(bound, batch_size))

    def _table_supported(self, expressions: List[Expr]) raises -> Bool:
        """Whether the aggregate table (`aggregate_table.mojo`) groups these
        keys and reduces these expressions."""
        if len(self._keys) == 0 or not _stream_reductions(
            expressions, True, True
        ):
            return False
        for expression in expressions:
            for node in expression._nodes:
                if is_reduction(node.op) and not table_reduction(node.op):
                    return False
        for key in self._keys:
            if not key_supported(key):
                return False
        return True

    def _agg_table(
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """Grouped reductions through the aggregate table: each worker
        hashes its morsels' keys where they are and probes its own table,
        updating its states by group; partitions of the groups then merge
        across workers in parallel (`aggregate_table.mojo`). None when some reduction's
        state does not fit inline in a group row."""
        # A float key's words are canonical (-0.0 and 0.0, or two NaNs, are
        # one group), so output uses the first row's original value. That
        # value comes from a FIRST state and replaces the key column.
        var float_keys = List[Int]()
        var all_expressions = expressions.copy()
        for k in range(len(self._keys)):
            var name = self._keys[k].name()
            if (
                self._keys[k].dtype().is_float()
                and name in self._frame.columns()
            ):
                float_keys.append(k)
                all_expressions.append(
                    col(name).first().alias("__key_first_" + String(k))
                )
        if len(float_keys) > 0:
            var grouped = self._agg_canonical_table(
                all_expressions, workers, batch_size
            )
            if not grouped:
                return None
            var frame = grouped.take()
            var hidden = List[String]()
            for k in float_keys:
                var name = "__key_first_" + String(k)
                frame = frame.with_column(
                    frame.column(name).renamed(self._keys[k].name())
                )
                hidden.append(name)
            return frame.drop(hidden)
        return self._agg_canonical_table(expressions, workers, batch_size)

    def _agg_canonical_table(
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """`_agg_table` with float keys as their canonical words."""
        var empty_keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            empty_keys.append(key.slice(0, 0))
        var template = _StreamReduction(
            self._frame.clear(), expressions, empty_keys, self._key_names()
        )
        var specs = table_specs(template)
        var empty = table_inputs(self._frame.clear(), empty_keys, expressions)
        if not table_states_supported(specs, empty):
            return None
        trace_path("group_by.aggregate_table")
        if not _plain_inputs(expressions, self._frame):
            # Computed reduction inputs are evaluated a batch of
            # `batch_size` rows at a time, each batch then updating the
            # states: an input error past the batch comes after the
            # batch's update errors, as the batch size says.
            return self._agg_batches(
                expressions, specs, empty, template, workers, batch_size
            )
        var inputs = table_inputs(self._frame, self._keys, expressions)
        # Morsels within chunks: the columns' common chunks, or one
        # contiguous copy of each chunked column when they differ.
        var keys = self._keys.copy()
        var all_columns = keys.copy()
        for item in inputs:
            all_columns.append(item.copy())
        var ends = List[Int]()
        if can_filter_aligned_chunks(all_columns):
            ends = all_columns[0]._chunked.value()[].ends.copy()
        else:
            for k in range(len(keys)):
                if keys[k].is_chunked():
                    keys[k] = keys[k].rechunk()
            for k in range(len(inputs)):
                if inputs[k].is_chunked():
                    inputs[k] = inputs[k].rechunk()
            ends.append(self._frame.height())
        var ranges = morsel_ranges(ends, workers)
        var result = aggregate_table(
            keys, inputs, specs, ranges, workers, self._maintain_order
        )
        return table_frame(template, result^, self._maintain_order)

    def _agg_batches(
        self,
        expressions: List[Expr],
        specs: List[ReducerSpec],
        empty: List[Series],
        template: _StreamReduction,
        workers: Int,
        batch_size: Int,
    ) raises -> Optional[DataFrame]:
        """The aggregate table over morsels whose reduction inputs are
        evaluated `batch_size` rows at a time on the workers."""
        var frame = self._frame.rechunk()
        # The sink's input: the key columns (named apart from the frame's),
        # then the frame's columns, which the input expressions read.
        var columns = List[Series](capacity=len(self._keys) + frame.width())
        var key_names = List[String](capacity=len(self._keys))
        var keys = List[Series](capacity=len(self._keys))
        for k in range(len(self._keys)):
            var key = (
                self._keys[k]
                .rechunk() if self._keys[k]
                .is_chunked() else self._keys[k]
                .copy()
            )
            var name = "__agg_key_" + String(k)
            key_names.append(name)
            columns.append(key.renamed(name))
            keys.append(key^)
        for column in frame._columns:
            columns.append(column.copy())
        var empty_keys = List[Series](capacity=len(keys))
        for key in keys:
            empty_keys.append(key.slice(0, 0))
        var layout = aggregate_layout(empty_keys, empty, specs)
        layout.use_perfect_hash(keys)
        var schema = List[Series](capacity=len(columns))
        for column in columns:
            schema.append(column.slice(0, 0))
        var shared = TableSinkLayout(
            layout, DataFrame(schema^), key_names, expressions, empty
        )
        # Morsels end on whole input batches: an input error anywhere in a
        # batch precedes that batch's state updates, whichever worker has it.
        var ends: List[Int] = [frame.height()]
        var batch = max(1, batch_size)
        var ranges = morsel_ranges(
            ends, workers, batch if batch % 64 == 0 else 64 * batch
        )
        var cursor = _Cursor.new()
        var jobs = List[_BatchSinkJob](capacity=workers)
        for _ in range(max(1, workers)):
            jobs.append(
                _BatchSinkJob(
                    columns,
                    layout,
                    shared,
                    ranges,
                    cursor,
                    workers,
                    batch_size,
                )
            )
        try:
            run_jobs(jobs)
        except e:
            cursor.free()
            raise e^
        cursor.free()
        # The error of the earliest morsel, as a serial scan would meet it.
        var failed = -1
        for j in range(len(jobs)):
            if jobs[j].failed_morsel >= 0 and (
                failed < 0 or jobs[j].failed_morsel < jobs[failed].failed_morsel
            ):
                failed = j
        if failed >= 0:
            raise Error(jobs[failed].failure)
        var sinks = List[AggregateSink](capacity=len(jobs))
        for j in range(len(jobs)):
            while len(jobs[j].sinks) > 0:
                sinks.append(jobs[j].sinks.pop())
        var result = aggregate_finish(
            sinks, layout, ordered=self._maintain_order
        )
        return table_frame(template, result^, self._maintain_order)

    def _agg_groups(
        self, bound: List[BoundExpr], batch_size: Int
    ) raises -> DataFrame:
        """Expressions the aggregate table has no state for, over the
        groups it finds: each group's row numbers come out of the table as
        a list in input order, and the grouped
        evaluator reduces every expression over those groups."""
        trace_path("group_by.row_lists")
        var height = self._frame.height()
        var numbers = List[Int64](capacity=height)
        for r in range(height):
            numbers.append(Int64(r))
        var listed = GroupBy(
            self._frame.with_column(
                Series(_GROUP_ROWS, Column[Int64](numbers^))
            ),
            self._keys.copy(),
            self._maintain_order,
        )._agg_table(
            [col(_GROUP_ROWS).implode().alias(_GROUP_ROWS)],
            group_by_workers(height),
            batch_size,
        )
        if not listed:
            raise Error("aggregate table: no state for a group's rows")
        var grouped = listed.take()
        var count = grouped.height()
        var lists = grouped.column(_GROUP_ROWS).rechunk().list_column()
        var rows = lists.child().rechunk()
        ref values = rows._data[Column[Int64]]
        var ids = List[Int](length=height, fill=0)
        for g in range(count):
            for e in range(lists._start(g), lists._end(g)):
                ids[Int(values._get(e))] = g
        var columns = List[Series](capacity=len(self._keys) + len(bound))
        for c in range(len(self._keys)):
            columns.append(grouped._columns[c].copy())
        for expression in bound:
            columns.append(
                evaluate(
                    expression,
                    self._frame._columns,
                    height,
                    batch_size=batch_size,
                    grouped=True,
                    groups=ids,
                    group_count=count,
                )
            )
        return DataFrame(columns^, height=count)

    def len(self, name: String = "len") raises -> DataFrame:
        """Row count per group, including rows with null values."""
        if name in self._key_names():
            raise Error(
                "Aggregate output name collides with grouping key: " + name
            )
        var groups = encode_rows(self._keys, nulls_equal=True)
        var counts = List[Int64](length=groups.count(), fill=0)
        for id in groups.ids:
            counts[id] += 1
        var columns = take_sorted_chunked(
            self._keys, groups.representatives.copy(), 1
        )
        columns.append(Series(name, Column[Int64](counts^)))
        return _pack_struct_keys(DataFrame(columns^, height=groups.count()))

    def _each(
        self,
        reduction: String,
        quantile: Float64 = 0.5,
        interpolation: String = "nearest",
    ) raises -> DataFrame:
        """Apply `reduction` to every non-key column."""
        var keys = self._key_names()
        var exprs = List[Expr]()
        for column in self._frame._columns:
            if column.name() not in keys:
                exprs.append(
                    _reduce_column(
                        column.name(), reduction, quantile, interpolation
                    )
                )
        return self.agg(exprs)

    def sum(self) raises -> DataFrame:
        """Each other column's sum per group."""
        return self._each("sum")

    def mean(self) raises -> DataFrame:
        """Each other column's mean per group, as Float64."""
        return self._each("mean")

    def min(self) raises -> DataFrame:
        """Each other column's minimum per group."""
        return self._each("min")

    def max(self) raises -> DataFrame:
        """Each other column's maximum per group."""
        return self._each("max")

    def median(self) raises -> DataFrame:
        """Each other column's median per group, as Float64."""
        return self._each("median")

    def first(self) raises -> DataFrame:
        """Each other column's first value per group (null when that row's
        value is null)."""
        return self._each("first")

    def last(self) raises -> DataFrame:
        """Each other column's last value per group (null when that row's
        value is null)."""
        return self._each("last")

    def n_unique(self) raises -> DataFrame:
        """Each other column's distinct values per group, null counted once,
        as Int64."""
        return self._each("n_unique")

    def quantile(
        self, quantile: Float64, interpolation: String = "nearest"
    ) raises -> DataFrame:
        """Each other column's quantile per group, as Float64; interpolation
        defaults to "nearest"."""
        return self._each("quantile", quantile, interpolation)


def _reduce_column(
    name: String, reduction: String, quantile: Float64, interpolation: String
) raises -> Expr:
    """`col(name)` reduced by a GroupBy convenience method's reduction."""
    var c = col(name)
    if reduction == "sum":
        return c.sum()
    if reduction == "mean":
        return c.mean()
    if reduction == "min":
        return c.min()
    if reduction == "max":
        return c.max()
    if reduction == "median":
        return c.median()
    if reduction == "first":
        return c.first()
    if reduction == "last":
        return c.last()
    if reduction == "n_unique":
        return c.n_unique()
    if reduction == "quantile":
        return c.quantile(quantile, interpolation)
    raise Error("Unknown group reduction: " + reduction)


# The column of row numbers whose implode gives each group's rows
# (`_agg_groups`).
comptime _GROUP_ROWS = "__group_rows"


def group_by_workers(height: Int) -> Int:
    """Workers the eager group-by divides `height` rows among: at most one
    per morsel (`morsel_size`), at most the configured threads."""
    var workers = configured_workers()
    return max(1, min(workers, -(-height // morsel_size(height, workers))))
