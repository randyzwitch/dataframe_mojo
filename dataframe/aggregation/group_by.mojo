"""Eager grouping requests, strategy dispatch, and worker jobs."""
from dataframe.aggregation.hash_table import (
    table_states_supported,
)
from dataframe.aggregation.table_sink import (
    aggregate_table,
)
from dataframe.aggregation.expressions import (
    _bind_all,
    without_filters,
)
from dataframe.aggregation.state import (
    _StreamReduction,
    _merge_ascending_runs,
    table_supported,
    table_frame,
    table_inputs,
    table_specs,
)
from dataframe.row_expressions import len as row_len
from dataframe.selectors import expand_all
from dataframe.planning.expression_inputs import _references
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
from dataframe.dtype import DataType
from dataframe.storage.column import (
    Column,
)
from dataframe.execution import (
    evaluate,
)
from dataframe.expr import (
    Expr,
    SELECTOR,
    group_sequence_rows,
    col,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.kernels.gather import (
    take_sorted_chunked,
    can_filter_aligned_chunks,
    take_parallel,
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
    source_tasks,
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

    def _key_names(self) -> Dict[String, Bool]:
        var names = Dict[String, Bool]()
        for key in self._keys:
            names[key.name()] = True
        return names^

    def agg(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> DataFrame:
        return self.agg([expression.copy()], batch_size=batch_size)

    def agg(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> DataFrame:
        """One row per group: the keys, then the aggregations.

        Runs as the lazy plan's group_by does: a plan of one aggregation
        over this frame, executed through the same pipelines and sinks as
        `LazyFrame.collect`. The aggregate table sink takes every reduction
        it has a state for; the plan's executor evaluates the rest over each
        group's rows. `batch_size` is the plan's batch size."""
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var prepared = List[Expr](capacity=len(expressions))
        for expression in expressions:
            if group_sequence_rows(expression):
                prepared.append(expression.implode())
            else:
                prepared.append(expression.copy())
        # Selectors expand against this frame's columns, before the key
        # columns join them.
        var expanded = expand_all(prepared, self._frame._columns)
        var bound = _bind_all(expanded, self._frame._columns)
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
        return self._planned(expanded, batch_size)

    def _planned(
        self, expressions: List[Expr], batch_size: Int
    ) raises -> DataFrame:
        """`agg` of checked, expanded expressions. The plan's input is the
        frame with each key column added under a name of its own, so that
        computed keys and keys named like a column the aggregations read
        group as given while the aggregations still see the frame's
        columns. Of those, it holds only the ones the aggregations read:
        the pipeline cuts every input column into each morsel."""
        var taken = Dict[String, Bool]()
        for column in self._frame._columns:
            taken[column.name()] = True
        var reads = Dict[String, Bool]()
        var every_column = False
        for expression in expressions:
            var r = _references(expression)
            if not r:
                every_column = True
                break
            for name in r.value():
                reads[name] = True
        var columns = List[Series]()
        for column in self._frame._columns:
            if every_column or column.name() in reads:
                columns.append(column.copy())
        for expression in expressions:
            taken[expression._name] = True
        var grouped = List[String](capacity=len(self._keys))
        var coded = List[Bool](capacity=len(self._keys))
        var firsts = List[String](capacity=len(self._keys))
        var all_expressions = expressions.copy()
        for k in range(len(self._keys)):
            ref key = self._keys[k]
            var name = String("__group_key_") + String(k)
            while name in taken:
                name += "_"
            taken[name] = True
            # A String key that carries its source's dictionary codes (a
            # Parquet scan) groups by the codes, as a categorical key does,
            # and comes back as strings: integer codes hash and compare far
            # faster than the strings, which are then read once per group.
            var codes = key._dictionary_codes()
            if codes:
                columns.append(canonical_labels(codes.take()).renamed(name))
                coded.append(key.dtype() == DataType.STRING)
            else:
                columns.append(key.renamed(name))
                coded.append(False)
            grouped.append(name)
            # A float key's groups compare canonical words (-0.0 and 0.0,
            # or two NaNs, are one group); the output keeps the value of
            # the group's first row.
            var first = String()
            if key.dtype().is_float():
                first = String("__key_first_") + String(k)
                while first in taken:
                    first += "_"
                taken[first] = True
                all_expressions.append(col(name).first().alias(first))
            firsts.append(first^)
        if len(all_expressions) == 0:
            # The groups alone: a table whose one state, a count, is not
            # output.
            var count = String("__group_count")
            while count in taken:
                count += "_"
            all_expressions.append(col(grouped[0]).len().alias(count))
        var input = DataFrame(columns^, height=self._frame.height())
        var result = (
            input.lazy()
            .group_by(grouped, maintain_order=self._maintain_order)
            .agg(all_expressions)
            ._collect(optimize=False, streaming=True, batch_size=batch_size)
        )
        var out = List[Series](capacity=len(self._keys) + len(expressions))
        for k in range(len(self._keys)):
            var key: Series
            if firsts[k].byte_length() > 0:
                key = result.column(firsts[k])
            else:
                key = result._columns[k].copy()
            if coded[k] and key.dtype().is_categorical():
                key = decode(key)
            out.append(key.renamed(self._keys[k].name()))
        for j in range(len(expressions)):
            out.append(result._columns[len(self._keys) + j].copy())
        return DataFrame(out^, height=result.height())

    def _declined(
        self, expressions: List[Expr], batch_size: Int, planned: Bool
    ) raises -> DataFrame:
        """An aggregation node the plan's executor runs over its executed
        input. With `planned`, when the aggregate table holds every state
        (or there are none), that is the one-node plan over the frame,
        which runs in the table sink. Otherwise each group's rows are
        reduced by the grouped evaluator: so are a plan's schema runs (no
        rows) and a node a pipeline declined, which runs inside that
        pipeline's task (where no other schedule may start) and which a
        one-node plan would decline again."""
        var names = List[String](capacity=len(self._keys))
        for key in self._keys:
            names.append(key.name())
        if planned and (
            len(expressions) == 0
            or table_supported(self._frame.clear(), names, expressions)
        ):
            return self.agg(expressions, batch_size=batch_size)
        return self._agg_groups(
            _bind_all(expressions, self._frame._columns), batch_size
        )

    def _agg_table(
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """Reductions of the frame's columns as they are (`_agg_groups`'
        row lists) through the aggregate table: each worker hashes its
        morsels' keys where they are and probes its own table, updating
        its states by group; partitions of the groups then merge across
        workers in parallel (`aggregate_table.mojo`). None when some
        reduction's state does not fit inline in a group row."""
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
        # The states' types are the unfiltered reductions'.
        var template = _StreamReduction(
            self._frame.clear(),
            without_filters(expressions),
            empty_keys,
            self._key_names(),
        )
        var specs = table_specs(template)
        var empty = table_inputs(self._frame.clear(), empty_keys, expressions)
        if not table_states_supported(specs, empty):
            return None
        trace_path("group_by.aggregate_table")
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
        return self.agg([row_len().alias(name)])

    def _counts(self, name: String) raises -> DataFrame:
        """`len` in first-occurrence order on the calling thread, for a
        value's counts within an expression (`Series.value_counts`), which
        may run once per group over a few rows."""
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
        return DataFrame(columns^, height=groups.count())

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
    """Workers the eager group-by divides `height` rows among: one a
    thread, up to its morsels (`source_tasks`)."""
    var workers = configured_workers()
    return source_tasks(-(-height // morsel_size(height, workers)), workers)
