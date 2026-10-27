"""Logical plan nodes and operation tags; no optimizer or executor dependencies."""
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    Expr,
)
from dataframe.value import (
    AnyValue,
)
from std.collections import (
    Optional,
)
from std.memory import (
    ArcPointer,
)


comptime SCAN_FRAME = 0
comptime SCAN_CSV = 1
comptime FILTER = 2
comptime SELECT = 3
comptime WITH_COLUMNS = 4
comptime AGG = 5
comptime JOIN = 6
comptime SORT = 7
comptime SLICE = 8
comptime UNIQUE = 9
comptime DROP = 10
comptime SCAN_PARQUET = 11
# File IPC scan: offset remains the input slot; names2[0] is the row offset,
# length is the row limit; flags hold mapped input and explicit projection.
comptime SCAN_IPC = 44
comptime SCAN_NDJSON = 45
# A whole-frame strategy fill; text is strategy, length its optional limit.
comptime FILL_NULL_STRATEGY = 46
comptime EXPLODE = 12
comptime UNNEST = 13
comptime ASOF = 14
# UNION ALL of two plans: the left input's rows, then the right's.
comptime CONCAT = 15
# Wide to long (DataFrame.unpivot): `names` the value columns (every
# non-index column when empty), `names2` the index columns and
# `right_keys` the variable and value column names. A breaker: its rows
# come out by value column, then input row, so it needs its whole input,
# as Polars' unpivot is a map function over a materialized frame.
comptime UNPIVOT = 20
# Time windows (`DataFrame.group_by_dynamic` and `DataFrame.rolling`, then
# `agg`). `names` holds the index column then the `by` keys, `names2` the
# window settings (every, period, offset, closed, label) with their
# defaults filled in as the eager methods fill them, `exprs` the
# aggregations and `length` their batch size. Both read their whole
# input in index order, so they run as pipeline breakers, and filters and
# slices never move below them.
comptime GROUP_BY_DYNAMIC = 28
comptime ROLLING = 29
# Row-order operators, each run by its eager DataFrame method on its whole
# input in order (a pipeline breaker under streaming); filters, slices and
# other row-changing rewrites do not cross them.
# The last `offset` rows (DataFrame.tail; a negative count drops rows).
comptime TAIL = 16
# The rows in reverse order (DataFrame.reverse).
comptime REVERSE = 17
# Every column moved `offset` rows later (DataFrame.shift); `exprs` holds
# the fill value when one was given.
comptime SHIFT = 18
# A leading Int64 row number named `names[0]`, starting at `offset`
# (DataFrame.with_row_index).
comptime ROW_INDEX = 19
# The left input's columns, then the right's, side by side (a horizontal
# concat, Polars' HCONCAT); the heights must match.
comptime HCONCAT = 40
# The left input's rows at the positions the right input's one integer
# column lists (Polars' Gather); `flags[0]` is null_on_oob.
comptime GATHER = 41
# Whole-input row selection: offset is the first row, length is the stride.
comptime GATHER_EVERY = 47
# Insert the right input's one column at offset in the left schema.
comptime INSERT_COLUMN = 48
# Grouping sets (`group_by_rollup`, `group_by_cube`,
# `group_by_grouping_sets`, then `agg`). `names` holds every key any set
# names, `flags` which keys each set groups by (one row of flags per set,
# a flag per key), `exprs` the aggregations and `grouping` columns as
# written. The result is each set's rows in turn, a key a set does not
# group by null. It reads its input once: the stream sinks every morsel
# into a table per set, as DuckDB's one sink feeds every set's table.
comptime GROUPING_SETS = 43
# How the executor offers nodes to an accelerator provider (`_handoff`).
comptime HANDOFF_OFF = 0
comptime HANDOFF_FORCED = 1
comptime HANDOFF_AUTO = 2


@fieldwise_init
struct PlanNode(Copyable, Deinitable, Movable):
    var kind: Int
    var left: Int
    var right: Int
    var exprs: List[Expr]
    var names: List[String]
    var names2: List[String]
    var flags: List[Bool]
    var text: String
    var offset: Int
    var length: Int
    var maintain_order: Bool
    # A JOIN's type code (see join_type); -1 for an unknown name, which the
    # eager join reports from the original text when the plan executes.
    var how: Int
    # A JOIN's right-side key names, paired with `names` (the left keys),
    # and whether key columns coalesce into the left names.
    var right_keys: List[String]
    var coalesce: Bool
    # Optional internal source-execution counter, shared across plan copies.
    # No counter storage is allocated unless profiling/tests attach one.
    var _executions: Optional[ArcPointer[Int]]
    var asof_tolerance: Optional[AnyValue]
    var source_failure: String
    var json_schema: Optional[DataType]

    def _record_execution(self):
        if self._executions:
            var counter = self._executions.value()
            counter[] += 1


def _plan_node(
    kind: Int,
    left: Int = -1,
    right: Int = -1,
    exprs: List[Expr] = List[Expr](),
    names: List[String] = List[String](),
    names2: List[String] = List[String](),
    flags: List[Bool] = List[Bool](),
    text: String = "",
    offset: Int = 0,
    length: Int = -1,
    maintain_order: Bool = False,
    how: Int = -1,
    right_keys: List[String] = List[String](),
    coalesce: Bool = True,
    source_failure: String = "",
    json_schema: Optional[DataType] = None,
) -> PlanNode:
    return PlanNode(
        kind,
        left,
        right,
        exprs.copy(),
        names.copy(),
        names2.copy(),
        flags.copy(),
        text,
        offset,
        length,
        maintain_order,
        how,
        right_keys.copy(),
        coalesce,
        Optional[ArcPointer[Int]](),
        Optional[AnyValue](),
        source_failure,
        json_schema.copy(),
    )


# Long to wide with the output columns given up front (Polars 1.44's
# QueryContext.pivot): `names` the index columns, then the column whose values
# name the new columns, then the values column; `offset` the number of
# index columns; `text` the aggregate function; `right_keys` the new
# columns' names in order. One row per distinct index key, so it runs on
# its whole input.
comptime PIVOT = 42
