"""Logical plan nodes and operation tags; no optimizer or executor dependencies."""
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    Expr,
)
from dataframe.io.dataset_options import DatasetSpec
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
<<<<<<< HEAD
    # A JOIN whose build side a swap already chose: no swap rule flips it
    # again, as DuckDB's build/probe optimizer decides a join's sides
    # once. Without this a swap made on a low estimate could flip back
    # once the other side's real height is known, and again after that.
    var sides_fixed: Bool
=======
    var dataset: Optional[DatasetSpec]
    var dataset_projected: Bool

    def has_dataset(self) -> Bool:
        return Bool(self.dataset)

    def set_dataset(mut self, spec: DatasetSpec):
        self.dataset = spec.copy()

    def dataset_spec(self) -> ref[self.dataset._value] DatasetSpec:
        return self.dataset.value()
>>>>>>> origin/main

    # These views borrow the stored lists: reading a payload does not add a
    # copy-on-write owner. String elements share their list's element origin
    # (the index in the return origin is not a runtime read). Operation dispatch
    # remains the caller's responsibility.

    def scan_projection(self) -> ref[self.names] List[String]:
        return self.names

    def csv_settings(self) -> ref[self.names2] List[String]:
        return self.names2

    def csv_switches(self) -> ref[self.flags] List[Bool]:
        return self.flags

    def csv_null_values(self) -> ref[self.right_keys] List[String]:
        return self.right_keys

    def sort_keys(self) -> ref[self.names] List[String]:
        return self.names

    def sort_options(self) -> ref[self.flags] List[Bool]:
        return self.flags

    def group_keys(self) -> ref[self.names] List[String]:
        return self.names

    def grouping_membership(self) -> ref[self.flags] List[Bool]:
        return self.flags

    def join_left_keys(self) -> ref[self.names] List[String]:
        return self.names

    def join_right_keys(self) -> ref[self.right_keys] List[String]:
        return self.right_keys

    def unpivot_columns(self) -> ref[self.names] List[String]:
        return self.names

    def unpivot_index(self) -> ref[self.names2] List[String]:
        return self.names2

    def unpivot_output_names(self) -> ref[self.right_keys] List[String]:
        return self.right_keys

    def window_columns(self) -> ref[self.names] List[String]:
        return self.names

    def window_settings(self) -> ref[self.names2] List[String]:
        return self.names2

    def pivot_columns(self) -> ref[self.names] List[String]:
        return self.names

    def pivot_output_columns(self) -> ref[self.right_keys] List[String]:
        return self.right_keys

    def selected_columns(self) -> ref[self.names] List[String]:
        return self.names

    def source_slot(self) -> Int:
        return self.offset

    def scan_row_limit(self) -> Int:
        return self.length

    def ipc_row_offset(self) -> ref[self.names2[0]] String:
        return self.names2[0]

    def ipc_memory_map(self) -> Bool:
        return self.flags[0]

    def ipc_has_projection(self) -> Bool:
        return self.flags[1]

    def ndjson_infer_schema_length(self) -> ref[self.names2[0]] String:
        return self.names2[0]

    def ndjson_ignore_errors(self) -> Bool:
        return self.flags[0]

    def join_suffix(self) -> ref[self.names2[0]] String:
        return self.names2[0]

    def join_build_slot(self) -> Int:
        return self.offset

    def asof_allow_exact_matches(self) -> Bool:
        return self.flags[0]

    def sort_limit(self) -> Int:
        return self.length

    def slice_offset(self) -> Int:
        return self.offset

    def slice_length(self) -> Int:
        return self.length

    def tail_count(self) -> Int:
        return self.offset

    def shift_count(self) -> Int:
        return self.offset

    def gather_stride(self) -> Int:
        return self.length

    def gather_offset(self) -> Int:
        return self.offset

    def insert_position(self) -> Int:
        return self.offset

    def gather_null_on_oob(self) -> Bool:
        return self.flags[0]

    def row_index_name(self) -> ref[self.names[0]] String:
        return self.names[0]

    def row_index_offset(self) -> Int:
        return self.offset

    def fill_limit(self) -> Int:
        return self.length

    def window_batch_size(self) -> Int:
        return self.length

    def pivot_index_count(self) -> Int:
        return self.offset

    def pivot_on(self) -> ref[self.names[0]] String:
        return self.names[self.offset]

    def pivot_values(self) -> ref[self.names[0]] String:
        return self.names[self.offset + 1]

    def filter_is_pinned(self) -> Bool:
        return len(self.flags) > 0 and self.flags[0]

    def sort_descending(self, key: Int) -> Bool:
        return self.flags[key]

    def sort_nulls_last(self, key: Int) -> Bool:
        return self.flags[len(self.names) + key]

    def asof_group_count(self) -> Int:
        return len(self.names2) - 1

    def asof_group(self, index: Int) -> ref[self.names2[0]] String:
        return self.names2[index + 1]

    def window_every(self) -> ref[self.names2[0]] String:
        return self.names2[0]

    def window_period(self) -> ref[self.names2[0]] String:
        return self.names2[1]

    def window_offset(self) -> ref[self.names2[0]] String:
        return self.names2[2]

    def window_closed(self) -> ref[self.names2[0]] String:
        return self.names2[3]

    def window_label(self) -> ref[self.names2[0]] String:
        return self.names2[4]

    def set_source_slot(mut self, slot: Int):
        self.offset = slot

    def set_scan_projection(mut self, var columns: List[String]):
        self.names = columns^
        if self.has_dataset():
            self.dataset_projected = True
        if self.kind == SCAN_IPC:
            self.flags[1] = True

    def set_scan_row_limit(mut self, limit: Int):
        self.length = limit

    def set_ipc_slice(mut self, row_offset: String, row_limit: Int):
        self.names2 = [row_offset]
        self.length = row_limit

    def set_sort_limit(mut self, limit: Int):
        self.length = limit

    def set_join_build_slot(mut self, slot: Int):
        self.offset = slot

    def pin_filter(mut self):
        self.flags = [True]

    def set_window_aggregations(
        mut self, var exprs: List[Expr], batch_size: Int
    ):
        self.exprs = exprs^
        self.length = batch_size

    def set_asof_options(
        mut self,
        strategy: String,
        allow_exact_matches: Bool,
        tolerance: Optional[AnyValue],
        groups: List[String],
    ):
        # Convert the already connected join while retaining its suffix and keys.
        self.kind = ASOF
        self.text = strategy
        self.flags = [allow_exact_matches]
        self.asof_tolerance = tolerance.copy()
        for name in groups:
            self.names2.append(name)
        debug_assert(self.payload_layout_valid(), "invalid asof payload layout")

    def csv_separator(self) -> ref[self.names2[0]] String:
        return self.names2[0]

    def csv_quote_char(self) -> ref[self.names2[0]] String:
        return self.names2[1]

    def csv_comment_prefix(self) -> ref[self.names2[0]] String:
        return self.names2[2]

    def csv_encoding(self) -> ref[self.names2[0]] String:
        return self.names2[3]

    def csv_skip_rows(self) -> ref[self.names2[0]] String:
        return self.names2[4]

    def csv_infer_schema_length(self) -> ref[self.names2[0]] String:
        return self.names2[5]

    def csv_has_header(self) -> Bool:
        return self.flags[0]

    def csv_ignore_errors(self) -> Bool:
        return self.flags[1]

    def csv_truncate_ragged_lines(self) -> Bool:
        return self.flags[2]

    def csv_try_parse_dates(self) -> Bool:
        return self.flags[3]

    def csv_has_options(self) -> Bool:
        return len(self.names2) >= 6

    def csv_override_count(self) -> Int:
        return max(0, (len(self.names2) - 6) // 2)

    def csv_override_name(self, index: Int) -> ref[self.names2[0]] String:
        return self.names2[6 + 2 * index]

    def csv_override_dtype(self, index: Int) -> ref[self.names2[0]] String:
        return self.names2[7 + 2 * index]

    def payload_layout_valid(self) -> Bool:
        """Check storage shape only, leaving user-option validation where it was.

        Legacy _plan_node intentionally permits partially built nodes. Named
        builders use this check once their operation payload is complete.
        """
        if self.kind == SCAN_CSV:
            if len(self.names2) == 0:
                return len(self.flags) == 0
            return (
                len(self.names2) >= 6
                and (len(self.names2) - 6) % 2 == 0
                and len(self.flags) == 4
            )
        if self.kind == SCAN_IPC:
            return len(self.names2) == 1 and len(self.flags) == 2
        if self.kind == SCAN_NDJSON:
            return len(self.names2) == 1 and len(self.flags) == 1
        if self.kind == SORT:
            return len(self.flags) == 2 * len(self.names)
        if self.kind == JOIN:
            return len(self.names2) == 1
        if self.kind == ASOF:
            return len(self.names2) >= 1 and len(self.flags) == 1
        if self.kind == GATHER:
            return len(self.flags) == 1
        if self.kind == ROW_INDEX:
            return len(self.names) == 1
        if self.kind == PIVOT:
            return self.offset >= 0 and len(self.names) == self.offset + 2
        if self.kind == UNPIVOT:
            return len(self.right_keys) == 2
        if self.kind == GROUP_BY_DYNAMIC or self.kind == ROLLING:
            return len(self.names) >= 1 and len(self.names2) == 5
        if self.kind == GROUPING_SETS:
            return (
                len(self.names) > 0 and len(self.flags) % len(self.names) == 0
            )
        return True

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
    dataset: Optional[DatasetSpec] = None,
    dataset_projected: Bool = False,
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
<<<<<<< HEAD
        False,
=======
        dataset.copy(),
        dataset_projected,
>>>>>>> origin/main
    )


# Long to wide with the output columns given up front (Polars 1.44's
# QueryContext.pivot): `names` the index columns, then the column whose values
# name the new columns, then the values column; `offset` the number of
# index columns; `text` the aggregate function; `right_keys` the new
# columns' names in order. One row per distinct index key, so it runs on
# its whole input.
comptime PIVOT = 42
