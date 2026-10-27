"""Operation-specific payload construction over the stable flat PlanNode layout.

Builders encode storage only. Public API validation and deferred execution errors
stay at their existing call sites. _plan_node remains the compatibility factory
for generic graph code and external internal-API consumers.
"""
from dataframe.planning.model import (
    PlanNode,
    _plan_node,
    SCAN_FRAME,
    SCAN_CSV,
    SCAN_PARQUET,
    SCAN_IPC,
    SCAN_NDJSON,
    FILTER,
    SELECT,
    WITH_COLUMNS,
    AGG,
    JOIN,
    SORT,
    SLICE,
    UNIQUE,
    DROP,
    EXPLODE,
    UNNEST,
    CONCAT,
    HCONCAT,
    GATHER,
    GATHER_EVERY,
    INSERT_COLUMN,
    TAIL,
    REVERSE,
    SHIFT,
    ROW_INDEX,
    UNPIVOT,
    PIVOT,
    GROUPING_SETS,
    GROUP_BY_DYNAMIC,
    ROLLING,
    FILL_NULL_STRATEGY,
)
from dataframe.dtype import DataType
from dataframe.expr import Expr
from std.collections import Optional


def _frame_scan_node(slot: Int = 0) -> PlanNode:
    var node = _plan_node(SCAN_FRAME, offset=slot)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _parquet_scan_node(path: String, slot: Int = 0) -> PlanNode:
    var node = _plan_node(SCAN_PARQUET, text=path, offset=slot)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _csv_source_node(
    path: String,
    settings: List[String] = List[String](),
    switches: List[Bool] = List[Bool](),
    null_values: List[String] = List[String](),
    slot: Int = 0,
) -> PlanNode:
    var node = _plan_node(
        SCAN_CSV,
        text=path,
        names2=settings,
        flags=switches,
        right_keys=null_values,
        offset=slot,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _ipc_scan_node(
    path: String, memory_map: Bool = True, slot: Int = 0
) -> PlanNode:
    var node = _plan_node(
        SCAN_IPC,
        text=path,
        names2=["0"],
        flags=[memory_map, False],
        offset=slot,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _ndjson_scan_node(
    path: String,
    schema: Optional[DataType],
    infer_schema_length: Int,
    ignore_errors: Bool,
    slot: Int = 0,
) -> PlanNode:
    var node = _plan_node(
        SCAN_NDJSON,
        text=path,
        json_schema=schema,
        names2=[String(infer_schema_length)],
        flags=[ignore_errors],
        offset=slot,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _filter_node(exprs: List[Expr], left: Int = -1) -> PlanNode:
    var node = _plan_node(FILTER, left=left, exprs=exprs)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _select_node(exprs: List[Expr], left: Int = -1) -> PlanNode:
    var node = _plan_node(SELECT, left=left, exprs=exprs)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _with_columns_node(exprs: List[Expr], left: Int = -1) -> PlanNode:
    var node = _plan_node(WITH_COLUMNS, left=left, exprs=exprs)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _aggregate_node(
    keys: List[String],
    exprs: List[Expr],
    maintain_order: Bool = False,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        AGG, left=left, names=keys, exprs=exprs, maintain_order=maintain_order
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _join_node(
    left_keys: List[String],
    right_keys: List[String],
    join_type: String,
    type_code: Int,
    suffix: String = "_right",
    coalesce: Bool = True,
    left: Int = -1,
    right: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        JOIN,
        left=left,
        right=right,
        names=left_keys,
        right_keys=right_keys,
        text=join_type,
        how=type_code,
        names2=[suffix],
        coalesce=coalesce,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _sort_node(
    keys: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
    maintain_order: Bool = False,
    left: Int = -1,
) -> PlanNode:
    debug_assert(
        len(descending) == len(keys) and len(nulls_last) == len(keys),
        "sort requires one direction and null placement per key",
    )
    var node = _plan_node(
        SORT,
        left=left,
        names=keys,
        flags=descending + nulls_last.copy(),
        maintain_order=maintain_order,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _slice_node(row_offset: Int, row_count: Int, left: Int = -1) -> PlanNode:
    var node = _plan_node(SLICE, left=left, offset=row_offset, length=row_count)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _unique_node(
    subset: List[String],
    keep: String,
    maintain_order: Bool = False,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        UNIQUE,
        left=left,
        names=subset,
        text=keep,
        maintain_order=maintain_order,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _drop_node(columns: List[String], left: Int = -1) -> PlanNode:
    var node = _plan_node(DROP, left=left, names=columns)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _explode_node(columns: List[String], left: Int = -1) -> PlanNode:
    var node = _plan_node(EXPLODE, left=left, names=columns)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _unnest_node(column: String, left: Int = -1) -> PlanNode:
    var node = _plan_node(UNNEST, left=left, text=column)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _concat_node(left: Int = -1, right: Int = -1) -> PlanNode:
    var node = _plan_node(CONCAT, left=left, right=right)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _hconcat_node(left: Int = -1, right: Int = -1) -> PlanNode:
    var node = _plan_node(HCONCAT, left=left, right=right)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _gather_node(
    null_on_oob: Bool, left: Int = -1, right: Int = -1
) -> PlanNode:
    var node = _plan_node(GATHER, left=left, right=right, flags=[null_on_oob])
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _gather_every_node(
    stride: Int, row_offset: Int, left: Int = -1
) -> PlanNode:
    var node = _plan_node(
        GATHER_EVERY, left=left, length=stride, offset=row_offset
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _insert_column_node(
    position: Int, left: Int = -1, right: Int = -1
) -> PlanNode:
    var node = _plan_node(
        INSERT_COLUMN, left=left, right=right, offset=position
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _tail_node(count: Int, left: Int = -1) -> PlanNode:
    var node = _plan_node(TAIL, left=left, offset=count)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _reverse_node(left: Int = -1) -> PlanNode:
    var node = _plan_node(REVERSE, left=left)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _shift_node(count: Int, fill: List[Expr], left: Int = -1) -> PlanNode:
    var node = _plan_node(SHIFT, left=left, offset=count, exprs=fill)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _row_index_node(name: String, start: Int, left: Int = -1) -> PlanNode:
    var node = _plan_node(ROW_INDEX, left=left, names=[name], offset=start)
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _unpivot_node(
    columns: List[String],
    index: List[String],
    variable_name: String,
    value_name: String,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        UNPIVOT,
        left=left,
        names=columns,
        names2=index,
        right_keys=[variable_name, value_name],
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _pivot_node(
    index: List[String],
    on: String,
    values: String,
    output_columns: List[String],
    aggregate_function: String,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        PIVOT,
        left=left,
        names=index + [on, values],
        offset=len(index),
        text=aggregate_function,
        right_keys=output_columns,
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _grouping_sets_node(
    keys: List[String],
    exprs: List[Expr],
    membership: List[Bool],
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        GROUPING_SETS, left=left, names=keys, exprs=exprs, flags=membership
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _dynamic_window_node(
    index_column: String,
    groups: List[String],
    every: String,
    period: String,
    offset: String,
    closed: String,
    label: String,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        GROUP_BY_DYNAMIC,
        left=left,
        names=List[String]([index_column]) + groups.copy(),
        names2=[every, period, offset, closed, label],
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _rolling_window_node(
    index_column: String,
    groups: List[String],
    period: String,
    offset: String,
    closed: String,
    left: Int = -1,
) -> PlanNode:
    var node = _plan_node(
        ROLLING,
        left=left,
        names=List[String]([index_column]) + groups.copy(),
        names2=[String(""), period, offset, closed, String("datapoint")],
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^


def _fill_null_strategy_node(
    strategy: String, limit: Int, subset: List[String], left: Int = -1
) -> PlanNode:
    var node = _plan_node(
        FILL_NULL_STRATEGY, left=left, text=strategy, length=limit, names=subset
    )
    debug_assert(node.payload_layout_valid(), "invalid plan payload layout")
    return node^
