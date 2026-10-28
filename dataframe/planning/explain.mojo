"""Logical plan labels and streaming capability descriptions."""
from dataframe.aggregation.expressions import (
    _stream_reductions,
)
from dataframe.join_hash import (
    prefer_left_build,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_known_height,
)
from dataframe.planning.model import (
    AGG,
    ASOF,
    CONCAT,
    EXPLODE,
    FILL_NULL_STRATEGY,
    FILTER,
    GATHER,
    GATHER_EVERY,
    GROUPING_SETS,
    GROUP_BY_DYNAMIC,
    HCONCAT,
    INSERT_COLUMN,
    JOIN,
    PIVOT,
    REVERSE,
    ROLLING,
    ROW_INDEX,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SHIFT,
    SLICE,
    SORT,
    TAIL,
    UNIQUE,
    UNNEST,
    UNPIVOT,
    WITH_COLUMNS,
)
from dataframe.planning.properties import (
    _is_scan,
    _joined,
    _key_pairs,
    _output_names,
    _stream_rows,
)


def plan_label(self: QueryContext, index: Int) -> String:
    """The operator name `explain` prints for a node."""
    ref node = self._nodes[index]
    var label: String
    if node.kind == SCAN_FRAME:
        label = "SCAN frame"
    elif node.kind == SCAN_CSV:
        label = "SCAN CSV " + node.text
    elif node.kind == SCAN_PARQUET:
        label = "SCAN PARQUET " + node.text
    elif node.kind == SCAN_NDJSON:
        label = "SCAN NDJSON " + node.text
    elif node.kind == SCAN_IPC:
        label = "SCAN IPC " + node.text
    elif node.kind == HCONCAT:
        label = "HCONCAT"
    elif node.kind == GATHER:
        label = "GATHER" + (" null_on_oob" if node.gather_null_on_oob() else "")
    elif node.kind == FILTER:
        label = "FILTER"
    elif node.kind == SELECT:
        label = "SELECT " + _joined(_output_names(node.exprs))
    elif node.kind == WITH_COLUMNS:
        label = "WITH_COLUMNS " + _joined(_output_names(node.exprs))
    elif node.kind == AGG:
        label = (
            "GROUP_BY "
            + _joined(node.group_keys())
            + " AGG "
            + _joined(_output_names(node.exprs))
        )
    elif node.kind == GROUPING_SETS:
        label = (
            "GROUPING_SETS "
            + _joined(node.group_keys())
            + " ("
            + String(len(node.grouping_membership()) // len(node.group_keys()))
            + " sets) AGG "
            + _joined(_output_names(node.exprs))
        )
    elif node.kind == ASOF:
        label = "ASOF " + node.text + " " + _joined(node.join_left_keys())
    elif node.kind == JOIN:
        label = "JOIN " + node.text + " on " + _key_pairs(node)
    elif node.kind == SORT and node.sort_limit() >= 0:
        label = (
            "TOP_K "
            + String(node.sort_limit())
            + " by "
            + _joined(node.sort_keys())
        )
    elif node.kind == SORT:
        label = "SORT " + _joined(node.sort_keys())
    elif node.kind == SLICE:
        label = (
            "SLICE "
            + String(node.slice_offset())
            + " "
            + String(node.slice_length())
        )
    elif node.kind == FILL_NULL_STRATEGY:
        label = "FILL_NULL strategy=" + node.text
    elif node.kind == UNIQUE:
        label = "UNIQUE " + _joined(node.names)
    elif node.kind == PIVOT:
        label = (
            "PIVOT on "
            + node.pivot_on()
            + " index "
            + _joined(
                List[String](node.pivot_columns()[: node.pivot_index_count()])
            )
            + " values "
            + node.pivot_values()
            + " columns "
            + _joined(node.pivot_output_columns())
        )
        if node.text.byte_length() > 0:
            label += " AGG " + node.text
    elif node.kind == EXPLODE:
        label = "EXPLODE " + _joined(node.names)
    elif node.kind == UNNEST:
        label = "UNNEST " + node.text
    elif node.kind == CONCAT:
        label = "CONCAT"
    elif node.kind == UNPIVOT:
        label = (
            "UNPIVOT on "
            + (
                _joined(node.unpivot_columns()) if len(
                    node.unpivot_columns()
                ) else "*"
            )
            + " index "
            + _joined(node.unpivot_index())
            + " as "
            + _joined(node.unpivot_output_names())
        )
    elif node.kind == GROUP_BY_DYNAMIC:
        label = (
            "GROUP_BY_DYNAMIC "
            + node.window_columns()[0]
            + " every "
            + node.window_every()
            + " period "
            + node.window_period()
            + " offset "
            + node.window_offset()
            + " closed "
            + node.window_closed()
            + " label "
            + node.window_label()
        )
        if len(node.window_columns()) > 1:
            label += " by " + _joined(List[String](node.window_columns()[1:]))
        label += " AGG " + _joined(_output_names(node.exprs))
    elif node.kind == ROLLING:
        label = (
            "ROLLING "
            + node.window_columns()[0]
            + " period "
            + node.window_period()
            + " offset "
            + node.window_offset()
            + " closed "
            + node.window_closed()
        )
        if len(node.window_columns()) > 1:
            label += " by " + _joined(List[String](node.window_columns()[1:]))
        label += " AGG " + _joined(_output_names(node.exprs))
    elif node.kind == TAIL:
        label = "TAIL " + String(node.tail_count())
    elif node.kind == GATHER_EVERY:
        label = (
            "GATHER_EVERY "
            + String(node.gather_stride())
            + " from "
            + String(node.gather_offset())
        )
    elif node.kind == INSERT_COLUMN:
        label = "INSERT_COLUMN at " + String(node.insert_position())
    elif node.kind == REVERSE:
        label = "REVERSE"
    elif node.kind == SHIFT:
        label = "SHIFT " + String(node.shift_count())
        if len(node.exprs) > 0:
            label += " fill " + node.exprs[0]._name
    elif node.kind == ROW_INDEX:
        label = (
            "WITH_ROW_INDEX "
            + node.row_index_name()
            + " from "
            + String(node.row_index_offset())
        )
    else:
        label = "DROP " + _joined(node.names)
    return label^


def plan_describe(
    self: QueryContext,
    index: Int,
    depth: Int,
    mut out: String,
    streaming: Bool = True,
):
    ref node = self._nodes[index]
    var pad = String("  ") * depth
    var label = plan_label(self, index)
    if _is_scan(node.kind):
        if len(node.names) > 0:
            label += " [project " + _joined(node.names) + "]"
        elif node.kind == SCAN_IPC and node.ipc_has_projection():
            label += " [project <none>]"
        if node.kind == SCAN_IPC:
            if node.ipc_row_offset() != "0":
                label += " [offset " + node.ipc_row_offset() + "]"
            if node.scan_row_limit() >= 0:
                label += " [n_rows " + String(node.scan_row_limit()) + "]"
        if node.kind in [SCAN_CSV, SCAN_NDJSON] and node.scan_row_limit() >= 0:
            label += " [n_rows " + String(node.scan_row_limit()) + "]"
    if streaming:
        if _is_scan(node.kind) or _stream_rows(node):
            label += " [stream]"
        elif (node.kind == AGG or node.kind == SELECT) and _stream_reductions(
            node.exprs
        ):
            label += " [stream aggregate state]"
        elif node.kind == JOIN and node.how in [
            JOIN_INNER,
            JOIN_LEFT,
            JOIN_SEMI,
            JOIN_ANTI,
            JOIN_CROSS,
        ]:
            if (
                plan_known_height(self, node.left) <= 65536
                and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                and prefer_left_build(
                    plan_known_height(self, node.left),
                    plan_known_height(self, node.right),
                )
            ):
                label += " [small left input; materialize unless compact progression]"
            else:
                label += " [stream probe; materialize build]"
        elif node.kind == SORT and node.sort_limit() >= 0:
            label += " [stream top-k state]"
        elif node.kind == SLICE and node.slice_offset() >= 0:
            label += " [ordered slice]"
        else:
            label += " [materialize]"
    out += pad + label + "\n"
    if node.left >= 0:
        plan_describe(self, node.left, depth + 1, out, streaming)
    if node.right >= 0:
        plan_describe(self, node.right, depth + 1, out, streaming)
