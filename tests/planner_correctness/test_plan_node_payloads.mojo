"""Flat payload compatibility, borrowed views, and operation-specific mutations."""
# contract-group: planner.node_payloads
from std.testing import assert_equal, assert_true, assert_false
from std.collections import Optional
from dataframe import col, Expr
from dataframe.dtype import DataType
from dataframe.lazy import _plan_node
from dataframe.planning.model import (
    PlanNode,
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
    ASOF,
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
from dataframe.planning.node_builders import (
    _frame_scan_node,
    _csv_source_node,
    _parquet_scan_node,
    _ipc_scan_node,
    _ndjson_scan_node,
    _filter_node,
    _select_node,
    _with_columns_node,
    _aggregate_node,
    _join_node,
    _sort_node,
    _slice_node,
    _unique_node,
    _drop_node,
    _explode_node,
    _unnest_node,
    _concat_node,
    _hconcat_node,
    _gather_node,
    _gather_every_node,
    _insert_column_node,
    _tail_node,
    _reverse_node,
    _shift_node,
    _row_index_node,
    _unpivot_node,
    _pivot_node,
    _grouping_sets_node,
    _dynamic_window_node,
    _rolling_window_node,
    _fill_null_strategy_node,
)


def _same_payload(actual: PlanNode, legacy: PlanNode) raises:
    assert_true(actual.payload_layout_valid())
    assert_equal(actual.kind, legacy.kind)
    assert_equal(actual.left, legacy.left)
    assert_equal(actual.right, legacy.right)
    assert_equal(actual.names, legacy.names)
    assert_equal(actual.names2, legacy.names2)
    assert_equal(actual.flags, legacy.flags)
    assert_equal(actual.offset, legacy.offset)
    assert_equal(actual.length, legacy.length)
    assert_equal(actual.text, legacy.text)
    assert_equal(actual.how, legacy.how)
    assert_equal(actual.right_keys, legacy.right_keys)
    assert_equal(actual.coalesce, legacy.coalesce)
    assert_equal(actual.maintain_order, legacy.maintain_order)
    assert_equal(actual.source_failure, legacy.source_failure)
    assert_equal(Bool(actual._executions), Bool(legacy._executions))
    assert_equal(Bool(actual.asof_tolerance), Bool(legacy.asof_tolerance))
    assert_equal(Bool(actual.json_schema), Bool(legacy.json_schema))
    assert_equal(len(actual.exprs), len(legacy.exprs))
    for i in range(len(actual.exprs)):
        assert_equal(actual.exprs[i]._name, legacy.exprs[i]._name)
        assert_equal(len(actual.exprs[i]._nodes), len(legacy.exprs[i]._nodes))
        for j in range(len(actual.exprs[i]._nodes)):
            assert_equal(
                actual.exprs[i]._nodes[j].op, legacy.exprs[i]._nodes[j].op
            )
            assert_equal(
                actual.exprs[i]._nodes[j].text, legacy.exprs[i]._nodes[j].text
            )


def test_source_builders_preserve_slots_defaults_and_reader_options() raises:
    _same_payload(_frame_scan_node(7), _plan_node(SCAN_FRAME, offset=7))
    _same_payload(
        _parquet_scan_node("data.parquet", 3),
        _plan_node(SCAN_PARQUET, text="data.parquet", offset=3),
    )
    _same_payload(
        _csv_source_node("data.csv"), _plan_node(SCAN_CSV, text="data.csv")
    )
    var csv = _csv_source_node(
        "data.csv",
        ["|", "'", "#", "utf8", "9", "23", "v", "Int64"],
        [False, True, False, True],
        ["NA"],
        2,
    )
    _same_payload(
        csv,
        _plan_node(
            SCAN_CSV,
            text="data.csv",
            offset=2,
            names2=["|", "'", "#", "utf8", "9", "23", "v", "Int64"],
            flags=[False, True, False, True],
            right_keys=["NA"],
        ),
    )
    assert_equal(csv.csv_separator(), "|")
    assert_equal(csv.csv_quote_char(), "'")
    assert_equal(csv.csv_comment_prefix(), "#")
    assert_equal(csv.csv_encoding(), "utf8")
    assert_equal(csv.csv_skip_rows(), "9")
    assert_equal(csv.csv_infer_schema_length(), "23")
    assert_false(csv.csv_has_header())
    assert_true(csv.csv_ignore_errors())
    assert_false(csv.csv_truncate_ragged_lines())
    assert_true(csv.csv_try_parse_dates())
    assert_equal(csv.csv_override_count(), 1)
    assert_equal(csv.csv_override_name(0), "v")
    assert_equal(csv.csv_override_dtype(0), "Int64")
    assert_equal(csv.csv_null_values(), List[String](["NA"]))
    var ipc = _ipc_scan_node("data.arrow", False, 4)
    _same_payload(
        ipc,
        _plan_node(
            SCAN_IPC,
            text="data.arrow",
            offset=4,
            names2=["0"],
            flags=[False, False],
        ),
    )
    assert_false(ipc.ipc_memory_map())
    assert_false(ipc.ipc_has_projection())
    assert_equal(ipc.ipc_row_offset(), "0")
    var json = _ndjson_scan_node(
        "data.ndjson", Optional[DataType](), 17, True, 5
    )
    _same_payload(
        json,
        _plan_node(
            SCAN_NDJSON,
            text="data.ndjson",
            offset=5,
            names2=["17"],
            flags=[True],
        ),
    )
    assert_equal(json.ndjson_infer_schema_length(), "17")
    assert_true(json.ndjson_ignore_errors())


def test_relational_builders_preserve_expression_keys_and_sort_layout() raises:
    var exprs: List[Expr] = [col("v").sum().alias("total")]
    _same_payload(
        _filter_node([col("k")], 2), _plan_node(FILTER, 2, exprs=[col("k")])
    )
    _same_payload(_select_node(exprs, 2), _plan_node(SELECT, 2, exprs=exprs))
    _same_payload(
        _with_columns_node(exprs, 2), _plan_node(WITH_COLUMNS, 2, exprs=exprs)
    )
    _same_payload(
        _aggregate_node(["k"], exprs, True, 2),
        _plan_node(AGG, 2, names=["k"], exprs=exprs, maintain_order=True),
    )
    var join = _join_node(["a"], ["b"], "unknown", -1, "_rhs", False, 2, 6)
    _same_payload(
        join,
        _plan_node(
            JOIN,
            2,
            6,
            names=["a"],
            right_keys=["b"],
            text="unknown",
            how=-1,
            names2=["_rhs"],
            coalesce=False,
        ),
    )
    assert_equal(join.join_left_keys(), List[String](["a"]))
    assert_equal(join.join_right_keys(), List[String](["b"]))
    assert_equal(join.join_suffix(), "_rhs")
    var sorted = _sort_node(["a", "b"], [True, False], [False, True], True, 2)
    _same_payload(
        sorted,
        _plan_node(
            SORT,
            2,
            names=["a", "b"],
            flags=[True, False, False, True],
            maintain_order=True,
        ),
    )
    assert_true(sorted.sort_descending(0))
    assert_false(sorted.sort_descending(1))
    assert_false(sorted.sort_nulls_last(0))
    assert_true(sorted.sort_nulls_last(1))
    _same_payload(
        _unique_node(["a"], "last", True, 2),
        _plan_node(UNIQUE, 2, names=["a"], text="last", maintain_order=True),
    )
    _same_payload(_drop_node(["a"], 2), _plan_node(DROP, 2, names=["a"]))
    _same_payload(
        _grouping_sets_node(
            ["a", "b"], exprs, [True, True, True, False, False, False], 2
        ),
        _plan_node(
            GROUPING_SETS,
            2,
            names=["a", "b"],
            exprs=exprs,
            flags=[True, True, True, False, False, False],
        ),
    )


def test_row_and_reshape_builders_keep_distinct_integer_and_name_meanings() raises:
    _same_payload(
        _slice_node(-3, 2, 1), _plan_node(SLICE, 1, offset=-3, length=2)
    )
    _same_payload(_tail_node(-3, 1), _plan_node(TAIL, 1, offset=-3))
    _same_payload(
        _shift_node(-3, [col("v")], 1),
        _plan_node(SHIFT, 1, offset=-3, exprs=[col("v")]),
    )
    _same_payload(_reverse_node(1), _plan_node(REVERSE, 1))
    _same_payload(
        _row_index_node("row", 41, 1),
        _plan_node(ROW_INDEX, 1, names=["row"], offset=41),
    )
    _same_payload(
        _gather_every_node(3, 2, 1),
        _plan_node(GATHER_EVERY, 1, length=3, offset=2),
    )
    _same_payload(
        _insert_column_node(2, 1, 4), _plan_node(INSERT_COLUMN, 1, 4, offset=2)
    )
    _same_payload(
        _gather_node(True, 1, 4), _plan_node(GATHER, 1, 4, flags=[True])
    )
    _same_payload(_concat_node(1, 4), _plan_node(CONCAT, 1, 4))
    _same_payload(_hconcat_node(1, 4), _plan_node(HCONCAT, 1, 4))
    _same_payload(
        _explode_node(["items"], 1), _plan_node(EXPLODE, 1, names=["items"])
    )
    _same_payload(_unnest_node("obj", 1), _plan_node(UNNEST, 1, text="obj"))
    _same_payload(
        _unpivot_node(["v"], ["k"], "metric", "reading", 1),
        _plan_node(
            UNPIVOT,
            1,
            names=["v"],
            names2=["k"],
            right_keys=["metric", "reading"],
        ),
    )
    var pivot = _pivot_node(
        ["k", "j"], "metric", "reading", ["x", "y"], "sum", 1
    )
    _same_payload(
        pivot,
        _plan_node(
            PIVOT,
            1,
            names=["k", "j", "metric", "reading"],
            offset=2,
            right_keys=["x", "y"],
            text="sum",
        ),
    )
    assert_equal(pivot.pivot_index_count(), 2)
    assert_equal(pivot.pivot_on(), "metric")
    assert_equal(pivot.pivot_values(), "reading")
    _same_payload(
        _fill_null_strategy_node("forward", 2, ["v"], 1),
        _plan_node(
            FILL_NULL_STRATEGY, 1, text="forward", length=2, names=["v"]
        ),
    )


def test_window_builders_and_asof_conversion_preserve_complete_settings() raises:
    var dynamic = _dynamic_window_node(
        "t", ["k"], "2d", "3d", "-1d", "both", "right", 2
    )
    _same_payload(
        dynamic,
        _plan_node(
            GROUP_BY_DYNAMIC,
            2,
            names=["t", "k"],
            names2=["2d", "3d", "-1d", "both", "right"],
        ),
    )
    assert_equal(dynamic.window_every(), "2d")
    assert_equal(dynamic.window_period(), "3d")
    assert_equal(dynamic.window_offset(), "-1d")
    assert_equal(dynamic.window_closed(), "both")
    assert_equal(dynamic.window_label(), "right")
    _same_payload(
        _rolling_window_node("t", ["k"], "3d", "-3d", "right", 2),
        _plan_node(
            ROLLING,
            2,
            names=["t", "k"],
            names2=["", "3d", "-3d", "right", "datapoint"],
        ),
    )
    dynamic.set_window_aggregations([col("v").sum()], 37)
    assert_equal(dynamic.window_batch_size(), 37)
    assert_equal(len(dynamic.exprs), 1)
    var asof = _join_node(["t"], ["r_t"], "inner", 0, "_rhs", True, 2, 6)
    asof.set_asof_options("nearest", False, None, ["k", "j"])
    _same_payload(
        asof,
        _plan_node(
            ASOF,
            2,
            6,
            names=["t"],
            right_keys=["r_t"],
            text="nearest",
            how=0,
            names2=["_rhs", "k", "j"],
            flags=[False],
        ),
    )
    assert_equal(asof.join_suffix(), "_rhs")
    assert_equal(asof.asof_group_count(), 2)
    assert_equal(asof.asof_group(0), "k")
    assert_equal(asof.asof_group(1), "j")
    assert_false(asof.asof_allow_exact_matches())


def test_named_mutations_preserve_copies_and_independent_slot_domains() raises:
    var ipc = _ipc_scan_node("data.arrow", False, 7)
    var original = ipc.copy()
    ipc.set_scan_projection(List[String]())
    ipc.set_ipc_slice("123", 19)
    ipc.set_source_slot(9)
    assert_true(ipc.ipc_has_projection())
    assert_equal(len(ipc.scan_projection()), 0)
    assert_equal(ipc.ipc_row_offset(), "123")
    assert_equal(ipc.scan_row_limit(), 19)
    assert_equal(ipc.source_slot(), 9)
    assert_false(original.ipc_has_projection())
    assert_equal(original.ipc_row_offset(), "0")
    assert_equal(original.scan_row_limit(), -1)
    assert_equal(original.source_slot(), 7)
    var sorted = _sort_node(["a"], [False], [True])
    sorted.set_sort_limit(13)
    assert_equal(sorted.sort_limit(), 13)
    assert_false(sorted.sort_descending(0))
    assert_true(sorted.sort_nulls_last(0))
    var join = _join_node(["a"], ["b"], "inner", 0)
    join.set_join_build_slot(21)
    assert_equal(join.join_build_slot(), 21)
    assert_equal(join.join_suffix(), "_right")
    var predicate = _filter_node([col("a")])
    assert_false(predicate.filter_is_pinned())
    predicate.pin_filter()
    assert_true(predicate.filter_is_pinned())


def test_layout_validation_is_structural_and_legacy_factory_stays_permissive() raises:
    # Malformed storage is detectable without treating invalid user options as
    # malformed storage: those retain their existing eager/deferred error timing.
    assert_false(
        _plan_node(SORT, names=["a"], flags=[True]).payload_layout_valid()
    )
    assert_false(
        _plan_node(SCAN_IPC, names2=["0"], flags=[True]).payload_layout_valid()
    )
    assert_false(_plan_node(SCAN_CSV, names2=["x"]).payload_layout_valid())
    assert_false(
        _plan_node(
            GROUP_BY_DYNAMIC, names=["t"], names2=["1d"]
        ).payload_layout_valid()
    )
    assert_false(
        _plan_node(
            GROUPING_SETS, names=["a", "b"], flags=[True]
        ).payload_layout_valid()
    )
    assert_false(
        _plan_node(PIVOT, names=["k", "v"], offset=1).payload_layout_valid()
    )
    assert_false(
        _plan_node(UNPIVOT, right_keys=["variable"]).payload_layout_valid()
    )
    assert_true(_slice_node(0, -9).payload_layout_valid())
    assert_true(
        _dynamic_window_node(
            "t", [], "invalid", "invalid", "invalid", "invalid", "invalid"
        ).payload_layout_valid()
    )
    assert_true(_join_node(["a"], ["b"], "invalid", -1).payload_layout_valid())
