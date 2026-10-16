"""Structural file-failure prefixes never guess types or probe future files."""
# contract-group: public.accelerator_staging
# contract-group: public.apple_parity

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    scan_csv,
)
from dataframe.decimal import pow10
from dataframe.lazy import LazyFrame, SCAN_FRAME, JOIN, SELECT, _plan_node
from dataframe.join_type import JOIN_CROSS
from dataframe.accelerator.sources import (
    source_prefixes,
    file_source_nodes,
    prepare_file_sources,
    stage_file_sources,
)
from dataframe.accelerator.metadata import prune_projections


def _missing() raises -> LazyFrame:
    return scan_csv(
        "/tmp/mojo-source-prefixes-do-not-open/missing.csv",
        CsvSchema([CsvField("right", DataType.INT64)]),
    )


def _number(name: String, value: Int64) raises -> LazyFrame:
    return (
        DataFrame([Series(name, Column[Int64]([value]))])
        .lazy()
        .select(col(name).sum())
    )


def test_file_source_prefixes_closed_left_subtrees_keep_observation_order() raises:
    var left = _number("a", 11)
    var middle = _number("b", 22)
    var query = left.join(middle.join(_missing(), how="cross"), how="cross")
    var source = file_source_nodes(query)
    assert_equal(len(source), 1)
    var prefixes = source_prefixes(query, source[0])
    assert_equal(len(prefixes), 2)
    for index in range(2):
        assert_equal(len(file_source_nodes(prefixes[index])), 0)
        assert_equal(
            prefixes[index]._nodes[len(prefixes[index]._nodes) - 1].kind, SELECT
        )
    var first = prefixes[0].collect(
        engine="cpu", optimize=False, streaming=False
    )
    var second = prefixes[1].collect(
        engine="cpu", optimize=False, streaming=False
    )
    assert_equal(first.columns(), List[String](["a"]))
    assert_equal(first.item(0, "a").int64(), Int64(11))
    assert_equal(second.columns(), List[String](["b"]))
    assert_equal(second.item(0, "b").int64(), Int64(22))


def test_file_source_prefixes_preserve_earlier_shared_child_boundary() raises:
    var query = DataFrame([Series("a", Column[Int64]([11]))]).lazy()
    query._nodes[0].source_failure = "earlier source failure"
    query._frames.append(DataFrame([Series("b", Column[Int64]([22]))]))
    query._schemas.append(Optional[CsvSchema]())
    query._nodes.append(_plan_node(SCAN_FRAME, offset=1))
    query._nodes.append(_plan_node(SELECT, 1, exprs=[col("b")]))
    query._nodes.append(
        _plan_node(JOIN, 2, 0, names2=["_right"], text="cross", how=JOIN_CROSS)
    )
    var missing = _missing()
    var failed = missing._nodes[0].copy()
    failed.offset = len(query._frames)
    query._frames.extend(missing._frames.copy())
    query._schemas.extend(missing._schemas.copy())
    query._nodes.append(failed^)
    query._nodes.append(
        _plan_node(JOIN, 3, 4, names2=["_right"], text="cross", how=JOIN_CROSS)
    )
    query._nodes.append(
        _plan_node(JOIN, 0, 5, names2=["_right"], text="cross", how=JOIN_CROSS)
    )
    var prefixes = source_prefixes(query, 4)
    assert_equal(len(prefixes), 2)
    assert_equal(len(prefixes[0]._nodes), 1)
    assert_equal(prefixes[0]._nodes[0].source_failure, "earlier source failure")
    assert_equal(prefixes[1]._nodes[len(prefixes[1]._nodes) - 1].kind, JOIN)
    assert_equal(len(file_source_nodes(prefixes[1])), 0)
    # Removing the first prefix merely because the second also references
    # it would move its failure behind the second prefix's left operand.
    with assert_raises(contains="earlier source failure"):
        _ = prefixes[0].collect(engine="cpu", optimize=False, streaming=False)


def test_file_source_prefixes_empty_unreachable_and_invalid_indices() raises:
    var source = _missing()
    assert_equal(len(source_prefixes(source, 0)), 0)
    for index in [-1, 99]:
        with assert_raises(contains="outside the execution DAG"):
            _ = source_prefixes(source, index)
    source._frames.append(DataFrame([Series("x", Column[Int64]([1]))]))
    source._schemas.append(Optional[CsvSchema]())
    source._nodes.append(_plan_node(SCAN_FRAME, offset=1))
    with assert_raises(contains="not reachable"):
        _ = source_prefixes(source, 0)
    with assert_raises(contains="original file scan"):
        _ = source_prefixes(source, 1)


def _overflow() raises -> LazyFrame:
    var amount = Int128(6) * pow10(37)
    return (
        DataFrame(
            [
                Series("v", Column[Int128]([amount, amount])).with_dtype(
                    DataType.decimal(38, 0)
                ),
                Series("k", Column[Int64]([0, 0])),
            ]
        )
        .lazy()
        .select_exprs([col("v").sum().over("k").alias("total"), col("k")])
    )


def _error(query: LazyFrame, optimize: Bool = True) raises -> String:
    try:
        _ = query.collect(engine="cpu", optimize=optimize, streaming=False)
    except error:
        return String(error)
    raise Error("CPU fixture unexpectedly succeeded")


def _planning_error(query: LazyFrame) raises -> String:
    try:
        _ = query._optimized()
    except error:
        return String(error)
    raise Error("CPU planning fixture unexpectedly succeeded")


def test_file_source_prefixes_cpu_projection_metadata_precedes_data_errors() raises:
    var raw = _overflow().join(_missing(), how="cross")
    # A plain cross join needs no schema IO in these CPU optimizer passes:
    # its left producer reaches value evaluation before the missing right file.
    _ = raw._optimized()
    var pruned = raw.copy()
    prune_projections(pruned)
    assert_true("decimal precision" in _error(raw))
    # A projection above that same join asks the CPU optimizer which input
    # supplies the selected names. The missing right schema is now observed
    # during planning, before any decimal arithmetic. Prefix observation must
    # not run merely because a structurally earlier subtree is available.
    var projected = raw.select(col("total"))
    var planning = _planning_error(projected)
    assert_true("decimal precision" not in planning)
    assert_equal(_error(projected), planning)
    # Forced AcceleratorBackend.execute currently receives no optimize flag;
    # its source guard targets the default optimize=True CPU contract.
    assert_true("decimal precision" in _error(projected, optimize=False))
    var projected_pruned = projected.copy()
    with assert_raises():
        prune_projections(projected_pruned)


def test_file_source_prefixes_cpu_predicate_metadata_can_precede_gpu_pruning() raises:
    var missing = scan_csv(
        "/tmp/mojo-source-prefixes-do-not-open/missing.csv",
        CsvSchema([CsvField("k", DataType.INT64)]),
    )
    var query = _overflow().join(missing, on="k").filter(col("total") > 0)
    # CPU _push_predicates calls _passes -> _columns_of(left/right) for this
    # join filter, before _push_projections. The shared GPU projection pass
    # alone does not perform that earlier metadata observation.
    var pruned = query.copy()
    prune_projections(pruned)
    var planning = _planning_error(query)
    assert_true("decimal precision" not in planning)
    assert_equal(_error(query), planning)
    assert_true("decimal precision" in _error(query, optimize=False))


def test_file_source_preparation_unavailable_schema_keeps_real_prefixes() raises:
    var raw = _overflow().join(_missing(), how="cross")
    var expected_source = file_source_nodes(raw)[0]
    var prepared = prepare_file_sources(raw)
    assert_equal(prepared.failed_source, expected_source)
    assert_true(prepared.failure != "")
    assert_true("decimal precision" not in prepared.failure)
    assert_equal(prepared.staging_bytes, 0)
    assert_equal(prepared.scan_count, 1)
    assert_equal(len(prepared.prefixes), 1)
    # The unknown source stays a file scan; no empty column or guessed dtype
    # may replace it merely to make whole-query metadata binding succeed.
    assert_equal(len(file_source_nodes(prepared.query)), 1)
    assert_equal(len(file_source_nodes(prepared.prefixes[0])), 0)
    with assert_raises(contains="decimal precision"):
        _ = prepared.prefixes[0].collect(
            engine="cpu", optimize=False, streaming=False
        )
    var safe = _number("a", 11).join(_missing(), how="cross")
    var ready = prepare_file_sources(safe)
    assert_equal(len(ready.prefixes), 1)
    assert_equal(
        ready.prefixes[0]
        .collect(engine="cpu", optimize=False, streaming=False)
        .item(0, "a")
        .int64(),
        Int64(11),
    )
    assert_equal(ready.failure, _error(safe))
    # Compatibility callers without a GPU observer still receive the source
    # error; they never get a partially prepared successful query.
    with assert_raises():
        _ = stage_file_sources(raw)


def test_file_source_preparation_preserves_earlier_cpu_metadata_failure() raises:
    var raw = _overflow().join(_missing(), how="cross")
    var projected = raw.select(col("total"))
    var planning = _planning_error(projected)
    var observed = String()
    try:
        _ = prepare_file_sources(projected)
    except error:
        observed = String(error)
    assert_equal(observed, planning)
    var missing = scan_csv(
        "/tmp/mojo-source-prefixes-do-not-open/missing.csv",
        CsvSchema([CsvField("k", DataType.INT64)]),
    )
    var filtered = _overflow().join(missing, on="k").filter(col("total") > 0)
    planning = _planning_error(filtered)
    observed = ""
    try:
        _ = prepare_file_sources(filtered)
    except error:
        observed = String(error)
    assert_equal(observed, planning)


def _prefix_budget_left(failing: Bool) raises -> LazyFrame:
    var amount = Int128(6) * pow10(37)
    var right = amount if failing else -amount
    return (
        DataFrame(
            [
                Series("v", Column[Int128]([amount, right])).with_dtype(
                    DataType.decimal(38, 0)
                ),
                Series("k", Column[Int64]([0, 0])),
            ]
        )
        .lazy()
        .select_exprs([col("v").sum().over("k").alias("total"), col("k")])
    )


def _prefix_budget_query(failing: Bool) raises -> LazyFrame:
    # One distinct 1-MiB string belongs to a source visited AFTER the failed
    # CSV. It is an existing immutable input, not this prefix's dictionary.
    var pieces = List[String](
        length=16384,
        fill="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )
    var large = String("").join(pieces)
    var future = DataFrame([Series("future", StringColumn([large]))]).lazy()
    var tail = _missing().join(future, how="cross")
    return _prefix_budget_left(failing).join(tail, how="cross")


def test_file_source_prefixes_drop_future_dictionary_frames_and_keep_live_owner() raises:
    from dataframe.accelerator.dictionaries import (
        DictionaryCodes,
        has_dictionary_columns,
    )

    var query = _prefix_budget_query(True)
    var prepared = prepare_file_sources(query)
    assert_equal(len(prepared.prefixes), 1)
    var prefix = prepared.prefixes[0].copy()
    assert_equal(len(prefix._frames), 1)
    assert_equal(len(prefix._schemas), 1)
    assert_false(has_dictionary_columns(prefix))
    var dictionary = DictionaryCodes(prefix)
    # An empty dictionary owns its one 8-byte offset, no values or codes.
    assert_equal(dictionary.bytes, 8)
    assert_equal(dictionary.staging_bytes(prefix), 8)
    assert_true(
        prefix._frames[0].column("v")._data[Column[Int128]]._data.ptr()
        == query._frames[0].column("v")._data[Column[Int128]]._data.ptr()
    )
    for node in prefix._nodes:
        if node.kind == SCAN_FRAME:
            assert_equal(node.offset, 0)
    with assert_raises(contains="decimal precision"):
        _ = prefix.collect(engine="cpu", optimize=False, streaming=False)


def test_file_source_prefixes_remap_shared_scan_slots_and_preserve_failures() raises:
    var query = DataFrame([Series("x", Column[Int64]([11]))]).lazy()
    query._frames.append(query._frames[0].copy())
    query._schemas.append(query._schemas[0].copy())
    query._frames[0] = DataFrame([Series("unused", StringColumn(["future"]))])
    query._nodes[0].offset = 1
    query._nodes[0].source_failure = "shared prefix source sentinel"
    # Both live scans share shifted slot1; slot0 belongs to discarded work.
    # Preserve its measured cardinality once while dropping dead origins.
    query._origin_rows[0] = 999
    query._origin_rows[1] = 41
    query._nodes.append(_plan_node(SCAN_FRAME, offset=1))
    query._nodes.append(
        _plan_node(JOIN, 0, 1, names2=["_right"], text="cross", how=JOIN_CROSS)
    )
    var missing = _missing()
    var node = missing._nodes[0].copy()
    node.offset = 2
    query._origin_rows[2] = 321
    query._frames.extend(missing._frames.copy())
    query._schemas.extend(missing._schemas.copy())
    query._nodes.append(node^)
    query._nodes.append(
        _plan_node(JOIN, 2, 3, names2=["_right"], text="cross", how=JOIN_CROSS)
    )
    var prefixes = source_prefixes(query, 3)
    assert_equal(len(prefixes), 1)
    assert_equal(len(prefixes[0]._frames), 1)
    assert_equal(len(prefixes[0]._schemas), 1)
    assert_equal(len(prefixes[0]._origin_rows), 1)
    assert_equal(prefixes[0]._origin_rows[0], 41)
    assert_false(1 in prefixes[0]._origin_rows)
    assert_false(2 in prefixes[0]._origin_rows)
    assert_equal(prefixes[0]._spine_rows(0), 41)
    assert_equal(
        prefixes[0]._frames[0].column("x")._data[Column[Int64]]._get(0),
        Int64(11),
    )
    assert_equal(len(query._origin_rows), 3)
    assert_equal(query._nodes[0].offset, 1)
    assert_equal(query._origin_rows[1], 41)
    var scans = 0
    var marked = 0
    for source in prefixes[0]._nodes:
        if source.kind == SCAN_FRAME:
            scans += 1
            assert_equal(source.offset, 0)
            if source.source_failure:
                marked += 1
                assert_equal(
                    source.source_failure, "shared prefix source sentinel"
                )
    assert_equal(scans, 2)
    assert_equal(marked, 1)
    with assert_raises(contains="shared prefix source sentinel"):
        _ = prefixes[0].collect(engine="cpu", optimize=False, streaming=False)
