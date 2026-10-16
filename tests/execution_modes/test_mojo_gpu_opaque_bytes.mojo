"""Public Binary/Geometry GPU transport and raw-key contracts."""
from std.memory import ArcPointer, bitcast
from std.math import abs
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    null,
    when,
    from_wkb,
)
from dataframe.metal import MetalRuntime
from dataframe.lazy import LazyFrame, SCAN_FRAME
from dataframe.accelerator.opaque import (
    prepare,
    restoration_bytes,
    has_opaque_columns,
)
from dataframe.accelerator.windows import lower_partition_aggregates
from dataframe.accelerator.joins import lower_join_dag, DAG_ROWS
from dataframe.accelerator.rows import lower_rows, ROW_GROUP_KEY
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.accelerator.opaque_dictionary import hex_key
from tests.support.coverage_generators import offset_layout, chunk_layout


def _binary() raises -> Series:
    var rows: List[List[UInt8]] = [
        [255],
        [],
        [0],
        [0, 0],
        [0, 255],
        [127],
        [128],
        [195, 40],
        [237, 160, 128],
        [97, 0, 98],
        [255],
        [222, 173],
    ]
    var long_a = List[UInt8](length=24, fill=97)
    long_a.extend([UInt8(0), UInt8(255)])
    rows.append(long_a^)
    var long_b = List[UInt8](length=24, fill=97)
    long_b.extend([UInt8(0), UInt8(254)])
    rows.append(long_b^)
    rows.append(List[UInt8](length=25, fill=97))
    var valid = List[Bool](length=len(rows), fill=True)
    valid[11] = False
    return Series.binary("b", rows^, valid^)


def _assert_bytes(actual: Series, expected: Series) raises:
    assert_equal(actual.dtype(), expected.dtype())
    assert_equal(len(actual), len(expected))
    var left = actual.rechunk()
    var right = expected.rechunk()
    ref a = left._data[StringColumn]
    ref b = right._data[StringColumn]
    for row in range(len(actual)):
        assert_equal(a._valid(row), b._valid(row))
        if a._valid(row):
            assert_equal(len(a._row_bytes(row)), len(b._row_bytes(row)))
            for byte in range(len(a._row_bytes(row))):
                assert_equal(a._row_bytes(row)[byte], b._row_bytes(row)[byte])


def _ids(frame: DataFrame, expected: List[Int64]) raises:
    assert_equal(frame.column("id").int64().to_list(), expected)


def test_mojo_gpu_binary_opaque_raw_bytes_literals_stable_orders_layouts_and_metadata() raises:
    var source = _binary()
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    var ids = List[Int64]()
    var labels = List[String]()
    for row in range(len(source)):
        ids.append(Int64(row))
        labels.append("00" if row % 2 else "ff")
    var original = DataFrame(
        [
            source.copy(),
            Series("id", Column[Int64](ids^)),
            Series("text", StringColumn(labels^)),
        ]
    )
    var expected: List[Int64] = [
        1,
        2,
        3,
        4,
        9,
        13,
        12,
        14,
        5,
        6,
        7,
        8,
        0,
        10,
        11,
    ]
    for variant in range(3):
        var frame = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var query = frame.lazy().sort(["b"], nulls_last=True)
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        _ids(actual, expected)
        assert_true(actual.equals(query.collect(engine="cpu", streaming=False)))
        assert_true(
            actual.column("b")._field_metadata.value().ptr()
            == source._field_metadata.value().ptr()
        )
        var descending = frame.lazy().sort(
            ["b"], descending=True, nulls_last=False
        )
        actual = descending.collect(accelerator=MetalRuntime(), engine="accel")
        _ids(
            actual,
            [
                Int64(11),
                Int64(0),
                Int64(10),
                Int64(8),
                Int64(7),
                Int64(6),
                Int64(5),
                Int64(14),
                Int64(12),
                Int64(13),
                Int64(9),
                Int64(4),
                Int64(3),
                Int64(2),
                Int64(1),
            ],
        )
        assert_true(
            actual.equals(descending.collect(engine="cpu", streaming=False))
        )
        for literal in ["\x00", "a\x00b", "absent"]:
            var selected = (
                frame.lazy()
                .filter(col("b") == lit(literal))
                .select_exprs([col("b"), col("id"), col("text")])
            )
            actual = selected.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_true(
                actual.equals(selected.collect(engine="cpu", streaming=False))
            )
            if literal == "\x00":
                _ids(actual, [Int64(2)])
            elif literal == "a\x00b":
                _ids(actual, [Int64(9)])
            else:
                assert_equal(actual.height(), 0)
        var compared = frame.lazy().select_exprs(
            [
                (col("b") < lit("a\x00b")).alias("less"),
                (col("text") == lit("ff")).alias("text_equal"),
                col("b").alias("renamed"),
            ]
        )
        actual = compared.collect(accelerator=MetalRuntime(), engine="accel")
        assert_true(
            actual.equals(compared.collect(engine="cpu", streaming=False))
        )
        assert_equal(actual.column("renamed").dtype(), DataType.BINARY)
        assert_true(
            actual.column("renamed")._field_metadata.value().ptr()
            == source._field_metadata.value().ptr()
        )
        var pipeline = (
            frame.lazy()
            .filter(col("id") != lit(Int64(11)))
            .sort(["b"], nulls_last=True)
            .slice(2, 4)
            .select_exprs([col("b"), col("id")])
        )
        actual = pipeline.collect(accelerator=MetalRuntime(), engine="accel")
        _ids(actual, [Int64(3), Int64(4), Int64(9), Int64(13)])
        assert_true(
            actual.equals(pipeline.collect(engine="cpu", streaming=False))
        )


def test_mojo_gpu_binary_join_grouped_windows_and_raw_hash_provenance() raises:
    var source = _binary()
    var ids = List[Int64]()
    for row in range(len(source)):
        ids.append(Int64(row))
    var left = DataFrame([source.copy(), Series("id", Column[Int64](ids^))])
    var right = DataFrame(
        [
            Series.binary(
                "b",
                [[UInt8(255)], [UInt8(0), UInt8(0)], []],
                [True, True, False],
            ),
            Series("rid", Column[Int64]([0, 1, 2])),
        ]
    )
    for how in ["inner", "left", "full", "semi", "anti"]:
        var query = left.lazy().join(right.lazy(), on="b", how=how)
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_true(actual.equals(query.collect(engine="cpu", streaming=False)))
        assert_equal(actual.column("b").dtype(), DataType.BINARY)
        if how == "inner":
            _ids(actual, [Int64(0), Int64(3), Int64(10)])
    var grouped = (
        left.lazy()
        .group_by("b", maintain_order=True)
        .agg([col("id").len().alias("count"), col("id").sum().alias("total")])
    )
    var actual = grouped.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(actual.equals(grouped.collect(engine="cpu", streaming=False)))
    assert_equal(actual.height(), 14)
    assert_equal(actual.item(0, "count").int64(), Int64(2))
    assert_equal(actual.item(0, "total").int64(), Int64(10))
    var reductions = left.lazy().select_exprs(
        [
            col("b").min().alias("min"),
            col("b").max().alias("max"),
            col("b").first().alias("first"),
            col("b").last().alias("last"),
            col("b").count().alias("count"),
            col("b").null_count().alias("nulls"),
            col("b").arg_min().alias("arg_min"),
            col("b").arg_max().alias("arg_max"),
        ]
    )
    var reduced = reductions.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(
        reduced.equals(reductions.collect(engine="cpu", streaming=False))
    )
    _assert_bytes(reduced.column("min"), Series.binary("min", [[]]))
    _assert_bytes(reduced.column("max"), Series.binary("max", [[UInt8(255)]]))
    _assert_bytes(
        reduced.column("first"), Series.binary("first", [[UInt8(255)]])
    )
    _assert_bytes(
        reduced.column("last"),
        Series.binary("last", [List[UInt8](length=25, fill=97)]),
    )
    assert_equal(reduced.item(0, "count").int64(), Int64(14))
    assert_equal(reduced.item(0, "nulls").int64(), Int64(1))
    assert_equal(reduced.item(0, "arg_min").uint32(), UInt32(1))
    assert_equal(reduced.item(0, "arg_max").uint32(), UInt32(0))
    # Original raw key hashes, not generated hex or numeric/categorical codes,
    # remain in the window descriptors after opaque recoding.
    var key_values: List[List[UInt8]] = [[255], [], [0], [195, 40]]
    var keys = List[List[UInt8]]()
    var values = List[Float64]()
    for row in range(65537):
        keys.append(key_values[row % 4].copy())
        values.append(Float64(row % 17) / 8 + 1048576)
    var frame = DataFrame(
        [Series.binary("b", keys^), Series("x", Column[Float64](values^))]
    )
    var query = frame.lazy().select_exprs(
        [
            col("b"),
            col("x").mean().over("b").alias("mean"),
            col("x").var().over("b").alias("var"),
        ]
    )
    var normalized = lower_partition_aggregates(query)
    var missing_hash = normalized.copy()
    for operation in range(len(missing_hash._nodes)):
        for key in range(len(missing_hash._nodes[operation].window_keys)):
            missing_hash._nodes[operation].window_keys[key].string_hash = False
    with assert_raises(contains="missing raw byte hash provenance"):
        _ = prepare(missing_hash)
    var staged = prepare(normalized)
    var captured = False
    for operation in staged.query._nodes:
        for key in operation.window_keys:
            if key.dtype._is_bytes():
                captured = True
                assert_equal(key.dtype, DataType.BINARY)
                assert_true(key.string_hash)
                assert_equal(len(key.values[]), 4)
                var code = Int(
                    staged.dictionary[].code(
                        Span[UInt8, ImmutAnyOrigin](key_values[0])
                    )
                )
                assert_equal(key.values[][code], UInt64(0xCAF29CE4842223DA))
    assert_true(captured)
    var dag = lower_join_dag(staged.query, capabilities())
    var descriptor = False
    for task in dag.tasks:
        if task.kind == DAG_ROWS:
            var plan = lower_rows(
                task.query, capabilities(), terminal_boundary=task.terminal
            )
            for node in range(len(plan.node_dtypes)):
                if plan.code[4 * node] == ROW_GROUP_KEY:
                    var packed = bitcast[DType.uint64](plan.literals[node])
                    if packed & (UInt64(1) << 63):
                        descriptor = True
                        assert_false(Bool(packed & (UInt64(1) << 62)))
    assert_true(descriptor)
    var cpu = query.collect(engine="cpu", streaming=False)
    actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    _assert_bytes(actual.column("b"), cpu.column("b"))
    for name in ["mean", "var"]:
        for row in range(frame.height()):
            assert_true(
                abs(
                    actual.item(row, name).float64()
                    - cpu.item(row, name).float64()
                )
                < 1e-12
            )


def _geometry(crs: String) raises -> Series:
    var little: List[UInt8] = [1, 1, 0, 0, 0]
    little.extend(List[UInt8](length=16, fill=0))
    var big: List[UInt8] = [0, 0, 0, 0, 1]
    big.extend(List[UInt8](length=16, fill=0))
    var negative = little.copy()
    negative[12] = 128
    return from_wkb(
        Series.binary(
            "g",
            [little.copy(), big.copy(), negative.copy(), []],
            [True, True, True, False],
        ),
        crs,
    )


def test_mojo_gpu_geometry_transport_preserves_wkb_identity_crs_and_field_metadata() raises:
    var source = _geometry('{"crs":"EPSG:4326"}')
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    var frame = DataFrame(
        [source.copy(), Series("id", Column[Int64]([0, 1, 2, 3]))]
    )
    for variant in range(3):
        var input = (
            frame.copy() if variant
            == 0 else offset_layout(frame) if variant
            == 1 else chunk_layout(frame)
        )
        var query = input.lazy().sort(["g"], nulls_last=True)
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        _ids(actual, [Int64(1), Int64(0), Int64(2), Int64(3)])
        assert_true(actual.equals(query.collect(engine="cpu", streaming=False)))
        assert_equal(actual.column("g").dtype(), source.dtype())
        assert_true(
            actual.column("g")._field_metadata.value().ptr()
            == source._field_metadata.value().ptr()
        )
    var grouped = (
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(col("id").len().alias("count"))
    )
    var result = grouped.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(result.height(), 4)
    for row in range(4):
        assert_equal(result.item(row, "count").int64(), Int64(1))
    assert_true(result.equals(grouped.collect(engine="cpu", streaming=False)))
    var mismatch = frame.lazy().join(
        DataFrame([_geometry('{"crs":"EPSG:3857"}')]).lazy(), on="g"
    )
    with assert_raises(contains="Join key dtypes differ"):
        _ = mismatch.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
    with assert_raises(contains="Join key dtypes differ"):
        _ = mismatch.collect(engine="cpu", streaming=False)


def _dead_byte_branch(live: LazyFrame) raises -> LazyFrame:
    # Dead invalid conversion and a substantial payload must affect neither
    # capability admission nor live staging/budget/source-origin ownership.
    var raw = List[UInt8](length=8192, fill=255)
    var result = (
        DataFrame([Series.binary("dead", [raw.copy(), raw.copy()])])
        .lazy()
        .select(col("dead").cast(DataType.STRING))
    )
    result._nodes[0].source_failure = "unreachable byte source failure"
    result._origin_rows[0] = 999999
    var node_offset = len(result._nodes)
    var frame_offset = len(result._frames)
    for frame in live._frames:
        result._frames.append(frame.copy())
    for schema in live._schemas:
        result._schemas.append(schema.copy())
    for entry in live._origin_rows.items():
        result._origin_rows[frame_offset + entry.key] = entry.value
    for operation in live._nodes:
        var node = operation.copy()
        if node.left >= 0:
            node.left += node_offset
        if node.right >= 0:
            node.right += node_offset
        if node.kind == SCAN_FRAME:
            node.offset += frame_offset
        result._nodes.append(node^)
    return result^


def _dead_byte_branches_preserve_live_results_and_budget() raises:
    var live = DataFrame([_binary()]).lazy().select(col("b"))
    live._origin_rows[0] = 12345
    var combined = _dead_byte_branch(live)
    assert_true(has_opaque_columns(combined))
    var expected = prepare(live)
    var actual = prepare(combined, memory_limit_bytes=expected.staging_bytes)
    assert_equal(actual.staging_bytes, expected.staging_bytes)
    assert_equal(len(actual.query._frames), 1)
    assert_equal(len(actual.query._nodes), len(live._nodes))
    assert_equal(actual.query._nodes[0].offset, 0)
    assert_equal(actual.query._nodes[0].source_failure, "")
    assert_equal(actual.query._origin_rows[0], 12345)
    assert_equal(actual.dictionary[].keys, expected.dictionary[].keys)
    with assert_raises(contains="opaque staging"):
        _ = prepare(combined, memory_limit_bytes=expected.staging_bytes - 1)
    var cpu = combined.collect(engine="cpu", streaming=False)
    var measured = live.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var gpu = combined.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
    )
    _assert_bytes(gpu.column("b"), cpu.column("b"))
    _assert_bytes(gpu.column("b"), measured[0].column("b"))
    # A root without bytes must not activate opaque normalization at all.
    var numeric = (
        DataFrame([Series("x", Column[Int32]([2, 5, 8]))])
        .lazy()
        .select((col("x") + lit(Int32(1))).alias("x"))
    )
    var dead_numeric = _dead_byte_branch(numeric)
    assert_false(has_opaque_columns(dead_numeric))
    var compact = prepare(dead_numeric, memory_limit_bytes=8)
    assert_equal(compact.staging_bytes, 8)
    assert_equal(len(compact.dictionary[].keys), 0)
    assert_equal(len(compact.query._frames), 1)
    var numeric_measured = numeric.profile(
        accelerator=MetalRuntime(), engine="accel"
    )
    var numeric_peak_field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in numeric_measured[1].columns() else "peak_requested_bytes"
    )
    var numeric_peak = Int(
        numeric_measured[1].item(0, numeric_peak_field).int64()
    )
    var numeric_result = dead_numeric.collect(
        accelerator=MetalRuntime(memory_limit_bytes=numeric_peak),
        engine="accel",
    )
    assert_true(
        numeric_result.equals(
            dead_numeric.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(
        numeric_result.column("x").int32().to_list(),
        [Int32(3), Int32(6), Int32(9)],
    )


def test_mojo_gpu_opaque_budgets_observers_empty_inputs_and_explicit_refusals() raises:
    _dead_byte_branches_preserve_live_results_and_budget()
    var source = _binary()
    var query = (
        DataFrame([source.copy()])
        .lazy()
        .select(
            when(col("b").is_not_null())
            .then(col("b"))
            .otherwise(null(DataType.BINARY))
            .alias("b")
        )
    )
    var plan = prepare(query)
    assert_true(plan.staging_bytes > 0)
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    assert_equal(
        measured[1].item(0, "opaque_staging_bytes").int64(),
        Int64(plan.staging_bytes),
    )
    assert_true(measured[1].item(0, "opaque_restoration_bytes").int64() > 0)
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ).equals(measured[0])
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=limited, engine="accel")
    assert_equal(
        DataFrame([Series("x", Column[Int64]([7]))])
        .lazy()
        .select(col("x"))
        .collect(accelerator=limited, engine="accel")
        .item()
        .int64(),
        Int64(7),
    )
    with assert_raises(contains="opaque staging"):
        _ = prepare(query, memory_limit_bytes=plan.staging_bytes - 1)
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = query.profile(accelerator=observer, engine="accel")
    assert_equal(observed[0].height(), 0)
    assert_equal(observed[1].item(0, "host_result_bytes").int64(), Int64(0))
    assert_equal(
        observed[1].item(0, "opaque_restoration_bytes").int64(), Int64(0)
    )
    assert_equal(
        observed[1].item(0, "download_bytes").int64(), Int64(len(source) * 4)
    )
    assert_true(observed[1].item(0, "kernel_launches").int64() > 0)
    var empty = (
        DataFrame([source.head(0)])
        .lazy()
        .select(col("b"))
        .collect(accelerator=MetalRuntime(device_id=999999), engine="accel")
    )
    assert_equal(empty.height(), 0)
    assert_equal(empty.column("b").dtype(), DataType.BINARY)
    assert_true(
        "opaque byte transport"
        in MetalRuntime(device_id=999999).describe(query)
    )
    with assert_raises(contains="byte operation"):
        _ = prepare(
            DataFrame([source.copy()])
            .lazy()
            .select(col("b").cast(DataType.STRING))
        )
    var incompatible = DataFrame(
        [
            source.copy(),
            Series(
                "s", StringColumn(List[String](length=len(source), fill="ff"))
            ),
        ]
    )
    with assert_raises(contains="matching dtypes"):
        _ = prepare(incompatible.lazy().select(col("b") == col("s")))
    var nulls = (
        DataFrame([Series.binary("b", [[], []], [False, False])])
        .lazy()
        .select(col("b"))
    )
    var null_result = nulls.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(null_result.height(), 2)
    assert_equal(null_result.column("b").null_count(), 2)
