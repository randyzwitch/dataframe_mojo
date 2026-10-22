"""Public scoped List EXPLODE; requires the logical parent-count CPU repair."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, ListColumn, Series, col, lit
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.gpu.runtime import capabilities
from dataframe_metal.accelerator.joins import (
    lower_join_dag,
    DAG_SOURCE_FIELDS,
    DAG_EXPLODE,
)
from tests.support.resident_explode_reference import _source
from tests.support.coverage_generators import offset_layout, chunk_layout


def _literal(output: DataFrame) raises:
    assert_equal(output.height(), 5)
    assert_equal(
        output.column("id").int64().to_list(),
        List[Int64]([Int64(10), Int64(11), Int64(12), Int64(12), Int64(13)]),
    )
    for row in range(5):
        assert_equal(output.item(row, "xs").is_null(), row != 3)
        assert_equal(output.item(row, "wide").is_null(), row != 3)
    assert_equal(output.item(3, "xs").int64(), Int64(10))
    assert_equal(output.column("wide")._decimal_raw(3), (Int128(1) << 80) + 7)


# requires: metal
def test_mojo_gpu_explode_source_layouts_hidden_spans_high_words_and_metadata() raises:
    var original = _source()
    for variant in range(3):
        var source = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var query = source.lazy().explode(["xs", "wide"])
        var counter = ArcPointer(Int(0))
        query._nodes[0]._executions = counter
        var bound = lower_join_dag(query, capabilities())
        assert_equal(len(bound.tasks), 2)
        assert_equal(bound.tasks[0].kind, DAG_SOURCE_FIELDS)
        assert_equal(bound.tasks[1].kind, DAG_EXPLODE)
        assert_equal(counter[], 0)
        var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
        assert_equal(counter[], 1)
        assert_equal(
            measured[1].item(1, "peak_requested_bytes").int64(),
            measured[1].item(0, "query_peak_requested_bytes").int64(),
        )
        _literal(measured[0])
        var cpu = query.collect(engine="cpu", streaming=False, optimize=False)
        _literal(cpu)
        assert_true(measured[0].equals(cpu))
        for name in ["xs", "wide"]:
            assert_true(
                measured[0].column(name)._field_metadata.value().ptr()
                == source.column(name)
                .list_column()
                .child()
                ._field_metadata.value()
                .ptr()
            )
        assert_equal(
            measured[1].item(0, "algorithm").string(), "mojo_gpu_source_fields"
        )
        assert_equal(
            measured[1].item(1, "algorithm").string(),
            "mojo_gpu_primitive_explode",
        )
        assert_equal(measured[1].item(0, "host_result_bytes").int64(), Int64(0))
        assert_equal(
            measured[1].item(1, "host_result_bytes").int64(),
            Int64(5 * (8 + 16 + 8) + 3),
        )
        assert_equal(
            measured[1].item(0, "query_upload_bytes").int64(),
            Int64(456 if variant == 1 else 372),
        )
        assert_equal(
            measured[1].item(0, "query_download_bytes").int64(), Int64(379)
        )
        assert_equal(
            measured[1].item(0, "query_host_result_bytes").int64(), Int64(163)
        )
        if variant == 2:
            # Use a fresh source because the CPU reference may cache chunks.
            var fresh = chunk_layout(original)
            _ = (
                fresh.lazy()
                .explode(["xs", "wide"])
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            for field in fresh._columns:
                assert_equal(field._chunked.value()[].cache[].state.load(), 0)


# requires: metal
def test_mojo_gpu_explode_resident_row_join_group_and_complete_cardinality_observation() raises:
    var source = _source()
    var expanded = source.lazy().explode(["xs", "wide"])
    var query = (
        expanded.filter(col("xs").is_not_null())
        .with_columns(
            (col("xs").cast(DataType.FLOAT64) + lit(Float64(1))).alias("plus")
        )
        .sort("id")
        .slice(0, 1)
        .select(col("plus").sum())
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(actual.item().float64(), Float64(11))
    assert_true(
        actual.equals(
            query.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var right = DataFrame(
        [
            Series("id", Column[Int64]([12, 10])),
            Series("tag", Column[Int32]([120, 100])),
        ]
    )
    var joined = expanded.join(right.lazy(), on=["id"], how="inner")
    var measured = joined.profile(accelerator=MetalRuntime(), engine="accel")
    assert_equal(
        measured[0].column("id").int64().to_list(),
        List[Int64]([Int64(10), Int64(12), Int64(12)]),
    )
    assert_equal(
        measured[0].column("tag").int32().to_list(),
        List[Int32]([Int32(100), Int32(120), Int32(120)]),
    )
    assert_equal(
        measured[0].column("wide")._decimal_raw(2), (Int128(1) << 80) + 7
    )
    assert_true(
        measured[0].equals(
            joined.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var grouped = joined.group_by("id", maintain_order=True).agg(
        col("xs").count().alias("count")
    )
    var counts = grouped.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(
        counts.column("id").int64().to_list(),
        List[Int64]([Int64(10), Int64(12)]),
    )
    assert_equal(
        counts.column("count").int64().to_list(),
        List[Int64]([Int64(0), Int64(1)]),
    )
    assert_true(
        counts.equals(
            grouped.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    # Empty and singleton each emit one row, but their logical counts differ.
    var offsets: List[Int64] = [0, 1, 2, 4, 4]
    var bits: List[UInt8] = [6]
    var bad = source.with_column(
        Series(
            "wide",
            ListColumn(
                offsets^, source.column("wide").list_column().child(), bits^
            ),
        )
    )
    var rejected = bad.lazy().explode(["xs", "wide"]).head(0).select(col("id"))
    with assert_raises(contains="different element counts at row 1"):
        _ = rejected.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="different element counts at row 1"):
        _ = rejected.collect(accelerator=MetalRuntime(), engine="accel")
    _literal(expanded.collect(accelerator=MetalRuntime(), engine="accel"))


# requires: metal
def test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery() raises:
    var source = _source()
    var query = chunk_layout(source).lazy().explode(["xs", "wide"])
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ).equals(measured[0])
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=limited, engine="accel")
    _literal(query.collect(accelerator=MetalRuntime(), engine="accel"))
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = query.profile(accelerator=observer, engine="accel")
    assert_equal(observed[0].height(), 0)
    assert_equal(
        observed[1].item(0, "query_host_result_bytes").int64(), Int64(0)
    )
    # Two parent controls32 + shared shape control16 + primitive error16 + final error60.
    assert_equal(
        observed[1].item(0, "query_download_bytes").int64(), Int64(124)
    )
    var empty = source.clear().lazy().explode(["xs", "wide"])
    var result = empty.profile(
        accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
        engine="accel",
    )
    assert_equal(result[0].height(), 0)
    assert_equal(
        result[1].item(0, "query_peak_requested_bytes").int64(), Int64(0)
    )
    assert_equal(result[1].item(0, "query_kernel_launches").int64(), Int64(0))
    var caps = capabilities()
    caps.primitive_list_explode = False
    with assert_raises(contains="primitive List expansion is unavailable"):
        _ = lower_join_dag(query, caps)
    for duplicate in [False, True]:
        var names = List[String](["xs", "xs"]) if duplicate else List[String](
            ["id"]
        )
        with assert_raises(
            contains="column listed twice" if duplicate else "needs list columns"
        ):
            _ = (
                source.lazy()
                .explode(names)
                .collect(
                    accelerator=MetalRuntime(device_id=999999), engine="accel"
                )
            )
    var unselected = source.lazy().explode("xs")
    with assert_raises(contains="unselected nested outputs"):
        _ = unselected.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
    var failed = query.copy()
    failed._nodes[0].source_failure = "missing List source prefix"
    with assert_raises(contains="missing List source prefix"):
        _ = failed.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )


# requires: metal
def test_mojo_gpu_explode_ieee_words_and_child_metadata_survive_projection() raises:
    var words: List[UInt64] = [
        0x8000000000000000,
        1,
        0x7FF8000000001234,
        0xFFF0000000000000,
    ]
    var values = List[Float64]()
    for word in words:
        values.append(bitcast[DType.float64](word))
    var offsets: List[Int64] = [0, 2, 4]
    var child = Series("item", Column[Float64](values^))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var owner = child._field_metadata.value()
    var source = DataFrame([Series("f", ListColumn(offsets^, child^))])
    var query = source.lazy().explode("f").select(col("f").alias("copy"))
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(actual.height(), 4)
    for row in range(4):
        assert_equal(
            bitcast[DType.uint64](actual.column("copy").float64()._get(row)),
            words[row],
        )
    assert_true(
        actual.column("copy")._field_metadata.value().ptr() == owner.ptr()
    )
    assert_true(
        actual.equals(
            query.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
