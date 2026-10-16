"""Grouped Lists cross public DAG regions as GPU descriptor/child owners."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.joins import (
    lower_join_dag,
    DAG_GROUPED_FIELDS,
    DAG_EXPLODE,
)
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.accelerator.gpu.dag_runtime import execute as execute_dag
from tests.support.resident_grouped_publication_reference import (
    _source,
    _expressions,
    _literal,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


# requires: metal
def test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer() raises:
    var original = _source()
    for variant in range(3):
        var source = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var query = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg(_expressions())
            .explode(["xs", "wide"])
        )
        var counter = ArcPointer(Int(0))
        query._nodes[0]._executions = counter
        var dag = lower_join_dag(query, capabilities(), ordered_source=True)
        assert_equal(len(dag.tasks), 2)
        assert_equal(dag.tasks[0].kind, DAG_GROUPED_FIELDS)
        assert_true(Bool(dag.tasks[0].grouped))
        assert_equal(dag.tasks[1].kind, DAG_EXPLODE)
        assert_equal(counter[], 0)
        var measured = query.profile(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        )
        assert_equal(counter[], 1)
        _literal(measured[0])
        var cpu = query.collect(engine="cpu", streaming=False, optimize=False)
        _literal(cpu)
        assert_true(measured[0].equals(cpu))
        for pair in List[Tuple[String, String]]([("xs", "x"), ("wide", "w")]):
            assert_true(
                measured[0].column(pair[0])._field_metadata.value().ptr()
                == source.column(pair[1])._field_metadata.value().ptr()
            )
        assert_equal(
            measured[1].item(0, "algorithm").string(),
            "mojo_gpu_grouped_list_producer",
        )
        assert_equal(
            measured[1].item(1, "algorithm").string(),
            "mojo_gpu_primitive_explode",
        )
        assert_equal(measured[1].item(0, "host_result_bytes").int64(), Int64(0))
        assert_equal(
            measured[1].item(0, "query_host_result_bytes").int64(), Int64(165)
        )
        assert_equal(
            measured[1].item(1, "peak_requested_bytes").int64(),
            measured[1].item(0, "query_peak_requested_bytes").int64(),
        )
        assert_true(
            measured[1].item(1, "retained_intermediate_bytes").int64() > 0
        )
        assert_true(
            "resident fields"
            in measured[1].item(0, "materialization_boundary").string()
        )


# requires: metal
def test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure() raises:
    var source = _source()
    var grouped = (
        source.lazy().group_by("k", maintain_order=True).agg(_expressions())
    )
    var expanded = grouped.explode(["xs", "wide"])
    var query = (
        expanded.filter(col("xs").is_not_null())
        .with_columns(
            (col("xs").cast(DataType.FLOAT64) + lit(Float64(1))).alias("plus")
        )
        .sort("k", descending=True)
        .select(col("plus").sum())
    )
    var result = query.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_equal(result.item().float64(), Float64(31))
    assert_true(
        result.equals(
            query.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64]([2, 3])),
            Series("tag", Column[Int32]([200, 300])),
        ]
    )
    var joined = expanded.join(right.lazy(), on=["k"], how="inner")
    var actual = joined.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_equal(
        actual.column("k").int64().to_list(),
        List[Int64]([Int64(2), Int64(2), Int64(3)]),
    )
    assert_equal(
        actual.column("tag").int32().to_list(),
        List[Int32]([Int32(200), Int32(200), Int32(300)]),
    )
    assert_true(
        actual.equals(
            joined.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var regrouped = joined.group_by("k", maintain_order=True).agg(
        [col("xs").count().alias("count"), col("xs").sum().alias("sum")]
    )
    var counts = regrouped.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_equal(
        counts.column("count").int64().to_list(),
        List[Int64]([Int64(2), Int64(1)]),
    )
    assert_equal(
        counts.column("sum").int32().to_list(),
        List[Int32]([Int32(17), Int32(11)]),
    )
    assert_true(
        counts.equals(
            regrouped.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var sliced = grouped.slice(-2, 1).explode(["xs", "wide"])
    var one = sliced.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_equal(one.height(), 1)
    assert_equal(one.item(0, "k").int64(), Int64(1))
    assert_true(one.item(0, "xs").is_null())
    assert_true(
        one.equals(
            sliced.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var failed = expanded.copy()
    failed._nodes[0].source_failure = "missing grouped List input prefix"
    with assert_raises(contains="missing grouped List input prefix"):
        _ = failed.collect(
            accelerator=MetalRuntime(device_id=999999),
            engine="accel",
            optimize=False,
        )


# requires: metal
def test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget() raises:
    var source = _source()
    var query = (
        chunk_layout(source)
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions())
        .explode(["xs", "wide"])
    )
    var measured = query.profile(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak),
            engine="accel",
            optimize=False,
        ).equals(measured[0])
    )
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
            optimize=False,
        )
    _literal(
        query.collect(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        )
    )
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = query.profile(
        accelerator=observer, engine="accel", optimize=False
    )
    assert_equal(observed[0].height(), 0)
    assert_equal(
        observed[1].item(0, "query_host_result_bytes").int64(), Int64(0)
    )
    assert_equal(
        measured[1].item(0, "query_download_bytes").int64()
        - observed[1].item(0, "query_download_bytes").int64(),
        Int64(17 * 4 * 5),
    )
    for row in range(observed[1].height()):
        assert_equal(
            observed[1].item(row, "host_result_bytes").int64(), Int64(0)
        )
    var empty = (
        source.clear()
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions())
        .explode(["xs", "wide"])
    )
    var zero = empty.profile(
        accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
        engine="accel",
        optimize=False,
    )
    assert_equal(zero[0].height(), 0)
    assert_equal(
        zero[1].item(0, "query_peak_requested_bytes").int64(), Int64(0)
    )
    assert_equal(zero[1].item(0, "query_kernel_launches").int64(), Int64(0))
    var caps = capabilities()
    caps.grouped = False
    with assert_raises(
        contains="grouped primitive List publication is unavailable"
    ):
        _ = lower_join_dag(query, caps, ordered_source=True)
    var a = Int128(6)
    for _ in range(37):
        a *= 10
    var bad = source.with_column(
        Series("x", Column[Int32]([128, 9, 10, 11]))
    ).with_column(
        Series("w", Column[Int128](List[Int128](length=4, fill=a))).with_dtype(
            DataType.decimal(38, 0)
        )
    )
    for state_first in [False, True]:
        var list = col("x").cast(DataType.INT8).implode().alias("xs")
        var sum = col("w").sum().alias("sum")
        var expressions = List[Expr](
            [sum.copy(), list.copy()]
        ) if state_first else List[Expr]([list.copy(), sum.copy()])
        var failing = (
            bad.lazy()
            .group_by("k", maintain_order=True)
            .agg(expressions)
            .explode("xs")
            .head(0)
            .select(col("k"))
        )
        with assert_raises(
            contains="decimal precision" if state_first else "cast from int32 to int8 failed"
        ):
            _ = failing.collect(engine="cpu", streaming=False, optimize=False)
        with assert_raises(
            contains="decimal precision" if state_first else "strict cast failed"
        ):
            _ = failing.collect(
                accelerator=MetalRuntime(), engine="accel", optimize=False
            )
    # An internal negative-provenance recursion cannot certify a checked feed.
    var uncertified = (
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").implode().alias("xs"),
                col("x").cast(DataType.INT8).sum().alias("sum"),
            ]
        )
        .explode("xs")
    )
    with assert_raises(contains="original raw source provenance"):
        _ = execute_dag(uncertified, 999999, 0, False, capabilities())


# requires: metal
def test_mojo_gpu_grouped_publication_layouts_slices_mixed_siblings_and_resident_consumers() raises:
    from tests.support.resident_grouped_publication_reference import (
        layouts_slices_mixed_siblings_and_resident_consumers,
    )

    layouts_slices_mixed_siblings_and_resident_consumers()


# requires: metal
def test_mojo_gpu_grouped_publication_empty_foreign_owner_checked_phase_order_and_exact_budget() raises:
    from tests.support.resident_grouped_publication_reference import (
        empty_foreign_owner_checked_phase_order_and_exact_budget,
    )

    empty_foreign_owner_checked_phase_order_and_exact_budget()
