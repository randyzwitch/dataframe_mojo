"""Original successful SUM bounds authorize only infallible consumers."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.dtype import DataType, CategoricalDictionary
from dataframe.lazy import LazyFrame, SELECT, _plan_node
from dataframe.binding import bind
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.joins import lower_join_dag, _mark_decimal_producers
from dataframe.accelerator.metadata import prune_projections
from dataframe.accelerator.code_domains import (
    CategoryOwner,
    CategoryField,
    TextField,
)
from dataframe.accelerator import decimal_sum_bounds as raw
from dataframe.accelerator import decimal_sum_receipts
from dataframe.accelerator.gpu.runtime import capabilities, _preflight
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def source() raises -> DataFrame:
    var dtype = DataType.decimal(18, 2, 64)
    var keys: List[Int32] = [-2, -2, 0, 1]
    var key_valid: List[Bool] = [True, True, True, False]
    var sale: List[Int64] = [1000, 0, 1000, Int64.MIN]
    var refund: List[Int64] = [200, 200, 600, Int64.MAX]
    var zero: List[Int64] = [0, 0, 0, 1]
    var valid: List[Bool] = [True, True, True, False]
    return DataFrame(
        [
            Series("g", Column[Int32](keys^, key_valid^)),
            Series("sale", Column[Int64](sale^, valid)).with_dtype(dtype),
            Series("refund", Column[Int64](refund^, valid^)).with_dtype(dtype),
            Series("zero", Column[Int64](zero^)).with_dtype(dtype),
        ]
    )


def query(frame: DataFrame) raises -> LazyFrame:
    var dtype = frame.column("sale").dtype()
    return (
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            [
                col("sale").sum(min_count=1).alias("sale"),
                (col("refund") + col("zero")).sum(min_count=1).alias("refund"),
            ]
        )
        .filter(col("sale") > lit("2").cast(dtype) * col("refund"))
        .select(["g"])
    )


def test_sum_output_bound_raw_math_and_original_slots() raises:
    var cap = raw.sum_bound(
        raw.Magnitude(Int128(1) << 63, True), Int(UInt32.MAX)
    )
    assert_true(cap.known)
    assert_equal(cap.maximum, Int128(UInt32.MAX) * (Int128(1) << 63))
    assert_true(
        raw.precision_safe(
            raw.product(cap, raw.Magnitude(200, True)), DataType.decimal(38, 4)
        )
    )
    assert_true(
        not raw.product(
            raw.Magnitude(Int128.MAX, True), raw.Magnitude(2, True)
        ).known
    )
    assert_true(not raw.sum_bound(cap, Int(UInt32.MAX) + 1).known)
    assert_true(not raw.physical(DataType.decimal(38, 2, 128)).known)
    var normalized = fold_literal_text_parsing(query(source()))
    var dag = lower_join_dag(
        normalized._with_accelerator_settings(False, True, 17), capabilities()
    )
    var receipts = 0
    for task in dag.tasks:
        if task.sum_receipt:
            receipts += 1
            var receipt = task.sum_receipt.value()
            assert_equal(receipt[].origin[].original_root, 1)
            assert_equal(receipt[].origin[].maxima[1].maximum, cap.maximum)
            assert_true(receipt[].origin[].retained_bytes > 0)
            assert_equal(
                receipt[].origin[].outputs[1].dtype(),
                DataType.decimal(38, 2, 128),
            )
            decimal_sum_receipts.validate(
                receipt[], task.source, dag.tasks[task.source].query, task.query
            )
            with assert_raises(contains="task lineage"):
                decimal_sum_receipts.validate(
                    receipt[],
                    task.source + 1,
                    dag.tasks[task.source].query,
                    task.query,
                )
            var forged = task.query.copy()
            forged._frames[0] = forged._frames[0].select(
                ["refund", "sale", "g"]
            )
            with assert_raises(contains="slot or dtype"):
                decimal_sum_receipts.validate(
                    receipt[], task.source, dag.tasks[task.source].query, forged
                )
    assert_equal(receipts, 1)
    _guarded_prefix_cases()


def test_mojo_gpu_sum_output_bounds_scaled_literal_nulls_layouts() raises:
    var frame = source()
    for layout in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        var planned = query(layout)
        for workers in [1, 2]:
            var setting = WorkerSetting(workers)
            for streaming in [False, True]:
                for gpu in [False, True]:
                    var result: DataFrame
                    if gpu:
                        result = planned.collect(
                            engine="accel",
                            accelerator=MetalRuntime(),
                            optimize=False,
                            streaming=streaming,
                            batch_size=1,
                        )
                    else:
                        result = planned.collect(
                            engine="cpu",
                            optimize=False,
                            streaming=streaming,
                            batch_size=1,
                        )
                    assert_equal(result.height(), 1)
                    assert_equal(result.column("g").dtype(), DataType.INT32)
                    assert_equal(
                        result.column("g").int32().to_list(), [Int32(-2)]
                    )
            _ = setting^


def test_mojo_gpu_sum_output_bounds_preserve_producer_fault_and_budget() raises:
    var good = query(source())
    var measured = good.profile(
        engine="accel",
        accelerator=MetalRuntime(),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    var field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in measured[1].columns() else "peak_requested_bytes"
    )
    var peak = Int(measured[1].item(0, field).int64())
    var result = good.collect(
        engine="accel",
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    assert_equal(result.column("g").int32().to_list(), [Int32(-2)])
    with assert_raises(contains="budget"):
        _ = good.collect(
            engine="accel",
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
            optimize=False,
            streaming=True,
        )
    var frame = source().with_column(
        Series("refund", Column[Int64]([Int64.MAX, 200, 600, 0])).with_dtype(
            DataType.decimal(18, 2, 64)
        )
    )
    for gpu in [False, True]:
        for streaming in [False, True]:
            with assert_raises(contains="decimal precision"):
                if gpu:
                    _ = (
                        query(frame)
                        .head(0)
                        .collect(
                            engine="accel",
                            accelerator=MetalRuntime(),
                            optimize=False,
                            streaming=streaming,
                        )
                    )
                else:
                    _ = (
                        query(frame)
                        .head(0)
                        .collect(
                            engine="cpu", optimize=False, streaming=streaming
                        )
                    )


def test_sum_output_bounds_refuse_arbitrary_wide_columns_and_scale18_products() raises:
    var payload: List[Int128] = [1, 2]
    var wide = DataFrame(
        [
            Series("v", Column[Int128](payload^)).with_dtype(
                DataType.decimal(38, 2, 128)
            )
        ]
    )
    var arbitrary = wide.lazy().filter(
        col("v") > lit("2").cast(DataType.decimal(18, 2, 64)) * col("v")
    )
    with assert_raises(contains="checked decimal"):
        _ = _preflight(fold_literal_text_parsing(arbitrary))
    var dtype = DataType.decimal(18, 18, 64)
    var frame = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("v", Column[Int64]([1, 1])).with_dtype(dtype),
            Series("z", Column[Int64]([0, 0])).with_dtype(dtype),
        ]
    )
    var unsafe = (
        frame.lazy()
        .group_by("g")
        .agg((col("v") + col("z")).sum(min_count=1).alias("s"))
        .filter(
            col("s") > col("s") * lit("2").cast(DataType.decimal(19, 18, 128))
        )
    )
    with assert_raises(contains="checked decimal"):
        _ = lower_join_dag(
            fold_literal_text_parsing(unsafe)._with_accelerator_settings(
                False, True, 17
            ),
            capabilities(),
        )


def _guarded_prefix_cases() raises:
    var frame = source().with_column(
        Series("unused", Column[UInt32]([0, 1, 0, 1]))
    )
    var right = DataFrame(
        [
            Series("g", Column[Int32]([-2, 0])),
            Series("right_unused", Column[Int32]([70, 80])),
        ]
    )
    var dtype = frame.column("sale").dtype()
    var original = fold_literal_text_parsing(
        frame.lazy()
        .filter(col("g") >= lit(Int32(-10)))
        .join(right.lazy(), on="g")
        .group_by("g", maintain_order=True)
        .agg(
            [
                col("sale").sum(min_count=1).alias("sale"),
                (col("refund") + col("zero")).sum(min_count=1).alias("refund"),
            ]
        )
        .filter(col("sale") > lit("2").cast(dtype) * col("refund"))
        .select(["g"])
    )
    var dictionary = CategoricalDictionary()
    dictionary.append("a")
    dictionary.append("b")
    var category_dtype = DataType.categorical(dictionary^)
    var codes = List[Column[UInt32]]()
    codes.append(frame.column("unused").uint32().copy())
    var owner = ArcPointer(CategoryOwner(category_dtype, codes^))
    original._nodes[0].text_fields = [
        TextField("unused", DataType.STRING, CategoryField(owner.copy(), True))
    ]
    for at in range(len(original._nodes)):
        original._nodes[at].lineage_anchor = ArcPointer(at)
    var group = 4
    var origin = decimal_sum_receipts.capture(original, group)
    var projected = original.copy()
    prune_projections(projected)
    _mark_decimal_producers(projected)
    var mapped = -1
    var prefix_filter = -1
    var transport = -1
    for at in range(len(projected._nodes)):
        if decimal_sum_receipts._same_anchor(
            projected._nodes[at], original._nodes[group]
        ):
            mapped = at
        if decimal_sum_receipts._same_anchor(
            projected._nodes[at], original._nodes[1]
        ):
            prefix_filter = at
        if decimal_sum_receipts._plain_transport(projected._nodes[at]):
            transport = at
    assert_true(mapped > group)
    assert_true(prefix_filter >= 0)
    assert_true(transport >= 0)
    assert_true(decimal_sum_receipts.matches(origin[], projected, mapped))
    # This real prune bridge inserts SELECT above a filtered JOIN arm,
    # narrows SCAN.names, and reorders indices. No original computation moves.
    var dag = lower_join_dag(
        original._with_accelerator_settings(False, True, 17), capabilities()
    )
    var count = 0
    for task in dag.tasks:
        count += Int(Bool(task.sum_receipt))
    assert_equal(count, 1)
    # Equal content in a newly allocated ORIGINAL source column is rejected,
    # including an unused column. Values/dtype equality never waives owners.
    var changed_source = projected.copy()
    changed_source._frames[0] = frame.with_column(
        Series("unused", Column[UInt32]([0, 1, 0, 1]))
    )
    assert_true(
        not decimal_sum_receipts.matches(origin[], changed_source, mapped)
    )
    var changed_prefix = projected.copy()
    changed_prefix._nodes[prefix_filter].exprs = [col("g") >= lit(Int32(0))]
    assert_true(
        not decimal_sum_receipts.matches(origin[], changed_prefix, mapped)
    )
    var removed_checked = projected.copy()
    removed_checked._nodes[mapped].exprs[1] = (
        col("refund").sum(min_count=1).alias("refund")
    )
    assert_true(
        not decimal_sum_receipts.matches(origin[], removed_checked, mapped)
    )
    var changed_transport = projected.copy()
    changed_transport._nodes[transport].exprs[0] = (
        col("g") + lit(Int32(0))
    ).alias("g")
    assert_true(
        not decimal_sum_receipts.matches(origin[], changed_transport, mapped)
    )
    var changed_transport_metadata = projected.copy()
    changed_transport_metadata._nodes[transport].code_domain = DataType.INT32
    assert_true(
        not decimal_sum_receipts.matches(
            origin[], changed_transport_metadata, mapped
        )
    )

    var changed_category = projected.copy()
    changed_category._nodes[0].text_fields[0].category = CategoryField(
        owner.copy(), False
    )
    assert_true(
        not decimal_sum_receipts.matches(origin[], changed_category, mapped)
    )
    var replaced_owner = ArcPointer(
        CategoryOwner(category_dtype, owner[].codes.copy())
    )
    var changed_owner = projected.copy()
    changed_owner._nodes[0].text_fields[0].category = CategoryField(
        replaced_owner.copy(), True
    )
    assert_true(
        not decimal_sum_receipts.matches(origin[], changed_owner, mapped)
    )
    # Same visible STRING field/name and exact COL AST: only private category
    # semantics differ. The pruning transport must reject each independently.
    var transported = frame.lazy()
    transported._nodes[0].text_fields = original._nodes[0].text_fields.copy()
    transported._nodes.append(
        _plan_node(
            SELECT,
            0,
            exprs=[col("unused")],
            text_fields=transported._nodes[0].text_fields,
        )
    )
    assert_true(decimal_sum_receipts._transport_matches(transported, 1))
    var transport_flags = transported.copy()
    transport_flags._nodes[1].text_fields[0].category = CategoryField(
        owner.copy(), False
    )
    assert_true(not decimal_sum_receipts._transport_matches(transport_flags, 1))
    var transport_owner = transported.copy()
    transport_owner._nodes[1].text_fields[0].category = CategoryField(
        replaced_owner, True
    )
    assert_true(not decimal_sum_receipts._transport_matches(transport_owner, 1))
    var transport_presence = transported.copy()
    transport_presence._nodes[1].text_fields[0].category = None
    assert_true(
        not decimal_sum_receipts._transport_matches(transport_presence, 1)
    )
