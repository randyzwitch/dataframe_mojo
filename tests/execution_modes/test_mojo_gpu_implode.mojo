"""Terminal primitive IMPLODE keeps GPU operand evaluation and error order."""
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.lists import lower_terminal_implode
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var n = Series(
        "n",
        Column[Int64](
            [4, Int64.MAX, -2, 7, 1, -3, 9],
            [True, False, True, True, True, True, True],
        ),
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    n._field_metadata = ArcPointer(metadata^)
    var wide: List[Int128] = [
        Int128(1) << 80,
        0,
        -(Int128(1) << 80),
        7,
        8,
        9,
        10,
    ]
    return DataFrame(
        [
            n^,
            Series(
                "use", BoolColumn([True, True, True, False, True, True, False])
            ),
            Series(
                "wide",
                Column[Int128](
                    wide^, [True, False, True, True, True, True, True]
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series(
                "time", Column[Int64]([0, Int64.MIN, -1, 1, 7, Int64.MAX, 42])
            ).with_dtype(DataType.datetime("ns", "UTC")),
        ]
    )


def _check_child(actual: Series, expected: Series) raises:
    assert_equal(actual.dtype(), DataType.list(expected.dtype()))
    assert_equal(len(actual), 1)
    var lists = actual.list_column()
    assert_false(lists.is_null(0))
    assert_equal(lists._offsets[][0], Int64(0))
    assert_equal(lists._offsets[][1], Int64(len(expected)))
    assert_equal(len(lists._bits[]), 0)
    var child = lists.row(0)
    assert_true(child.equals(expected.renamed("item")))
    assert_false(Bool(actual._field_metadata))


def test_mojo_gpu_implode_primitive_literals_nulls_layouts_metadata_and_empty() raises:
    var original = _source()
    for variant in range(3):
        var frame = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var masked = (
            when(col("use"))
            .then(col("n") + 1)
            .otherwise(lit(Int64(0)))
            .alias("chosen")
        )
        var expressions: List[Expr] = [
            col("n").implode().alias("n"),
            masked.implode().alias("chosen"),
            col("use").implode().alias("flags"),
            col("wide").implode().alias("wide"),
            col("time").implode().alias("time"),
        ]
        var query = frame.lazy().select_exprs(expressions)
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.shape(), (1, 5))
        _check_child(actual.column("n"), original.column("n"))
        _check_child(actual.column("flags"), original.column("use"))
        _check_child(actual.column("wide"), original.column("wide"))
        _check_child(actual.column("time"), original.column("time"))
        var expected = Series(
            "item",
            Column[Int64](
                [5, 0, -1, 0, 2, -2, 0],
                [True, False, True, True, True, True, True],
            ),
        )
        _check_child(actual.column("chosen"), expected)
        assert_true(
            actual.column("n")
            .list_column()
            .child()
            ._field_metadata.value()
            .ptr()
            == original.column("n")._field_metadata.value().ptr()
        )
        assert_false(
            Bool(actual.column("chosen").list_column().child()._field_metadata)
        )
        for batch in [1, 64]:
            assert_true(
                actual.equals(frame.select_exprs(expressions, batch_size=batch))
            )
        var empty_query = frame.clear().lazy().select_exprs(expressions)
        var empty = empty_query.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
        assert_equal(empty.shape(), (1, 5))
        for column in empty._columns:
            assert_false(column.list_column().is_null(0))
            assert_equal(len(column.list_column().row(0)), 0)
        assert_true(
            empty.equals(empty_query.collect(engine="cpu", streaming=False))
        )
    var nulls = (
        DataFrame(
            [Series("x", Column[Int64]([Int64.MAX, Int64.MIN], [False, False]))]
        )
        .lazy()
        .select((col("x") + 1).implode())
    )
    var all_null = nulls.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(len(all_null.item().list()), 2)
    assert_equal(all_null.item().list().null_count(), 2)
    assert_true(all_null.equals(nulls.collect(engine="cpu", streaming=False)))
    # Bit transport, independent of host floating arithmetic/NaN equality.
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000001234,
    ]
    var values = List[Float64]()
    for word in words:
        values.append(bitcast[DType.float64](word))
    var floats = (
        DataFrame([Series("f", Column[Float64](values^))])
        .lazy()
        .select(col("f").implode())
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    var child = floats.item().list()
    for row in range(len(words)):
        assert_equal(
            bitcast[DType.uint64](child.float64()._get(row)), words[row]
        )


def test_mojo_gpu_implode_prior_rows_trailing_slices_and_profiles_budget() raises:
    var frame = DataFrame(
        [
            Series(
                "n",
                Column[Int32](
                    [4, 99, -2, 7, 1, -3, 9],
                    [True, False, True, True, True, True, True],
                ),
            )
        ]
    )
    var input = (
        frame.lazy()
        .filter(col("n") >= lit(Int32(0)))
        .sort("n", descending=True)
        .slice(1, 2)
    )
    var query = input.select(
        (col("n") + lit(Int32(1))).implode().alias("values")
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    _check_child(
        measured[0].column("values"), Series("item", Column[Int32]([8, 5]))
    )
    assert_true(
        measured[0].equals(query.collect(engine="cpu", streaming=False))
    )
    var plan = lower_terminal_implode(query)
    assert_equal(plan.assembly_bytes, 16)
    assert_equal(
        measured[1].item(0, "implode_assembly_bytes").int64(), Int64(16)
    )
    assert_true(measured[1].item(0, "kernel_launches").int64() > 0)
    var inner = plan.query.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var inner_peak_field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in inner[1].columns() else "peak_requested_bytes"
    )
    assert_equal(peak, 16 + Int(inner[1].item(0, inner_peak_field).int64()))
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ).equals(measured[0])
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=limited, engine="accel")
    assert_equal(
        DataFrame([Series("safe", Column[Int64]([7]))])
        .lazy()
        .select(col("safe"))
        .collect(accelerator=limited, engine="accel")
        .item()
        .int64(),
        Int64(7),
    )
    for offset in [-2, -1, 0, 1, 3]:
        for length in [-1, 0, 1, 3]:
            var sliced = query.slice(offset, length)
            assert_true(
                sliced.collect(
                    accelerator=MetalRuntime(), engine="accel"
                ).equals(sliced.collect(engine="cpu", streaming=False))
            )
    assert_equal(
        query.head(0)
        .collect(accelerator=MetalRuntime(), engine="accel")
        .height(),
        0,
    )
    var empty = frame.clear().lazy().select(col("n").implode())
    with assert_raises(contains="assembly exceeds memory budget"):
        _ = empty.collect(
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=15),
            engine="accel",
        )
    assert_equal(
        empty.collect(
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=16),
            engine="accel",
        ).height(),
        1,
    )

    # Multiple DAG regions: first-region counters are not whole-query totals.
    var left = DataFrame([Series("k", Column[Int32]([1, 2, 1]))])
    var right = DataFrame(
        [
            Series("k", Column[Int32]([1, 2])),
            Series("v", Column[Int32]([10, 20])),
        ]
    )
    var joined = (
        left.lazy()
        .join(right.lazy(), on="k")
        .select((col("k") + col("v")).implode().alias("joined"))
    )
    var dag_result = joined.profile(accelerator=MetalRuntime(), engine="accel")
    _check_child(
        dag_result[0].column("joined"),
        Series("item", Column[Int32]([11, 22, 11])),
    )
    assert_true(
        dag_result[0].equals(joined.collect(engine="cpu", streaming=False))
    )
    var dag_plan = lower_terminal_implode(joined)
    var dag_inner = dag_plan.query.profile(
        accelerator=MetalRuntime(), engine="accel"
    )
    assert_true(dag_result[1].height() > 1)
    assert_equal(dag_result[1].height(), dag_inner[1].height())
    var region_sum = Int64(0)
    for region in range(dag_result[1].height()):
        var before_bytes = (
            dag_inner[1].item(region, "host_result_bytes").int64()
        )
        var after_bytes = (
            dag_result[1].item(region, "host_result_bytes").int64()
        )
        assert_equal(
            after_bytes,
            before_bytes
            + Int64(16 if region == dag_result[1].height() - 1 else 0),
        )
        region_sum += after_bytes
    assert_equal(dag_result[1].item(0, "host_result_bytes").int64(), Int64(0))
    assert_equal(
        dag_result[1].item(0, "query_implode_child_export_bytes").int64(),
        Int64(13),
    )
    assert_equal(
        dag_result[1].item(0, "query_host_result_bytes").int64(), Int64(29)
    )
    assert_equal(region_sum, Int64(29))
    var dag_peak = Int(
        dag_result[1].item(0, "query_peak_requested_bytes").int64()
    )
    assert_equal(
        dag_peak,
        16 + Int(dag_inner[1].item(0, "query_peak_requested_bytes").int64()),
    )
    assert_true(
        joined.collect(
            accelerator=MetalRuntime(memory_limit_bytes=dag_peak),
            engine="accel",
        ).equals(dag_result[0])
    )
    with assert_raises(contains="budget"):
        _ = joined.collect(
            accelerator=MetalRuntime(memory_limit_bytes=dag_peak - 1),
            engine="accel",
        )


def _check_implode_eager_phases_and_deferred_slices() raises:
    var decimal_type = DataType.decimal(3, 0, 128)
    var decimals: List[Int128] = [999, 1]
    var zeros: List[Int128] = [0, 0]
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([128, 0])),
            Series("d", Column[Int128](decimals^)).with_dtype(decimal_type),
            Series("zero", Column[Int128](zeros^)).with_dtype(decimal_type),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    for partitioned in [False, True]:
        var state = col("d").sum().over("k") if partitioned else col("d").sum()
        var eager = (col("zero") + state).implode().alias("decimal")
        var strict = col("x").cast(DataType.INT8).implode().alias("strict")
        for strict_first in [False, True]:
            var fields = List[Expr]()
            if strict_first:
                fields.append(strict.copy())
                fields.append(eager.copy())
            else:
                fields.append(eager.copy())
                fields.append(strict.copy())
            var query = frame.lazy().select_exprs(fields^)
            for head in [False, True]:
                var checked = query.head(0) if head else query.copy()
                # Shared binding validates every projection before evaluating
                # operands. A bare nested aggregate is invalid in either order;
                # OVER resets aggregate shape and exercises the eager phases.
                if not partitioned:
                    for gpu in [False, True]:
                        with assert_raises(
                            contains="Aggregate input must be row-valued without nested aggregates"
                        ):
                            if gpu:
                                _ = checked.collect(
                                    accelerator=MetalRuntime(), optimize=False
                                )
                            else:
                                _ = checked.collect(
                                    engine="cpu",
                                    optimize=False,
                                    streaming=False,
                                )
                    continue
                with assert_raises(
                    contains="cast from int64 to int8 failed" if strict_first else "decimal precision"
                ):
                    _ = checked.collect(
                        engine="cpu", optimize=False, streaming=False
                    )
                with assert_raises(
                    contains="strict cast failed" if strict_first else "decimal precision"
                ):
                    _ = checked.collect(
                        accelerator=MetalRuntime(), optimize=False
                    )
    var collision = (
        DataFrame(
            [
                Series("x", Column[Int64]([1, 2])),
                Series("__dataframe_implode_0", Column[Int64]([7, 8])),
            ]
        )
        .lazy()
        .select_exprs(
            [
                col("x").implode().alias("first"),
                col("__dataframe_implode_0").implode().alias("second"),
            ]
        )
    )
    var collision_result = collision.collect(accelerator=MetalRuntime())
    _check_child(
        collision_result.column("first"), Series("item", Column[Int64]([1, 2]))
    )
    _check_child(
        collision_result.column("second"), Series("item", Column[Int64]([7, 8]))
    )
    for empty in [False, True]:
        for overflow in [False, True]:
            var values = DataFrame(
                [
                    Series(
                        "x",
                        Column[Int64]([Int64.MAX if overflow else Int64(1)]),
                    )
                ]
            )
            if empty:
                values = values.clear()
            var base = values.lazy().select(
                (col("x") + lit(Int64(1))).implode()
            )
            for head in [False, True]:
                var query = (base.head(0) if head else base.copy()).slice(0, -2)
                var anchor = (
                    "overflow" if overflow
                    and not empty else "Slice length must be nonnegative"
                )
                with assert_raises(contains=anchor):
                    _ = query.collect(
                        engine="cpu", optimize=False, streaming=False
                    )
                with assert_raises(contains=anchor):
                    _ = query.collect(
                        accelerator=MetalRuntime(
                            device_id=999999 if empty else 0
                        ),
                        optimize=False,
                    )


def test_mojo_gpu_implode_gpu_errors_masks_projection_order_head_and_observer() raises:
    _check_implode_eager_phases_and_deferred_slices()
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 4])),
            Series("u", Column[UInt64]([UInt64.MAX, 7])),
            Series("mask", BoolColumn([False, True])),
        ]
    )
    for cast_first in [False, True]:
        var overflow = (col("x") + 1).implode().alias("overflow")
        var cast = col("u").cast(DataType.INT32).implode().alias("cast")
        var expressions = List[Expr]()
        if cast_first:
            expressions.append(cast.copy())
            expressions.append(overflow.copy())
        else:
            expressions.append(overflow.copy())
            expressions.append(cast.copy())
        var query = frame.lazy().select_exprs(expressions)
        for head in [False, True]:
            var checked = query.head(0) if head else query.copy()
            with assert_raises(
                contains="cast from uint64 to int32 failed" if cast_first else "overflow"
            ):
                _ = checked.collect(engine="cpu", streaming=False)
            with assert_raises(
                contains="strict cast failed" if cast_first else "overflow"
            ):
                _ = checked.collect(accelerator=MetalRuntime(), engine="accel")
    var safe = frame.lazy().select(
        when(col("mask"))
        .then(col("x") + 1)
        .otherwise(lit(Int64(0)))
        .implode()
        .alias("safe")
    )
    _check_child(
        safe.collect(accelerator=MetalRuntime(), engine="accel").column("safe"),
        Series("item", Column[Int64]([0, 5])),
    )
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = safe.profile(accelerator=observer, engine="accel")
    assert_equal(observed[0].height(), 0)
    assert_equal(
        observed[0].column("safe").dtype(), DataType.list(DataType.INT64)
    )
    assert_equal(
        observed[1].item(0, "implode_assembly_bytes").int64(), Int64(0)
    )
    # Every observed producer downloads errors only. The final projection has
    # two Int64 cells: exactly two UInt32 errors, with no child-value export.
    assert_true(observed[1].height() > 1)
    for region in range(observed[1].height()):
        assert_equal(
            observed[1].item(region, "host_result_bytes").int64(), Int64(0)
        )
    assert_equal(
        observed[1].item(observed[1].height() - 1, "download_bytes").int64(),
        Int64(2 * 4),
    )
    assert_equal(
        observed[1].item(0, "query_implode_child_export_bytes").int64(),
        Int64(0),
    )
    assert_true(observed[1].item(0, "kernel_launches").int64() > 0)


def test_mojo_gpu_implode_explicit_boundaries_and_metadata_only_description() raises:
    var source = _source()
    var query = source.lazy().select((col("n") + 1).implode())
    var description = MetalRuntime(device_id=999999).describe(query)
    assert_true("terminal primitive implode" in description)
    assert_false("invalid device" in description)
    with assert_raises(contains="only IMPLODE output roots"):
        _ = lower_terminal_implode(
            source.lazy().select_exprs(
                [col("n").implode(), col("n").sum().alias("sum")]
            )
        )
    var scalar_operand = source.lazy().select(lit(Int64(7)).implode())
    with assert_raises(
        contains="Aggregate input must be row-valued without nested aggregates"
    ):
        _ = scalar_operand.collect(engine="cpu", streaming=False)
    with assert_raises(
        contains="Aggregate input must be row-valued without nested aggregates"
    ):
        _ = lower_terminal_implode(scalar_operand)
    with assert_raises(
        contains="Aggregate input must be row-valued without nested aggregates"
    ):
        _ = scalar_operand.collect(accelerator=MetalRuntime())
    with assert_raises(contains="terminal SELECT"):
        _ = lower_terminal_implode(
            source.lazy().group_by("use").agg(col("n").implode())
        )
    with assert_raises(contains="only IMPLODE output roots"):
        _ = lower_terminal_implode(
            source.lazy().select(col("n").implode().list().len())
        )
    with assert_raises(contains="fixed-width primitive"):
        _ = lower_terminal_implode(
            DataFrame([Series("s", StringColumn(["a", "b"]))])
            .lazy()
            .select(col("s").implode())
        )
