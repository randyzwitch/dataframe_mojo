"""Primitive source List len/get remain resident through row expressions."""
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
    col,
    lit,
    when,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.source_lists import lower_source_lists
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var child = Series(
        "item",
        Column[Int64](
            [5, 6, Int64.MAX, Int64.MAX, 9, Int64.MAX, 11, 12],
            [True, True, True, True, True, False, True, True],
        ),
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var offsets: List[Int64] = [0, 2, 4, 4, 6, 8]
    var bits: List[UInt8] = [29]
    var xs = Series("xs", ListColumn(offsets.copy(), child^, bits.copy()))
    var wide_values: List[Int128] = [
        Int128(1) << 80,
        -(Int128(1) << 80),
        0,
        0,
        7,
        8,
        9,
        10,
    ]
    var wide = Series("item", Column[Int128](wide_values^)).with_dtype(
        DataType.decimal(38, 2)
    )
    var lists = Series("wide", ListColumn(offsets^, wide^, bits^))
    return DataFrame(
        [
            xs^,
            lists^,
            Series("use", BoolColumn([True, True, True, False, True])),
        ]
    )


def _check(actual: DataFrame, source: DataFrame, start: Int) raises:
    assert_equal(
        actual.columns(), ["len", "first", "identity", "last", "masked", "wide"]
    )
    var first: List[Int64] = [5, 0, 0, 9, 11]
    var last: List[Int64] = [6, 0, 0, 0, 12]
    var masked: List[Int64] = [6, 0, 0, 0, 12]
    var child_owner = (
        source.column("xs").list_column().child()._field_metadata.value()
    )
    for name in ["first", "identity", "last"]:
        assert_true(
            actual.column(name)._field_metadata.value().ptr()
            == child_owner.ptr()
        )
    assert_false(Bool(actual.column("len")._field_metadata))
    assert_false(Bool(actual.column("masked")._field_metadata))
    assert_equal(actual.column("wide").dtype(), DataType.decimal(38, 2))
    for row in range(actual.height()):
        var at = start + row
        assert_equal(actual.item(row, "len").is_null(), at == 1)
        if at != 1:
            assert_equal(
                actual.item(row, "len").int64(), Int64(0 if at == 2 else 2)
            )
        for name in ["first", "identity", "wide"]:
            assert_equal(actual.item(row, name).is_null(), at == 1 or at == 2)
        assert_equal(
            actual.item(row, "last").is_null(), at == 1 or at == 2 or at == 3
        )
        assert_equal(actual.item(row, "masked").is_null(), at == 1 or at == 2)
        if at != 1 and at != 2:
            assert_equal(actual.item(row, "first").int64(), first[at])
            assert_equal(actual.item(row, "identity").int64(), first[at])
            assert_equal(actual.item(row, "masked").int64(), masked[at])
            var wide = actual.column("wide")._data[Column[Int128]]._get(row)
            assert_equal(
                wide,
                Int128(1) << 80 if at == 0 else Int128(7 if at == 3 else 9),
            )
        if at == 0 or at == 4:
            assert_equal(actual.item(row, "last").int64(), last[at])


def test_mojo_gpu_source_list_get_len_null_parents_layouts_metadata_and_computed_rows() raises:
    var original = _source()
    var first = col("xs").list().get(0)
    var expressions: List[Expr] = [
        col("xs").list().len().alias("len"),
        first.alias("first"),
        first.cast(DataType.INT64).cast(DataType.INT64).alias("identity"),
        col("xs").list().get(-1).alias("last"),
        when(col("use"))
        .then(first + 1)
        .otherwise(lit(Int64(0)))
        .alias("masked"),
        col("wide").list().get(0).alias("wide"),
    ]
    for sliced in [False, True]:
        var source = original.slice(1, 3) if sliced else original.copy()
        for variant in range(3):
            var frame = (
                source.copy() if variant
                == 0 else offset_layout(source) if variant
                == 1 else chunk_layout(source)
            )
            var query = frame.lazy().select_exprs(expressions)
            var measured = query.profile(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_equal(measured[0].height(), 3 if sliced else 5)
            _check(measured[0], original, 1 if sliced else 0)
            for batch in [1, 64]:
                var cpu = frame.select_exprs(expressions, batch_size=batch)
                _check(cpu, original, 1 if sliced else 0)
                assert_true(measured[0].equals(cpu))
            assert_equal(
                measured[1].item(0, "source_list_backend").string(),
                "mojo_gpu_resident_list",
            )
            assert_equal(
                measured[1]
                .item(0, "source_list_preprocess_download_bytes")
                .int64(),
                Int64(0),
            )
            assert_equal(
                measured[1].item(0, "source_list_consumers").int64(), Int64(4)
            )
            assert_true(
                measured[1].item(0, "query_kernel_launches").int64()
                > measured[1].item(0, "kernel_launches").int64()
            )
    # IEEE child words are transported unchanged, including a NaN payload.
    var raw: List[UInt64] = [0x8000000000000000, 1, 0x7FF8000000001234]
    var floats = List[Float64]()
    for word in raw:
        floats.append(bitcast[DType.float64](word))
    var offsets: List[Int64] = [0, 1, 2, 3]
    var frame = DataFrame(
        [
            Series(
                "f",
                ListColumn(offsets^, Series("item", Column[Float64](floats^))),
            )
        ]
    )
    var actual = (
        frame.lazy()
        .select(col("f").list().get(0))
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    for row in range(3):
        assert_equal(
            bitcast[DType.uint64](actual.column("f").float64()._get(row)),
            raw[row],
        )


def test_mojo_gpu_source_list_oob_empty_error_order_observer_and_exact_budget() raises:
    var source = _source()
    for index in [Int64.MIN, Int64.MAX, Int64(2), Int64(-3)]:
        var query = source.lazy().select(col("xs").list().get(Int(index)))
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.column("xs").null_count(), 5)
        assert_true(actual.equals(query.collect(engine="cpu", streaming=False)))
    var empty = (
        source.clear()
        .lazy()
        .select_exprs([col("xs").list().len(), col("wide").list().get(0)])
    )
    assert_equal(
        empty.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        ).height(),
        0,
    )
    var query = (
        chunk_layout(source)
        .lazy()
        .select((col("xs").list().get(0) + 1).alias("x"))
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ).equals(measured[0])
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="memory budget before submission"):
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
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = query.profile(accelerator=observer, engine="accel")
    assert_equal(observed[0].height(), 0)
    assert_equal(observed[1].item(0, "host_result_bytes").int64(), Int64(0))
    assert_equal(
        observed[1].item(0, "query_download_bytes").int64(), Int64(5 * 4)
    )
    assert_equal(
        observed[1].item(0, "source_list_preprocess_download_bytes").int64(),
        Int64(0),
    )
    var offsets: List[Int64] = [0, 1]
    var poison = DataFrame(
        [
            Series(
                "xs",
                ListColumn(
                    offsets^, Series("item", Column[Int64]([Int64.MAX]))
                ),
            ),
            Series("u", Column[UInt64]([UInt64.MAX])),
        ]
    )
    for cast_first in [False, True]:
        var overflow = (col("xs").list().get(0) + 1).alias("overflow")
        var cast = col("u").cast(DataType.INT32).alias("cast")
        var expressions = List[Expr]()
        if cast_first:
            expressions.append(cast.copy())
            expressions.append(overflow.copy())
        else:
            expressions.append(overflow.copy())
            expressions.append(cast.copy())
        var checked = poison.lazy().select_exprs(expressions)
        with assert_raises(
            contains="cast from uint64 to int32 failed" if cast_first else "overflow"
        ):
            _ = checked.collect(engine="cpu", streaming=False)
        with assert_raises(
            contains="strict cast failed" if cast_first else "overflow"
        ):
            _ = checked.collect(accelerator=MetalRuntime(), engine="accel")


def test_mojo_gpu_source_list_metadata_only_admission_and_retained_boundaries() raises:
    var source = _source()
    var query = source.lazy().select((col("xs").list().get(0) + 1).alias("x"))
    assert_true(
        "resident source List len/get"
        in MetalRuntime(device_id=999999).describe(query)
    )
    var plan = lower_source_lists(query)
    assert_equal(len(plan.consumers), 1)
    assert_equal(plan.source.height(), 5)
    assert_equal(plan.query._frames[0].height(), 0)
    with assert_raises(contains="primitive row outputs"):
        _ = lower_source_lists(source.lazy().select(col("xs")))
    with assert_raises(contains="bare or intermediate List consumer"):
        _ = lower_source_lists(source.lazy().select(col("xs").list().sum()))
    with assert_raises(contains="direct scan and terminal SELECT"):
        _ = lower_source_lists(query.head(0))
    with assert_raises(contains="primitive row outputs"):
        _ = lower_source_lists(source.lazy().select(lit(Int64(7))))
