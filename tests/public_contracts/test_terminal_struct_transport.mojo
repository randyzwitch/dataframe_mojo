"""Terminal struct assembly shares primitive outputs and keeps field validity."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    StructColumn,
    as_struct,
    col,
    lit,
    when,
)
from dataframe.accelerator.structs import (
    has_terminal_structs,
    lower_terminal_structs,
    assemble,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def source_frame() raises -> DataFrame:
    var magnitude = Int128(1) << 90
    var number = Series("x", Column[Int64]([2, 99, 5], [True, False, True]))
    # One packed Arrow application metadata entry: tag -> raw.
    var metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        3,
        0,
        0,
        0,
        116,
        97,
        103,
        3,
        0,
        0,
        0,
        114,
        97,
        119,
    ]
    number._field_metadata = ArcPointer(metadata^)
    var text = Series(
        "s", StringColumn(["z", "poison", "a"], [True, False, True])
    ).cast(DataType.CATEGORICAL)
    return DataFrame(
        [
            number^,
            text^,
            Series(
                "d",
                Column[Int128](
                    [magnitude + 1, Int128.MIN, -magnitude], [True, False, True]
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series("__dataframe_struct_0_0", Column[Int64]([10, 11, 12])),
        ]
    )


def expressions() raises -> List[Expr]:
    return [
        as_struct(
            [
                col("x").alias("value"),
                col("s").alias("word"),
                col("d").alias("wide"),
                lit(Int64(7)).alias("constant"),
            ],
            "record",
        ),
        as_struct(
            [col("x").alias("number"), col("d").alias("decimal")], "allnull"
        ),
        col("__dataframe_struct_0_0").alias("id"),
    ]


def check_values(actual: DataFrame, rows: Int) raises:
    assert_equal(actual.height(), rows)
    assert_equal(actual.columns(), ["record", "allnull", "id"])
    assert_equal(
        actual.column("record").dtype().field_names(),
        ["value", "word", "wide", "constant"],
    )
    assert_equal(
        actual.column("record").dtype().field_dtypes(),
        [
            DataType.INT64,
            DataType.STRING,
            DataType.decimal(38, 2),
            DataType.INT64,
        ],
    )
    for row in range(rows):
        var record = actual.item(row, "record")
        assert_false(record.is_null())
        assert_false(actual.item(row, "allnull").is_null())
        assert_equal(record.struct_field("value").is_null(), row == 1)
        assert_equal(record.struct_field("word").is_null(), row == 1)
        assert_equal(record.struct_field("wide").is_null(), row == 1)
        assert_equal(record.struct_field("constant").int64(), Int64(7))
        assert_equal(actual.item(row, "id").int64(), Int64(10 + row))
        if row != 1:
            assert_equal(
                record.struct_field("value").int64(),
                Int64(2 if row == 0 else 5),
            )
            assert_equal(
                record.struct_field("word").string(), "z" if row == 0 else "a"
            )
            assert_equal(
                String(record.struct_field("wide")),
                "12379400392853802748991242.25" if row
                == 0 else "-12379400392853802748991242.24",
            )


def test_terminal_struct_pack_flatten_assembly_nullable_fields_and_metadata() raises:
    var raw = source_frame()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for empty in [False, True]:
            var source = frame.clear() if empty else frame.copy()
            var query = source.lazy().select_exprs(expressions())
            assert_true(has_terminal_structs(query))
            var plan = lower_terminal_structs(query)
            assert_equal(plan.schema.height(), 0)
            for output in plan.outputs:
                assert_equal(len(output.prototype), 0)
            var flat = plan.query.collect(engine="cpu", streaming=False)
            var actual = assemble(flat, plan)
            check_values(actual, 0 if empty else 3)
            assert_frame(actual, query.collect(engine="cpu", streaming=False))
            var child = actual.column("record").struct_column().field("value")
            var staged = flat.column(plan.outputs[0].fields[0])
            assert_true(Bool(child._field_metadata))
            assert_equal(
                child._field_metadata.value()[],
                raw.column("x")._field_metadata.value()[],
            )
            assert_equal(
                Int(child._data[Column[Int64]]._data.ptr()),
                Int(staged._data[Column[Int64]]._data.ptr()),
            )
            for batch in [1, 64]:
                var cpu = source.select_exprs(expressions(), batch_size=batch)
                assert_frame(cpu, actual)
                var cpu_child = (
                    cpu.column("record").struct_column().field("value")
                )
                assert_true(Bool(cpu_child._field_metadata))
                assert_equal(
                    cpu_child._field_metadata.value()[],
                    raw.column("x")._field_metadata.value()[],
                )
    # Metadata probing must not evaluate a scalar checked payload on zero rows.
    var bad = as_struct(
        [
            col("x").alias("value"),
            lit(Int64(128)).cast(DataType.INT8).alias("checked"),
        ],
        "record",
    )
    var empty = raw.clear().select_exprs([bad.copy()], batch_size=1)
    assert_equal(empty.height(), 0)
    assert_true(
        Bool(
            empty.column("record")
            .struct_column()
            .field("value")
            ._field_metadata
        )
    )
    var guarded = as_struct(
        [
            col("x").alias("value"),
            when(lit(False))
            .then(lit(Int64(128)).cast(DataType.INT8))
            .otherwise(lit(Int8(7)))
            .alias("checked"),
        ],
        "record",
    )
    var selected = raw.select_exprs([guarded.copy()], batch_size=1)
    assert_equal(selected.height(), 3)
    for row in range(3):
        assert_equal(
            selected.item(row, "record").struct_field("checked").int8(),
            Int8(7),
        )
    with assert_raises(contains="range"):
        _ = raw.select_exprs([bad.copy()], batch_size=1)


def test_terminal_struct_scalar_only_fields_keep_one_row_on_empty_source() raises:
    var frame = DataFrame([Series("x", Column[Int64]([]))])
    var query = frame.lazy().select(
        as_struct([lit(Int64(9)).alias("n"), lit("z").alias("s")], "record")
    )
    var plan = lower_terminal_structs(query)
    var result = assemble(
        plan.query.collect(engine="cpu", streaming=False), plan
    )
    assert_equal(plan.schema.height(), 0)
    assert_equal(result.height(), 1)
    assert_false(result.item().is_null())
    assert_equal(result.item().struct_field("n").int64(), Int64(9))
    assert_equal(result.item().struct_field("s").string(), "z")


def test_terminal_struct_rejects_source_parent_nulls_and_intermediate_consumers() raises:
    # The ordinary primitive path must not run CPU schema evaluation merely
    # to decide that no struct adapter is involved.
    var plain = DataFrame([Series("x", Column[Int64]([1]))])
    assert_false(has_terminal_structs(plain.lazy().select(col("missing"))))
    var parent_bits: List[UInt8] = [1]
    var record = Series(
        "record",
        StructColumn([Series("hidden", Column[Int64]([1, 42]))], parent_bits^),
    )
    var frame = DataFrame([record.copy()])
    assert_true(frame.item(1, "record").is_null())
    assert_equal(
        record.struct_column().field("hidden").get(1).int64(), Int64(42)
    )
    with assert_raises(contains="nullable source struct transport"):
        _ = lower_terminal_structs(frame.lazy().select(col("record")))
    with assert_raises(contains="nested child fields"):
        _ = lower_terminal_structs(
            frame.lazy().select(as_struct([col("record")], "nested"))
        )
    var primitive = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var intermediate = (
        primitive.lazy()
        .with_columns(as_struct([col("x")], "record"))
        .select(col("record").field("x"))
    )
    with assert_raises(contains="intermediate struct consumers"):
        _ = lower_terminal_structs(intermediate)
    var plan = lower_terminal_structs(
        primitive.lazy().select(as_struct([col("x")], "record"))
    )
    with assert_raises(contains="exported field schema differs"):
        _ = assemble(DataFrame([Series("wrong", Column[Int64]([1, 2]))]), plan)
