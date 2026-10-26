"""Null logical fields retain their schema through JSON and Arrow storage."""
# contract-group: dtype.null
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    export_arrow_series,
    import_arrow_series,
)
from dataframe.io.arrow import _read_c_string
from dataframe.json_decode import decode_json
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_null_dtype_all_null_storage_casts_and_frame_operations() raises:
    assert_equal(DataType.parse("null"), DataType.NULL)
    assert_equal(DataType.NULL.name(), "null")
    assert_equal(DataType.NULL.bit_width(), 0)
    assert_true(not DataType.NULL.is_numeric())
    with assert_raises(contains="every value"):
        _ = Series("n", Column[Int8]([0])).with_dtype(DataType.NULL)
    for n in [0, 11]:
        var raw = DataFrame([Series.full_null("n", DataType.NULL, n)])
        for layout in range(3):
            var frame = raw.copy() if layout == 0 else (
                offset_layout(raw) if layout == 1 else chunk_layout(raw)
            )
            var series = frame.column("n")
            assert_equal(series.dtype(), DataType.NULL)
            assert_equal(series.null_count(), n)
            for i in range(n):
                assert_true(series.get(i).is_null())
            var casted = series.cast(DataType.INT64)
            assert_equal(casted.dtype(), DataType.INT64)
            assert_equal(casted.null_count(), n)
            var counted = frame.select_exprs(
                [
                    col("n").count().alias("count"),
                    col("n").n_unique().alias("unique"),
                    col("n").null_count().alias("nulls"),
                ]
            )
            assert_equal(counted.column("count").get(0).int64(), Int64(0))
            assert_equal(counted.column("unique").get(0).int64(), Int64(n != 0))
            assert_equal(counted.column("nulls").get(0).int64(), Int64(n))
            var lazy = (
                frame.lazy()
                .select(col("n"))
                .collect(engine="cpu", batch_size=3)
            )
            assert_equal(lazy.column("n").dtype(), DataType.NULL)
            assert_equal(lazy.height(), n)
            assert_equal(frame.lazy().collect_schema(), [String("n: null")])


def test_json_decoder_null_fields_and_lists_retain_logical_type() raises:
    var dtype = DataType.struct(
        ["n", "items"], [DataType.NULL, DataType.list(DataType.NULL)]
    )
    var input = Series(
        "json",
        Column[String](
            [
                '{"n":null,"items":[null,null]}',
                '{"items":[]}',
                '{"items":null}',
            ]
        ),
    )
    var decoded = decode_json(input, dtype)
    assert_equal(decoded.dtype(), dtype)
    var frame = DataFrame([decoded.renamed("row")]).unnest("row")
    assert_equal(frame.column("n").dtype(), DataType.NULL)
    assert_equal(frame.column("n").null_count(), 3)
    var lists = frame.column("items").list_column()
    assert_equal(lists.child().dtype(), DataType.NULL)
    assert_equal(lists.child().null_count(), 2)
    var bad = Series("json", Column[String](["1"]))
    with assert_raises(contains="not null"):
        _ = decode_json(bad, DataType.NULL)
    var none = decode_json(
        Series("json", Column[String](["null"])), DataType.NULL
    )
    assert_equal(none.dtype(), DataType.NULL)
    assert_equal(none.null_count(), 1)


def test_arrow_null_arrays_have_no_payload_and_survive_consumed_producers() raises:
    for size in [0, 13]:
        for copy in [False, True]:
            var source = Series.full_null("n", DataType.NULL, size + 3).slice(
                3, size
            )
            var array = ArrowArray()
            var schema = ArrowSchema()
            export_arrow_series(source, array, schema)
            assert_equal(_read_c_string(schema.format), "n")
            assert_equal(array.n_buffers, Int64(0))
            assert_equal(array.buffers, 0)
            assert_equal(array.null_count, Int64(size))
            var result = import_arrow_series(array, schema, copy=copy)
            assert_equal(array.release, 0)
            assert_equal(schema.release, 0)
            assert_equal(result.dtype(), DataType.NULL)
            assert_equal(len(result), size)
            assert_equal(result.null_count(), size)
