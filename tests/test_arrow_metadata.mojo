"""Lossless field metadata and unknown extension storage through row operations."""
from std.memory import Pointer
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    concat,
    export_arrow_series,
    import_arrow_series,
)
from dataframe.arrow import (
    _at,
    _SchemaState,
    _set_metadata,
    _export_series,
    _copy_metadata,
)
from dataframe.field_metadata import _equal_metadata
from dataframe.gather import take_parallel


def tagged(reverse: Bool = False, value: String = "payload") raises -> Series:
    var array = ArrowArray()
    var schema = ArrowSchema()
    var releases = 0
    _export_series(
        Series("x", Column[Int64]([3, 1, 2])),
        array,
        schema,
        Int(Pointer(to=releases)),
    )
    var keys = List[String](
        ["ARROW:extension:name", "ARROW:extension:metadata", "custom"]
    )
    var values = List[String](["example.unknown", value, "\0binary!"])
    if reverse:
        keys = ["custom", "ARROW:extension:metadata", "ARROW:extension:name"]
        values = ["\0binary!", value, "example.unknown"]
    _set_metadata(schema, keys, values)
    # A binary value must never pass through UTF-8 decoding.
    if not reverse:
        ref bytes = _at[_SchemaState](schema.private_data)[].metadata
        bytes[len(bytes) - 1] = 255
    var result = import_arrow_series(array, schema)
    assert_equal(releases, 1)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    return result^


def check_metadata(actual: Series, expected: Series) raises:
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(actual, array, schema)
    var exported = _copy_metadata(schema)
    assert_true(Bool(exported))
    assert_true(
        _equal_metadata(exported.value()[], expected._field_metadata.value()[])
    )
    var back = import_arrow_series(array, schema)
    assert_true(back.equals(actual))
    assert_true(
        _equal_metadata(
            back._field_metadata.value()[], expected._field_metadata.value()[]
        )
    )


def test_column_views_chunks_and_gathers() raises:
    var source = tagged()
    check_metadata(source.renamed("renamed"), source)
    check_metadata(source.slice(1, 2), source)
    check_metadata(source.take([2, 0]), source)
    check_metadata(source.take_or_null([2, -1, 0]), source)
    var chunked = Series._from_chunks([source.copy(), source.copy()])
    check_metadata(chunked, source)
    check_metadata(chunked.rechunk(), source)
    check_metadata(chunked.rechunk(), source)  # Cached materialization.
    check_metadata(chunked.slice(2, 3), source)
    check_metadata(chunked.slice(0, 0), source)
    check_metadata(chunked.take([5, 0, 4]), source)
    for chunk in chunked.chunks():
        check_metadata(chunk, source)
    var many = List[Series]()
    for _ in range(12):
        many.append(source.copy())
    var sparse = Series._from_chunks(many)
    check_metadata(sparse.take([35, 0]), source)
    check_metadata(take_parallel([source.copy()], [2, 0], 2)[0], source)


def test_frame_operations_and_conflicts() raises:
    var source = tagged()
    var frame = DataFrame([source.copy()])
    check_metadata(frame.select(["x"]).column("x"), source)
    check_metadata(frame.select(col("x").alias("y")).column("y"), source)
    check_metadata(frame.filter(col("x") > 1).column("x"), source)
    check_metadata(frame.sort("x").column("x"), source)
    check_metadata(concat([frame.copy(), frame.copy()]).column("x"), source)
    var other = DataFrame([Series("y", Column[Int64]([4]))])
    check_metadata(
        concat([frame.copy(), other^], "diagonal").column("x"), source
    )
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = concat([frame.copy(), DataFrame([tagged(value="different")])])
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = concat([frame.copy(), DataFrame([Series("x", Column[Int64]([4]))])])
    # Retagging changes the storage contract of an unknown extension.
    assert_true(not source.with_dtype(DataType.DATE)._field_metadata)


def test_pair_order_does_not_change_agreement() raises:
    var source = tagged(True)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    _set_metadata(
        schema,
        ["ARROW:extension:name", "ARROW:extension:metadata", "custom"],
        ["example.unknown", "payload", "\0binary!"],
    )
    var reordered = import_arrow_series(array, schema)
    check_metadata(Series._from_chunks([source.copy(), reordered^]), source)


def test_boolean_string_and_parallel_metadata() raises:
    var metadata = tagged()._field_metadata
    var columns = List[Series](
        [
            Series("b", Column[Bool]([True, False, True])),
            Series("s", Column[String](["c", "a", "b"])),
        ]
    )
    var rows = List[Int](length=131_072, fill=0)
    for i in range(len(rows)):
        rows[i] = i % 3
    for column in columns:
        var source = column.copy()
        source._field_metadata = metadata
        var gathered = take_parallel([source.copy()], rows.copy(), 2)[0].copy()
        check_metadata(gathered, source)
        var frame = DataFrame([gathered.copy()])
        var mask = List[Bool](length=len(gathered), fill=True)
        for i in range(0, len(mask), 3):
            mask[i] = False
        check_metadata(
            frame.filter(Column[Bool](mask^)).column(source.name()), source
        )
        check_metadata(
            Series._from_chunks([source.copy(), source.copy()]).rechunk(),
            source,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
