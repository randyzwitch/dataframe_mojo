"""Independent logical List counts versus immutable hidden physical backing."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, ListColumn, Series, col
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var child = Series(
        "item", Column[Int64]([7, 8, 9, 10], [True, True, False, True])
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var offsets: List[Int64] = [0, 2, 2, 4, 4]
    var bits: List[UInt8] = [6]
    return DataFrame(
        [
            Series("xs", ListColumn(offsets^, child^, bits^)),
            Series("id", Column[Int64]([10, 11, 12, 13])),
        ]
    )


def _check(frame: DataFrame) raises:
    var output = frame.explode("xs")
    assert_equal(output.height(), 5)
    assert_equal(
        output.column("id").int64().to_list(),
        List[Int64]([Int64(10), Int64(11), Int64(12), Int64(12), Int64(13)]),
    )
    for row in range(5):
        assert_equal(output.item(row, "xs").is_null(), row != 3)
    assert_equal(output.item(3, "xs").int64(), Int64(10))
    assert_true(
        output.column("xs")._field_metadata.value().ptr()
        == frame.column("xs")
        .list_column()
        .child()
        ._field_metadata.value()
        .ptr()
    )
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [1, 64]:
                var lazy = (
                    frame.lazy()
                    .explode("xs")
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch,
                    )
                )
                assert_true(lazy.equals(output))


def test_list_logical_count_null_hidden_span_empty_and_layouts() raises:
    var source = _source()
    var lists = source.column("xs").list_column()
    assert_equal(lists._physical_count(0), 2)
    assert_equal(lists.element_count(0), 0)
    assert_equal(
        lists.lengths(), List[Int64]([Int64(0), Int64(0), Int64(2), Int64(0)])
    )
    assert_equal(len(lists.row(0)), 0)
    assert_true(
        lists.row(0)._field_metadata.value().ptr()
        == lists.child()._field_metadata.value().ptr()
    )
    assert_equal(lists.child().int64().value(0), Int64(7))
    assert_equal(lists.child().int64().value(1), Int64(8))
    for variant in range(3):
        var frame = (
            source.copy() if variant
            == 0 else offset_layout(source) if variant
            == 1 else chunk_layout(source)
        )
        _check(frame)
    var empty = source.clear().explode("xs")
    assert_equal(empty.height(), 0)
    assert_true(
        empty.column("xs")._field_metadata.value().ptr()
        == lists.child()._field_metadata.value().ptr()
    )


def test_list_physical_backing_take_concat_slice_preserves_logical_counts() raises:
    var source = _source()
    var xs = source.column("xs")
    var taken = xs.take(List[Int]([2, 0, 1]))
    var lists = taken.list_column()
    assert_equal(
        lists._offsets[], List[Int64]([Int64(0), Int64(2), Int64(4), Int64(4)])
    )
    assert_equal(lists.lengths(), List[Int64]([Int64(2), Int64(0), Int64(0)]))
    assert_equal(lists.child().int64().value(2), Int64(7))
    assert_equal(len(lists.row(1)), 0)
    var missing = xs.take_or_null(List[Int]([-1, 0, 2]))
    assert_equal(
        missing.list_column().lengths(),
        List[Int64]([Int64(0), Int64(0), Int64(2)]),
    )
    var joined = xs.slice(0, 2).append(xs.slice(2, 2))
    assert_true(joined.equals(xs))
    assert_equal(
        joined.list_column()._offsets[],
        List[Int64]([Int64(0), Int64(2), Int64(2), Int64(4), Int64(4)]),
    )
    assert_equal(len(joined.list_column().child()), 4)
    _check(DataFrame([joined^, source.column("id")]))
    var sliced = source.slice(1, 3).explode("xs")
    assert_equal(
        sliced.column("id").int64().to_list(),
        List[Int64]([Int64(11), Int64(12), Int64(12), Int64(13)]),
    )
    assert_true(sliced.item(0, "xs").is_null())
    assert_equal(sliced.item(2, "xs").int64(), Int64(10))


def test_explode_multiple_lists_compare_logical_counts_before_placeholders() raises:
    var source = _source()
    var offsets: List[Int64] = [0, 1, 1, 3, 3]
    var bits: List[UInt8] = [6]
    var ys = Series(
        "ys",
        ListColumn(
            offsets^, Series("item", Column[Int64]([999, 100, 200])), bits^
        ),
    )
    var frame = source.with_column(ys^)
    var actual = frame.explode(["xs", "ys"])
    assert_equal(actual.height(), 5)
    assert_true(actual.item(0, "xs").is_null())
    assert_true(actual.item(0, "ys").is_null())
    assert_equal(actual.item(2, "ys").int64(), Int64(100))
    assert_equal(actual.item(3, "ys").int64(), Int64(200))
    var bad_offsets: List[Int64] = [0, 1, 2, 4, 4]
    var bad_bits: List[UInt8] = [6]
    var bad = source.with_column(
        Series(
            "ys",
            ListColumn(
                bad_offsets^,
                Series("item", Column[Int64]([999, 1, 100, 200])),
                bad_bits^,
            ),
        )
    )
    with assert_raises(contains="different element counts at row 1"):
        _ = bad.explode(["xs", "ys"])
    with assert_raises(contains="different element counts at row 1"):
        _ = bad.lazy().explode(["xs", "ys"]).head(0).collect(engine="cpu")
