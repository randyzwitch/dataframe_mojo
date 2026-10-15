"""Source transport proof layered on the resident EXPLODE storage milestone."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, ListColumn, Series
from dataframe.accelerator.gpu.residency import Scope
from dataframe.accelerator.gpu.source_fields import stage, staging_bytes
from dataframe.accelerator.gpu.explode import execute as explode
from dataframe.accelerator.gpu.runtime import materialize
from tests.support.resident_explode_reference import _source
from tests.support.coverage_generators import chunk_layout, offset_layout


def layouts_clean_metadata_and_private_cache() raises:
    var original = _source()
    for variant in range(3):
        var source = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var scope = Scope()
        if variant == 2:
            for field in source._columns:
                assert_equal(field._chunked.value()[].cache[].state.load(), 0)
        var fields = stage(source, scope)
        if variant == 2:
            assert_true(staging_bytes(source) > 0)
            for field in source._columns:
                assert_equal(field._chunked.value()[].cache[].state.load(), 0)
        for field in fields.fields:
            assert_equal(len(field.prototype), 0)
            if field.lists:
                assert_equal(len(field.prototype.list_column().child()), 0)
                var child = field.lists.value().child.schema._columns[0].copy()
                if child.dtype().is_decimal():
                    assert_equal(len(child.numeric[DType.int128]()._data[]), 0)
                else:
                    assert_equal(len(child.int64()._data[]), 0)
            else:
                assert_equal(len(field.prototype.int64()._data[]), 0)
        var output = materialize(explode(fields, ["xs", "wide"], scope))
        assert_true(output.equals(source.explode(["xs", "wide"])))
        assert_true(
            output.column("wide")._field_metadata.value().ptr()
            == source.column("wide")
            .list_column()
            .child()
            ._field_metadata.value()
            .ptr()
        )
        _ = fields^
    # Empty source metadata never merges or stages retained hidden child arrays.
    var empty_scope = Scope(999999, 0)
    var empty = stage(original.clear(), empty_scope)
    assert_equal(explode(empty, ["xs", "wide"], empty_scope).rows, 0)
    assert_equal(empty_scope.info()[1], 0)
    assert_equal(empty_scope.transfers()[2], 0)


def _run(budget: Int) raises -> Tuple[DataFrame, Int]:
    var source = chunk_layout(_source())
    var scope = Scope(budget=budget)
    var fields = stage(source, scope)
    var result = explode(fields, ["xs", "wide"], scope)
    var output = materialize(result)
    var peak = scope.info()[1]
    _ = result^
    _ = fields^
    return (output^, peak)


def reservations_exact_budget_and_early_admission() raises:
    var source = chunk_layout(_source())
    var bytes = staging_bytes(source)
    var invalid = Scope(999999, bytes - 1)
    with assert_raises(contains="memory budget"):
        _ = stage(source, invalid)
    assert_equal(invalid.transfers()[2], 0)
    for field in source._columns:
        assert_equal(field._chunked.value()[].cache[].state.load(), 0)
    var initial = _run(-1)
    var exact = _run(initial[1])
    assert_true(initial[0].equals(exact[0]))
    with assert_raises(contains="memory budget"):
        _ = _run(initial[1] - 1)
    assert_true(_run(-1)[0].equals(initial[0]))


def simultaneous_scratch_and_output_literal_budget() raises:
    # One Int64 child: 21 device bytes. One parent: 16 offsets + 1 validity
    # + 4 error = 21 descriptor bytes. Input owners therefore retain 42.
    var offsets: List[Int64] = [0, 1]
    var source = DataFrame(
        [Series("xs", ListColumn(offsets^, Series("item", Column[Int64]([7]))))]
    )
    # EXPLODE scratch = 8 logical + 16 prefix + 16 device controls + 16
    # host controls = 56. Output allocation needs 4 parent-map + 21 matrix.
    # All coexist: 42 + 56 + 4 + 21 = 123, independently of reported peak.
    var limited = Scope(budget=122)
    var input = stage(source, limited)
    assert_equal(limited.info()[0], 42)
    var before = limited.transfers()[2]
    with assert_raises(contains="memory budget"):
        _ = explode(input, ["xs"], limited)
    # Parent validation and full cardinality ran, but repeat/gather did not.
    assert_equal(limited.transfers()[2] - before, 2)
    assert_equal(limited.info()[0], 42)
    _ = input^
    assert_equal(limited.info()[0], 0)
    var exact = Scope(budget=123)
    var fields = stage(source, exact)
    var output = explode(fields, ["xs"], exact)
    assert_equal(output.rows, 1)
    assert_equal(exact.info()[0], 63)
    assert_equal(exact.info()[1], 123)
    _ = output^
    assert_equal(exact.info()[0], 42)
    _ = fields^
    assert_equal(exact.info()[0], 0)
