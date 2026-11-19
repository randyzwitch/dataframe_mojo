"""Retained row mappings keep source storage alive and preserve row semantics."""
# contract-group: execution.lifecycle
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.vector.selection_batches import SelectionBatch, SelectionBatches


def _selected() raises -> SelectionBatch:
    var source = Series(
        "a",
        Column[Int64]([90, 10, 20, 30, 40], [True, True, False, True, True]),
    ).slice(1, 4)
    var other = Series("b", Column[Int32]([7, 8, 9]))
    var batch = SelectionBatch()
    batch.morsel = 3
    batch.columns = [source.copy(), other.copy(), source.renamed("c")]
    batch.starts = [0, 0, 0]
    batch.lists = [0, 1, 0]
    batch.rows.append([UInt32(3), 1, 3, 0])
    batch.rows.append([UInt32(0), 2, 1, 2])
    batch.count = 4
    return batch^


def test_selection_batch_owns_sources_and_distinct_join_mappings() raises:
    # The local source series and the source window in _selected are gone.
    # Two columns share one selection; the joined column has its own order.
    var batch = _selected()
    batch.validate()
    assert_equal(len(batch.rows), 2)
    assert_true(batch.columns[0]._shares_buffers_with(batch.columns[2]))
    var result = batch.materialize()
    var expected = DataFrame(
        [
            Series(
                "a", Column[Int64]([40, 0, 40, 10], [True, False, True, True])
            ),
            Series("b", Column[Int32]([7, 9, 8, 9])),
            Series(
                "c", Column[Int64]([40, 0, 40, 10], [True, False, True, True])
            ),
        ]
    )
    assert_true(result.equals(expected))


def test_selection_batch_rejects_invalid_maps_before_submission() raises:
    var batch = _selected()
    batch.rows[0][0] = UInt32(4)
    with assert_raises(contains="outside"):
        batch.validate()
    batch.rows[0][0] = UInt32(3)
    batch.lists[1] = 2
    with assert_raises(contains="mapping"):
        batch.validate()
    batch.lists[1] = 1
    batch.starts[0] = Int.MAX
    with assert_raises(contains="outside"):
        batch.validate()
    batch.starts[0] = 0
    batch.count = 3
    with assert_raises(contains="length"):
        batch.validate()


def test_selection_batches_empty_output_preserves_schema() raises:
    var schema = DataFrame([Series("a", Column[Int64](List[Int64]()))])
    var batches = SelectionBatches(schema)
    with assert_raises(contains="ready"):
        batches.materialize()
    batches.ready = True
    assert_true(batches.materialize().equals(schema))


def test_selection_batch_flat_window_and_constant() raises:
    var batch = SelectionBatch()
    batch.columns = [
        Series("a", Column[Int32]([4, 5, 6, 7])),
        Series("b", Column[Int64]._repeated(9, 20, True)),
    ]
    batch.starts = [1, 0]
    batch.lists = [-1, 0]
    batch.rows.append([UInt32(0), 0, 0])
    batch.count = 3
    var expected = DataFrame(
        [
            Series("a", Column[Int32]([5, 6, 7])),
            Series("b", Column[Int64]([9, 9, 9])),
        ]
    )
    assert_true(batch.materialize().equals(expected))
    assert_true(batch.columns[1].is_constant())
