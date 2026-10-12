"""Snapshots and nested aliases remain intact when append reassembly detaches."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from tests.support.reference_model import generated_model, ReferenceTable
from tests.support.coverage_generators import frame_from_model
from tests.support.coverage_compare import assert_model


def test_append_reassembly_preserves_nested_aliases() raises:
    for n in [1, 7, 8, 9, 63, 64, 65]:
        for mode in range(3):
            for offset in [0, 1, 7]:
                var model = generated_model(n, 3, mode)
                var extra = generated_model(9, 101, 1)
                var source = frame_from_model(model, offset)
                var alias = source.slice(1, n - 1).slice(0, n - 1)
                var right = frame_from_model(extra, 3)
                var i = source.column("i").int64()
                var f = source.column("f").float64()
                var s = source.column("s").string()
                var b = source.column("b").bool()
                assert_true(i._shares_buffers_with(source.column("i").int64()))
                i._append_column(right.column("i").int64())
                f._append_column(right.column("f").float64())
                s._append_column(right.column("s").string())
                b._append_column(right.column("b").bool())
                var appended = DataFrame(
                    [
                        Series("i", i^),
                        Series("f", f^),
                        Series("s", s^),
                        Series("b", b^),
                    ]
                )
                var rows = model.rows.copy()
                rows.extend(extra.rows.copy())
                assert_model(
                    appended, ReferenceTable(model.names.copy(), rows^)
                )
                assert_model(source, model)
                assert_model(alias, model.slice(1, n - 1))
                assert_model(right, extra)
                for name in source.columns():
                    assert_true(
                        source.column(name)._shares_buffers_with(
                            alias.column(name)
                        )
                    )


def test_payload_snapshot_and_unaligned_validity_are_separate_contracts() raises:
    var payloads = List[Int64]()
    var valid = List[Bool]()
    for row in range(100):
        payloads.append(Int64(1000 + row))
        valid.append(row % 3 != 1)
    var owner = Column[Int64](payloads.copy(), valid.copy())
    for offset in range(16):
        var view = owner.slice(offset, 65)
        assert_equal(view.validity_offset(), offset)
        assert_true(
            view.unsafe_values() == owner.unsafe_values().unsafe_offset(offset)
        )
        var raw = view.to_list()
        assert_equal(len(raw), 65)
        for row in range(65):
            assert_equal(raw[row], payloads[offset + row])
            assert_equal(view.is_valid(row), valid[offset + row])
            var bit = (
                Int(view.unsafe_validity()[unsafe_offset=(offset + row) // 8])
                >> ((offset + row) % 8)
            ) & 1
            assert_equal(bit != 0, valid[offset + row])
        # The owned snapshot is independent even though the view shares storage.
        raw[0] = -1
        assert_equal(view.to_list()[0], payloads[offset])
