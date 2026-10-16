"""Row-index groups expose first-observed identities and complete memberships."""
# contract-group: table.group_indices
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series
from tests.support.key_kernel_reference import KeyCell, expected_key_ids
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_group_indices_match_first_observed_scalar_keys_and_owned_memberships() raises:
    var nan = Float64(0) / Float64(0)
    var floats: List[Float64] = [nan, 0.0, -0.0, nan, 3, 3, nan, 99, 101, 0, -5]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    var strings: List[String] = [
        "é",
        "",
        "",
        "é",
        "a\x00b",
        "a\x00b",
        "hidden",
        "different",
        "é",
        "",
        "東京",
    ]
    var sv: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    var integers: List[Int64] = [
        Int64.MAX,
        -1,
        -1,
        Int64.MAX,
        3,
        3,
        Int64.MIN,
        99,
        0,
        -1,
        Int64.MIN,
    ]
    var narrow = List[Float32]()
    var payload = List[Int64]()
    for row in range(11):
        narrow.append(floats[row].cast[DType.float32]())
        payload.append(Int64(4 * row - 9))
    var source = DataFrame(
        [
            Series("f", Column[Float64](floats.copy(), valid.copy())),
            Series("f32", Column[Float32](narrow^, valid.copy())),
            Series("s", Column[String](strings.copy(), sv.copy())),
            Series("i", Column[Int64](integers.copy(), valid.copy())),
            Series("payload", Column[Int64](payload^)),
        ]
    )
    for length in [0, 1, 11]:
        var raw = source.head(length)
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            for kind in range(5):
                var keys: List[String] = [
                    "s" if kind
                    == 1 else "i" if kind
                    == 2 else "f32" if kind
                    == 4 else "f"
                ]
                if kind == 3:
                    keys.append("s")
                var cells = List[List[KeyCell]]()
                for row in range(length):
                    var key = List[KeyCell]()
                    if kind == 0 or kind == 3 or kind == 4:
                        key.append(KeyCell(1, valid[row], floating=floats[row]))
                    if kind == 1 or kind == 3:
                        key.append(
                            KeyCell(2, sv[row], text=strings[row].copy())
                        )
                    if kind == 2:
                        key.append(
                            KeyCell(
                                0, valid[row], integer=Int128(integers[row])
                            )
                        )
                    cells.append(key^)
                var expected = expected_key_ids(cells, True)
                if length == 11 and kind == 0:
                    assert_equal(expected[0], [0, 1, 1, 0, 2, 2, 3, 3, 4, 1, 5])
                    assert_equal(expected[1], [0, 1, 4, 6, 8, 10])
                var actual = frame.group_indices(keys[0]) if len(
                    keys
                ) == 1 else frame.group_indices(keys)
                assert_equal(actual.height(), length)
                assert_equal(actual.count(), len(expected[1]))
                assert_equal(actual.ids(), expected[0])
                assert_equal(actual.representatives(), expected[1])
                var all_rows = actual.all_rows()
                var sizes = actual.sizes()
                assert_equal(len(all_rows), actual.count())
                assert_equal(len(sizes), actual.count())
                for row in range(length):
                    assert_equal(actual.group_of(row), expected[0][row])
                for group in range(actual.count()):
                    var members = List[Int]()
                    for row in range(length):
                        if expected[0][row] == group:
                            members.append(row)
                    assert_equal(
                        actual.representative(group), expected[1][group]
                    )
                    assert_equal(actual.rows(group), members)
                    assert_equal(all_rows[group], members)
                    assert_equal(sizes[group], len(members))
                    var panel = frame.take(actual.rows(group))
                    assert_equal(panel.columns(), frame.columns())
                    assert_equal(panel.height(), len(members))
                    assert_true(
                        panel.column("payload").dtype() == DataType.INT64
                    )
                    for position in range(len(members)):
                        assert_equal(
                            panel.item(position, "payload").int64(),
                            Int64(4 * members[position] - 9),
                        )
                # Returned arrays are caller-owned values, not mutable aliases.
                if length:
                    var ids = actual.ids()
                    ids[0] = 999
                    var representatives = actual.representatives()
                    representatives[0] = 999
                    all_rows[0].append(999)
                    sizes[0] = 999
                    assert_equal(actual.group_of(0), 0)
                    assert_equal(actual.representative(0), 0)
                    assert_true(999 not in actual.rows(0))
                    assert_true(actual.sizes()[0] != 999)
                for bad in [-1, actual.count()]:
                    with assert_raises(contains="Group index out of bounds"):
                        _ = actual.rows(bad)
                    with assert_raises(contains="Group index out of bounds"):
                        _ = actual.representative(bad)
                for bad in [-1, length]:
                    with assert_raises(contains="Row index out of bounds"):
                        _ = actual.group_of(bad)
            with assert_raises(contains="requires at least one key"):
                _ = frame.group_indices(List[String]())
            with assert_raises(contains="Duplicate group_indices key: f"):
                _ = frame.group_indices(["f", "f"])
            with assert_raises(contains="Unknown column: missing"):
                _ = frame.group_indices("missing")
