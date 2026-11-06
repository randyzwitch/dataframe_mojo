"""Native tuple-key matching and ticket identity before GPU integration."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false
from std.collections import Dict
from std.memory import Pointer
from dataframe import Column, DataFrame, Series
from dataframe_metal.group_keys import compile_group_keys
from dataframe_metal.accelerator.gpu.multi_reduce import Bytes
from dataframe_metal.accelerator.gpu.group_tuple_hash import (
    claim_tuple,
    tuple_hash,
    GROUP_CAPACITY_FAULT,
    MAX_GENERAL_CAPACITY,
    MAX_GENERAL_GROUPS,
)
from dataframe_metal.accelerator.gpu.group_hash import NO_GROUP


def test_metal_group_tuple_key_schema_keeps_native_integer_types() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([0])),
            Series("b", Column[Int64]([0])),
            Series("c", Column[Int32]([0])),
            Series("d", Column[Int64]([0])),
            Series("f", Column[Float32]([0])),
        ]
    )
    assert_true(Bool(compile_group_keys(["a"], frame)))
    assert_true(Bool(compile_group_keys(["a", "b", "c", "d"], frame)))
    assert_false(Bool(compile_group_keys(["a", "f"], frame)))
    assert_false(Bool(compile_group_keys(["a", "missing"], frame)))
    assert_false(Bool(compile_group_keys(["a", "a"], frame)))
    assert_false(Bool(compile_group_keys(List[String](), frame)))
    assert_false(Bool(compile_group_keys(["a", "b", "c", "d", "a"], frame)))


def test_metal_group_tuple_ticket_identity_ignores_null_payloads() raises:
    var a = List[Int32](length=133, fill=0)
    var b = List[Int64](length=133, fill=0)
    var c = List[Int32](length=133, fill=0)
    var d = List[Int64](length=133, fill=0)
    var ab = List[UInt8](length=32, fill=0)
    var bb = List[UInt8](length=32, fill=0)
    var cb = List[UInt8](length=32, fill=0)
    var db = List[UInt8](length=32, fill=0)
    for row in range(128):
        var g = row % 32
        var mask = g % 16
        a[row + 5] = Int32(g - 16 if mask & 1 else row + 4096)
        b[row + 5] = Int64.MIN + Int64(g) if mask & 2 else Int64.MAX - Int64(
            row
        )
        c[row + 5] = Int32(g // 8 - 2 if mask & 4 else row + 8192)
        d[row + 5] = Int64.MAX - Int64(g) if mask & 8 else Int64.MIN + Int64(
            row
        )
        for key in range(4):
            var bit = row + 3 + key * 2
            if mask & (1 << key):
                if key == 0:
                    ab[bit // 8] |= UInt8(1) << UInt8(bit % 8)
                elif key == 1:
                    bb[bit // 8] |= UInt8(1) << UInt8(bit % 8)
                elif key == 2:
                    cb[bit // 8] |= UInt8(1) << UInt8(bit % 8)
                else:
                    db[bit // 8] |= UInt8(1) << UInt8(bit % 8)
    var ap = Bytes(unsafe_from_address=Int(a.unsafe_ptr().unsafe_offset(5)))
    var bp = Bytes(unsafe_from_address=Int(b.unsafe_ptr().unsafe_offset(5)))
    var cp = Bytes(unsafe_from_address=Int(c.unsafe_ptr().unsafe_offset(5)))
    var dp = Bytes(unsafe_from_address=Int(d.unsafe_ptr().unsafe_offset(5)))
    var av = Bytes(unsafe_from_address=Int(ab.unsafe_ptr()))
    var bv = Bytes(unsafe_from_address=Int(bb.unsafe_ptr()))
    var cv = Bytes(unsafe_from_address=Int(cb.unsafe_ptr()))
    var dv = Bytes(unsafe_from_address=Int(db.unsafe_ptr()))
    for keys in range(1, 5):
        var words = List[UInt64](length=16, fill=0)
        words[3] = 512
        words[4] = 256
        words[11] = UInt64(keys)
        for key in range(4):
            words[12 + key] = UInt64(key % 2) | 2 | (UInt64(3 + key * 2) << 2)
        var metadata = Pointer[UInt64, UntrackedOrigin[mut=False]](
            unsafe_from_address=Int(words.unsafe_ptr())
        )
        var table = List[UInt32](length=512, fill=0)
        var control = List[UInt32](length=4, fill=0)
        var expected = Dict[String, UInt32]()
        var occupied = Dict[UInt32, Bool]()
        for row in range(128):
            var g = row % 32
            var mask = g % 16
            var identity = String()
            for key in range(keys):
                if not (mask & (1 << key)):
                    identity += ":null"
                else:
                    var value = (
                        Int64(g - 16) if key
                        == 0 else Int64.MIN + Int64(g) if key
                        == 1 else Int64(g // 8 - 2) if key
                        == 2 else Int64.MAX - Int64(g)
                    )
                    identity += ":" + String(value)
            var slot = claim_tuple(
                metadata,
                row,
                ap,
                bp,
                cp,
                dp,
                av,
                bv,
                cv,
                dv,
                Pointer[UInt32, MutUntrackedOrigin](
                    unsafe_from_address=Int(table.unsafe_ptr())
                ),
                Pointer[UInt32, MutUntrackedOrigin](
                    unsafe_from_address=Int(control.unsafe_ptr())
                ),
            )
            assert_true(slot != NO_GROUP)
            if identity in expected:
                assert_equal(slot, expected[identity])
            else:
                assert_false(slot in occupied)
                expected[identity] = slot
                occupied[slot] = True
            assert_equal(control[1], UInt32(0))
        assert_equal(control[0], UInt32(len(expected)))
        assert_equal(
            tuple_hash(metadata, 0, ap, bp, cp, dp, av, bv, cv, dv),
            tuple_hash(metadata, 32, ap, bp, cp, dp, av, bv, cv, dv),
        )
        _ = (words^, table^, control^)
    _ = (a^, b^, c^, d^, ab^, bb^, cb^, db^)


def test_metal_group_tuple_capacity_status_distinguishes_data_faults() raises:
    var values = List[Int64]()
    for row in range(MAX_GENERAL_GROUPS):
        values.append(
            Int64.MIN if row == 0 else Int64.MAX if row == 1 else Int64(row)
        )
    values.append(Int64.MIN)
    values.append(Int64(MAX_GENERAL_GROUPS + 1))
    var words = List[UInt64](length=16, fill=0)
    words[3] = UInt64(MAX_GENERAL_CAPACITY)
    words[4] = UInt64(MAX_GENERAL_GROUPS)
    words[11] = 1
    words[12] = 1
    var metadata = Pointer[UInt64, UntrackedOrigin[mut=False]](
        unsafe_from_address=Int(words.unsafe_ptr())
    )
    var data = Bytes(unsafe_from_address=Int(values.unsafe_ptr()))
    var empty = data
    var table = List[UInt32](length=MAX_GENERAL_CAPACITY, fill=0)
    var control = List[UInt32](length=4, fill=0)
    var first_slot = UInt32(0)
    for row in range(MAX_GENERAL_GROUPS + 1):
        var slot = claim_tuple(
            metadata,
            row,
            data,
            data,
            data,
            data,
            empty,
            empty,
            empty,
            empty,
            Pointer[UInt32, MutUntrackedOrigin](
                unsafe_from_address=Int(table.unsafe_ptr())
            ),
            Pointer[UInt32, MutUntrackedOrigin](
                unsafe_from_address=Int(control.unsafe_ptr())
            ),
        )
        assert_true(slot != NO_GROUP)
        if row == 0:
            first_slot = slot
        elif row == MAX_GENERAL_GROUPS:
            assert_equal(slot, first_slot)
    assert_equal(control[0], UInt32(MAX_GENERAL_GROUPS))
    assert_equal(control[1], UInt32(0))
    assert_equal(
        claim_tuple(
            metadata,
            MAX_GENERAL_GROUPS + 1,
            data,
            data,
            data,
            data,
            empty,
            empty,
            empty,
            empty,
            Pointer[UInt32, MutUntrackedOrigin](
                unsafe_from_address=Int(table.unsafe_ptr())
            ),
            Pointer[UInt32, MutUntrackedOrigin](
                unsafe_from_address=Int(control.unsafe_ptr())
            ),
        ),
        NO_GROUP,
    )
    assert_equal(control[1], GROUP_CAPACITY_FAULT)
    _ = (values^, words^, table^, control^)


def test_metal_group_general_layout_bounds_state_without_row_arrays() raises:
    from dataframe_metal.group_layout import GroupLayout
    from std.testing import assert_raises

    var low = GroupLayout(512, 4, 128, False)
    low.check()
    var general = GroupLayout(131072, 4, 128, True)
    general.check()
    assert_equal(low.state_blocks(), 128)
    assert_equal(general.state_blocks(), 1)
    assert_equal(general.entries(), 131072 * 4)
    assert_equal(general.words(), 4 + 131072 * 3 + 131072 * 4 * 32)
    assert_true(general.words() < low.words() * 3)
    assert_equal(general.resolved_values(), general.entries() * 2)
    assert_true(
        general.words() * 4 + general.resolved_values() * 8 < 80 * 1024 * 1024
    )
    with assert_raises(contains="low-cardinality table"):
        GroupLayout(131072, 4, 128, False).check()
    with assert_raises(contains="power of two"):
        GroupLayout(513, 4, 128, True).check()
    with assert_raises(contains="invalid field"):
        GroupLayout(512, 5, 128, True).check()


def test_metal_group_tuple_finish_copies_flat_keys_and_preserves_row_counts() raises:
    from dataframe import col
    from dataframe_metal.group_plan import plan_grouped, grouped_key
    from dataframe_metal.group_finish import finish_grouped
    from dataframe_metal.accelerator.gpu.group_state import GROUP_STATE_WORDS

    var frame = DataFrame(
        [
            Series(
                "a", Column[Int32]([0, 3, 0, 3], [False, True, False, True])
            ),
            Series(
                "b",
                Column[Int64](
                    [Int64.MIN, Int64.MIN, 999, 1000],
                    [True, True, False, False],
                ),
            ),
            Series("v", Column[Int32]([1, 2, 3, 4])),
        ]
    )
    # General matching remains explicit until its GPU executor is integrated.
    var matched = plan_grouped(
        frame.lazy()
        .group_by(["a", "b"])
        .agg(
            [
                col("v").sum().alias("sum"),
                col("v").len().alias("rows"),
            ]
        ),
        max_keys=4,
    )
    assert_true(Bool(matched))
    assert_equal(matched.value().keys.names, List[String](["a", "b"]))
    assert_true(
        grouped_key(matched.value()).startswith("grouped_reductions:v2:tuple:")
    )
    var fields = len(matched.value().selected.bundle.fields)
    var first = List[UInt32](length=5, fill=UInt32.MAX)
    var rows = List[UInt32](length=5, fill=0)
    var states = List[UInt32](length=5 * fields * GROUP_STATE_WORDS, fill=0)
    var extrema = List[Int64](length=5 * fields * 2, fill=0)
    var slots = List[Int]([3, 0, 4, 1])
    for row in range(4):
        var slot = slots[row]
        first[slot] = UInt32(row)
        rows[slot] = 3
        for field in range(fields):
            var at = (slot * fields + field) * GROUP_STATE_WORDS
            if matched.value().selected.bundle.fields[field].left == "v":
                states[at] = 2
                states[at + 1] = UInt32(2 * (row + 1))
    var answer = finish_grouped(
        matched.value(), frame, 1, first, rows, states, extrema
    )
    var expected = DataFrame(
        [
            Series(
                "a", Column[Int32]([0, 3, 0, 3], [False, True, False, True])
            ),
            Series(
                "b",
                Column[Int64](
                    [Int64.MIN, Int64.MIN, 0, 0], [True, True, False, False]
                ),
            ),
            Series("sum", Column[Int32]([2, 4, 6, 8])),
            Series("rows", Column[Int64]([3, 3, 3, 3])),
        ]
    )
    assert_true(answer.equals(expected))
    for name in answer.columns():
        assert_equal(len(answer.column(name).chunks()), 1)
    for name in ["a", "b"]:
        if name == "a":
            assert_false(
                answer.column(name)
                .numeric[DType.int32]()
                ._shares_buffers_with(frame.column(name).numeric[DType.int32]())
            )
        else:
            assert_false(
                answer.column(name)
                .numeric[DType.int64]()
                ._shares_buffers_with(frame.column(name).numeric[DType.int64]())
            )
