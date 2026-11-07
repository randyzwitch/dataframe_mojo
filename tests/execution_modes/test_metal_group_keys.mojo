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


def test_metal_group_row_counts_do_not_consume_value_or_predicate_slots() raises:
    from dataframe import Expr, col
    from dataframe_metal.group_plan import plan_grouped

    var columns = List[Series]()
    for i in range(8):
        columns.append(Series("v" + String(i), Column[Int64]([Int64(i + 1)])))
    columns.append(Series("p", Column[Int32]([1])))
    columns.append(Series("a", Column[Int32]([0])))
    columns.append(Series("b", Column[Int64]([0])))
    var frame = DataFrame(columns^)
    var outputs = List[Expr]()
    outputs.append(col("v0").len().alias("rows0"))
    for i in range(4):
        outputs.append(
            (col("v" + String(i * 2)) + col("v" + String(i * 2 + 1)))
            .sum()
            .alias("s" + String(i))
        )
        if i == 1:
            outputs.append(col("v1").len().alias("rows1"))
    outputs.append(col("v7").len().alias("rows2"))
    var matched = plan_grouped(
        frame.lazy().filter(col("p") > 0).group_by(["a", "b"]).agg(outputs)
    )
    assert_true(Bool(matched))
    assert_equal(len(matched.value().selected.bundle.fields), 4)
    assert_equal(len(matched.value().selected.bundle.inputs), 8)
    assert_equal(matched.value().selected.bundle.predicate.filter_column, "p")
    assert_equal(matched.value().selected.program.slots, List[Int]([8]))
    assert_equal(len(matched.value().outputs), 7)
    assert_equal(
        matched.value().row_counts, List[String](["rows0", "rows1", "rows2"])
    )
    var expected_names = List[String](
        ["rows0", "s0", "s1", "rows1", "s2", "s3", "rows2"]
    )
    for i in range(7):
        assert_equal(matched.value().outputs[i].name, expected_names[i])
        assert_equal(
            matched.value().outputs[i].field < 0, i == 0 or i == 3 or i == 6
        )
    var only_rows = plan_grouped(
        frame.lazy()
        .group_by(["a", "b"])
        .agg(
            [
                col("v0").len().alias("rows0"),
                col("v1").len().alias("rows1"),
            ]
        )
    )
    assert_true(Bool(only_rows))
    assert_equal(len(only_rows.value().selected.bundle.fields), 1)
    assert_equal(len(only_rows.value().outputs), 2)
    assert_equal(only_rows.value().outputs[0].field, -1)
    assert_equal(only_rows.value().outputs[1].field, -1)
    assert_false(
        Bool(
            plan_grouped(
                frame.lazy()
                .group_by("a")
                .agg(
                    [
                        col("v0").len().alias("r"),
                        col("v1").sum().alias("r"),
                    ]
                )
            )
        )
    )
    assert_false(
        Bool(plan_grouped(frame.lazy().group_by("a").agg(col("missing").len())))
    )


def test_metal_group_capacity_marker_preserves_checked_fault_bits() raises:
    from dataframe_metal.accelerator.gpu.group_hash import mark_capacity_fault

    var fault = List[UInt32]([UInt32(3)])
    var pointer = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(fault.unsafe_ptr())
    )
    mark_capacity_fault(pointer)
    assert_equal(fault[0], GROUP_CAPACITY_FAULT | UInt32(3))
    mark_capacity_fault(pointer)
    assert_equal(fault[0], GROUP_CAPACITY_FAULT | UInt32(3))
    _ = fault^


def test_metal_group_tuple_sampling_does_not_prove_actual_cardinality() raises:
    from dataframe_metal.group_properties import sample_group_tuples

    var sampled = List[Bool](length=8192, fill=False)
    var random = UInt64(0x9E3779B97F4A7C15)
    for i in range(4096):
        random = random * UInt64(6364136223846793005) + UInt64(1)
        sampled[Int((random >> 32) % 8192)] = True
    var a = List[Int32](length=8192, fill=0)
    var b = List[Int64](length=8192, fill=0)
    var rare = 0
    for row in range(1, 8192):
        if not sampled[row] and rare < 300:
            rare += 1
            b[row] = Int64(rare)
    assert_equal(rare, 300)
    var frame = DataFrame(
        [Series("a", Column[Int32](a^)), Series("b", Column[Int64](b^))]
    )
    var sample = sample_group_tuples(frame, ["a", "b"])
    assert_equal(sample.distinct, 1)
    assert_equal(sample.samples, 4096)
    assert_false(sample.global_state())
    assert_equal(sample.capacity(), 512)
    assert_true(":stateworkgroup:capacity512" in sample.key())
    # The exact small-frame sample can identify a global-state workload.
    var keys = List[Int32]()
    for row in range(257):
        keys.append(Int32(row))
    var distinct = sample_group_tuples(
        DataFrame([Series("a", Column[Int32](keys^))]), ["a"]
    )
    assert_equal(distinct.distinct, 257)
    assert_true(distinct.global_state())
    assert_equal(distinct.capacity(), 1024)


def _matrix_tuple_keys(frame: DataFrame) raises -> DataFrame:
    var columns = List[Series]()
    for name in frame.columns():
        columns.append(frame.column(name))
    var k2 = List[Int32]()
    var k3 = List[Int64]()
    var k4 = List[Int32]()
    var valid = List[Bool]()
    var odd = List[Bool]()
    for row in range(frame.height()):
        var item = frame.item(row, "k")
        var present = not item.is_null()
        var key = item.int64() if present else Int64(0)
        k2.append(Int32(key + 8) if present else Int32(0))
        k3.append(Int64.MIN + key if present else Int64(0))
        k4.append(Int32(-key) if present else Int32(0))
        valid.append(present)
        odd.append(present and key % 2 != 0)
    columns.append(Series("k2", Column[Int32](k2^, valid)))
    columns.append(Series("k3", Column[Int64](k3^, valid)))
    columns.append(Series("k4", Column[Int32](k4^, odd)))
    return DataFrame(columns^)


def _conditional_matrix_answer() raises -> DataFrame:
    from tests.support.metal_matrix_reference import _present, _value, _typed

    var sums = List[Float64](length=20, fill=0)
    var counts = List[Int64](length=20, fill=0)
    var low = List[Float64](length=20, fill=Float64.MAX_FINITE)
    var high = List[Float64](length=20, fill=-Float64.MAX_FINITE)
    var seen = List[Bool](length=5, fill=False)
    var order = List[Int]()
    for row in range(32):
        if _present(8, row) and _value(8, row) <= 0:
            continue
        var group = row % 4 if _present(9, row) else 4
        if not seen[group]:
            seen[group] = True
            order.append(group)
        for field in range(4):
            var value = Float64(0)
            if _present(8, row):
                if not _present(field * 2, row) or not _present(
                    field * 2 + 1, row
                ):
                    continue
                var a = _value(field * 2, row)
                var b = _value(field * 2 + 1, row)
                value = Float64(
                    a * b + a if field
                    == 0 else a - b + a if field
                    == 1 else a * b + 1 if field
                    == 2 else (a + b) * a
                )
            var at = group * 4 + field
            sums[at] += value
            counts[at] += 1
            low[at] = min(low[at], value)
            high[at] = max(high[at], value)
    var keys = List[Int64]()
    var key_valid = List[Bool]()
    for group in order:
        keys.append(Int64(group if group < 4 else 0))
        key_valid.append(group < 4)
    var columns = List[Series]([Series("k", Column[Int64](keys^, key_valid))])
    for field in range(4):
        var totals = List[Float64]()
        var ns = List[Int64]()
        var means = List[Float64]()
        var lows = List[Float64]()
        var highs = List[Float64]()
        var sum_valid = List[Bool]()
        var valid = List[Bool]()
        for group in order:
            var at = group * 4 + field
            totals.append(sums[at])
            ns.append(counts[at])
            means.append(
                sums[at] / Float64(counts[at]) if counts[at] else Float64(0)
            )
            lows.append(low[at] if counts[at] else Float64(0))
            highs.append(high[at] if counts[at] else Float64(0))
            sum_valid.append(counts[at] >= 2 if field == 3 else True)
            valid.append(counts[at] > 0)
        var suffix = String(field)
        columns.append(_typed(field, "s" + suffix, totals, sum_valid))
        columns.append(Series("n" + suffix, Column[Int64](ns^)))
        columns.append(Series("avg" + suffix, Column[Float64](means^, valid)))
        columns.append(_typed(field, "lo" + suffix, lows, valid))
        columns.append(_typed(field, "hi" + suffix, highs, valid))
    return DataFrame(columns^)


# requires: metal
def test_metal_group_tuple_four_values_ninth_predicate_case_membership_and_rows() raises:
    from dataframe import Expr, col, lit, when
    from tests.support.metal_matrix_reference import (
        matrix_frame,
        matrix_expressions,
        matrix_reference,
    )
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed
    from dataframe_metal.group_plan import plan_grouped
    from dataframe_metal.composition_plan import has_conditions

    var runtime = require_metal()
    for conditional in [False, True]:
        var base = (
            _conditional_matrix_answer() if conditional else matrix_reference(
                True
            )
        )
        var full = _matrix_tuple_keys(base)
        var row_counts = List[Int64](length=base.height(), fill=0)
        for row in range(32):
            if row % 5 != 0 and row % 3 != 2:
                continue
            for group in range(base.height()):
                var item = base.item(group, "k")
                if (item.is_null() and row % 7 == 0) or (
                    not item.is_null()
                    and row % 7 != 0
                    and item.int64() == Int64(row % 4)
                ):
                    row_counts[group] += 1
        var expected_columns = List[Series]()
        for name in ["k", "k2", "k3", "k4"]:
            expected_columns.append(full.column(name))
        expected_columns.append(
            Series("rows_first", Column[Int64](row_counts.copy()))
        )
        for name in base.columns():
            if name != "k":
                expected_columns.append(base.column(name))
        expected_columns.append(Series("rows_last", Column[Int64](row_counts^)))
        var expected = DataFrame(expected_columns^)
        for fragmented in [False, True]:
            var frame = _matrix_tuple_keys(matrix_frame(fragmented))
            var outputs = List[Expr]([col("a").len().alias("rows_first")])
            var member = col("p").is_in([lit(Int32(1))])
            if conditional:
                var values = List[Expr](
                    [
                        col("a") * col("b") + col("a"),
                        col("l") - col("m") + col("l"),
                        col("f") * col("g") + 1,
                        (col("h") + col("j")) * col("h"),
                    ]
                )
                for field in range(4):
                    var value = (
                        when(member.copy())
                        .then(values[field].copy())
                        .otherwise(0)
                    )
                    var suffix = String(field)
                    outputs.append(
                        value.sum(min_count=2 if field == 3 else 0).alias(
                            "s" + suffix
                        )
                    )
                    outputs.append(value.count().alias("n" + suffix))
                    outputs.append(value.mean().alias("avg" + suffix))
                    outputs.append(value.min().alias("lo" + suffix))
                    outputs.append(value.max().alias("hi" + suffix))
            else:
                for expression in matrix_expressions():
                    outputs.append(expression.copy())
            outputs.append(col("j").len().alias("rows_last"))
            var query = (
                frame.lazy()
                .filter(member | col("p").is_null())
                .group_by(["k", "k2", "k3", "k4"], maintain_order=True)
                .agg(outputs)
            )
            var matched = plan_grouped(query)
            assert_true(Bool(matched))
            assert_equal(len(matched.value().selected.bundle.fields), 4)
            assert_equal(len(matched.value().selected.bundle.inputs), 8)
            assert_equal(
                has_conditions(matched.value().selected.bundle), conditional
            )
            for copied in [False, True]:
                var answer = execute_timed(query, zero_copy=not copied)[
                    0
                ].copy()
                assert_true(answer.equals(expected))
                for name in answer.columns():
                    assert_equal(len(answer.column(name).chunks()), 1)
            assert_true(query.collect(accelerator=runtime).equals(expected))


def _rare_tuple_fixture() raises -> Tuple[DataFrame, DataFrame]:
    var sampled = List[Bool](length=8192, fill=False)
    var random = UInt64(0x9E3779B97F4A7C15)
    for i in range(4096):
        random = random * UInt64(6364136223846793005) + UInt64(1)
        sampled[Int((random >> 32) % 8192)] = True
    var a = List[Int32](length=8197, fill=0)
    var b = List[Int64](length=8197, fill=0)
    var values = List[Float32](length=8197, fill=0)
    var valid = List[Bool](length=8197, fill=False)
    var totals = List[Float64](length=301, fill=0)
    var counts = List[Int64](length=301, fill=0)
    var rows = List[Int64](length=301, fill=0)
    var minimum = List[Float32](length=301, fill=Float32.MAX_FINITE)
    var maximum = List[Float32](length=301, fill=-Float32.MAX_FINITE)
    var rare = 0
    for row in range(8192):
        var group = 0
        if row > 0 and not sampled[row] and rare < 300:
            rare += 1
            group = rare
        b[row + 5] = Int64(group)
        values[row + 5] = Float32(row % 8 - 4)
        valid[row + 5] = row % 11 != 3
        rows[group] += 1
        if valid[row + 5]:
            totals[group] += Float64(row % 8 - 4)
            counts[group] += 1
            minimum[group] = min(minimum[group], Float32(row % 8 - 4))
            maximum[group] = max(maximum[group], Float32(row % 8 - 4))
    assert_equal(rare, 300)
    var input = DataFrame(
        [
            Series("a", Column[Int32](a^)),
            Series("b", Column[Int64](b^)),
            Series("v", Column[Float32](values^, valid)),
        ]
    ).slice(5, 8192)
    var keys = List[Int64]()
    var sum_values = List[Float32]()
    var means = List[Float64]()
    var nonempty = List[Bool]()
    for group in range(301):
        keys.append(Int64(group))
        sum_values.append(Float32(totals[group]))
        means.append(
            totals[group]
            / Float64(counts[group]) if counts[group] else Float64(0)
        )
        nonempty.append(counts[group] > 0)
    var expected = DataFrame(
        [
            Series("a", Column[Int32](List[Int32](length=301, fill=0))),
            Series("b", Column[Int64](keys^)),
            Series("s", Column[Float32](sum_values^)),
            Series("n", Column[Int64](counts^)),
            Series("avg", Column[Float64](means^, nonempty)),
            Series("lo", Column[Float32](minimum^, nonempty)),
            Series("hi", Column[Float32](maximum^, nonempty)),
            Series("rows", Column[Int64](rows^)),
        ]
    )
    return (input^, expected^)


# requires: metal
def test_metal_group_tuple_capacity_retry_handles_unsampled_groups() raises:
    from dataframe import col
    from dataframe_metal.group_plan import plan_grouped
    from dataframe_metal.group_properties import sample_group_tuples
    from dataframe_metal.group_general_reduce import general_partials
    from dataframe_metal.group_finish import finish_grouped
    from dataframe_metal.handoff import execute_timed
    from dataframe_metal.accelerator.gpu.query_scope import (
        Scope,
        shared_service,
    )
    from dataframe_metal.accelerator.gpu.reduction_inputs import HandoffTimings
    from tests.support.metal_outcomes import require_metal
    from std.testing import assert_raises

    var runtime = require_metal()
    var fixture = _rare_tuple_fixture()
    for fragmented in [False, True]:
        var frame = fixture[0].copy()
        if fragmented:
            var columns = List[Series]()
            for name in frame.columns():
                var parts = List[Series]()
                for part in range(8):
                    parts.append(frame.column(name).slice(part * 1024, 1024))
                columns.append(Series._from_chunks(parts))
            frame = DataFrame(columns^)
        assert_equal(sample_group_tuples(frame, ["a", "b"]).distinct, 1)
        var query = (
            frame.lazy()
            .group_by(["a", "b"], maintain_order=True)
            .agg(
                [
                    col("v").sum().alias("s"),
                    col("v").count().alias("n"),
                    col("v").mean().alias("avg"),
                    col("v").min().alias("lo"),
                    col("v").max().alias("hi"),
                    col("v").len().alias("rows"),
                ]
            )
        )
        var matched = plan_grouped(query)
        assert_true(Bool(matched))
        var scope = Scope(service=shared_service())
        var timings = HandoffTimings()
        var partials = general_partials(
            matched.value(), frame, scope, True, timings
        )
        # Two initialization/scan pairs prove a device retry, rather than a
        # precomputed CPU grouping or a single sampled-capacity scan.
        assert_equal(scope.transfers()[2], 4)
        assert_equal(scope.transfers()[3], 2)
        var answer = finish_grouped(
            matched.value(),
            frame,
            partials.blocks,
            partials.first,
            partials.rows,
            partials.states,
            partials.extrema,
        )
        assert_true(answer.equals(fixture[1]))
        assert_equal(scope.info()[0], 0)
        scope.mark_success()
        assert_true(execute_timed(query, zero_copy=False)[0].equals(fixture[1]))
        assert_true(query.collect(accelerator=runtime).equals(fixture[1]))
        var limited = Scope(service=shared_service(), budget=32)
        with assert_raises(contains="memory budget"):
            _ = general_partials(matched.value(), frame, limited, True, timings)
        assert_equal(limited.info()[0], 0)
        assert_equal(limited.transfers()[2], 0)


# requires: metal
def test_metal_group_tuple_upper_cardinality_boundary_is_checked_on_device() raises:
    from dataframe import col
    from dataframe_metal.handoff import execute_timed
    from tests.support.metal_outcomes import require_metal
    from std.testing import assert_raises

    var runtime = require_metal()
    for groups in [65536, 65537]:
        var a = List[Int32]()
        var b = List[Int64]()
        var values = List[Int64](length=groups, fill=1)
        for row in range(groups):
            a.append(Int32(row))
            b.append(Int64.MIN + Int64(row))
        var frame = DataFrame(
            [
                Series("a", Column[Int32](a^)),
                Series("b", Column[Int64](b^)),
                Series("v", Column[Int64](values^)),
            ]
        )
        var query = (
            frame.lazy()
            .group_by(["a", "b"], maintain_order=True)
            .agg([col("v").sum().alias("s"), col("v").count().alias("n")])
        )
        if groups == 65536:
            var answer = execute_timed(query)[0].copy()
            assert_equal(answer.height(), groups)
            for row in range(groups):
                assert_equal(answer.item(row, "a").int32(), Int32(row))
                assert_equal(
                    answer.item(row, "b").int64(), Int64.MIN + Int64(row)
                )
                assert_equal(answer.item(row, "s").int64(), Int64(1))
                assert_equal(answer.item(row, "n").int64(), Int64(1))
            assert_true(query.collect(accelerator=runtime).equals(answer))
        else:
            with assert_raises(contains="65536 groups"):
                _ = execute_timed(query)
            var answer = query.collect(accelerator=runtime)
            assert_equal(answer.height(), groups)
            assert_equal(
                answer.item(groups - 1, "a").int32(), Int32(groups - 1)
            )
            assert_equal(answer.item(groups - 1, "s").int64(), Int64(1))


def test_metal_group_conditional_tuple_oracle_matches_cpu() raises:
    from dataframe import Expr, col, lit, when
    from tests.support.metal_matrix_reference import matrix_frame

    var base = _conditional_matrix_answer()
    var full = _matrix_tuple_keys(base)
    var expected_columns = List[Series]()
    for name in ["k", "k2", "k3", "k4"]:
        expected_columns.append(full.column(name))
    for name in base.columns():
        if name != "k":
            expected_columns.append(base.column(name))
    var expected = DataFrame(expected_columns^)
    for fragmented in [False, True]:
        var frame = _matrix_tuple_keys(matrix_frame(fragmented))
        var member = col("p").is_in([lit(Int32(1))])
        var values = List[Expr](
            [
                col("a") * col("b") + col("a"),
                col("l") - col("m") + col("l"),
                col("f") * col("g") + 1,
                (col("h") + col("j")) * col("h"),
            ]
        )
        var outputs = List[Expr]()
        for field in range(4):
            var value = (
                when(member.copy()).then(values[field].copy()).otherwise(0)
            )
            var suffix = String(field)
            outputs.append(
                value.sum(min_count=2 if field == 3 else 0).alias("s" + suffix)
            )
            outputs.append(value.count().alias("n" + suffix))
            outputs.append(value.mean().alias("avg" + suffix))
            outputs.append(value.min().alias("lo" + suffix))
            outputs.append(value.max().alias("hi" + suffix))
        var answer = (
            frame.lazy()
            .filter(member | col("p").is_null())
            .group_by(["k", "k2", "k3", "k4"], maintain_order=True)
            .agg(outputs)
            .collect(engine="cpu")
        )
        assert_true(answer.equals(expected))


# requires: metal
def test_metal_group_tuple_checked_branches_and_ieee_adjudication() raises:
    from dataframe import col, when
    from std.testing import assert_raises
    from std.memory import bitcast
    from std.utils.numerics import inf
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([0, 0])),
            Series("b", Column[Int64]([Int64.MIN, Int64.MIN])),
            Series("p", Column[Int32]([0, 1])),
            Series("v", Column[Int64]([Int64.MAX, 1])),
            Series("w", Column[Int64]([0, 0], [False, True])),
        ]
    )
    var safe = when(col("p") > 0).then(col("v") + 1).otherwise(0)
    var query = (
        frame.lazy()
        .group_by(["a", "b"])
        .agg(
            [
                safe.sum().alias("s"),
                safe.count().alias("n"),
                safe.mean().alias("avg"),
                safe.min().alias("lo"),
                safe.max().alias("hi"),
                col("v").len().alias("rows"),
            ]
        )
    )
    var answer = execute_timed(query)[0].copy()
    assert_equal(answer.item(0, "s").int64(), Int64(2))
    assert_equal(answer.item(0, "n").int64(), Int64(2))
    assert_equal(answer.item(0, "avg").float64(), Float64(1))
    assert_equal(answer.item(0, "lo").int64(), Int64(0))
    assert_equal(answer.item(0, "hi").int64(), Int64(2))
    assert_equal(answer.item(0, "rows").int64(), Int64(2))
    assert_true(query.collect(accelerator=runtime).equals(answer))
    var selected_fault = (
        frame.lazy()
        .group_by(["a", "b"])
        .agg(when(col("p") == 0).then(col("v") + 1).otherwise(0).count())
    )
    var later_null = (
        frame.lazy()
        .group_by(["a", "b"])
        .agg(((col("v") + 1) + col("w")).count())
    )
    for fault in [selected_fault^, later_null^]:
        with assert_raises(contains="checked-value"):
            _ = execute_timed(fault)
        same_outcome(fault)
    var filtered = (
        frame.lazy()
        .filter(col("p") > 0)
        .group_by(["a", "b"])
        .agg((col("v") + 1).count())
    )
    assert_equal(execute_timed(filtered)[0].item(0, "v").int64(), Int64(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC00017))
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var ieee = DataFrame(
        [
            Series("a", Column[Int32]([0, 0, 1, 1, 2])),
            Series(
                "b",
                Column[Int64]([Int64.MIN, Int64.MIN, Int64.MAX, Int64.MAX, 0]),
            ),
            Series(
                "v",
                Column[Float32](
                    [nan, 3, negative_zero, 0.0, inf[DType.float32]()]
                ),
            ),
        ]
    )
    var extrema = (
        ieee.lazy()
        .group_by(["a", "b"])
        .agg(
            [
                col("v").min().alias("lo"),
                col("v").max().alias("hi"),
                col("v").count().alias("n"),
            ]
        )
    )
    with assert_raises(contains="signed-zero"):
        _ = execute_timed(extrema)
    same_outcome(extrema)
    var unambiguous = (
        ieee.lazy()
        .filter(col("a") != 1)
        .group_by(["a", "b"], maintain_order=True)
        .agg(
            [
                col("v").min().alias("lo"),
                col("v").max().alias("hi"),
                col("v").count().alias("n"),
            ]
        )
    )
    var stable = execute_timed(unambiguous)[0].copy()
    assert_equal(stable.height(), 2)
    assert_equal(stable.item(0, "lo").float32(), Float32(3))
    assert_equal(stable.item(0, "hi").float32(), Float32(3))
    assert_equal(stable.item(0, "n").int64(), Int64(2))
    assert_equal(stable.item(1, "lo").float32(), inf[DType.float32]())
    assert_true(unambiguous.collect(accelerator=runtime).equals(stable))
    var only_nan = (
        DataFrame(
            [
                Series("a", Column[Int32]([0])),
                Series("b", Column[Int64]([0])),
                Series("v", Column[Float32]([nan])),
            ]
        )
        .lazy()
        .group_by(["a", "b"])
        .agg(col("v").min())
    )
    with assert_raises(contains="all-NaN"):
        _ = execute_timed(only_nan)
    same_outcome(only_nan)
