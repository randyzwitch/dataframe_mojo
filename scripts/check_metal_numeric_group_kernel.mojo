"""One production grouped kernel with hand-packed metadata; no CPU planner.

Diagnostic state checks, not public-query or performance evidence. The kernel
specialization matches the 32-group, three-dtype numeric benchmark shape.
"""
from std.math import abs
from std.memory import bitcast
from std.testing import assert_equal
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import COL, LIT_INT, ABS, NEG, CLIP_LOW, CLIP_HIGH, FLOORDIV
from dataframe_metal.expression_format import (
    TREE_OP,
    TYPED_HEADER,
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    VALUE_TYPE_SHIFT,
)
from dataframe_metal.accelerator.gpu.group_cached import (
    cached_grouped_partials,
)
from dataframe_metal.group_layout import GroupLayout
from dataframe_metal.exact_float import finish_float_bins
from dataframe_metal.accelerator.gpu.query_scope import Scope
from dataframe_metal.accelerator.gpu.group_general import (
    initialize_general_groups,
    general_grouped_partials,
    resolve_general_int64,
)


def stage(value: String) raises:
    with open("kernel-stage.txt", "w") as target:
        target.write(value)


def node(
    mut words: List[UInt64],
    at: Int,
    op: Int,
    tag: Int,
    input: Int = 0,
    literal: UInt64 = 0,
):
    words[at] = UInt64(op)
    words[at + 1] = UInt64(input)
    words[at + 2] = UInt64(tag + 1) << VALUE_TYPE_SHIFT
    words[at + 3] = literal


def run_check[
    cached: Bool = False
](nullable: Bool, wide_case: Bool = False, many: Bool = False) raises:
    var rows = 65536
    var blocks = 128
    var layout = GroupLayout(512, 3, blocks, False)
    layout.check()
    var words = List[UInt64](length=144, fill=0)
    words[0] = UInt64(rows)
    words[1] = UInt64(blocks * 256)
    words[2] = UInt64(blocks)
    words[3] = 512
    words[4] = 256
    words[6] = 3
    words[7] = 1
    words[8] = 1
    words[9] = 0x777
    words[10] = 48
    words[11] = 2
    words[13] = 1
    words[49] = 1
    for field in range(3):
        words[16 + field * 8] = UInt64(TREE_OP)
    words[22] = 57
    words[23] = 8
    words[30] = 104
    words[31] = 7
    words[38] = 132
    words[39] = 3
    # Legacy CASE: n>=0 ? f : -clip_low(f,-8), numerically ABS(f).
    words[57] = UInt64(CASE_HEADER)
    words[61] = UInt64(CASE_TEST)
    words[62] = 32
    words[64] = 4
    words[65] = UInt64(COL)
    words[69] = UInt64(CASE_JUMP)
    words[72] = 8
    words[73] = UInt64(COL)
    words[77] = UInt64(LIT_INT)
    words[80] = 0xC1000000
    words[81] = UInt64(CLIP_LOW)
    words[85] = UInt64(NEG)
    # Embedded n>=0 predicate: one range clause, true when its bit is set.
    words[89] = 1
    words[90] = 2
    words[98] = 1
    words[100] = 0
    words[101] = UInt64(Int32.MAX)
    words[103] = 3
    # Typed Int32 ABS/CLIP.
    words[104] = UInt64(TYPED_HEADER)
    # ABS, CLIP_LOW and CLIP_HIGH use the production scalar schedule.
    words[107] = 2 | (UInt64(3) << 8) | (UInt64(1) << 16)
    node(words, 108, COL, 0, 1)
    node(words, 112, ABS, 0)
    node(words, 116, LIT_INT, 0)
    node(words, 120, CLIP_LOW, 0)
    node(words, 124, LIT_INT, 0, literal=12)
    node(words, 128, CLIP_HIGH, 0)
    # Untagged legacy Int64 arithmetic.
    words[132] = UInt64(COL)
    words[133] = 2
    words[136] = UInt64(LIT_INT)
    words[139] = 3
    words[140] = UInt64(FLOORDIV)
    if nullable:
        # Independent value and validity bit origins, shared by all three
        # input dtypes. Alternate 32-row runs give eight valid rows per
        # group/block, including conditional branches and Int64 extrema.
        for at in [67, 75, 110, 134]:
            words[at] |= (UInt64(1) << 32) | UInt64(3)
    var bitmap = List[UInt8](length=(rows + 10) // 8, fill=0)
    for row in range(rows):
        if (row // 32) % 2 == 0:
            bitmap[(row + 3) // 8] |= UInt8(1) << UInt8((row + 3) % 8)
    var floating = List[Float32](capacity=rows)
    var narrow = List[Int32](capacity=rows)
    var wide = List[Int64](capacity=rows)
    var key = List[Int32](capacity=rows)
    var second = List[Int64](capacity=rows)
    for row in range(rows):
        var p = row % 32
        var value = p - 16
        floating.append(Float32(value) * Float32(0.5))
        narrow.append(Int32(value))
        wide.append(
            Int64(value * 257 + p % 8)
            + (Int64(1) << 50 if wide_case else Int64(0))
        )
        key.append(Int32(row % 257 if many else p % 8))
        second.append(Int64(p // 8))
    var scope = Scope()
    var ctx = scope.context()
    var f = ctx.enqueue_create_buffer[DType.float32](rows)
    var n = ctx.enqueue_create_buffer[DType.int32](rows)
    var w = ctx.enqueue_create_buffer[DType.int64](rows)
    var k = ctx.enqueue_create_buffer[DType.int32](rows)
    var j = ctx.enqueue_create_buffer[DType.int64](rows)
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var validity = ctx.enqueue_create_buffer[DType.uint8](len(bitmap))
    ctx.enqueue_copy(validity, bitmap.unsafe_ptr())
    var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var scratch = ctx.enqueue_create_buffer[DType.uint32](layout.words())
    var resolved = ctx.enqueue_create_buffer[DType.int64](layout.entries() * 2)
    ctx.enqueue_copy(f, floating.unsafe_ptr())
    ctx.enqueue_copy(n, narrow.unsafe_ptr())
    ctx.enqueue_copy(w, wide.unsafe_ptr())
    ctx.enqueue_copy(k, key.unsafe_ptr())
    ctx.enqueue_copy(j, second.unsafe_ptr())
    ctx.enqueue_copy(metadata, words.unsafe_ptr())
    ctx.enqueue_memset(resolved, Int64(0))
    var fb = f.create_sub_buffer[DType.uint8](0, rows * 4)
    var nb = n.create_sub_buffer[DType.uint8](0, rows * 4)
    var wb = w.create_sub_buffer[DType.uint8](0, rows * 8)
    var kb = k.create_sub_buffer[DType.uint8](0, rows * 4)
    var jb = j.create_sub_buffer[DType.uint8](0, rows * 8)
    stage("drain uploads before pipeline creation")
    scope.synchronize()
    stage("compile general before any dispatch")
    comptime kernel = cached_grouped_partials[
        True, False, False
    ] if cached else general_grouped_partials[True, False, False, 7]
    var general = ctx.compile_function[kernel]()
    stage("compile resolver before any dispatch")
    var resolver = ctx.compile_function[
        resolve_general_int64[True, False, False]
    ]()
    stage("initialize")
    ctx.enqueue_function[initialize_general_groups[False]](
        metadata,
        scratch,
        grid_dim=(layout.words() + 255) // 256,
        block_dim=256,
    )
    stage("general_grouped_partials[True,False,False,7]")
    ctx.enqueue_function(
        general,
        fb,
        nb,
        wb,
        empty,
        empty,
        empty,
        empty,
        empty,
        validity,
        validity,
        validity,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        kb,
        jb,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        metadata,
        scratch,
        grid_dim=blocks,
        block_dim=256,
    )
    stage("resolve_general_int64[True,False,False]")
    ctx.enqueue_function(
        resolver,
        fb,
        nb,
        wb,
        empty,
        empty,
        empty,
        empty,
        empty,
        validity,
        validity,
        validity,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        metadata,
        scratch,
        resolved,
        scratch.create_sub_buffer[DType.uint32](0, 4),
        grid_dim=(layout.entries() + 255) // 256,
        block_dim=256,
    )
    var first = List[UInt32](length=512, fill=UInt32.MAX)
    var counts = List[UInt32](length=512 * blocks, fill=0)
    var states = List[UInt32](length=layout.entries() * 32, fill=0)
    var extrema = List[Int64](length=layout.entries() * 2, fill=0)
    var control = List[UInt32](length=4, fill=0)
    ctx.enqueue_copy(
        first.unsafe_ptr(),
        scratch.create_sub_buffer[DType.uint32](layout.first_offset(), 512),
    )
    ctx.enqueue_copy(
        counts.unsafe_ptr(),
        scratch.create_sub_buffer[DType.uint32](
            layout.rows_offset(), len(counts)
        ),
    )
    ctx.enqueue_copy(
        states.unsafe_ptr(),
        scratch.create_sub_buffer[DType.uint32](
            layout.states_offset(), len(states)
        ),
    )
    ctx.enqueue_copy(extrema.unsafe_ptr(), resolved)
    ctx.enqueue_copy(
        control.unsafe_ptr(), scratch.create_sub_buffer[DType.uint32](0, 4)
    )
    stage("synchronize")
    scope.synchronize()
    comptime if cached:
        if many:
            assert_equal(control[1], UInt32(0x80000000))
            scope.mark_success()
            print("PASS original group-capacity retry request")
            return
    assert_equal(control[0], UInt32(32))
    assert_equal(control[1], UInt32(0))
    assert_equal(control[2], UInt32(blocks))
    assert_equal(control[3], UInt32((layout.entries() + 255) // 256))
    var expected_count = 8 if nullable else 16
    var groups = 0
    for slot in range(512):
        if first[slot] == UInt32.MAX:
            continue
        groups += 1
        var p = Int(first[slot]) % 32
        var expected_n = Int64(min(abs(p - 16), 12))
        var expected_w = (
            Int64(p - 16) * 257
            + Int64(p % 8)
            + (Int64(1) << 50 if wide_case else Int64(0))
        ) // 3
        var expected_f = Float64(abs(p - 16)) * Float64(0.5)
        for block in range(blocks):
            assert_equal(counts[slot * blocks + block], UInt32(16))
            for field in range(3):
                var index = (slot * blocks + block) * 3 + field
                var at = index * 32
                assert_equal(states[at], UInt32(expected_count))
                if field == 0:
                    var bins = List[Int128](length=9, fill=0)
                    for i in range(9):
                        var raw = UInt64(states[at + 10 + i * 2]) | (
                            UInt64(states[at + 11 + i * 2]) << 32
                        )
                        bins[i] = Int128(bitcast[DType.int64](raw))
                    assert_equal(
                        finish_float_bins(bins, False, False, False),
                        expected_f * Float64(expected_count),
                    )
                    var ordered = bitcast[DType.uint32](
                        Float32(expected_f)
                    ) | UInt32(0x80000000)
                    assert_equal(states[at + 4], ordered)
                    assert_equal(states[at + 5], ~ordered)
                else:
                    var low = UInt64(states[at + 1]) | (
                        UInt64(states[at + 2]) << 32
                    )
                    var total = (
                        Int128(bitcast[DType.int32](states[at + 3])) << 64
                    ) + Int128(low)
                    assert_equal(
                        total,
                        Int128(expected_n if field == 1 else expected_w)
                        * Int128(expected_count),
                    )
                    if field == 1:
                        var ordered = bitcast[DType.uint32](
                            Int32(expected_n)
                        ) ^ UInt32(0x80000000)
                        assert_equal(states[at + 4], ordered)
                        assert_equal(states[at + 5], ~ordered)
                    else:
                        assert_equal(extrema[index * 2], expected_w)
                        assert_equal(extrema[index * 2 + 1], expected_w)
    assert_equal(groups, 32)
    scope.mark_success()
    stage("PASS")
    print(
        "PASS 12288 production grouped numeric states on Metal, nullable=",
        nullable,
    )


def main() raises:
    run_check(False)
    run_check(True)
    run_check[True](False)
    run_check[True](True)
    run_check[True](False, wide_case=True)
    run_check[True](False, many=True)
    run_check(False, wide_case=True)
    run_check[True](True, wide_case=True)
