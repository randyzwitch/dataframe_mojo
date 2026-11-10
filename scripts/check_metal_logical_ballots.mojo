"""Independent selection/Kleene ballot oracle across offsets and native types."""
from std.memory import bitcast, Pointer
from std.sys import size_of, argv
from std.time import perf_counter_ns
from std.testing import assert_equal
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import LIT_INT
from dataframe_metal.expression_format import (
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    CASE_NULL,
    BOOL_COLUMN_COMPARISON,
)
from dataframe_metal.accelerator.gpu.logical_predicates import (
    predicate_logical_partials,
)
from dataframe_metal.accelerator.gpu.compound_predicate import predicate_matches
from dataframe_metal.accelerator.gpu.typed_expression_value import (
    typed_expression_value,
)
from dataframe_metal.accelerator.gpu.multi_reduce import Bytes


def case_partials(
    n: Bytes,
    b: Bytes,
    nb: Bytes,
    bb: Bytes,
    empty: Bytes,
    meta: Pointer[UInt64, UntrackedOrigin[mut=False]],
    rows: UInt32,
    stride: UInt32,
    case_at: UInt32,
    output: Pointer[UInt32, MutUntrackedOrigin],
):
    var row = global_idx.x
    var nk = UInt32(0)
    var nt = UInt32(0)
    var nf = UInt32(0)
    while row < Int(rows):
        if predicate_matches(
            meta.unsafe_offset(Int(meta[unsafe_offset=1])),
            0,
            row,
            n,
            b,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            nb,
            bb,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        ):
            var answer = typed_expression_value(
                meta.unsafe_offset(Int(case_at)),
                8,
                0,
                row,
                True,
                n,
                b,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                nb,
                bb,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                UInt32(1),
                empty,
                empty,
            )
            nk += 1
            nt += UInt32(answer[1] and answer[0] != 0)
            nf += UInt32(answer[1] and answer[0] == 0)
        row += Int(stride)
    var k = block.sum[block_size=256, broadcast=False](nk)
    var t = block.sum[block_size=256, broadcast=False](nt)
    var f = block.sum[block_size=256, broadcast=False](nf)
    if thread_idx.x == 0:
        output[unsafe_offset=block_idx.x * 3] = k
        output[unsafe_offset=block_idx.x * 3 + 1] = t
        output[unsafe_offset=block_idx.x * 3 + 2] = f


def numeric_clause(
    mut words: List[UInt64],
    at: Int,
    tag: Int,
    nullable: Bool,
    selection: Bool,
    inverted: Bool = False,
):
    words[at + 1] = 8 | (UInt64(nullable) << 32) | (UInt64(tag) << 33)
    words[at + 2] = bitcast[DType.uint64](Int64(-7 if selection else 1))
    words[at + 3] = UInt64(Int64.MAX)
    words[at + 4] = UInt64(
        bitcast[DType.uint32](Float32(-8 if selection else 0))
    ) | (UInt64(0x7F800000) << 32)
    words[at + 5] = UInt64(2 if tag == 2 else 3) | (UInt64(inverted) << 4)


def boolean_clause(
    mut words: List[UInt64], at: Int, nullable: Bool, inverted: Bool = False
):
    words[at] = 1
    words[at + 1] = (
        12 | (UInt64(nullable) << 32) | (UInt64(3) << 33) | (UInt64(2) << 35)
    )
    words[at + 5] = (
        (UInt64(BOOL_COLUMN_COMPARISON) << 5) | 4 | (UInt64(inverted) << 4)
    )


def run_check[
    D: DType
](
    ctx: DeviceContext,
    rows: Int,
    nullable: Bool,
    filtered: Bool,
    benchmark: Bool = False,
) raises:
    comptime TAG = 0 if D == DType.int32 else 1 if D == DType.int64 else 2
    var values = List[Scalar[D]](capacity=rows + 5)
    var bvalues = List[UInt8](length=(rows + 17) // 8, fill=0)
    var nbits = List[UInt8](length=(rows + 15) // 8, fill=0)
    var bbits = List[UInt8](length=(rows + 19) // 8, fill=0)
    var expected = List[UInt32](length=3, fill=0)
    for physical in range(rows + 5):
        var value = Scalar[D](physical % 32 - 16)
        comptime if D == DType.float32:
            if physical % 8 == 0:
                value = Scalar[D](bitcast[DType.float32](UInt32(0x7FC12345)))
        var nv = not nullable or physical % 16 not in [1, 2]
        var bv = not nullable or physical % 8 != 3
        var b = (physical // 2) % 2 == 0
        values.append(value)
        if b:
            bvalues[(physical + 5) // 8] |= UInt8(1) << UInt8(
                (physical + 5) % 8
            )
        if nv:
            nbits[(physical + 3) // 8] |= UInt8(1) << UInt8((physical + 3) % 8)
        if bv:
            bbits[(physical + 7) // 8] |= UInt8(1) << UInt8((physical + 7) % 8)
        if physical >= 5 and (not filtered or (nv and value > Scalar[D](-8))):
            expected[0] += 1
            if (nv and value > 0) or (bv and b):
                expected[1] += 1
            if nv and not (value > 0) and bv and not b:
                expected[2] += 1
    var selection_words = 15 if filtered else 9
    var pos = 4 + selection_words
    var neg = pos + 21
    var old = neg + 21
    var words = List[UInt64](length=old + 32 + 42, fill=0)
    words[0] = 1
    words[1] = 4
    words[2] = UInt64(pos)
    words[3] = UInt64(neg)
    words[4] = UInt64(filtered)
    words[5] = 2 if filtered else 1
    if filtered:
        numeric_clause(words, 13, TAG, nullable, True)
    for offset in [pos, neg]:
        words[offset] = 2
    words[pos + 1] = 14
    words[neg + 1] = 8
    boolean_clause(words, pos + 9, nullable)
    numeric_clause(words, pos + 15, TAG, nullable, False)
    boolean_clause(words, neg + 9, nullable, True)
    numeric_clause(words, neg + 15, TAG, nullable, False, True)
    words[old] = UInt64(CASE_HEADER)
    words[old + 4] = UInt64(CASE_TEST)
    words[old + 5] = 32
    words[old + 7] = 4
    words[old + 8] = UInt64(LIT_INT)
    words[old + 11] = 1
    words[old + 12] = UInt64(CASE_JUMP)
    words[old + 15] = 8
    words[old + 16] = UInt64(CASE_TEST)
    words[old + 17] = 53
    words[old + 19] = 7
    words[old + 20] = UInt64(LIT_INT)
    words[old + 24] = UInt64(CASE_JUMP)
    words[old + 27] = 8
    words[old + 28] = UInt64(CASE_NULL)
    for i in range(21):
        words[old + 32 + i] = words[pos + i]
        words[old + 53 + i] = words[neg + i]
    var blocks = 128 if benchmark else 2
    var dv = ctx.enqueue_create_buffer[D](rows + 5)
    var db = ctx.enqueue_create_buffer[DType.uint8](len(bvalues))
    var dn = ctx.enqueue_create_buffer[DType.uint8](len(nbits))
    var dvalid = ctx.enqueue_create_buffer[DType.uint8](len(bbits))
    var dm = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var output = ctx.enqueue_create_buffer[DType.uint32](blocks * 3)
    ctx.enqueue_copy(dv, values.unsafe_ptr())
    ctx.enqueue_copy(db, bvalues.unsafe_ptr())
    ctx.enqueue_copy(dn, nbits.unsafe_ptr())
    ctx.enqueue_copy(dvalid, bbits.unsafe_ptr())
    ctx.enqueue_copy(dm, words.unsafe_ptr())
    ctx.synchronize()
    var nview = dv.create_sub_buffer[DType.uint8](
        5 * size_of[Scalar[D]]() if rows else 0,
        max(1, rows * size_of[Scalar[D]]()),
    )
    var bview = db.create_sub_buffer[DType.uint8](1, (rows + 9) // 8)
    # Zero-row probes still pass valid one-byte argument windows.
    if not rows:
        nview = dv.create_sub_buffer[DType.uint8](0, 1)
    for mode in range(2):
        for rep in range(4 if benchmark else 1):
            var started = Int(perf_counter_ns())
            if mode:
                ctx.enqueue_function[predicate_logical_partials](
                    nview,
                    bview,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    dn,
                    dvalid,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    dm,
                    UInt32(rows),
                    UInt32(blocks * 256),
                    UInt32(0),
                    output,
                    grid_dim=blocks,
                    block_dim=256,
                )
            else:
                ctx.enqueue_function[case_partials](
                    nview,
                    bview,
                    dn,
                    dvalid,
                    empty,
                    dm,
                    UInt32(rows),
                    UInt32(blocks * 256),
                    UInt32(old),
                    output,
                    grid_dim=blocks,
                    block_dim=256,
                )
            var host = List[UInt32](length=blocks * 3, fill=UInt32.MAX)
            ctx.enqueue_copy(host.unsafe_ptr(), output)
            ctx.synchronize()
            var elapsed = Int(perf_counter_ns()) - started
            for field in range(3):
                var total = UInt32(0)
                for block in range(blocks):
                    total += host[block * 3 + field]
                assert_equal(total, expected[field])
            if benchmark and rep:
                print(
                    '{"rows":'
                    + String(rows)
                    + ',"tag":'
                    + String(TAG)
                    + ',"nullable":'
                    + String(Int(nullable))
                    + ',"filtered":'
                    + String(Int(filtered))
                    + ',"ballot":'
                    + String(mode)
                    + ',"rep":'
                    + String(rep)
                    + ',"ns":'
                    + String(elapsed)
                    + ',"verified":true}'
                )
    _ = (
        values^,
        bvalues^,
        nbits^,
        bbits^,
        words^,
        dv^,
        db^,
        dn^,
        dvalid^,
        dm^,
        empty^,
        output^,
        nview^,
        bview^,
    )


def main() raises:
    var ctx = DeviceContext(0, api="metal")
    var args = argv()
    if len(args) > 1:
        var tag = Int(String(args[1]))
        var nullable = Bool(Int(String(args[2])))
        var filtered = Bool(Int(String(args[3])))
        comptime for i in range(3):
            comptime D = DType.int32 if i == 0 else DType.int64 if i == 1 else DType.float32
            if tag == i:
                run_check[D](ctx, 50000000, nullable, filtered, True)
        return
    comptime for i in range(3):
        comptime D = DType.int32 if i == 0 else DType.int64 if i == 1 else DType.float32
        for rows in [0, 1, 31, 32, 33, 129, 257, 65536]:
            for nullable in [False, True]:
                for filtered in [False, True]:
                    run_check[D](ctx, rows, nullable, filtered)
    print(
        "PASS logical ballot/reference selection, true and false counts:96cases, Int32/Int64/Float32, independent offsets/tails/nulls/NaNs"
    )
