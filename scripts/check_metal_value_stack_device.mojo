"""Device regression for shared value interpreter control flow and faults.

Explicit expected results; no production helper is used as an oracle.
"""
from std.memory import Pointer
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import (
    COL,
    LIT_INT,
    ADD,
    MUL,
    NEG,
    FLOORDIV,
    DIV,
    CLIP_LOW,
    FILL_NAN,
    CAST,
)
from dataframe_metal.expression_format import (
    TYPED_HEADER,
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    CASE_NULL,
    VALUE_TYPE_SHIFT,
)
from dataframe_metal.accelerator.gpu.multi_reduce import Bytes
from dataframe_metal.accelerator.gpu.typed_expression_value import (
    typed_expression_value,
)
from dataframe_metal.accelerator.gpu.query_scope import Scope
from dataframe_metal.accelerator.gpu.expression_value import expression_value


def probe[
    fast: Bool
](
    programs: Pointer[UInt64, UntrackedOrigin[mut=False]],
    output: Pointer[UInt64, MutUntrackedOrigin],
):
    var index = global_idx.x
    if index >= 4608:
        return
    var scenario = index % 18
    var at = scenario * 64
    var empty = programs.unsafe_offset(18 * 64).unsafe_bitcast[UInt8]()
    var answer: Tuple[UInt64, Bool, UInt32]
    comptime if fast:
        answer = expression_value[False, True](
            programs.unsafe_offset(at + 4),
            Int(programs[unsafe_offset=at]),
            0,
            0,
            UInt32(programs[unsafe_offset=at + 1]),
            Bool(programs[unsafe_offset=at + 2]),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )
    else:
        answer = typed_expression_value(
            programs.unsafe_offset(at + 4),
            Int(programs[unsafe_offset=at]),
            0,
            0,
            Bool(programs[unsafe_offset=at + 2]),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            UInt32(programs[unsafe_offset=at + 1]),
            empty,
            empty,
        )
    output[unsafe_offset=index * 3] = answer[0]
    output[unsafe_offset=index * 3 + 1] = UInt64(answer[1])
    output[unsafe_offset=index * 3 + 2] = UInt64(answer[2])


def main() raises:
    var words = List[UInt64](length=18 * 64 + 1, fill=0)
    var expected = List[UInt64]()
    var validity = List[UInt64]()
    var faults = List[UInt64]()
    # 0: i32 overflow
    words[0] = 4
    words[1] = 0
    words[2] = 1
    words[4] = UInt64(TYPED_HEADER)
    words[8] = UInt64(LIT_INT)
    words[10] = UInt64(1) << VALUE_TYPE_SHIFT
    words[11] = 2147483647
    words[12] = UInt64(LIT_INT)
    words[14] = UInt64(1) << VALUE_TYPE_SHIFT
    words[15] = 1
    words[16] = UInt64(ADD)
    words[18] = UInt64(1) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(1)
    # 1: i64 neg minimum
    words[64] = 3
    words[65] = 1
    words[66] = 1
    words[68] = UInt64(TYPED_HEADER)
    words[72] = UInt64(LIT_INT)
    words[74] = UInt64(2) << VALUE_TYPE_SHIFT
    words[75] = 9223372036854775808
    words[76] = UInt64(NEG)
    words[78] = UInt64(2) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(1)
    # 2: i64 quotient overflow
    words[128] = 4
    words[129] = 1
    words[130] = 1
    words[132] = UInt64(TYPED_HEADER)
    words[136] = UInt64(LIT_INT)
    words[138] = UInt64(2) << VALUE_TYPE_SHIFT
    words[139] = 9223372036854775808
    words[140] = UInt64(LIT_INT)
    words[142] = UInt64(2) << VALUE_TYPE_SHIFT
    words[143] = 18446744073709551615
    words[144] = UInt64(FLOORDIV)
    words[146] = UInt64(2) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(1)
    # 3: integer zero divisor
    words[192] = 4
    words[193] = 1
    words[194] = 1
    words[196] = UInt64(TYPED_HEADER)
    words[200] = UInt64(LIT_INT)
    words[202] = UInt64(2) << VALUE_TYPE_SHIFT
    words[203] = 3
    words[204] = UInt64(LIT_INT)
    words[206] = UInt64(2) << VALUE_TYPE_SHIFT
    words[208] = UInt64(FLOORDIV)
    words[210] = UInt64(2) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(0)
    faults.append(0)
    # 4: strict narrowing cast
    words[256] = 3
    words[257] = 0
    words[258] = 1
    words[260] = UInt64(TYPED_HEADER)
    words[264] = UInt64(LIT_INT)
    words[266] = UInt64(2) << VALUE_TYPE_SHIFT
    words[267] = 9223372036854775807
    words[268] = UInt64(CAST)
    words[270] = UInt64(1) << VALUE_TYPE_SHIFT
    words[271] = 257
    expected.append(0)
    validity.append(0)
    faults.append(1)
    # 5: nullable narrowing cast
    words[320] = 3
    words[321] = 0
    words[322] = 1
    words[324] = UInt64(TYPED_HEADER)
    words[328] = UInt64(LIT_INT)
    words[330] = UInt64(2) << VALUE_TYPE_SHIFT
    words[331] = 9223372036854775807
    words[332] = UInt64(CAST)
    words[334] = UInt64(1) << VALUE_TYPE_SHIFT
    words[335] = 256
    expected.append(0)
    validity.append(0)
    faults.append(0)
    # 6: float divide count skips arithmetic
    words[384] = 4
    words[385] = 2
    words[386] = 0
    words[388] = UInt64(TYPED_HEADER)
    words[392] = UInt64(LIT_INT)
    words[394] = UInt64(3) << VALUE_TYPE_SHIFT
    words[395] = 1065353216
    words[396] = UInt64(LIT_INT)
    words[398] = UInt64(3) << VALUE_TYPE_SHIFT
    words[400] = UInt64(DIV)
    words[402] = UInt64(3) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(0)
    # 7: float divide requests adjudication
    words[448] = 4
    words[449] = 2
    words[450] = 1
    words[452] = UInt64(TYPED_HEADER)
    words[456] = UInt64(LIT_INT)
    words[458] = UInt64(3) << VALUE_TYPE_SHIFT
    words[459] = 1065353216
    words[460] = UInt64(LIT_INT)
    words[462] = UInt64(3) << VALUE_TYPE_SHIFT
    words[464] = UInt64(DIV)
    words[466] = UInt64(3) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(2)
    # 8: header forces float for count
    words[512] = 4
    words[513] = 2
    words[514] = 0
    words[516] = UInt64(TYPED_HEADER)
    words[519] = 1
    words[520] = UInt64(LIT_INT)
    words[522] = UInt64(3) << VALUE_TYPE_SHIFT
    words[523] = 1065353216
    words[524] = UInt64(LIT_INT)
    words[526] = UInt64(3) << VALUE_TYPE_SHIFT
    words[528] = UInt64(DIV)
    words[530] = UInt64(3) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(1)
    faults.append(2)
    # 9: eager checked fault survives null
    words[576] = 6
    words[577] = 0
    words[578] = 1
    words[580] = UInt64(TYPED_HEADER)
    words[584] = UInt64(LIT_INT)
    words[586] = UInt64(1) << VALUE_TYPE_SHIFT
    words[587] = 2147483647
    words[588] = UInt64(LIT_INT)
    words[590] = UInt64(1) << VALUE_TYPE_SHIFT
    words[591] = 1
    words[592] = UInt64(ADD)
    words[594] = UInt64(1) << VALUE_TYPE_SHIFT
    words[596] = UInt64(CASE_NULL)
    words[600] = UInt64(MUL)
    words[602] = UInt64(1) << VALUE_TYPE_SHIFT
    expected.append(0)
    validity.append(0)
    faults.append(1)
    # 10: untagged integer arithmetic
    words[640] = 3
    words[641] = 1
    words[642] = 1
    words[644] = UInt64(LIT_INT)
    words[647] = 18446744073709551609
    words[648] = UInt64(LIT_INT)
    words[651] = 3
    words[652] = UInt64(FLOORDIV)
    expected.append(18446744073709551613)
    validity.append(1)
    faults.append(0)
    # 11: untagged Float32 clip
    words[704] = 3
    words[705] = 2
    words[706] = 1
    words[708] = UInt64(LIT_INT)
    words[711] = 3221225472
    words[712] = UInt64(LIT_INT)
    words[715] = 3212836864
    words[716] = UInt64(CLIP_LOW)
    expected.append(3212836864)
    validity.append(1)
    faults.append(0)
    # 12: lazy CASE
    words[768] = 8
    words[769] = 0
    words[770] = 1
    words[772] = UInt64(CASE_HEADER)
    words[776] = UInt64(CASE_TEST)
    words[777] = 48
    words[779] = 5
    words[780] = UInt64(LIT_INT)
    words[783] = 7
    words[784] = UInt64(CASE_JUMP)
    words[787] = 8
    words[788] = UInt64(LIT_INT)
    words[792] = UInt64(LIT_INT)
    words[795] = 2147483647
    words[796] = UInt64(LIT_INT)
    words[799] = 1
    words[800] = UInt64(ADD)
    words[820] = 0
    words[821] = 1
    expected.append(7)
    validity.append(1)
    faults.append(0)
    # 13: lazy CASE
    words[832] = 8
    words[833] = 0
    words[834] = 1
    words[836] = UInt64(CASE_HEADER)
    words[840] = UInt64(CASE_TEST)
    words[841] = 48
    words[843] = 5
    words[844] = UInt64(CASE_NULL)
    words[848] = UInt64(CASE_JUMP)
    words[851] = 8
    words[852] = UInt64(LIT_INT)
    words[856] = UInt64(LIT_INT)
    words[859] = 2147483647
    words[860] = UInt64(LIT_INT)
    words[863] = 1
    words[864] = UInt64(ADD)
    words[884] = 0
    words[885] = 1
    expected.append(0)
    validity.append(0)
    faults.append(0)
    # 14: lazy CASE
    words[896] = 8
    words[897] = 0
    words[898] = 1
    words[900] = UInt64(CASE_HEADER)
    words[904] = UInt64(CASE_TEST)
    words[905] = 48
    words[907] = 5
    words[908] = UInt64(LIT_INT)
    words[911] = 7
    words[912] = UInt64(CASE_JUMP)
    words[915] = 8
    words[916] = UInt64(LIT_INT)
    words[920] = UInt64(LIT_INT)
    words[923] = 2147483647
    words[924] = UInt64(LIT_INT)
    words[927] = 1
    words[928] = UInt64(ADD)
    words[948] = 0
    words[949] = 0
    expected.append(0)
    validity.append(1)
    faults.append(1)
    # 15: lazy CASE
    words[960] = 8
    words[961] = 0
    words[962] = 1
    words[964] = UInt64(CASE_HEADER)
    words[968] = UInt64(CASE_TEST)
    words[969] = 48
    words[971] = 5
    words[972] = UInt64(CASE_NULL)
    words[976] = UInt64(CASE_JUMP)
    words[979] = 8
    words[980] = UInt64(LIT_INT)
    words[984] = UInt64(LIT_INT)
    words[987] = 2147483647
    words[988] = UInt64(LIT_INT)
    words[991] = 1
    words[992] = UInt64(ADD)
    words[1012] = 0
    words[1013] = 0
    expected.append(0)
    validity.append(1)
    faults.append(1)
    # COUNT must load the NaN payload when either program header requests it.
    words[18 * 64] = 0x7FC12345
    for scenario in range(16, 18):
        var at = scenario * 64
        words[at] = 4
        words[at + 1] = 2
        words[at + 2] = 0
        words[at + 4] = UInt64(TYPED_HEADER if scenario == 16 else CASE_HEADER)
        words[at + 7] = 1
        words[at + 8] = UInt64(COL)
        words[at + 10] = (
            UInt64(3) << VALUE_TYPE_SHIFT
        ) if scenario == 16 else UInt64(0)
        words[at + 12] = UInt64(CASE_NULL)
        words[at + 16] = UInt64(FILL_NAN)
        words[at + 18] = (
            UInt64(3) << VALUE_TYPE_SHIFT
        ) if scenario == 16 else UInt64(0)
        expected.append(0)
        validity.append(0)
        faults.append(0)
    for scenario in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16]:
        var base = scenario * 64
        var at = 2
        var steps = 0
        var unary_mask = UInt64(0)
        while at < Int(words[base]):
            if words[base + 4 + at * 4] in [UInt64(NEG), UInt64(CAST)]:
                unary_mask |= UInt64(1) << UInt64(steps)
                at += 1
            else:
                at += 2
            steps += 1
        words[base + 7] |= 2 | (UInt64(steps) << 8) | (unary_mask << 16)
    var scope = Scope()
    var ctx = scope.context()
    var programs = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var result = ctx.enqueue_create_buffer[DType.uint64](4608 * 3)
    ctx.enqueue_copy(programs, words.unsafe_ptr())
    var actual = List[UInt64](length=4608 * 3, fill=0)
    comptime for fast in range(2):
        ctx.enqueue_function[probe[Bool(fast)]](
            programs, result, grid_dim=18, block_dim=256
        )
        ctx.enqueue_copy(actual.unsafe_ptr(), result)
        scope.synchronize()
        for i in range(4608):
            assert_equal(actual[i * 3], expected[i % 18])
            assert_equal(actual[i * 3 + 1], validity[i % 18])
            assert_equal(actual[i * 3 + 2], faults[i % 18])
    scope.mark_success()
    print(
        "PASS 9216 reference/optimized validity, fault and lazy CASE checks on Metal"
    )
