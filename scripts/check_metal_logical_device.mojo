"""Native packed Bool predicate ABI, bitmap offsets and SQL composition."""
from std.memory import Pointer
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import COL
from dataframe_metal.accelerator.gpu.compound_predicate import predicate_matches


def _logical_probe(
    program: Pointer[UInt64, MutUntrackedOrigin],
    boolean: Pointer[UInt8, MutUntrackedOrigin],
    integers: Pointer[Int32, MutUntrackedOrigin],
    boolean_bits: Pointer[UInt8, MutUntrackedOrigin],
    integer_bits: Pointer[UInt8, MutUntrackedOrigin],
    output: Pointer[UInt32, MutUntrackedOrigin],
    rows: UInt32,
    base: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var mask = UInt32(0)
    var numbers = integers.unsafe_bitcast[UInt8]()
    for scenario in range(8):
        var selected = predicate_matches(
            program.unsafe_offset(scenario * 21),
            Int(base),
            row,
            boolean,
            numbers,
            boolean,
            boolean,
            boolean,
            boolean,
            boolean,
            boolean,
            boolean,
            boolean_bits,
            integer_bits,
            boolean_bits,
            boolean_bits,
            boolean_bits,
            boolean_bits,
            boolean_bits,
            boolean_bits,
            boolean_bits,
        )
        mask |= UInt32(selected) << UInt32(scenario)
    output[unsafe_offset=row] = mask


def main() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var rows = 4096
    var capacity = rows + 128
    var booleans = List[UInt8](length=(capacity + 15) // 8, fill=0)
    var numbers = List[Int32]()
    var bool_bits = List[UInt8](length=(capacity + 15) // 8, fill=0)
    var int_bits = List[UInt8](length=(capacity + 15) // 8, fill=0)
    for i in range(capacity):
        if i % 5 < 2:
            booleans[(i + 5) // 8] |= UInt8(1 << ((i + 5) % 8))
        numbers.append(Int32(i % 7 - 3))
        if i % 11 != 0:
            bool_bits[(i + 3) // 8] |= UInt8(1 << ((i + 3) % 8))
        if i % 13 != 0:
            int_bits[(i + 7) // 8] |= UInt8(1 << ((i + 7) % 8))
    var program = List[UInt64](length=8 * 21, fill=0)
    for scenario in range(8):
        var at = scenario * 21
        program[at] = 1 if scenario < 4 else 2
        # Cases: b, ~b, is_null(b), is_not_null(b), b&n, b|n,
        # b|~n, ~(b&n). Each leaf carries its own SQL validity.
        program[at + 1] = 2 if scenario < 4 else 8 if scenario == 4 else 14
        var inverted = scenario == 1 or scenario == 3 or scenario == 7
        var comparison = COL if scenario != 2 and scenario != 3 else 0
        program[at + 9] = 0
        program[at + 10] = (
            UInt64(3)
            | (UInt64(1) << 32)
            | (UInt64(3) << 33)
            | (UInt64(5) << 35)
        )
        program[at + 14] = UInt64(
            3 | (1 << 2) | (Int(inverted) << 4) | (comparison << 5)
        )
        if scenario >= 4:
            program[at + 15] = 1
            program[at + 16] = UInt64(7) | (UInt64(1) << 32)
            program[at + 17] = 1
            program[at + 18] = UInt64(Int32.MAX)
            program[at + 20] = UInt64(
                3 | (Int(scenario == 6 or scenario == 7) << 4)
            )
    var scope = Scope()
    var ctx = scope.context()
    var metadata = ctx.enqueue_create_buffer[DType.uint64](len(program))
    var boolean = ctx.enqueue_create_buffer[DType.uint8](len(booleans))
    var integer = ctx.enqueue_create_buffer[DType.int32](capacity)
    var bbits = ctx.enqueue_create_buffer[DType.uint8](len(bool_bits))
    var nbits = ctx.enqueue_create_buffer[DType.uint8](len(int_bits))
    var output = ctx.enqueue_create_buffer[DType.uint32](rows)
    ctx.enqueue_copy(metadata, program.unsafe_ptr())
    ctx.enqueue_copy(boolean, booleans.unsafe_ptr())
    ctx.enqueue_copy(integer, numbers.unsafe_ptr())
    ctx.enqueue_copy(bbits, bool_bits.unsafe_ptr())
    ctx.enqueue_copy(nbits, int_bits.unsafe_ptr())
    var kernel = ctx.compile_function[_logical_probe]()
    for base in [0, 1, 5, 127]:
        ctx.enqueue_function(
            kernel,
            metadata,
            boolean,
            integer,
            bbits,
            nbits,
            output,
            UInt32(rows),
            UInt32(base),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        var actual = List[UInt32](length=rows, fill=0)
        ctx.enqueue_copy(actual.unsafe_ptr(), output)
        scope.synchronize()
        for row in range(rows):
            var i = base + row
            var bv = i % 11 != 0
            var nv = i % 13 != 0
            var b = i % 5 < 2
            var n = i % 7 - 3 > 0
            var expected = UInt32(0)
            var answers = [
                bv and b,
                bv and not b,
                not bv,
                bv,
                bv and b and nv and n,
                (bv and b) or (nv and n),
                (bv and b) or (nv and not n),
                (bv and not b) or (nv and not n),
            ]
            for scenario in range(8):
                expected |= UInt32(answers[scenario]) << UInt32(scenario)
            assert_equal(actual[row], expected)
    print("PASS 131072 actual Metal Bool predicate/offset/composition checks")
