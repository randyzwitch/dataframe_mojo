"""Bounded checked postfix evaluation inside a reduction scan."""
from std.memory import Pointer, bitcast
from dataframe.expr import COL, LIT_INT
from ...expression_format import EXPRESSION_NODE_WORDS
from .multi_reduce import Bytes, read_integer
from .compound_predicate import _pointer
from .filter_sum import _valid
from .arithmetic_sum import integer_binary
from .value_ops import checked_float_value


@always_inline
def _stack_value[
    floating: Bool
](
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    count: Int,
    row_base: Int,
    row: Int,
    tag: UInt32,
    want_value: Bool,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
) -> Tuple[UInt64, Bool, UInt32]:
    var values = SIMD[DType.uint64, 8](0)
    var valid = SIMD[DType.uint32, 8](0)
    var top = 0
    var fault = UInt32(0)
    for node in range(count):
        var at = node * EXPRESSION_NODE_WORDS
        var op = UInt32(program[unsafe_offset=at])
        var bits = UInt64(0)
        var present = True
        if op == UInt32(COL):
            var slot = Int(program[unsafe_offset=at + 1])
            var meta = program[unsafe_offset=at + 2]
            if meta >> 32:
                var bitmap = _pointer(slot, b0, b1, b2, b3, b4, b5, b6, b7, b7)
                present = _valid(bitmap, UInt32(meta) + UInt32(row_base), row)
            if present and (not floating or want_value):
                var data = _pointer(slot, v0, v1, v2, v3, v4, v5, v6, v7, v7)
                comptime if floating:
                    bits = UInt64(
                        data.unsafe_bitcast[UInt32]()[
                            unsafe_offset=row_base + row
                        ]
                    )
                else:
                    bits = bitcast[DType.uint64](
                        read_integer(data, tag, row_base + row)
                    )
        elif op == UInt32(LIT_INT):
            bits = program[unsafe_offset=at + 3]
        else:
            top -= 1
            var right = values[top]
            var right_valid = valid[top]
            top -= 1
            present = Bool(valid[top] & right_valid)
            if present:
                comptime if floating:
                    if want_value:
                        var result = checked_float_value(
                            bitcast[DType.float32](UInt32(values[top])),
                            bitcast[DType.float32](UInt32(right)),
                            op,
                        )
                        bits = UInt64(bitcast[DType.uint32](result[0]))
                        fault |= UInt32(result[1]) << 1
                else:
                    var result = integer_binary(
                        bitcast[DType.int64](values[top]),
                        bitcast[DType.int64](right),
                        op,
                        tag,
                    )
                    fault |= UInt32(result[1])
                    bits = bitcast[DType.uint64](result[0]) if not result[
                        1
                    ] else UInt64(0)
        values[top] = bits
        valid[top] = UInt32(present)
        top += 1
    return (values[0], Bool(valid[0]), fault)


@always_inline
def _leaf[
    floating: Bool, node: Int
](
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    row_base: Int,
    row: Int,
    tag: UInt32,
    want: Bool,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
) -> Tuple[UInt64, Bool, UInt32]:
    comptime at = node * EXPRESSION_NODE_WORDS
    var bits = program[unsafe_offset=at + 3]
    var present = True
    if UInt32(program[unsafe_offset=at]) == UInt32(COL):
        var slot = Int(program[unsafe_offset=at + 1])
        var meta = program[unsafe_offset=at + 2]
        if meta >> 32:
            var bitmap = _pointer(slot, b0, b1, b2, b3, b4, b5, b6, b7, b7)
            present = _valid(bitmap, UInt32(meta) + UInt32(row_base), row)
        bits = 0
        if present and (not floating or want):
            var data = _pointer(slot, v0, v1, v2, v3, v4, v5, v6, v7, v7)
            comptime if floating:
                bits = UInt64(
                    data.unsafe_bitcast[UInt32]()[unsafe_offset=row_base + row]
                )
            else:
                bits = bitcast[DType.uint64](
                    read_integer(data, tag, row_base + row)
                )
    return (bits, present, UInt32(0))


@always_inline
def _step[
    floating: Bool, node: Int
](
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    tag: UInt32,
    want: Bool,
    left: UInt64,
    left_valid: Bool,
    left_fault: UInt32,
    right: UInt64,
    right_valid: Bool,
    right_fault: UInt32,
) -> Tuple[UInt64, Bool, UInt32]:
    var present = left_valid and right_valid
    var fault = left_fault | right_fault
    var bits = UInt64(0)
    if present:
        var op = UInt32(program[unsafe_offset=node * EXPRESSION_NODE_WORDS])
        comptime if floating:
            if want:
                var result = checked_float_value(
                    bitcast[DType.float32](UInt32(left)),
                    bitcast[DType.float32](UInt32(right)),
                    op,
                )
                # Operation word2 is an opaque zero in the wire format. Keep
                # it in the integer dataflow to round each Float32 result
                # before its parent, even with fast FP contraction enabled.
                bits = UInt64(
                    bitcast[DType.uint32](result[0])
                    ^ UInt32(
                        program[unsafe_offset=node * EXPRESSION_NODE_WORDS + 2]
                    )
                )
                fault |= UInt32(result[1]) << 1
        else:
            var result = integer_binary(
                bitcast[DType.int64](left), bitcast[DType.int64](right), op, tag
            )
            fault |= UInt32(result[1])
            bits = bitcast[DType.uint64](result[0]) if not result[
                1
            ] else UInt64(0)
    return (bits, present, fault)


@always_inline
def _fixed[
    floating: Bool, shape: Int
](
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    row_base: Int,
    row: Int,
    tag: UInt32,
    want_value: Bool,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
) -> Tuple[UInt64, Bool, UInt32]:
    # Topology is compile-time; every register index and node offset is fixed.
    # Helpers receive flat pointers/scalars to preserve GPU address spaces.
    comptime THREE = shape >= 2
    comptime THIRD = 3 if shape == 0 or shape == 2 or shape == 3 else 2
    comptime FOURTH = 5 if shape == 2 or shape == 4 else 4 if shape == 3 or shape == 5 else 3
    comptime FIRST_OP = 2 if shape == 0 or shape == 2 or shape == 3 else 4 if shape == 6 else 3
    comptime SECOND_OP = 4 if shape < 3 or shape == 4 else 5
    comptime FIRST_LEFT = 0 if shape == 0 or shape == 2 or shape == 3 else 2 if shape == 6 else 1
    comptime FIRST_RIGHT = FIRST_LEFT + 1
    comptime SECOND_LEFT = 4 if shape == 0 or shape == 2 or shape == 5 else 2 if shape == 3 else 1 if shape == 6 else 0
    comptime SECOND_RIGHT = 2 if shape == 0 or shape == 2 else 3 if shape == 3 or shape == 5 else 4
    comptime THIRD_LEFT = 5 if shape == 2 or shape == 4 else 4 if shape == 3 else 0
    comptime THIRD_RIGHT = 3 if shape == 2 or shape == 4 else 5
    var registers = SIMD[DType.uint64, 8](0)
    var validity = SIMD[DType.uint32, 8](0)
    var faults = SIMD[DType.uint32, 8](0)
    var leaf0 = _leaf[floating, 0](
        program,
        row_base,
        row,
        tag,
        want_value,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
    )
    registers[0] = leaf0[0]
    validity[0] = UInt32(leaf0[1])
    faults[0] = leaf0[2]
    var leaf1 = _leaf[floating, 1](
        program,
        row_base,
        row,
        tag,
        want_value,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
    )
    registers[1] = leaf1[0]
    validity[1] = UInt32(leaf1[1])
    faults[1] = leaf1[2]
    var leaf2 = _leaf[floating, THIRD](
        program,
        row_base,
        row,
        tag,
        want_value,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
    )
    registers[2] = leaf2[0]
    validity[2] = UInt32(leaf2[1])
    faults[2] = leaf2[2]
    comptime if THREE:
        var leaf3 = _leaf[floating, FOURTH](
            program,
            row_base,
            row,
            tag,
            want_value,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
        )
        registers[3] = leaf3[0]
        validity[3] = UInt32(leaf3[1])
        faults[3] = leaf3[2]
    var step4 = _step[floating, FIRST_OP](
        program,
        tag,
        want_value,
        registers[FIRST_LEFT],
        Bool(validity[FIRST_LEFT]),
        faults[FIRST_LEFT],
        registers[FIRST_RIGHT],
        Bool(validity[FIRST_RIGHT]),
        faults[FIRST_RIGHT],
    )
    registers[4] = step4[0]
    validity[4] = UInt32(step4[1])
    faults[4] = step4[2]
    var step5 = _step[floating, SECOND_OP](
        program,
        tag,
        want_value,
        registers[SECOND_LEFT],
        Bool(validity[SECOND_LEFT]),
        faults[SECOND_LEFT],
        registers[SECOND_RIGHT],
        Bool(validity[SECOND_RIGHT]),
        faults[SECOND_RIGHT],
    )
    registers[5] = step5[0]
    validity[5] = UInt32(step5[1])
    faults[5] = step5[2]
    comptime if THREE:
        var step6 = _step[floating, 6](
            program,
            tag,
            want_value,
            registers[THIRD_LEFT],
            Bool(validity[THIRD_LEFT]),
            faults[THIRD_LEFT],
            registers[THIRD_RIGHT],
            Bool(validity[THIRD_RIGHT]),
            faults[THIRD_RIGHT],
        )
        return step6
    return (registers[5], Bool(validity[5]), faults[5])


@always_inline
def expression_value[
    floating: Bool
](
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    count: Int,
    row_base: Int,
    row: Int,
    tag: UInt32,
    want_value: Bool,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
) -> Tuple[UInt64, Bool, UInt32]:
    if count == 5:
        if program[unsafe_offset=8] > UInt64(1):
            return _fixed[floating, 0](
                program,
                row_base,
                row,
                tag,
                want_value,
                v0,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
            )
        return _fixed[floating, 1](
            program,
            row_base,
            row,
            tag,
            want_value,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
        )
    if count == 7:
        if program[unsafe_offset=8] > UInt64(1):
            if program[unsafe_offset=16] > UInt64(1):
                return _fixed[floating, 2](
                    program,
                    row_base,
                    row,
                    tag,
                    want_value,
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7,
                    b0,
                    b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    b6,
                    b7,
                )
            return _fixed[floating, 3](
                program,
                row_base,
                row,
                tag,
                want_value,
                v0,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
            )
        if program[unsafe_offset=12] > UInt64(1):
            if program[unsafe_offset=16] > UInt64(1):
                return _fixed[floating, 4](
                    program,
                    row_base,
                    row,
                    tag,
                    want_value,
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7,
                    b0,
                    b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    b6,
                    b7,
                )
            return _fixed[floating, 5](
                program,
                row_base,
                row,
                tag,
                want_value,
                v0,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
            )
        return _fixed[floating, 6](
            program,
            row_base,
            row,
            tag,
            want_value,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
        )
    return _stack_value[floating](
        program,
        count,
        row_base,
        row,
        tag,
        want_value,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
    )
