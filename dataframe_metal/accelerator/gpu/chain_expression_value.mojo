"""Scalar accumulator execution for proven straight-line typed expressions."""
from std.memory import Pointer, bitcast
from std.bit import pop_count
from dataframe.expr import (
    COL,
    LIT_INT,
    CAST,
    ADD,
    SUB,
    MUL,
    FILL_NULL,
    FILL_NAN,
)
from ...expression_format import VALUE_TYPE_SHIFT
from .multi_reduce import Bytes, read_integer
from .compound_predicate import _pointer
from .filter_sum import _valid
from .arithmetic_sum import integer_binary
from .value_ops import checked_float_value
from .native_cast import native_cast_bits
from .native_numeric import (
    numeric_unary,
    float32_unary_bits,
    integer_unary_bits,
    native_numeric_binary,
)
from .null_value import fill_value_bits


@always_inline
def _leaf(
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    at: Int,
    row: Int,
    need_float: Bool,
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
) -> Tuple[UInt64, Bool]:
    var op = UInt32(program[unsafe_offset=at])
    if op == UInt32(LIT_INT):
        return (program[unsafe_offset=at + 3], True)
    if op != UInt32(COL):
        return (UInt64(0), False)
    var meta = program[unsafe_offset=at + 2]
    var slot = Int(program[unsafe_offset=at + 1])
    var present = True
    if (meta >> 32) & 1:
        present = _valid(
            _pointer(slot, b0, b1, b2, b3, b4, b5, b6, b7, b7),
            UInt32(meta),
            row,
        )
    var bits = UInt64(0)
    if present:
        var tag = UInt32((meta >> VALUE_TYPE_SHIFT) & 3) - 1
        var data = _pointer(slot, v0, v1, v2, v3, v4, v5, v6, v7, v7)
        if tag == 2:
            if need_float:
                bits = UInt64(data.unsafe_bitcast[UInt32]()[unsafe_offset=row])
        else:
            bits = bitcast[DType.uint64](read_integer(data, tag, row))
    return (bits, present)


@always_inline
def chain_expression_value(
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    count: Int,
    row_base: Int,
    row: Int,
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
    # The host proves leaf (unary | leaf binary)*. Unlike the general
    # interpreter this carries only one scalar value and validity, and no
    # data-dependent stack indexing or conditional-program control flow.
    var need_float = want_value or Bool(program[unsafe_offset=3] & 1)
    var first = _leaf(
        program,
        4,
        row_base + row,
        need_float,
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
    var bits = first[0]
    var present = first[1]
    var fault = UInt32(0)
    var schedule = program[unsafe_offset=3]
    var steps = Int((schedule >> 8) & 255)
    if steps > 0 and steps <= 3:
        var unary_mask = UInt32(schedule >> 16)
        comptime for step in range(3):
            if step < steps:
                # The host records instruction arity. Fixed scalar offsets
                # let the compiler hoist metadata out of the reduction loop.
                var at = (
                    2
                    + step * 2
                    - Int(pop_count(unary_mask & UInt32((1 << step) - 1)))
                ) * 4
                var unary = Bool(unary_mask & UInt32(1 << step))
                var right = (UInt64(0), False)
                if not unary:
                    right = _leaf(
                        program,
                        at,
                        row_base + row,
                        need_float,
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
                    at += 4
                var result = _apply(
                    program,
                    at,
                    UInt32(program[unsafe_offset=at]),
                    unary,
                    need_float,
                    bits,
                    present,
                    right,
                )
                bits = result[0]
                present = result[1]
                fault |= result[2]
        return (bits, present, fault)
    var node = 2
    while node < count:
        var at = node * 4
        var op = UInt32(program[unsafe_offset=at])
        var unary = op == UInt32(CAST) or numeric_unary(Int(op))
        var right = (UInt64(0), False)
        if not unary:
            right = _leaf(
                program,
                at,
                row_base + row,
                need_float,
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
            node += 1
            at += 4
            op = UInt32(program[unsafe_offset=at])
        var result = _apply(
            program, at, op, unary, need_float, bits, present, right
        )
        bits = result[0]
        present = result[1]
        fault |= result[2]
        node += 1
    return (bits, present, fault)


@always_inline
def _apply(
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    at: Int,
    op: UInt32,
    unary: Bool,
    need_float: Bool,
    bits: UInt64,
    var present: Bool,
    right: Tuple[UInt64, Bool],
) -> Tuple[UInt64, Bool, UInt32]:
    var fault = UInt32(0)
    var meta = program[unsafe_offset=at + 2]
    var tag = UInt32((meta >> VALUE_TYPE_SHIFT) & 3) - 1
    var result_bits = UInt64(0)
    if op == UInt32(CAST):
        if present:
            var descriptor = program[unsafe_offset=at + 3]
            var converted = native_cast_bits(bits, UInt32(descriptor >> 8), tag)
            result_bits = converted[0]
            if converted[1]:
                fault |= UInt32(descriptor & 1)
                present = False
    elif unary:
        if present:
            if tag == 2:
                if need_float:
                    result_bits = UInt64(
                        float32_unary_bits(UInt32(bits), Int(op))
                    )
            else:
                var result = integer_unary_bits(bits, tag, Int(op))
                result_bits = result[0]
                fault |= UInt32(result[1])
    elif op == UInt32(FILL_NULL) or op == UInt32(FILL_NAN):
        var result = fill_value_bits(bits, present, right[0], right[1], Int(op))
        result_bits = result[0]
        present = result[1]
    else:
        present = present and right[1]
        if present:
            if op != UInt32(ADD) and op != UInt32(SUB) and op != UInt32(MUL):
                if tag != 2 or need_float:
                    var result = native_numeric_binary(
                        bits, right[0], tag, Int(op)
                    )
                    result_bits = result[0]
                    present = result[1]
                    fault |= result[2]
            elif tag == 2:
                if need_float:
                    var result = checked_float_value(
                        bitcast[DType.float32](UInt32(bits)),
                        bitcast[DType.float32](UInt32(right[0])),
                        op,
                    )
                    result_bits = UInt64(
                        bitcast[DType.uint32](result[0]) ^ UInt32(meta)
                    )
                    fault |= UInt32(result[1]) << 1
            else:
                var result = integer_binary(
                    bitcast[DType.int64](bits),
                    bitcast[DType.int64](right[0]),
                    op,
                    tag,
                )
                result_bits = bitcast[DType.uint64](result[0]) if not result[
                    1
                ] else UInt64(0)
                fault |= UInt32(result[1])
    return (result_bits, present, fault)
