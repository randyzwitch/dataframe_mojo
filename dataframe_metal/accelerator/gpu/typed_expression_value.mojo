"""Bounded typed stack for checked casts and explicit mixed-source arithmetic."""
from std.memory import Pointer, bitcast
from dataframe.expr import COL, LIT_INT, CAST
from ...expression_format import EXPRESSION_NODE_WORDS, VALUE_TYPE_SHIFT
from .multi_reduce import Bytes, read_integer
from .compound_predicate import _pointer
from .filter_sum import _valid
from .arithmetic_sum import integer_binary
from .value_ops import checked_float_value
from .native_cast import native_cast_bits


@always_inline
def typed_expression_value(
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
    var values = SIMD[DType.uint64, 8](0)
    var validity = SIMD[DType.uint32, 8](0)
    var top = 0
    var fault = UInt32(0)
    var need_float = want_value or Bool(program[unsafe_offset=3] & 1)
    for node in range(1, count):
        var at = node * EXPRESSION_NODE_WORDS
        var op = UInt32(program[unsafe_offset=at])
        var meta = program[unsafe_offset=at + 2]
        var tag = UInt32((meta >> VALUE_TYPE_SHIFT) & 3) - 1
        var bits = UInt64(0)
        var present = True
        if op == UInt32(COL):
            var slot = Int(program[unsafe_offset=at + 1])
            if (meta >> 32) & 1:
                present = _valid(
                    _pointer(slot, b0, b1, b2, b3, b4, b5, b6, b7, b7),
                    UInt32(meta) + UInt32(row_base),
                    row,
                )
            if present:
                var data = _pointer(slot, v0, v1, v2, v3, v4, v5, v6, v7, v7)
                if tag == 2:
                    if need_float:
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
        elif op == UInt32(CAST):
            top -= 1
            present = Bool(validity[top])
            if present:
                var descriptor = program[unsafe_offset=at + 3]
                var converted = native_cast_bits(
                    values[top], UInt32(descriptor >> 8), tag
                )
                bits = converted[0]
                if converted[1]:
                    if descriptor & 1:
                        fault |= 1
                    present = False
        else:
            top -= 1
            var right = values[top]
            var right_valid = Bool(validity[top])
            top -= 1
            present = Bool(validity[top]) and right_valid
            if present:
                if tag == 2:
                    if need_float:
                        var result = checked_float_value(
                            bitcast[DType.float32](UInt32(values[top])),
                            bitcast[DType.float32](UInt32(right)),
                            op,
                        )
                        # The low metadata word is opaque zero, preserving the
                        # Float32 rounding boundary before the next operation.
                        bits = UInt64(
                            bitcast[DType.uint32](result[0]) ^ UInt32(meta)
                        )
                        fault |= UInt32(result[1]) << 1
                else:
                    var result = integer_binary(
                        bitcast[DType.int64](values[top]),
                        bitcast[DType.int64](right),
                        op,
                        tag,
                    )
                    bits = bitcast[DType.uint64](result[0]) if not result[
                        1
                    ] else UInt64(0)
                    fault |= UInt32(result[1])
        values[top] = bits
        validity[top] = UInt32(present)
        top += 1
    return (values[0], Bool(validity[0]), fault)
