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
