"""Checked scalar Float32 arithmetic shared by fused value kernels."""
from std.memory import bitcast
from dataframe.expr import ADD, SUB, MUL


@always_inline
def checked_float_value(
    a: Float32, b: Float32, op: UInt32
) -> Tuple[Float32, Bool]:
    # Metal may flush subnormal operands/results during arithmetic. Reading
    # raw column bits into exact bins is safe, but arithmetic needs CPU
    # adjudication when the device cannot preserve gradual underflow.
    var ar = bitcast[DType.uint32](a) & 0x7FFF_FFFF
    var br = bitcast[DType.uint32](b) & 0x7FFF_FFFF
    var ae = ar >> 23
    var be = br >> 23
    var tiny = (ar != 0 and ae == 0) or (br != 0 and be == 0)
    var underflow = (ar != 0 and br != 0 and ae + be <= 127) if op == UInt32(
        MUL
    ) else ((ar != 0 and ae <= 24) or (br != 0 and be <= 24))
    var result = a + b if op == UInt32(ADD) else (
        a - b if op == UInt32(SUB) else a * b
    )
    return (result, tiny or underflow)
