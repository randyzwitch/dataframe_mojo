"""Host-scheduled scalar registers for bounded native expression trees."""
from std.memory import Pointer
from std.utils.static_tuple import StaticTuple
from dataframe.expr import COL, LIT_INT, CAST
from ...expression_format import CASE_NULL
from .multi_reduce import Bytes
from .native_numeric import numeric_unary
from .chain_expression_value import _leaf, _apply


@no_inline
def register_expression_value[
    registers: Int
](
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
    comptime assert (
        registers == 2 or registers == 3 or registers == 4 or registers == 8
    )
    var values = StaticTuple[UInt64, registers]()
    comptime for i in range(registers):
        values[i] = 0
    var valid = UInt32(0)
    var fault = UInt32(0)
    var slots = program[unsafe_offset=1]
    var need_float = want_value or Bool(program[unsafe_offset=3] & 1)
    for node in range(1, count):
        var at = node * 4
        var op = UInt32(program[unsafe_offset=at])
        var target = UInt32((slots >> UInt64((node - 1) * 3)) & 7)
        var result: Tuple[UInt64, Bool, UInt32]
        if (
            op == UInt32(COL)
            or op == UInt32(LIT_INT)
            or op == UInt32(CASE_NULL)
        ):
            var leaf = _leaf(
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
            result = (leaf[0], leaf[1], UInt32(0))
        else:
            var unary = op == UInt32(CAST) or numeric_unary(Int(op))
            var left = UInt64(0)
            var right = UInt64(0)
            # Every tuple index is a compile-time constant. Selection does
            # not create a per-row dynamically indexed value stack.
            comptime for i in range(registers):
                if target == UInt32(i):
                    left = values[i]
                if target + 1 == UInt32(i):
                    right = values[i]
            result = _apply(
                program,
                at,
                op,
                unary,
                need_float,
                left,
                Bool(valid & (UInt32(1) << target)),
                (right, Bool(valid & (UInt32(1) << (target + 1)))),
            )
        comptime for i in range(registers):
            if target == UInt32(i):
                values[i] = result[0]
        valid = (valid & ~(UInt32(1) << target)) | (UInt32(result[1]) << target)
        fault |= result[2]
    return (values[0], Bool(valid & 1), fault)
