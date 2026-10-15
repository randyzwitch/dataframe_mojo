"""Mojo GPU row programs over typed words and byte validity.

Only fixed-width scalars and device pointers cross a kernel boundary. Scratch
is bounded stack storage; kernels allocate no heap memory and call no host code.
Errors travel with values so an unselected conditional branch is unobserved.
"""
from max.gpu import global_idx
from dataframe.expr import (
    DT_YEAR,
    DT_TIME,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    FILL_NAN,
    CLIP_LOW,
    CLIP_HIGH,
    CAST,
    GT,
    EQ,
    LT,
    GE,
    LE,
    NE,
    AND,
    OR,
    XOR,
    FILL_NULL,
    KEEP_NULLS,
    INT_IS_IN,
    NEG,
    ABS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    WHEN,
)
from std.memory import Pointer, unsafe_stack_allocation
from . import floating, temporal, decimal, integers, iana_fields
from . import temporal_arithmetic
from .temporal_descriptors import ROW_TEMPORAL_BINARY, ROW_TEMPORAL_CAST
from .decimal_descriptors import (
    DECIMAL_FLOAT,
    DECIMAL_CAST,
    DECIMAL_COMPARE,
    DECIMAL_BINARY,
    DECIMAL_UNARY,
)
from dataframe.accelerator.rows import ROW_SLICE, ROW_IANA_ZONE_FIELDS


@always_inline
def integer_width(dtype: Int64) -> UInt32:
    if dtype == 5 or dtype == 7:
        return 8
    if dtype == 6 or dtype == 8:
        return 16
    if dtype == 2 or dtype == 9:
        return 32
    return 64


@always_inline
def unsigned_type(dtype: Int64) -> Bool:
    return dtype >= 7 and dtype <= 10


@always_inline
def integer_fits(value: UInt64, dtype: Int64) -> Bool:
    var width = integer_width(dtype)
    if unsigned_type(dtype):
        return width == 64 or value <= (UInt64(1) << UInt64(width)) - 1
    if width == 64:
        return True
    var signed = value.cast[DType.int64]()
    var high = (Int64(1) << Int64(width - 1)) - 1
    return signed >= -high - 1 and signed <= high


@always_inline
def integer_arithmetic(
    op: Int64, a: UInt64, b: UInt64, dtype: Int64
) -> Tuple[UInt64, UInt32]:
    var value: UInt64
    var fits: Bool
    if op == ADD:
        value = a + b
        if unsigned_type(dtype):
            fits = value >= a
        else:
            fits = ((a ^ value) & (b ^ value)) >> 63 == 0
    elif op == SUB:
        value = a - b
        if unsigned_type(dtype):
            fits = a >= b
        else:
            fits = ((a ^ b) & (a ^ value)) >> 63 == 0
    else:
        var negative = False
        var left = a
        var right = b
        var maximum = UInt64.MAX
        if not unsigned_type(dtype):
            negative = (a ^ b) >> 63 != 0
            left = UInt64(0) - a if a >> 63 else a
            right = UInt64(0) - b if b >> 63 else b
            maximum = UInt64(1) << 63
            maximum -= UInt64(not negative)
        fits = right == 0 or left <= maximum // right
        value = left * right
        if negative:
            value = UInt64(0) - value
    return (value, UInt32(not (fits and integer_fits(value, dtype))))


from . import ordered_faults


def _combine[ordered: Bool](left: UInt32, right: UInt32) -> UInt32:
    comptime if ordered:
        return ordered_faults.merge(left, right)
    else:
        return left | right


def _new_fault[ordered: Bool](category: UInt32, rank: UInt32) -> UInt32:
    comptime if ordered:
        return ordered_faults.pack(rank, category) if category else UInt32(0)
    else:
        return category


def _temporal_inputs[ordered: Bool](left: UInt32, right: UInt32) -> UInt32:
    comptime if ordered:
        return ordered_faults.merge(left, right)
    else:
        return left if left else right


def _row_program[
    ordered: Bool
](
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    fault_ranks: Pointer[UInt32, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    target: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var values = unsafe_stack_allocation[128, DType.uint64]()
    var highs = unsafe_stack_allocation[128, DType.uint64]()
    var valid = unsafe_stack_allocation[128, DType.uint8]()
    var failed = unsafe_stack_allocation[128, DType.uint32]()
    for node in range(Int(nodes)):
        var absolute = Int(start) + node
        var op = code[unsafe_offset=4 * absolute]
        var a = Int(code[unsafe_offset=4 * absolute + 1])
        var b = Int(code[unsafe_offset=4 * absolute + 2])
        var c = Int(code[unsafe_offset=4 * absolute + 3])
        var dtype = types[unsafe_offset=absolute]
        var value = UInt64(0)
        var high = UInt64(0)
        var present = True
        var error = UInt32(0)
        var own_rank = UInt32(0)
        comptime if ordered:
            own_rank = fault_ranks[unsafe_offset=absolute]
        if op == COL:
            var at = c * Int(stride) + row
            value = words[unsafe_offset=at]
            high = words[unsafe_offset=Int(stride) * Int(slots) + at]
            present = validity[unsafe_offset=at] != 0
            error = errors[unsafe_offset=at]
        elif op == ROW_SLICE:
            var offset = literals[unsafe_offset=a].cast[DType.int64]()
            var length = literals[unsafe_offset=b].cast[DType.int64]()
            var begin = min(
                Int64(rows),
                max(Int64(0), Int64(rows) + offset) if offset < 0 else offset,
            )
            var span = Int64(rows) - begin
            if length >= 0:
                span = min(span, length)
            value = UInt64(Int64(row) >= begin and Int64(row) - begin < span)
        elif op == LIT_INT or op == LIT_FLOAT or op == LIT_BOOL:
            value = literals[unsafe_offset=absolute]
            if dtype == 12 and op == LIT_INT:
                high = literals[unsafe_offset=c]
        elif op == LIT_NULL:
            present = False
        elif op == WHEN:
            var chosen = (
                b if valid[unsafe_offset=a]
                and (values[unsafe_offset=a] != 0) else c
            )
            value = values[unsafe_offset=chosen]
            high = highs[unsafe_offset=chosen]
            present = valid[unsafe_offset=chosen] != 0
            error = _combine[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=chosen]
            )
            var source_type = types[unsafe_offset=Int(start) + chosen]
            if dtype == 12 and source_type != 12:
                high = UInt64.MAX if value >> 63 else UInt64(0)
            if (dtype == 1 or dtype == 11) and source_type != dtype:
                value = floating.word(value, source_type)
                if dtype == 1:
                    value = UInt64(floating.to_float32(value))
        elif op == IS_NULL or op == IS_NOT_NULL:
            value = UInt64((valid[unsafe_offset=a] != 0) == (op == IS_NOT_NULL))
            error = failed[unsafe_offset=a]
        elif op == FILL_NULL or op == FILL_NAN:
            var use_left = valid[unsafe_offset=a] != 0
            if op == FILL_NAN:
                use_left = not floating.is_nan(
                    floating.word(
                        values[unsafe_offset=a],
                        types[unsafe_offset=Int(start) + a],
                    )
                )
            var chosen = a if use_left else b
            value = values[unsafe_offset=chosen]
            high = highs[unsafe_offset=chosen]
            present = valid[unsafe_offset=chosen] != 0
            # CPU evaluates both operands before choosing the output value.
            # An enclosing WHEN still masks this entire expression's fault.
            error = _combine[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=b]
            )
            var source_type = types[unsafe_offset=Int(start) + chosen]
            if dtype == 12 and source_type != 12:
                high = UInt64.MAX if value >> 63 else UInt64(0)
            if (dtype == 1 or dtype == 11) and source_type != dtype:
                value = floating.word(value, source_type)
                if dtype == 1:
                    value = UInt64(floating.to_float32(value))
        elif op == DECIMAL_FLOAT or op == DECIMAL_CAST or op == DECIMAL_UNARY:
            value = values[unsafe_offset=a]
            var source_type = types[unsafe_offset=Int(start) + a]
            high = highs[unsafe_offset=a] if source_type == 12 else (
                UInt64.MAX if not unsigned_type(source_type)
                and source_type != 4
                and value >> 63 else UInt64(0)
            )
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if present:
                var input = decimal.Wide(value, high)
                if op == DECIMAL_FLOAT or (
                    op == DECIMAL_CAST and (dtype == 1 or dtype == 11)
                ):
                    value = floating.div(
                        floating.from_wide(value, high, False),
                        literals[unsafe_offset=absolute],
                    )
                    if dtype == 1:
                        value = UInt64(floating.to_float32(value))
                    high = 0
                elif op == DECIMAL_UNARY:
                    var result = decimal.unary(
                        input,
                        UInt32(128) if dtype == 12 else integer_width(dtype),
                        c == ABS,
                    )
                    value = result.value.low
                    high = result.value.high
                    error = _combine[ordered](
                        error,
                        _new_fault[ordered](
                            decimal.error_flag(result.error), own_rank
                        ),
                    )
                    present = result.error == 0
                else:
                    var result = decimal.cast(
                        input,
                        UInt32(c & 63),
                        UInt32((c >> 6) & 63),
                        UInt32((c >> 12) & 63),
                        UInt32(128) if dtype == 12 else integer_width(dtype),
                        unsigned_type(dtype),
                        dtype == 4,
                    )
                    value = result.value.low
                    high = result.value.high if dtype == 12 else UInt64(0)
                    present = result.valid
                    if not present and (c & (1 << 18)):
                        error = _combine[ordered](
                            error, _new_fault[ordered](1, own_rank)
                        )
        elif op == DECIMAL_BINARY or op == DECIMAL_COMPARE:
            present = (
                valid[unsafe_offset=a] != 0 and valid[unsafe_offset=b] != 0
            )
            error = _combine[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=b]
            )
            if present:
                var left = values[unsafe_offset=a]
                var right = values[unsafe_offset=b]
                var left_type = types[unsafe_offset=Int(start) + a]
                var right_type = types[unsafe_offset=Int(start) + b]
                var lhs = decimal.Wide(
                    left,
                    highs[unsafe_offset=a] if left_type
                    == 12 else (UInt64.MAX if left >> 63 else UInt64(0)),
                )
                var rhs = decimal.Wide(
                    right,
                    highs[unsafe_offset=b] if right_type
                    == 12 else (UInt64.MAX if right >> 63 else UInt64(0)),
                )
                var left_scale = UInt32(c & 63)
                var right_scale = UInt32((c >> 6) & 63)
                var operation = UInt32(c >> 12)
                if op == DECIMAL_COMPARE:
                    var result = decimal.compare(
                        lhs, rhs, left_scale, right_scale, operation
                    )
                    value = UInt64(result.value)
                    present = result.valid
                    error = _combine[ordered](
                        error,
                        _new_fault[ordered](
                            UInt32(4) if not present else UInt32(0), own_rank
                        ),
                    )
                else:
                    var result: decimal.Arithmetic
                    if operation == MUL:
                        result = decimal.multiply(
                            lhs,
                            rhs,
                            left_scale,
                            right_scale,
                            UInt32(literals[unsafe_offset=absolute]),
                        )
                    elif operation == DIV:
                        result = decimal.divide_rows(
                            lhs, rhs, left_scale, right_scale
                        )
                    else:
                        result = decimal.add_subtract(
                            lhs,
                            rhs,
                            left_scale,
                            right_scale,
                            operation == SUB,
                            UInt32(literals[unsafe_offset=absolute]),
                        )
                    value = result.value.low
                    high = result.value.high
                    present = result.error == 0
                    error = _combine[ordered](
                        error,
                        _new_fault[ordered](
                            decimal.error_flag(result.error), own_rank
                        ),
                    )
        elif op == KEEP_NULLS:
            value = values[unsafe_offset=b]
            high = highs[unsafe_offset=b]
            present = (
                valid[unsafe_offset=a] != 0 and valid[unsafe_offset=b] != 0
            )
            error = _combine[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=b]
            )
        elif op == INT_IS_IN:
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if present:
                for member in range(c):
                    value |= UInt64(
                        values[unsafe_offset=a]
                        == literals[unsafe_offset=b + member]
                    )
        elif op == ROW_TEMPORAL_BINARY:
            present = (
                valid[unsafe_offset=a] != 0 and valid[unsafe_offset=b] != 0
            )
            error = _temporal_inputs[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=b]
            )
            if present and error == 0:
                var parameter = literals[unsafe_offset=absolute]
                var result = temporal_arithmetic.binary(
                    Int64(parameter >> 12),
                    values[unsafe_offset=a],
                    values[unsafe_offset=b],
                    parameter,
                )
                value = result[0]
                present = result[1]
                error = _new_fault[ordered](result[2], own_rank)
        elif op == ROW_TEMPORAL_CAST:
            value = values[unsafe_offset=a]
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if present and error == 0:
                var parameter = literals[unsafe_offset=absolute]
                var result = temporal_arithmetic.cast(
                    value, parameter, (parameter & (UInt64(1) << 12)) != 0
                )
                value = result[0]
                present = result[1]
                error = _new_fault[ordered](result[2], own_rank)
        elif op == CAST:
            value = values[unsafe_offset=a]
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            var source_type = types[unsafe_offset=Int(start) + a]
            var fits = True
            if present:
                if dtype == 1 or dtype == 11:
                    if source_type != dtype:
                        value = floating.word(value, source_type)
                        if dtype == 1:
                            value = UInt64(floating.to_float32(value))
                elif dtype == 4:
                    if source_type == 1 or source_type == 11:
                        var input = floating.word(value, source_type)
                        fits = not floating.is_nan(input)
                        value = UInt64(floating.absolute(input) != 0)
                    else:
                        value = UInt64(value != 0)
                elif source_type == 1 or source_type == 11:
                    var converted = floating.to_integer(
                        floating.word(value, source_type),
                        integer_width(dtype),
                        unsigned_type(dtype),
                    )
                    value = converted[0]
                    fits = converted[1]
                else:
                    fits = integer_fits(value, dtype)
                    if unsigned_type(source_type) and not unsigned_type(dtype):
                        fits = fits and value >> 63 == 0
                    elif not unsigned_type(source_type) and unsigned_type(
                        dtype
                    ):
                        fits = fits and value >> 63 == 0
                if not fits:
                    if c:
                        error = _combine[ordered](
                            error, _new_fault[ordered](1, own_rank)
                        )
                    else:
                        present = False
        elif (
            op == IS_NAN
            or op == IS_NOT_NAN
            or op == IS_FINITE
            or op == IS_INFINITE
        ):
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            var input = floating.word(
                values[unsafe_offset=a], types[unsafe_offset=Int(start) + a]
            )
            if op == IS_NAN or op == IS_NOT_NAN:
                value = UInt64(floating.is_nan(input) == (op == IS_NAN))
            elif op == IS_FINITE:
                value = UInt64(floating.absolute(input) < floating.INF)
            else:
                value = UInt64(floating.is_inf(input))
        elif temporal.supported(op):
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if present:
                var temporal_value = values[unsafe_offset=a]
                var temporal_parameter = Int64(b)
                var temporal_unit = Int64(c)
                if (
                    op >= DT_YEAR
                    and op <= DT_TIME
                    and (temporal_unit & ROW_IANA_ZONE_FIELDS)
                ):
                    temporal_unit &= 7
                    temporal_value = iana_fields.local_ticks(
                        literals,
                        b,
                        temporal_value,
                        temporal.per_second(temporal_unit),
                    )
                    temporal_parameter = 0
                var result = temporal.evaluate(
                    temporal_value,
                    op,
                    temporal_parameter,
                    temporal_unit,
                    literals[unsafe_offset=absolute],
                )
                value = result[0]
                error = _combine[ordered](
                    error, _new_fault[ordered](result[1], own_rank)
                )
        elif op == NEG or op == ABS or op == NOT:
            value = values[unsafe_offset=a]
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if op == NOT:
                value = UInt64(value == 0)
            elif dtype == 1 or dtype == 11:
                var sign = UInt64(1) << UInt64(31 if dtype == 1 else 63)
                value = value ^ sign if op == NEG else value & ~sign
            elif present:
                if op == NEG or (not unsigned_type(dtype) and value >> 63):
                    value = UInt64(0) - value
                    if (
                        not integer_fits(value, dtype)
                        or (value == UInt64(1) << 63)
                        or (unsigned_type(dtype) and value != 0)
                    ):
                        error = _combine[ordered](
                            error, _new_fault[ordered](1, own_rank)
                        )
        elif (
            op == SQRT
            or op == EXP
            or op == LOG
            or op == FLOOR
            or op == CEIL
            or op == ROUND
        ):
            present = valid[unsafe_offset=a] != 0
            error = failed[unsafe_offset=a]
            if present:
                var operand = types[unsafe_offset=Int(start) + a]
                var input = floating.word(values[unsafe_offset=a], operand)
                if (
                    dtype != 1
                    and dtype != 11
                    and (op == FLOOR or op == CEIL or op == ROUND)
                ):
                    value = values[unsafe_offset=a]
                elif op == SQRT:
                    value = floating.sqrt(input)
                elif op == EXP:
                    value = floating.exp(input)
                elif op == LOG:
                    value = floating.log(input)
                elif op == FLOOR or op == CEIL:
                    value = floating.integral(input, op == FLOOR)
                elif c == 0:
                    value = floating.round_integer(input)
                else:
                    value = floating.decimal_round(
                        input, literals[unsafe_offset=absolute], c < 0
                    )
                if dtype == 1:
                    value = UInt64(floating.to_float32(value))
        else:
            var left = values[unsafe_offset=a]
            var right = values[unsafe_offset=b]
            var av = valid[unsafe_offset=a] != 0
            var bv = valid[unsafe_offset=b] != 0
            present = av and bv
            error = _combine[ordered](
                failed[unsafe_offset=a], failed[unsafe_offset=b]
            )
            if op == AND or op == OR or op == XOR:
                if op == AND:
                    present = (
                        (av and left == 0) or (bv and right == 0) or present
                    )
                    value = UInt64(left != 0 and right != 0)
                elif op == OR:
                    present = (
                        (av and left != 0) or (bv and right != 0) or present
                    )
                    value = UInt64(left != 0 or right != 0)
                else:
                    value = UInt64((left != 0) != (right != 0))
            elif present:
                if (dtype == 1 or dtype == 11) and (
                    op == ADD
                    or op == SUB
                    or op == MUL
                    or op == DIV
                    or op == FLOORDIV
                    or op == MOD
                    or op == POW
                ):
                    var x = floating.word(
                        left, types[unsafe_offset=Int(start) + a]
                    )
                    var y = floating.word(
                        right, types[unsafe_offset=Int(start) + b]
                    )
                    if op == ADD:
                        value = floating.add(x, y)
                    elif op == SUB:
                        value = floating.sub(x, y)
                    elif op == MUL:
                        value = floating.mul(x, y)
                    elif op == MOD:
                        value = floating.modulo(x, y, dtype == 1)
                        if dtype == 1:
                            value = floating.from_float32(UInt32(value))
                    elif op == POW:
                        value = floating.power(x, y)
                    else:
                        value = floating.div(x, y)
                        # Binary32 floor division floors its rounded quotient.
                        if dtype == 1:
                            value = floating.from_float32(
                                floating.to_float32(value)
                            )
                        if op == FLOORDIV:
                            value = floating.integral(value, True)
                    if dtype == 1:
                        value = UInt64(floating.to_float32(value))
                elif op == ADD or op == SUB or op == MUL:
                    var result = integer_arithmetic(op, left, right, dtype)
                    value = result[0]
                    error = _combine[ordered](
                        error, _new_fault[ordered](result[1], own_rank)
                    )
                elif op == FLOORDIV or op == MOD or op == POW:
                    var result = integers.evaluate(
                        op,
                        left,
                        right,
                        integer_width(dtype),
                        unsigned_type(dtype),
                    )
                    value = result[0]
                    present = result[1]
                    error = _combine[ordered](
                        error,
                        _new_fault[ordered](
                            (
                                UInt32(32) if op == POW
                                and not unsigned_type(dtype)
                                and right >> 63 else result[2]
                            ),
                            own_rank,
                        ),
                    )
                else:
                    var operand = types[unsafe_offset=Int(start) + a]
                    var other_operand = types[unsafe_offset=Int(start) + b]
                    var less = left < right if unsigned_type(operand) else (
                        left.cast[DType.int64]() < right.cast[DType.int64]()
                    )
                    var equal = left == right
                    var greater = not less and not equal
                    if (
                        operand == 1
                        or operand == 11
                        or other_operand == 1
                        or other_operand == 11
                    ):
                        var x = floating.word(left, operand)
                        var y = floating.word(right, other_operand)
                        less = floating.less(x, y)
                        equal = floating.equal(x, y)
                        greater = floating.less(y, x)
                    if operand == 12:
                        var lhs = decimal.Wide(left, highs[unsafe_offset=a])
                        var rhs = decimal.Wide(right, highs[unsafe_offset=b])
                        less = decimal.signed_less(lhs, rhs)
                        equal = decimal.wide_equal(lhs, rhs)
                        greater = not less and not equal
                    if op == CLIP_LOW or op == CLIP_HIGH:
                        var chosen = b if (
                            less if op == CLIP_LOW else greater
                        ) else a
                        value = values[unsafe_offset=chosen]
                        high = highs[unsafe_offset=chosen]
                    elif op == GT:
                        value = UInt64(greater)
                    elif op == EQ:
                        value = UInt64(equal)
                    elif op == LT:
                        value = UInt64(less)
                    elif op == GE:
                        value = UInt64(greater or equal)
                    elif op == LE:
                        value = UInt64(less or equal)
                    elif op == NE:
                        value = UInt64(not equal)
        values[unsafe_offset=node] = value if present else UInt64(0)
        highs[unsafe_offset=node] = high if present and (
            dtype == 12 or dtype == 9
        ) else UInt64(0)
        valid[unsafe_offset=node] = UInt8(present)
        failed[unsafe_offset=node] = error
    var root = Int(nodes) - 1
    var destination = Int(target) * Int(stride) + row
    words[unsafe_offset=destination] = values[unsafe_offset=root]
    if slots:
        words[unsafe_offset=Int(stride) * Int(slots) + destination] = highs[
            unsafe_offset=root
        ]
    validity[unsafe_offset=destination] = valid[unsafe_offset=root]
    errors[unsafe_offset=destination] = failed[unsafe_offset=root]


def row_program(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    target: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    _row_program[False](
        words,
        validity,
        errors,
        code,
        literals,
        types,
        code.unsafe_bitcast[UInt32](),
        start,
        nodes,
        target,
        rows,
        stride,
        slots,
    )


def row_program_ordered(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    fault_ranks: Pointer[UInt32, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    target: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    _row_program[True](
        words,
        validity,
        errors,
        code,
        literals,
        types,
        fault_ranks,
        start,
        nodes,
        target,
        rows,
        stride,
        slots,
    )
