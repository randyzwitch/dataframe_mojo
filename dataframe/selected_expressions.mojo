"""Row-local arithmetic and casts computed for a selection's rows alone.

After a join or a filter a morsel keeps its probe-side columns where they
are and a selection over them. An expression over those columns used to
gather every column it read and then compute, or compute over every row of
the morsel, dropped ones included. DuckDB's expression executor does
neither: `ExpressionExecutor::Execute` is given the chunk's selection, and
its binary and unary executors read each input through its own selection
(`UnifiedVectorFormat`, `ScalarExecutor::ExecuteGenericBinaryNullable`),
writing one result per selected row.

This module does the same for the expressions that matter after selective
joins: decimal `+ - *`, Float64 `+ - * /`, Int64 `+ - *`, and the numeric
casts between them. It works a vector of 2,048 selected rows at a time.
Each column is read through the selection into a small buffer that stays in
cache. Each node then computes into its own buffer, and the root writes
either one value per selected row (compact) or each value at its row of
the morsel (scattered), so a morsel that is read in place keeps its
selection.

A dropped row is never computed, so it cannot raise. Any error, overflow
included, makes `evaluate_selected` return None. The caller then takes the
ordinary path, which raises the same error for the kept row.
"""
from dataframe.binding import BoundExpr, bind
from dataframe.decimal import divide_half_even, pow10, precision_limit
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.execution import evaluate
from dataframe.expr import ADD, CAST, COL, DIV, MUL, SUB, Expr, subtree
from dataframe.kernels.binary import (
    _decimal_mul,
    _round_div,
    _checked_mul,
    _checked_sub,
    checked_add,
)
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column

# Selected rows per vector, DuckDB's STANDARD_VECTOR_SIZE.
comptime _VECTOR = 2048

# How a node's values are held while computing: Int64 for integers,
# Int128 for decimals (unscaled), Float64 for floats, and Int64 for
# decimals when every decimal of the plan has at most 18 digits
# (`_narrow_decimals`).
comptime _LANE_INT = 0
comptime _LANE_DECIMAL = 1
comptime _LANE_FLOAT = 2
comptime _LANE_DECIMAL64 = 3

comptime _NODE_UNUSED = 0
comptime _NODE_COLUMN = 1
comptime _NODE_CONSTANT = 2
comptime _NODE_COMPUTED = 3


def _lane(dtype: DataType) -> Int:
    """The lane a dtype computes in, or -1 when it is not supported."""
    if dtype.is_categorical():
        return -1
    if dtype.is_decimal():
        return _LANE_DECIMAL
    if dtype == DataType.FLOAT64 or dtype == DataType.FLOAT32:
        return _LANE_FLOAT
    if (
        dtype == DataType.INT64
        or dtype == DataType.INT32
        or dtype == DataType.INT16
        or dtype == DataType.INT8
        or dtype == DataType.UINT32
        or dtype == DataType.UINT16
        or dtype == DataType.UINT8
    ):
        return _LANE_INT
    return -1


def _storage_supported(column: Series) -> Bool:
    """A contiguous numeric or decimal buffer this module reads."""
    if column.is_chunked():
        return False
    if column._data.isa[Column[Int128]]():
        return True
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D != DType.uint64:
            if column._data.isa[Column[Scalar[D]]]():
                return True
    return False


struct SelectedPlan(Copyable, Movable):
    """How each node of one bound expression is computed for selected
    rows: read from a column, a constant, or computed from its children."""

    var bound: BoundExpr
    var kinds: List[Int]
    var lanes: List[Int]
    var ints: List[Int64]
    var decimals: List[Int128]
    var floats: List[Float64]
    var constant_valid: List[Bool]

    def __init__(out self, var bound: BoundExpr):
        var n = len(bound.expr._nodes)
        self.kinds = List[Int](length=n, fill=_NODE_UNUSED)
        self.lanes = List[Int](length=n, fill=-1)
        self.ints = List[Int64](length=n, fill=0)
        self.decimals = List[Int128](length=n, fill=0)
        self.floats = List[Float64](length=n, fill=0)
        self.constant_valid = List[Bool](length=n, fill=True)
        self.bound = bound^


def _reads_column(bound: BoundExpr, index: Int) -> Bool:
    ref node = bound.expr._nodes[index]
    if node.op == COL:
        return True
    if node.left >= 0 and _reads_column(bound, node.left):
        return True
    return node.right >= 0 and _reads_column(bound, node.right)


def _constant(mut plan: SelectedPlan, index: Int) raises -> Bool:
    """Evaluate a node that reads no column once, as a one-row value."""
    var dtype = plan.bound.dtypes[index]
    var lane = _lane(dtype)
    if lane < 0:
        return False
    var empty = List[Series]()
    var value = evaluate(bind(subtree(plan.bound.expr, index), empty), empty, 1)
    if len(value) != 1 or value.is_chunked():
        return False
    if value.dtype() != dtype:
        if dtype.is_decimal() or value.dtype().is_decimal():
            return False
        value = value.cast(dtype)
    plan.kinds[index] = _NODE_CONSTANT
    plan.lanes[index] = lane
    if value._data.isa[Column[Int128]]():
        ref column = value._data[Column[Int128]]
        plan.constant_valid[index] = column._valid(0)
        plan.decimals[index] = column._get(0) if column._valid(0) else 0
        return lane == _LANE_DECIMAL
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if value._data.isa[Column[Scalar[D]]]():
            ref column = value._data[Column[Scalar[D]]]
            var valid = column._valid(0)
            plan.constant_valid[index] = valid
            var raw = column._get(0) if valid else Scalar[D](0)
            if lane == _LANE_FLOAT:
                plan.floats[index] = raw.cast[DType.float64]()
            elif lane == _LANE_DECIMAL:
                comptime if D.is_integral() and D != DType.uint64:
                    plan.decimals[index] = raw.cast[DType.int128]()
                else:
                    return False
            else:
                comptime if D.is_integral() and D != DType.uint64:
                    plan.ints[index] = raw.cast[DType.int64]()
                else:
                    return False
            return True
    return False


def _computed_supported(bound: BoundExpr, index: Int) raises -> Bool:
    """Whether node `index` (not a column or constant) computes here."""
    ref node = bound.expr._nodes[index]
    var dtype = bound.dtypes[index]
    if node.op == CAST:
        if node.left < 0 or len(node.dtypes) == 0 or not node.dtypes[0]:
            return False
        var source = bound.dtypes[node.left]
        if node.dtypes[0].value() != dtype:
            return False
        if _lane(source) < 0:
            return False
        if dtype.is_decimal():
            if source.is_decimal():
                # An upscale must stay within 38 digits before the check.
                return (
                    source.precision() + max(dtype.scale() - source.scale(), 0)
                    <= 38
                )
            return _lane(source) == _LANE_INT
        if dtype == DataType.FLOAT64:
            return True
        if dtype == DataType.INT64:
            return _lane(source) == _LANE_INT
        return False
    if not (
        node.op == ADD or node.op == SUB or node.op == MUL or node.op == DIV
    ):
        return False
    if node.left < 0 or node.right < 0:
        return False
    var left = bound.dtypes[node.left]
    var right = bound.dtypes[node.right]
    if dtype.is_decimal():
        return node.op != DIV and left.is_decimal() and right.is_decimal()
    if left != right:
        return False
    if dtype == DataType.FLOAT64:
        return left == DataType.FLOAT64
    if dtype == DataType.INT64:
        return node.op != DIV and left == DataType.INT64
    return False


def selected_plan(
    expr: Expr, columns: List[Series]
) raises -> Optional[SelectedPlan]:
    """A plan for computing `expr` over selected rows of `columns`, or None
    when it uses anything this module does not compute (or is a bare
    column or a constant, which have nothing to compute)."""
    var bound: BoundExpr
    try:
        bound = bind(expr, columns)
    except:
        return None
    var root = len(bound.expr._nodes) - 1
    if not _reads_column(bound, root) or bound.expr._nodes[root].op == COL:
        return None
    var plan = SelectedPlan(bound^)
    # Mark from the root down; a subtree that reads no column is one
    # constant.
    var needed = List[Bool](length=root + 1, fill=False)
    needed[root] = True
    for reverse in range(root + 1):
        var i = root - reverse
        if not needed[i]:
            continue
        var op = plan.bound.expr._nodes[i].op
        var left = plan.bound.expr._nodes[i].left
        var right = plan.bound.expr._nodes[i].right
        if op == COL:
            var source = plan.bound.sources[i]
            if source < 0:
                return None
            ref column = columns[source]
            var lane = _lane(plan.bound.dtypes[i])
            if (
                lane < 0
                or column.dtype() != plan.bound.dtypes[i]
                or not _storage_supported(column)
            ):
                return None
            plan.kinds[i] = _NODE_COLUMN
            plan.lanes[i] = lane
            continue
        if not _reads_column(plan.bound, i):
            try:
                if not _constant(plan, i):
                    return None
            except:
                return None
            continue
        if not _computed_supported(plan.bound, i):
            return None
        plan.kinds[i] = _NODE_COMPUTED
        plan.lanes[i] = _lane(plan.bound.dtypes[i])
        needed[left] = True
        if right >= 0:
            needed[right] = True
    _narrow_decimals(plan, columns)
    return plan^


def _narrow_decimals(mut plan: SelectedPlan, columns: List[Series]):
    """Every decimal node at 64 bits when each has at most 18 digits, as
    `_decimal_dense64` computes them: such values multiply, align and add
    exactly in 128-bit arithmetic, and a result checked against its
    precision fits 64 bits again. Left at 128 bits when any decimal is
    wider, a decimal column is stored at 128 bits, or a constant does not
    fit."""
    for i in range(len(plan.kinds)):
        if plan.kinds[i] == _NODE_UNUSED or plan.lanes[i] != _LANE_DECIMAL:
            continue
        if plan.bound.dtypes[i].precision() > 18:
            return
        if (
            plan.kinds[i] == _NODE_COLUMN
            and columns[plan.bound.sources[i]]._data.isa[Column[Int128]]()
        ):
            return
        if plan.kinds[i] == _NODE_CONSTANT:
            var value = plan.decimals[i]
            if value > Int128(Int64.MAX) or value < -Int128(Int64.MAX):
                return
    for i in range(len(plan.kinds)):
        if plan.kinds[i] == _NODE_UNUSED or plan.lanes[i] != _LANE_DECIMAL:
            continue
        plan.lanes[i] = _LANE_DECIMAL64
        if plan.kinds[i] == _NODE_CONSTANT:
            plan.ints[i] = Int64(plan.decimals[i])


struct _Vectors(Movable):
    """One vector of values and validity flags per node. `whole` is true
    where every row of the node's current vector is valid; its flags are
    then not read (nor kept up to date)."""

    var ints: List[List[Int64]]
    var decimals: List[List[Int128]]
    var floats: List[List[Float64]]
    var valid: List[List[UInt8]]
    var whole: List[Bool]

    def __init__(out self, plan: SelectedPlan):
        var n = len(plan.kinds)
        self.ints = List[List[Int64]](capacity=n)
        self.decimals = List[List[Int128]](capacity=n)
        self.floats = List[List[Float64]](capacity=n)
        self.valid = List[List[UInt8]](capacity=n)
        self.whole = List[Bool](length=n, fill=True)
        for i in range(n):
            var used = plan.kinds[i] != _NODE_UNUSED
            var lane = plan.lanes[i]
            self.ints.append(
                List[Int64](
                    length=_VECTOR if used
                    and (lane == _LANE_INT or lane == _LANE_DECIMAL64) else 0,
                    fill=0,
                )
            )
            self.decimals.append(
                List[Int128](
                    length=_VECTOR if used and lane == _LANE_DECIMAL else 0,
                    fill=0,
                )
            )
            self.floats.append(
                List[Float64](
                    length=_VECTOR if used and lane == _LANE_FLOAT else 0,
                    fill=0,
                )
            )
            var flag = UInt8(1)
            if plan.kinds[i] == _NODE_CONSTANT:
                flag = UInt8(1) if plan.constant_valid[i] else UInt8(0)
                self.whole[i] = plan.constant_valid[i]
                # A null constant's value is 0, as the plan holds it.
                if lane == _LANE_INT or lane == _LANE_DECIMAL64:
                    for k in range(_VECTOR):
                        self.ints[i][k] = plan.ints[i]
                elif lane == _LANE_DECIMAL:
                    for k in range(_VECTOR):
                        self.decimals[i][k] = plan.decimals[i]
                else:
                    for k in range(_VECTOR):
                        self.floats[i][k] = plan.floats[i]
            self.valid.append(
                List[UInt8](length=_VECTOR if used else 0, fill=flag)
            )


def _load[
    D: DType, T: DType, through: Bool
](
    column: Column[Scalar[D]],
    nulls: Bool,
    rows: List[Int],
    start: Int,
    count: Int,
    mut values: List[Scalar[T]],
    mut valid: List[UInt8],
):
    """Rows `start` to `start + count` of the selection, read from the
    column at their frame rows (`through`) or at their own positions (a
    column already holding only the selected rows). A null reads as 0;
    `nulls` is false when the column has none."""
    var source = column._ptr()
    var at = rows.unsafe_ptr()
    var out = values.unsafe_ptr()
    var flags = valid.unsafe_ptr()
    if not nulls:
        for k in range(count):
            var r: Int
            comptime if through:
                r = at[unsafe_offset=start + k]
            else:
                r = start + k
            out[unsafe_offset=k] = source[unsafe_offset=r].cast[T]()
        return
    var bits = column._bits[].unsafe_ptr()
    var offset = column._offset
    for k in range(count):
        var r: Int
        comptime if through:
            r = at[unsafe_offset=start + k]
        else:
            r = start + k
        var b = offset + r
        var flag = (bits[unsafe_offset=b >> 3] >> UInt8(b & 7)) & 1
        flags[unsafe_offset=k] = flag
        out[unsafe_offset=k] = source[unsafe_offset=r].cast[
            T
        ]() if flag != 0 else Scalar[T](0)


def _load_column[
    T: DType
](
    column: Series,
    through: Bool,
    nulls: Bool,
    rows: List[Int],
    start: Int,
    count: Int,
    mut values: List[Scalar[T]],
    mut valid: List[UInt8],
) raises:
    if column._data.isa[Column[Int128]]():
        comptime if T == DType.int128:
            if through:
                _load[DType.int128, T, True](
                    column._data[Column[Int128]],
                    nulls,
                    rows,
                    start,
                    count,
                    values,
                    valid,
                )
            else:
                _load[DType.int128, T, False](
                    column._data[Column[Int128]],
                    nulls,
                    rows,
                    start,
                    count,
                    values,
                    valid,
                )
            return
        raise Error("unsupported selected load")
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D != DType.uint64:
            if column._data.isa[Column[Scalar[D]]]():
                if through:
                    _load[D, T, True](
                        column._data[Column[Scalar[D]]],
                        nulls,
                        rows,
                        start,
                        count,
                        values,
                        valid,
                    )
                else:
                    _load[D, T, False](
                        column._data[Column[Scalar[D]]],
                        nulls,
                        rows,
                        start,
                        count,
                        values,
                        valid,
                    )
                return
    raise Error("unsupported selected load")


def _combine(
    mut vectors: _Vectors, left: Int, right: Int, index: Int, count: Int
):
    """Node `index` valid where both children are."""
    var left_whole = vectors.whole[left]
    var right_whole = vectors.whole[right]
    vectors.whole[index] = left_whole and right_whole
    if left_whole and right_whole:
        return
    if left_whole or right_whole:
        var one = vectors.valid[right if left_whole else left].unsafe_ptr()
        var to = vectors.valid[index].unsafe_ptr()
        for k in range(count):
            to[unsafe_offset=k] = one[unsafe_offset=k]
        return
    var a = vectors.valid[left].unsafe_ptr()
    var b = vectors.valid[right].unsafe_ptr()
    var o = vectors.valid[index].unsafe_ptr()
    for k in range(count):
        o[unsafe_offset=k] = a[unsafe_offset=k] & b[unsafe_offset=k]


def _decimal_product[
    divisor: Int
](
    mut vectors: _Vectors,
    left: Int,
    right: Int,
    index: Int,
    count: Int,
    rescale: Int128,
    limit: Int128,
) raises:
    """Exact products rounded half-even by `divisor` (0: by `rescale` at
    run time), as `_decimal_value` computes them. Null rows hold 0."""
    var a = vectors.decimals[left].unsafe_ptr()
    var b = vectors.decimals[right].unsafe_ptr()
    var o = vectors.decimals[index].unsafe_ptr()
    var flags = vectors.valid[index].unsafe_ptr()
    var whole = vectors.whole[index]
    var overflow = False
    for k in range(count):
        var value = _decimal_mul(a[unsafe_offset=k], b[unsafe_offset=k])
        comptime if divisor > 1:
            value = _round_div[divisor](value)
        elif divisor == 0:
            value = divide_half_even(value, rescale)
        value = value if whole or flags[unsafe_offset=k] != 0 else 0
        overflow |= value <= -limit or value >= limit
        o[unsafe_offset=k] = value
    if overflow:
        raise Error("value exceeds decimal precision")


def _decimal64_product[
    divisor: Int
](
    mut vectors: _Vectors,
    left: Int,
    right: Int,
    index: Int,
    count: Int,
    rescale: Int128,
    limit: Int128,
) raises:
    """`_decimal_product` for 64-bit operands: their product is exact in
    128 bits, and the rounded result is checked, so it fits 64 again."""
    var a = vectors.ints[left].unsafe_ptr()
    var b = vectors.ints[right].unsafe_ptr()
    var o = vectors.ints[index].unsafe_ptr()
    var overflow = False
    # A null row's operands are 0, so its product is too.
    for k in range(count):
        var value = Int128(a[unsafe_offset=k]) * Int128(b[unsafe_offset=k])
        comptime if divisor > 1:
            value = _round_div[divisor](value)
        elif divisor == 0:
            value = divide_half_even(value, rescale)
        overflow |= value <= -limit or value >= limit
        o[unsafe_offset=k] = Int64(value)
    if overflow:
        raise Error("value exceeds decimal precision")


def _decimal64_sum[
    op: Int, whole: Bool
](
    mut vectors: _Vectors,
    left: Int,
    right: Int,
    index: Int,
    count: Int,
    left_factor: Int128,
    right_factor: Int128,
    limit: Int128,
) raises:
    """A 64-bit decimal sum or difference, each side aligned in 128 bits
    (exact: a factor is at most 10^18). A null row holds 0."""
    var x = vectors.ints[left].unsafe_ptr()
    var y = vectors.ints[right].unsafe_ptr()
    var out = vectors.ints[index].unsafe_ptr()
    var flags = vectors.valid[index].unsafe_ptr()
    var overflow = False
    for k in range(count):
        var a = Int128(x[unsafe_offset=k]) * left_factor
        var b = Int128(y[unsafe_offset=k]) * right_factor
        var value: Int128
        comptime if op == ADD:
            value = a + b
        else:
            value = a - b
        comptime if not whole:
            value = value if flags[unsafe_offset=k] != 0 else 0
        overflow |= value <= -limit or value >= limit
        out[unsafe_offset=k] = Int64(value)
    if overflow:
        raise Error("value exceeds decimal precision")


def _decimal64(
    op: Int,
    mut vectors: _Vectors,
    left: Int,
    right: Int,
    index: Int,
    count: Int,
    left_type: DataType,
    right_type: DataType,
    dtype: DataType,
) raises:
    var limit = precision_limit(dtype.precision())
    if op == MUL:
        var shift = left_type.scale() + right_type.scale() - dtype.scale()
        var rescale = pow10(max(shift, 0))
        if shift <= 0:
            _decimal64_product[1](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 1:
            _decimal64_product[10](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 2:
            _decimal64_product[100](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 3:
            _decimal64_product[1000](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 4:
            _decimal64_product[10000](
                vectors, left, right, index, count, rescale, limit
            )
        else:
            _decimal64_product[0](
                vectors, left, right, index, count, rescale, limit
            )
        return
    var common = max(left_type.scale(), right_type.scale())
    var lf = pow10(common - left_type.scale())
    var rf = pow10(common - right_type.scale())
    var whole = vectors.whole[index]
    if op == ADD:
        if whole:
            _decimal64_sum[ADD, True](
                vectors, left, right, index, count, lf, rf, limit
            )
        else:
            _decimal64_sum[ADD, False](
                vectors, left, right, index, count, lf, rf, limit
            )
    elif whole:
        _decimal64_sum[SUB, True](
            vectors, left, right, index, count, lf, rf, limit
        )
    else:
        _decimal64_sum[SUB, False](
            vectors, left, right, index, count, lf, rf, limit
        )


def _float_op[
    op: Int, whole: Bool
](mut vectors: _Vectors, left: Int, right: Int, index: Int, count: Int):
    """Float64 `op` over one vector; a null row holds 0."""
    var x = vectors.floats[left].unsafe_ptr()
    var y = vectors.floats[right].unsafe_ptr()
    var out = vectors.floats[index].unsafe_ptr()
    var flags = vectors.valid[index].unsafe_ptr()
    for k in range(count):
        var a = x[unsafe_offset=k]
        var b = y[unsafe_offset=k]
        var value: Float64
        comptime if op == ADD:
            value = a + b
        elif op == SUB:
            value = a - b
        elif op == MUL:
            value = a * b
        else:
            value = a / b
        comptime if whole:
            out[unsafe_offset=k] = value
        else:
            out[unsafe_offset=k] = value if flags[unsafe_offset=k] != 0 else 0


def _int_op[
    op: Int, whole: Bool
](mut vectors: _Vectors, left: Int, right: Int, index: Int, count: Int) raises:
    """Checked Int64 `op` over one vector; a null row holds 0."""
    var x = vectors.ints[left].unsafe_ptr()
    var y = vectors.ints[right].unsafe_ptr()
    var out = vectors.ints[index].unsafe_ptr()
    var flags = vectors.valid[index].unsafe_ptr()
    for k in range(count):
        var a = x[unsafe_offset=k]
        var b = y[unsafe_offset=k]
        var value: Int64
        comptime if op == ADD:
            value = checked_add(a, b)
        elif op == SUB:
            value = _checked_sub(a, b)
        else:
            value = _checked_mul(a, b)
        comptime if whole:
            out[unsafe_offset=k] = value
        else:
            out[unsafe_offset=k] = value if flags[unsafe_offset=k] != 0 else 0


def _compute(
    plan: SelectedPlan, index: Int, mut vectors: _Vectors, count: Int
) raises:
    """Node `index` over one vector, from its children's vectors."""
    ref node = plan.bound.expr._nodes[index]
    var dtype = plan.bound.dtypes[index]
    var left = node.left
    var lane = plan.lanes[index]
    var source_lane = plan.lanes[left]
    if node.op == CAST:
        var source = plan.bound.dtypes[left]
        vectors.whole[index] = vectors.whole[left]
        if not vectors.whole[left]:
            var from_flags = vectors.valid[left].unsafe_ptr()
            var flags = vectors.valid[index].unsafe_ptr()
            for k in range(count):
                flags[unsafe_offset=k] = from_flags[unsafe_offset=k]
        if lane == _LANE_DECIMAL:
            var out = vectors.decimals[index].unsafe_ptr()
            var limit = precision_limit(dtype.precision())
            var overflow = False
            if source_lane == _LANE_INT:
                var factor = pow10(dtype.scale())
                var x = vectors.ints[left].unsafe_ptr()
                for k in range(count):
                    var value = Int128(x[unsafe_offset=k]) * factor
                    overflow |= value <= -limit or value >= limit
                    out[unsafe_offset=k] = value
            else:
                var x = vectors.decimals[left].unsafe_ptr()
                var shift = dtype.scale() - source.scale()
                var factor = pow10(abs(shift))
                for k in range(count):
                    var value = x[unsafe_offset=k]
                    if shift > 0:
                        value *= factor
                    elif shift < 0:
                        value = divide_half_even(value, factor)
                    overflow |= value <= -limit or value >= limit
                    out[unsafe_offset=k] = value
            if overflow:
                raise Error("value exceeds decimal precision")
            return
        if lane == _LANE_DECIMAL64:
            # From an integer or a 64-bit decimal, rescaled in 128 bits.
            var out = vectors.ints[index].unsafe_ptr()
            var x = vectors.ints[left].unsafe_ptr()
            var limit = precision_limit(dtype.precision())
            var shift = dtype.scale() - (
                source.scale() if source_lane == _LANE_DECIMAL64 else 0
            )
            var factor = pow10(abs(shift))
            var overflow = False
            for k in range(count):
                var value = Int128(x[unsafe_offset=k])
                if shift > 0:
                    value *= factor
                elif shift < 0:
                    value = divide_half_even(value, factor)
                overflow |= value <= -limit or value >= limit
                out[unsafe_offset=k] = Int64(value)
            if overflow:
                raise Error("value exceeds decimal precision")
            return
        if lane == _LANE_FLOAT:
            var out = vectors.floats[index].unsafe_ptr()
            if source_lane == _LANE_FLOAT:
                var x = vectors.floats[left].unsafe_ptr()
                for k in range(count):
                    out[unsafe_offset=k] = x[unsafe_offset=k]
            elif source_lane == _LANE_INT:
                var x = vectors.ints[left].unsafe_ptr()
                for k in range(count):
                    out[unsafe_offset=k] = Float64(x[unsafe_offset=k])
            elif source_lane == _LANE_DECIMAL64:
                var divisor = Float64(pow10(source.scale()))
                var x = vectors.ints[left].unsafe_ptr()
                for k in range(count):
                    out[unsafe_offset=k] = Float64(x[unsafe_offset=k]) / divisor
            else:
                # value / 10^scale, as `_decimal_to_float` converts.
                var divisor = Float64(pow10(source.scale()))
                var x = vectors.decimals[left].unsafe_ptr()
                for k in range(count):
                    out[unsafe_offset=k] = Float64(x[unsafe_offset=k]) / divisor
            return
        var out = vectors.ints[index].unsafe_ptr()
        var x = vectors.ints[left].unsafe_ptr()
        for k in range(count):
            out[unsafe_offset=k] = x[unsafe_offset=k]
        return
    var right = node.right
    _combine(vectors, left, right, index, count)
    var flags = vectors.valid[index].unsafe_ptr()
    if lane == _LANE_FLOAT:
        if node.op == ADD:
            if vectors.whole[index]:
                _float_op[ADD, True](vectors, left, right, index, count)
            else:
                _float_op[ADD, False](vectors, left, right, index, count)
        elif node.op == SUB:
            if vectors.whole[index]:
                _float_op[SUB, True](vectors, left, right, index, count)
            else:
                _float_op[SUB, False](vectors, left, right, index, count)
        elif node.op == MUL:
            if vectors.whole[index]:
                _float_op[MUL, True](vectors, left, right, index, count)
            else:
                _float_op[MUL, False](vectors, left, right, index, count)
        else:
            if vectors.whole[index]:
                _float_op[DIV, True](vectors, left, right, index, count)
            else:
                _float_op[DIV, False](vectors, left, right, index, count)
        return
    if lane == _LANE_INT:
        if node.op == ADD:
            if vectors.whole[index]:
                _int_op[ADD, True](vectors, left, right, index, count)
            else:
                _int_op[ADD, False](vectors, left, right, index, count)
        elif node.op == SUB:
            if vectors.whole[index]:
                _int_op[SUB, True](vectors, left, right, index, count)
            else:
                _int_op[SUB, False](vectors, left, right, index, count)
        else:
            if vectors.whole[index]:
                _int_op[MUL, True](vectors, left, right, index, count)
            else:
                _int_op[MUL, False](vectors, left, right, index, count)
        return
    # Decimals, as `_decimal_dense` computes them: products at the
    # operands' own scales, rounded to the result's; sums and differences
    # at the larger scale.
    var left_type = plan.bound.dtypes[left]
    var right_type = plan.bound.dtypes[right]
    var limit = precision_limit(dtype.precision())
    if lane == _LANE_DECIMAL64:
        _decimal64(
            node.op,
            vectors,
            left,
            right,
            index,
            count,
            left_type,
            right_type,
            dtype,
        )
        return
    if node.op == MUL:
        var shift = left_type.scale() + right_type.scale() - dtype.scale()
        var rescale = pow10(max(shift, 0))
        if shift <= 0:
            _decimal_product[1](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 1:
            _decimal_product[10](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 2:
            _decimal_product[100](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 3:
            _decimal_product[1000](
                vectors, left, right, index, count, rescale, limit
            )
        elif shift == 4:
            _decimal_product[10000](
                vectors, left, right, index, count, rescale, limit
            )
        else:
            _decimal_product[0](
                vectors, left, right, index, count, rescale, limit
            )
        return
    var common = max(left_type.scale(), right_type.scale())
    var left_factor = pow10(common - left_type.scale())
    var right_factor = pow10(common - right_type.scale())
    var x = vectors.decimals[left].unsafe_ptr()
    var y = vectors.decimals[right].unsafe_ptr()
    var out = vectors.decimals[index].unsafe_ptr()
    var whole = vectors.whole[index]
    var overflow = False
    for k in range(count):
        var a = x[unsafe_offset=k]
        var b = y[unsafe_offset=k]
        if left_factor != 1:
            a = _decimal_mul(a, left_factor)
        if right_factor != 1:
            b = _decimal_mul(b, right_factor)
        var value = a + b if node.op == ADD else a - b
        value = value if whole or flags[unsafe_offset=k] != 0 else 0
        overflow |= value <= -limit or value >= limit
        out[unsafe_offset=k] = value
    if overflow:
        raise Error("value exceeds decimal precision")


def _write[
    S: DType, T: DType, scatter: Bool
](
    values: List[Scalar[T]],
    valid: List[UInt8],
    whole: Bool,
    rows: List[Int],
    start: Int,
    count: Int,
    mut out: List[Scalar[S]],
    mut bits: List[UInt8],
    mut nulls: Int,
    mut next: Int,
):
    """Write one vector of the root's values: at positions `start` on
    (compact) or at their frame rows (`scatter`), the rows between those
    holding a valid 0 that nothing reads. `next` is the first row not yet
    written."""
    var source = values.unsafe_ptr()
    var flags = valid.unsafe_ptr()
    var at = rows.unsafe_ptr()
    var o = out.unsafe_ptr()
    var b = bits.unsafe_ptr()
    for k in range(count):
        var r: Int
        comptime if scatter:
            r = at[unsafe_offset=start + k]
            for gap in range(next, r):
                o[unsafe_offset=gap] = 0
        else:
            r = start + k
        o[unsafe_offset=r] = source[unsafe_offset=k].cast[S]()
        if not whole and flags[unsafe_offset=k] == 0:
            b[unsafe_offset=r >> 3] &= ~(UInt8(1) << UInt8(r & 7))
            nulls += 1
        next = r + 1


def _run[
    S: DType, scatter: Bool, dense: Bool
](
    plan: SelectedPlan,
    columns: List[Series],
    compact: List[Bool],
    rows: List[Int],
    mask: BoolColumn,
    height: Int,
    name: String,
) raises -> Series:
    """Every vector of selected rows through the plan; the root's values
    are stored as `S`. With `dense`, every row of the morsel instead, a
    window of 2,048 at a time, and a row `mask` drops is left valid."""
    var vectors = _Vectors(plan)
    var root = len(plan.kinds) - 1
    # Whether each column read has a null, counted once, not per vector.
    var has_nulls = List[Bool](length=root + 1, fill=False)
    for i in range(root + 1):
        if plan.kinds[i] == _NODE_COLUMN:
            has_nulls[i] = columns[plan.bound.sources[i]].null_count() > 0
            vectors.whole[i] = not has_nulls[i]
    var length = height if scatter or dense else len(rows)
    var values = List[Scalar[S]](unsafe_uninit_length=length)
    var bits = List[UInt8](length=(length + 7) // 8, fill=255)
    var nulls = 0
    var next = 0
    var total = height if dense else len(rows)
    for start in range(0, total, _VECTOR):
        var count = min(_VECTOR, total - start)
        for i in range(root + 1):
            var kind = plan.kinds[i]
            if kind == _NODE_COLUMN:
                var source = plan.bound.sources[i]
                var through = not compact[source] and not dense
                ref column = columns[source]
                var lane = plan.lanes[i]
                if lane == _LANE_INT or lane == _LANE_DECIMAL64:
                    _load_column[DType.int64](
                        column,
                        through,
                        has_nulls[i],
                        rows,
                        start,
                        count,
                        vectors.ints[i],
                        vectors.valid[i],
                    )
                elif lane == _LANE_DECIMAL:
                    _load_column[DType.int128](
                        column,
                        through,
                        has_nulls[i],
                        rows,
                        start,
                        count,
                        vectors.decimals[i],
                        vectors.valid[i],
                    )
                else:
                    _load_column[DType.float64](
                        column,
                        through,
                        has_nulls[i],
                        rows,
                        start,
                        count,
                        vectors.floats[i],
                        vectors.valid[i],
                    )
            elif kind == _NODE_COMPUTED:
                _compute(plan, i, vectors, count)
        var lane = plan.lanes[root]
        if lane == _LANE_INT or lane == _LANE_DECIMAL64:
            _write[S, DType.int64, scatter and not dense](
                vectors.ints[root],
                vectors.valid[root],
                vectors.whole[root],
                rows,
                start,
                count,
                values,
                bits,
                nulls,
                next,
            )
        elif lane == _LANE_DECIMAL:
            _write[S, DType.int128, scatter and not dense](
                vectors.decimals[root],
                vectors.valid[root],
                vectors.whole[root],
                rows,
                start,
                count,
                values,
                bits,
                nulls,
                next,
            )
        else:
            _write[S, DType.float64, scatter and not dense](
                vectors.floats[root],
                vectors.valid[root],
                vectors.whole[root],
                rows,
                start,
                count,
                values,
                bits,
                nulls,
                next,
            )
    var o = values.unsafe_ptr()
    for gap in range(next, length):
        o[unsafe_offset=gap] = 0
    comptime if dense:
        if nulls > 0:
            # Only a kept row's null counts; a dropped row reads as valid.
            nulls = 0
            var b = bits.unsafe_ptr()
            for r in range(length):
                if not (mask._valid(r) and mask._get(r)):
                    b[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
                elif (b[unsafe_offset=r >> 3] >> UInt8(r & 7)) & 1 == 0:
                    nulls += 1
    if nulls == 0:
        bits = List[UInt8]()
    return Series(
        name, Column[Scalar[S]](values=values^, bits=bits^)
    ).with_dtype(plan.bound.dtypes[root])


def _run_lanes[
    scatter: Bool, dense: Bool
](
    plan: SelectedPlan,
    columns: List[Series],
    compact: List[Bool],
    rows: List[Int],
    mask: BoolColumn,
    height: Int,
    name: String,
) raises -> Series:
    var dtype = plan.bound.dtypes[len(plan.kinds) - 1]
    if dtype.is_decimal():
        var width = dtype.decimal_width()
        if width == 32:
            return _run[DType.int32, scatter, dense](
                plan, columns, compact, rows, mask, height, name
            )
        if width == 64:
            return _run[DType.int64, scatter, dense](
                plan, columns, compact, rows, mask, height, name
            )
        return _run[DType.int128, scatter, dense](
            plan, columns, compact, rows, mask, height, name
        )
    if dtype == DataType.FLOAT64:
        return _run[DType.float64, scatter, dense](
            plan, columns, compact, rows, mask, height, name
        )
    return _run[DType.int64, scatter, dense](
        plan, columns, compact, rows, mask, height, name
    )


def evaluate_selected(
    plan: SelectedPlan,
    columns: List[Series],
    compact: List[Bool],
    rows: List[Int],
    height: Int,
    scatter: Bool,
    name: String,
) -> Optional[Series]:
    """The plan's expression for the selected rows `rows` (ascending frame
    rows of a morsel of `height` rows). A column flagged `compact` holds
    only the selected rows and is read at their positions; any other is
    read at their frame rows. The result has one value per selected row,
    or, with `scatter`, `height` values with each at its row and a valid 0
    at every other row. None when computing a kept row raises: the caller's
    ordinary path then raises it."""
    var mask = BoolColumn(List[Bool]())
    try:
        if scatter:
            return _run_lanes[True, False](
                plan, columns, compact, rows, mask, height, name
            )
        return _run_lanes[False, False](
            plan, columns, compact, rows, mask, height, name
        )
    except:
        return None


def evaluate_masked(
    plan: SelectedPlan,
    columns: List[Series],
    compact: List[Bool],
    mask: BoolColumn,
    height: Int,
    name: String,
) -> Optional[Series]:
    """The plan's expression at every row of a morsel of `height` rows
    whose selection is `mask`, for a morsel read in place: its kept rows
    are most of it, and listing them to read each through the list cost
    more than computing the dropped rows too, as `_with_columns_in_place`
    does. Each window of 2,048 rows is read straight from the columns. A
    dropped row's value goes unread and it is left valid. None when a
    column read holds only the kept rows or any row raises; the caller
    then computes the kept rows alone (`evaluate_selected`), so a dropped
    row's error is never raised."""
    for i in range(len(plan.kinds)):
        if plan.kinds[i] == _NODE_COLUMN and compact[plan.bound.sources[i]]:
            return None
    var rows = List[Int]()
    try:
        return _run_lanes[True, True](
            plan, columns, compact, rows, mask, height, name
        )
    except:
        return None
