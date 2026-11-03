"""Expressions over data chunks, read through the unified vector format.

DuckDB's `ExpressionExecutor` (execution/expression_executor.cpp) runs a
bound expression over a chunk: every function reads its inputs through
`UnifiedVectorFormat` (selection, data, validity), so a dictionary vector
from a filter or a join is read where it is and never gathered first, and
`Select` evaluates a filter's conjuncts one after another, each on the
rows the ones before kept (`ExpressionExecutor::Select`,
`SelectAnd`). This module does the same for one bound expression,
compiled once per worker for a step's input columns:

- a column reads its input vector; a subtree that reads no column is one
  constant, computed once;
- integer, decimal and float arithmetic and numeric casts compute in
  lanes (Int64, Int128, Float64), with the exact decimal rounding and
  overflow checks of the column kernels (`kernels/binary.mojo`,
  `decimal.mojo`);
- comparisons of numbers or strings (string_t), Kleene AND/OR, NOT and
  null tests compute in bytes;
- any other node is computed by the at-rest evaluator (`execution.mojo`)
  over the rows it needs, gathered: the adapter step 2's expression
  executor removes.

Rows are processed a chunk at a time: `evaluate` is given the chunk rows
to compute (`rows`, a selection over the chunk, or every row) and computes
each node for exactly those rows, in order. A dropped row is never
computed, so it never raises.
"""
from std.memory import Pointer, unsafe_memcpy

from std.ffi import external_call
from std.io import RawFileDescriptor
from std.os import getenv
from dataframe.binding import BoundExpr, bind, op_name
from dataframe.decimal import divide_half_even, pow10, precision_limit
from dataframe.dtype import DataType
from dataframe.execution import evaluate
from dataframe.expr import (
    ADD,
    AND,
    CAST,
    COL,
    DIV,
    EQ,
    GE,
    GT,
    INT_IS_IN,
    IS_NOT_NULL,
    IS_NULL,
    LE,
    LT,
    MUL,
    NE,
    NOT,
    OR,
    STR_CONTAINS,
    STR_ENDS_WITH,
    STR_IS_IN,
    STR_STARTS_WITH,
    SUB,
    WHEN,
    Expr,
    subtree,
)
from dataframe.kernels.binary import (
    _checked_mul,
    _checked_sub,
    _decimal_mul,
    _round_div,
    checked_add,
)
from dataframe.series import Series
from dataframe.text.predicates import decode_values
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    flat_over,
    gather_series,
)
from dataframe.vector.types import (
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    PHYSICAL_BOOL,
    PHYSICAL_DOUBLE,
    PHYSICAL_FLOAT,
    PHYSICAL_INT128,
    PHYSICAL_INT16,
    PHYSICAL_INT32,
    PHYSICAL_INT64,
    PHYSICAL_INT8,
    PHYSICAL_UINT16,
    PHYSICAL_UINT32,
    PHYSICAL_UINT8,
    PHYSICAL_VARCHAR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    StringT,
    UnifiedVectorFormat,
    ValidityMask,
    Vector,
    physical_of,
    set_invalid,
    string_compare,
    string_equals,
    string_data,
    zero_selection,
)

# How a node is computed.
comptime _KIND_UNUSED = 0
comptime _KIND_COLUMN = 1
comptime _KIND_CONSTANT = 2
comptime _KIND_LANE = 3
comptime _KIND_FALLBACK = 4

# How a node's values are held while computing (selected_expressions'
# lanes): Int64 for integers and temporal values, Int128 for decimals
# (unscaled), Float64 for floats, Int64 for decimals when every decimal of
# the expression has at most 18 digits, a byte for Booleans; strings are
# read as string_t through their vectors; ANY is a node only its validity
# is read of.
comptime _LANE_NONE = -1
comptime _LANE_INT = 0
comptime _LANE_DECIMAL = 1
comptime _LANE_FLOAT = 2
comptime _LANE_DECIMAL64 = 3
comptime _LANE_BOOL = 4
comptime _LANE_STRING = 5
comptime _LANE_ANY = 6


def _lane(dtype: DataType) -> Int:
    """The lane a value of `dtype` computes in, or _LANE_NONE."""
    if dtype.is_categorical():
        return _LANE_NONE
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
        or dtype.is_temporal()
    ):
        return _LANE_INT
    if dtype == DataType.BOOL:
        return _LANE_BOOL
    if dtype == DataType.STRING:
        return _LANE_STRING
    return _LANE_NONE


def _is_compare(op: Int) -> Bool:
    return op == EQ or op == NE or op == LT or op == LE or op == GT or op == GE


def _numeric_lane(lane: Int) -> Bool:
    return (
        lane == _LANE_INT
        or lane == _LANE_DECIMAL
        or lane == _LANE_FLOAT
        or lane == _LANE_DECIMAL64
    )


def _comparable(left: DataType, right: DataType) -> Bool:
    """Whether a comparison of these types computes here: the same type,
    integers of any width (compared exactly in Int64), floats of either
    width (exactly in Float64), or decimals of one scale."""
    var a = _lane(left)
    var b = _lane(right)
    if a == _LANE_NONE or b == _LANE_NONE or a == _LANE_BOOL:
        return False
    if a == _LANE_STRING or b == _LANE_STRING:
        return left == DataType.STRING and right == DataType.STRING
    if left.is_temporal() or right.is_temporal():
        return left == right
    if a != b:
        return False
    if a == _LANE_DECIMAL:
        return left.scale() == right.scale()
    if a == _LANE_FLOAT:
        # A Float32 compared with a Float64 is cast one way or the other
        # by the column kernels; only one type is compared here.
        return left == right
    return True


def _computed_supported(bound: BoundExpr, index: Int) raises -> Bool:
    """Whether arithmetic or cast node `index` computes in lanes: the
    cases `selected_expressions` computed, whose rounding and overflow
    match the column kernels."""
    ref node = bound.expr._nodes[index]
    var dtype = bound.dtypes[index]
    if node.op == CAST:
        if node.left < 0 or len(node.dtypes) == 0 or not node.dtypes[0]:
            return False
        var source = bound.dtypes[node.left]
        if node.dtypes[0].value() != dtype:
            return False
        if not _numeric_lane(_lane(source)) or source.is_temporal():
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
    if left.is_temporal() or right.is_temporal():
        return False
    if dtype.is_decimal():
        return node.op != DIV and left.is_decimal() and right.is_decimal()
    if left != right:
        return False
    if dtype == DataType.FLOAT64:
        return left == DataType.FLOAT64
    if dtype == DataType.INT64:
        return node.op != DIV and left == DataType.INT64
    return False


def _reads_column(bound: BoundExpr, index: Int) -> Bool:
    """Whether node `index`'s subtree reads a column, by every node of the
    subtree (constructors pack their fields' children outside `left` and
    `right`)."""
    for node in subtree(bound.expr, index)._nodes:
        if node.op == COL:
            return True
    return False


def _subtree_columns(
    bound: BoundExpr, index: Int, schema: List[Series], mut out: List[Int]
) raises:
    """The input columns node `index`'s subtree reads, by name."""
    for node in subtree(bound.expr, index)._nodes:
        if node.op != COL:
            continue
        var found = -1
        for c in range(len(schema)):
            if schema[c].name() == node.text:
                found = c
                break
        if found < 0:
            raise Error("Unknown expression column: " + node.text)
        if found not in out:
            out.append(found)


@fieldwise_init
struct _Shape(Copyable, ImplicitlyCopyable, Movable):
    """A node's operator and children, read once."""

    var op: Int
    var left: Int
    var right: Int
    var extra: Int


struct VectorExpression(Copyable, Movable):
    """One bound expression compiled for a step's input columns, with the
    per-node buffers it computes into (DuckDB's ExpressionState and its
    intermediate chunk), reused for every chunk."""

    var bound: BoundExpr
    var root: Int
    var kinds: List[Int]
    var lanes: List[Int]
    # Constants: their lane value, validity, and for a constant read as a
    # vector (a string, or the root) the vector and the column owning it.
    var ints: List[Int64]
    var decimals: List[Int128]
    var floats: List[Float64]
    var constant_valid: List[Bool]
    var constant_vectors: List[Vector]
    var held: List[Series]
    var strings: List[StringT]
    # Fallback nodes: the subtree bound to the columns it reads, and those
    # columns (input column indices), with an empty column of each type.
    var fallback_bound: List[Optional[BoundExpr]]
    var fallback_reads: List[List[Int]]
    var schema: List[Series]
    # Per node, per evaluation: lane buffers, validity bytes, and whether
    # every row is valid (`whole`, the bytes then unread); the unified
    # format of a node read as a vector, and whether it is indexed by
    # chunk row (a column) or by computed row (k).
    var lane_ints: List[List[Int64]]
    var lane_decimals: List[List[Int128]]
    var lane_floats: List[List[Float64]]
    var lane_bools: List[List[UInt8]]
    var valid: List[List[UInt8]]
    var whole: List[Bool]
    var formats: List[UnifiedVectorFormat]
    var by_chunk_row: List[Bool]
    # IN lists, per node: integer values, or string_t over `held`.
    var member_ints: List[List[Int64]]
    # WHEN nodes: the computed rows each branch takes and their chunk rows
    # (then positions, then rows, otherwise positions, otherwise rows).
    var when_rows: List[List[UInt32]]
    var member_strings: List[List[StringT]]
    # The composed selection `_entries` writes.
    var composed: List[UInt32]
    var scratch: DataChunk

    def __init__(out self, expr: Expr, schema: List[Series]) raises:
        """Compile `expr` (already expanded and row-local) for input
        columns typed as `schema` (column i of a chunk is schema[i])."""
        self.bound = bind(expr, schema)
        self.schema = schema.copy()
        var n = len(self.bound.expr._nodes)
        self.root = n - 1
        self.kinds = List[Int](length=n, fill=_KIND_UNUSED)
        self.lanes = List[Int](length=n, fill=_LANE_NONE)
        self.ints = List[Int64](length=n, fill=0)
        self.decimals = List[Int128](length=n, fill=0)
        self.floats = List[Float64](length=n, fill=0)
        self.constant_valid = List[Bool](length=n, fill=True)
        self.constant_vectors = List[Vector](length=n, fill=Vector())
        self.held = List[Series]()
        self.strings = List[StringT](length=n, fill=StringT(0, 0, 0))
        self.fallback_bound = List[Optional[BoundExpr]](length=n, fill=None)
        self.fallback_reads = List[List[Int]](length=n, fill=List[Int]())
        self.lane_ints = List[List[Int64]](length=n, fill=List[Int64]())
        self.lane_decimals = List[List[Int128]](length=n, fill=List[Int128]())
        self.lane_floats = List[List[Float64]](length=n, fill=List[Float64]())
        self.lane_bools = List[List[UInt8]](length=n, fill=List[UInt8]())
        self.valid = List[List[UInt8]](length=n, fill=List[UInt8]())
        self.whole = List[Bool](length=n, fill=True)
        self.formats = List[UnifiedVectorFormat](
            length=n,
            fill=UnifiedVectorFormat(
                SelectionVector(), 0, ValidityMask(), PHYSICAL_INT64
            ),
        )
        self.by_chunk_row = List[Bool](length=n, fill=False)
        self.member_ints = List[List[Int64]](length=n, fill=List[Int64]())
        self.when_rows = List[List[UInt32]](length=n, fill=List[UInt32]())
        self.member_strings = List[List[StringT]](
            length=n, fill=List[StringT]()
        )
        self.composed = List[UInt32](length=2 * STANDARD_VECTOR_SIZE, fill=0)
        self.scratch = DataChunk(n)
        self._plan(self.root)
        self._narrow_decimals()
        # Buffers for every node computed in lanes or read in one.
        for i in range(n):
            var kind = self.kinds[i]
            if kind == _KIND_UNUSED:
                continue
            self.valid[i] = List[UInt8](length=STANDARD_VECTOR_SIZE, fill=1)
            var lane = self.lanes[i]
            if lane == _LANE_INT or lane == _LANE_DECIMAL64:
                self.lane_ints[i] = List[Int64](
                    length=STANDARD_VECTOR_SIZE, fill=self.ints[i]
                )
            elif lane == _LANE_DECIMAL:
                self.lane_decimals[i] = List[Int128](
                    length=STANDARD_VECTOR_SIZE, fill=self.decimals[i]
                )
            elif lane == _LANE_FLOAT:
                self.lane_floats[i] = List[Float64](
                    length=STANDARD_VECTOR_SIZE, fill=self.floats[i]
                )
            elif lane == _LANE_BOOL:
                self.lane_bools[i] = List[UInt8](
                    length=STANDARD_VECTOR_SIZE,
                    fill=UInt8(1) if self.ints[i] != 0 else UInt8(0),
                )
            if kind == _KIND_CONSTANT:
                self.whole[i] = self.constant_valid[i]
                if not self.constant_valid[i]:
                    for k in range(STANDARD_VECTOR_SIZE):
                        self.valid[i][k] = 0

    # --- compilation -----------------------------------------------------

    def _plan(mut self, index: Int) raises:
        """Classify node `index` and, when it computes here, its children."""
        var node = _Shape(
            self.bound.expr._nodes[index].op,
            self.bound.expr._nodes[index].left,
            self.bound.expr._nodes[index].right,
            self.bound.expr._nodes[index].extra,
        )
        var dtype = self.bound.dtypes[index]
        if node.op == COL:
            self.kinds[index] = _KIND_COLUMN
            self.lanes[index] = _lane(dtype)
            if self.lanes[index] == _LANE_NONE:
                self.lanes[index] = _LANE_ANY
            return
        if not _reads_column(self.bound, index):
            if self._constant(index):
                return
            self._fallback(index)
            return
        if self._native(index):
            self.kinds[index] = _KIND_LANE
            if (
                _is_compare(node.op)
                or node.op == AND
                or node.op == OR
                or (
                    node.op == NOT
                    or node.op == IS_NULL
                    or node.op == IS_NOT_NULL
                )
                or node.op == STR_IS_IN
                or node.op == INT_IS_IN
                or _is_string_predicate(node.op)
            ):
                self.lanes[index] = _LANE_BOOL
                if node.op == STR_IS_IN or node.op == INT_IS_IN:
                    self._members(index)
            else:
                self.lanes[index] = _lane(dtype)
            self._plan(node.left)
            if node.right >= 0:
                self._plan(node.right)
            if node.extra >= 0:
                self._plan(node.extra)
            if node.op == WHEN:
                self.when_rows[index] = List[UInt32](
                    length=4 * STANDARD_VECTOR_SIZE, fill=0
                )
            # Children read in a lane they cannot be read in compute
            # elsewhere: a comparison of a string read through the
            # at-rest evaluator is still a string vector.
            if node.op == IS_NULL or node.op == IS_NOT_NULL:
                if self.lanes[node.left] == _LANE_NONE:
                    self.lanes[node.left] = _LANE_ANY
            return
        self._fallback(index)

    def _native(self, index: Int) raises -> Bool:
        ref node = self.bound.expr._nodes[index]
        var dtype = self.bound.dtypes[index]
        if (
            node.op == CAST
            or node.op == ADD
            or node.op == SUB
            or (node.op == MUL or node.op == DIV)
        ):
            return _computed_supported(self.bound, index)
        if _is_compare(node.op):
            if node.left < 0 or node.right < 0 or dtype != DataType.BOOL:
                return False
            return _comparable(
                self.bound.dtypes[node.left], self.bound.dtypes[node.right]
            )
        if node.op == AND or node.op == OR:
            return (
                node.left >= 0
                and node.right >= 0
                and dtype == DataType.BOOL
                and self.bound.dtypes[node.left] == DataType.BOOL
                and self.bound.dtypes[node.right] == DataType.BOOL
            )
        if node.op == NOT:
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == DataType.BOOL
            )
        if node.op == IS_NULL or node.op == IS_NOT_NULL:
            return node.left >= 0 and node.right < 0
        if _is_string_predicate(node.op):
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == DataType.STRING
                and (
                    node.op != STR_CONTAINS
                    or self.bound.expr._nodes[index].integer == 0
                )
            )
        if node.op == WHEN:
            if node.left < 0 or node.right < 0:
                return False
            if self.bound.dtypes[node.left] != DataType.BOOL:
                return False
            var lane = _lane(dtype)
            if lane == _LANE_NONE or lane == _LANE_STRING:
                return False
            if self.bound.dtypes[node.right] != dtype:
                return False
            return node.extra < 0 or self.bound.dtypes[node.extra] == dtype
        if node.op == STR_IS_IN:
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == DataType.STRING
            )
        if node.op == INT_IS_IN:
            if node.left < 0 or node.right >= 0:
                return False
            var input = self.bound.dtypes[node.left]
            return input.is_integer() and input != DataType.UINT64
        return False

    def _constant(mut self, index: Int) raises -> Bool:
        """Evaluate a node that reads no column once, as one value; False
        (leaving it to the at-rest evaluator) when its type has no lane."""
        var dtype = self.bound.dtypes[index]
        var lane = _lane(dtype)
        if lane == _LANE_NONE:
            return False
        var empty = List[Series]()
        var value: Series
        try:
            value = evaluate(
                bind(subtree(self.bound.expr, index), empty), empty, 1
            )
        except:
            return False
        if len(value) != 1:
            return False
        if value.is_chunked():
            value = value.rechunk()
        if value.dtype() != dtype:
            if dtype.is_decimal() or value.dtype().is_decimal():
                return False
            value = value.cast(dtype)
        self.kinds[index] = _KIND_CONSTANT
        self.lanes[index] = lane
        # The value as a vector, for a parent that reads vectors (strings)
        # or for the root of a projection.
        var arena = ColumnArena()
        _ = arena.add(value)
        var vector = flat_over(arena, 0, 0)
        vector.vector_type = CONSTANT_VECTOR
        vector.source = -1
        if lane == _LANE_STRING:
            ref strings = value._data[StringColumn]
            self.held.append(value.copy())
            var valid = strings._valid(0)
            self.constant_valid[index] = valid
            if valid:
                var text = strings._get(0)
                self.strings[index] = StringT.make(
                    Int(text.unsafe_ptr()), text.byte_length()
                )
            vector = Vector.constant(
                PHYSICAL_VARCHAR,
                Int(self.strings.unsafe_ptr()) + 16 * index,
                valid,
            )
            self.constant_vectors[index] = vector
            return True
        if lane == _LANE_BOOL:
            ref bools = value._data[BoolColumn]
            var valid = bools._valid(0)
            self.constant_valid[index] = valid
            self.ints[index] = Int64(1) if valid and bools._get(0) else 0
            self.held.append(
                Series("", Column[UInt8]([UInt8(self.ints[index])]))
            )
            self.constant_vectors[index] = Vector.constant(
                PHYSICAL_BOOL,
                Int(self.held[len(self.held) - 1]._data[Column[UInt8]]._ptr()),
                valid,
            )
            return True
        self.held.append(value.copy())
        ref kept = self.held[len(self.held) - 1]
        var held_vector = Vector()
        comptime for k in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[k]
            if kept._data.isa[Column[Scalar[D]]]():
                ref column = kept._data[Column[Scalar[D]]]
                var valid = column._valid(0)
                self.constant_valid[index] = valid
                var raw = column._get(0) if valid else Scalar[D](0)
                if lane == _LANE_FLOAT:
                    self.floats[index] = raw.cast[DType.float64]()
                elif lane == _LANE_DECIMAL:
                    self.decimals[index] = raw.cast[DType.int128]()
                else:
                    self.ints[index] = raw.cast[DType.int64]()
                held_vector = Vector.constant(
                    physical_of[D](), Int(column._ptr()), valid
                )
        if held_vector.data == 0:
            return False
        self.constant_vectors[index] = held_vector
        return True

    def _fallback(mut self, index: Int) raises:
        """Node `index` computed by the at-rest evaluator: its subtree
        bound to the columns it reads."""
        if getenv("DATAFRAME_TRACE_PATHS"):
            # Which nodes still take the adapter, per query (`trace_path`'s
            # format, with the operator's name).
            print(
                "dataframe-path: vector.fallback."
                + op_name(self.bound.expr._nodes[index].op),
                file=RawFileDescriptor(2),
            )
        var reads = List[Int]()
        _subtree_columns(self.bound, index, self.schema, reads)
        var columns = List[Series](capacity=len(reads))
        for r in reads:
            columns.append(self.schema[r].copy())
        var sub = subtree(self.bound.expr, index)
        # The subtree's column nodes read `columns` in `reads` order.
        self.fallback_bound[index] = bind(sub, columns)
        self.fallback_reads[index] = reads^
        self.kinds[index] = _KIND_FALLBACK
        var lane = _lane(self.bound.dtypes[index])
        self.lanes[index] = lane if lane != _LANE_NONE else _LANE_ANY

    def _narrow_decimals(mut self):
        """Every decimal node in 64-bit lanes when each has at most 18
        digits and is stored in at most 64 bits (selected_expressions'
        `_narrow_decimals`): such values multiply, align and add exactly
        in 128-bit arithmetic, and a checked result fits 64 bits."""
        for i in range(len(self.kinds)):
            if self.kinds[i] == _KIND_UNUSED:
                continue
            if self.lanes[i] != _LANE_DECIMAL:
                continue
            var dtype = self.bound.dtypes[i]
            if dtype.precision() > 18 or dtype.decimal_width() == 128:
                return
            if self.kinds[i] == _KIND_CONSTANT:
                var value = self.decimals[i]
                if value > Int128(Int64.MAX) or value < -Int128(Int64.MAX):
                    return
        for i in range(len(self.kinds)):
            if self.kinds[i] == _KIND_UNUSED:
                continue
            if self.lanes[i] != _LANE_DECIMAL:
                continue
            self.lanes[i] = _LANE_DECIMAL64
            if self.kinds[i] == _KIND_CONSTANT:
                self.ints[i] = Int64(self.decimals[i])

    def root_kind(self) -> Int:
        return self.kinds[self.root]

    def root_column(self) -> Int:
        """The input column a bare column expression reads."""
        return self.bound.sources[self.root]

    def root_dtype(self) -> DataType:
        return self.bound.dtypes[self.root]

    # --- evaluation -------------------------------------------------------

    def evaluate(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """Compute every node for the `count` chunk rows `rows` selects
        (every row when unset), into the node buffers, children first."""
        self.scratch.reset()
        self._evaluate(self.root, input, arena, rows, count)

    def _evaluate(
        mut self,
        index: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        var kind = self.kinds[index]
        if kind == _KIND_CONSTANT:
            if self.lanes[index] == _LANE_STRING:
                self.formats[index] = UnifiedVectorFormat(
                    zero_selection(),
                    self.constant_vectors[index].data,
                    self.constant_vectors[index].validity,
                    PHYSICAL_VARCHAR,
                )
                self.by_chunk_row[index] = False
            return
        if kind == _KIND_COLUMN:
            var column = self.bound.sources[index]
            if self.lanes[index] == _LANE_ANY:
                # Only validity is read: no decode.
                var v = input.data[column]
                var sel = SelectionVector()
                if v.vector_type == DICTIONARY_VECTOR:
                    sel = v.sel
                elif v.vector_type == CONSTANT_VECTOR:
                    sel = zero_selection()
                self.formats[index] = UnifiedVectorFormat(
                    sel, 0, v.validity, v.physical
                )
            else:
                self.formats[index] = input.unified(column, arena)
            self.by_chunk_row[index] = True
            self._load(index, rows, count)
            return
        if kind == _KIND_FALLBACK:
            self._run_fallback(index, input, arena, rows, count)
            self._load(index, rows, count)
            return
        var node = _Shape(
            self.bound.expr._nodes[index].op,
            self.bound.expr._nodes[index].left,
            self.bound.expr._nodes[index].right,
            self.bound.expr._nodes[index].extra,
        )
        var left = node.left
        var right = node.right
        if node.op == WHEN:
            self._when(index, node, input, arena, rows, count)
            return
        self._evaluate(left, input, arena, rows, count)
        if right >= 0:
            self._evaluate(right, input, arena, rows, count)
        var op = node.op
        if _is_compare(op):
            self._compare(index, left, right, rows, count)
        elif op == AND or op == OR:
            self._logic(index, left, right, count)
        elif op == NOT:
            self._not(index, left, count)
        elif op == IS_NULL or op == IS_NOT_NULL:
            self._null_test(index, left, rows, count, op == IS_NULL)
        elif op == STR_IS_IN or op == INT_IS_IN:
            self._is_in(index, left, rows, count)
        elif _is_string_predicate(op):
            self._string_predicate(index, left, rows, count)
        else:
            self._compute(index, count)

    def _run_fallback(
        mut self,
        index: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """The at-rest evaluator over the rows: each column the subtree
        reads gathered for them, the result held in the arena and read as
        a flat vector indexed by computed row."""
        ref reads = self.fallback_reads[index]
        var columns = List[Series](capacity=len(reads))
        for r in reads:
            columns.append(
                gather_series(
                    input,
                    r,
                    arena,
                    rows,
                    count,
                    self.schema[r].name(),
                    self.schema[r].dtype(),
                )
            )
        var result = evaluate(
            self.fallback_bound[index].value(), columns, count
        )
        if len(result) != count:
            if len(result) == 1:
                result = result._broadcast(count)
            else:
                raise Error("vector expression: fallback row count")
        var dtype = self.bound.dtypes[index]
        if result.dtype() != dtype and not dtype.is_categorical():
            result = result.with_dtype(dtype)
        var handle = arena.add(result)
        var vector = flat_over(arena, handle, 0)
        self.scratch.count = count
        self.scratch.data[index] = vector
        if self.lanes[index] == _LANE_ANY:
            self.formats[index] = UnifiedVectorFormat(
                SelectionVector(), 0, vector.validity, vector.physical
            )
        else:
            self.formats[index] = self.scratch.unified(index, arena)
        self.by_chunk_row[index] = False
        self.constant_vectors[index] = vector

    @always_inline
    def _at(self, index: Int, rows: SelectionVector, k: Int) -> Int:
        """The entry of node `index`'s unified format computed row `k`
        reads."""
        ref format = self.formats[index]
        var row = rows.get_index(k) if self.by_chunk_row[index] else k
        return format.sel.get_index(row)

    def _entries(
        mut self, index: Int, rows: SelectionVector, count: Int, slot: Int = 0
    ) -> Int:
        """The entries of node `index`'s unified format the computed rows
        read, as the address of `count` UInt32 indices, or 0 when computed
        row k reads entry k: the format's selection composed with the
        rows' when both are set (SelectionVector::Slice), else whichever
        is."""
        var own = self.formats[index].sel
        var through = self.by_chunk_row[index] and rows.is_set()
        if not through:
            return own.data
        if not own.is_set():
            return rows.data
        var out = self.composed.unsafe_ptr().unsafe_offset(
            slot * STANDARD_VECTOR_SIZE
        )
        var a = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows.data)
        var b = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=own.data)
        for k in range(count):
            out[unsafe_offset=k] = b[unsafe_offset=Int(a[unsafe_offset=k])]
        return Int(out)

    def _load(mut self, index: Int, rows: SelectionVector, count: Int):
        """Node `index`'s values for the computed rows into its lane (and
        validity bytes), from its unified format. Strings stay in their
        vector, read by the comparisons."""
        var lane = self.lanes[index]
        var entries = self._entries(index, rows, count)
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
        var validity = self.formats[index].validity
        # Validity first.
        var flags = self.valid[index].unsafe_ptr()
        if validity.all_valid():
            self.whole[index] = True
        else:
            self.whole[index] = False
            var bits = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=validity.bits
            )
            var offset = validity.offset
            if entries == 0 and offset & 7 == 0:
                # A byte of the bitmap spread to eight flags at a time.
                var lanes = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
                var from_byte = offset >> 3
                var whole_bytes = count >> 3
                for j in range(whole_bytes):
                    var byte = bits[unsafe_offset=from_byte + j]
                    var spread = (SIMD[DType.uint8, 8](byte) & lanes).ne(
                        SIMD[DType.uint8, 8](0)
                    )
                    flags.unsafe_offset(8 * j).unsafe_store(
                        spread.cast[DType.uint8]()
                    )
                for k in range(8 * whole_bytes, count):
                    var b = offset + k
                    flags[unsafe_offset=k] = (
                        bits[unsafe_offset=b >> 3] >> UInt8(b & 7)
                    ) & 1
            elif entries == 0:
                for k in range(count):
                    var b = offset + k
                    flags[unsafe_offset=k] = (
                        bits[unsafe_offset=b >> 3] >> UInt8(b & 7)
                    ) & 1
            else:
                for k in range(count):
                    var b = offset + Int(at[unsafe_offset=k])
                    flags[unsafe_offset=k] = (
                        bits[unsafe_offset=b >> 3] >> UInt8(b & 7)
                    ) & 1
        if lane == _LANE_STRING or lane == _LANE_ANY:
            return
        var physical = self.formats[index].physical
        var data = self.formats[index].data
        if lane == _LANE_BOOL:
            _gather[DType.uint8, DType.uint8](
                data, entries, count, Int(self.lane_bools[index].unsafe_ptr())
            )
            return
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if physical == physical_of[D]():
                if lane == _LANE_FLOAT:
                    _gather[D, DType.float64](
                        data,
                        entries,
                        count,
                        Int(self.lane_floats[index].unsafe_ptr()),
                    )
                elif lane == _LANE_DECIMAL:
                    _gather[D, DType.int128](
                        data,
                        entries,
                        count,
                        Int(self.lane_decimals[index].unsafe_ptr()),
                    )
                else:
                    _gather[D, DType.int64](
                        data,
                        entries,
                        count,
                        Int(self.lane_ints[index].unsafe_ptr()),
                    )
                return

    def _combine(mut self, left: Int, right: Int, index: Int, count: Int):
        """Node `index` valid where both children are."""
        var left_whole = self.whole[left]
        var right_whole = self.whole[right]
        self.whole[index] = left_whole and right_whole
        if left_whole and right_whole:
            return
        var o = self.valid[index].unsafe_ptr()
        if left_whole or right_whole:
            var one = self.valid[right if left_whole else left].unsafe_ptr()
            for k in range(count):
                o[unsafe_offset=k] = one[unsafe_offset=k]
            return
        _and_bytes(
            Int(o),
            Int(self.valid[left].unsafe_ptr()),
            Int(self.valid[right].unsafe_ptr()),
            count,
        )

    def _compare(
        mut self,
        index: Int,
        left: Int,
        right: Int,
        rows: SelectionVector,
        count: Int,
    ):
        """A comparison of two numbers in one lane, or of two strings;
        null where either side is (as the column kernels)."""
        self._combine(left, right, index, count)
        var op = self.bound.expr._nodes[index].op
        var out = self.lane_bools[index].unsafe_ptr()
        var lane = self.lanes[left]
        if lane == _LANE_STRING:
            var la = self._entries(left, rows, count, 0)
            var ra = self._entries(right, rows, count, 1)
            var target = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(self.lane_bools[index].unsafe_ptr())
            )
            var lp = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=la)
            var rp = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=ra)
            var ld = self.formats[left].data
            var rd = self.formats[right].data
            for k in range(count):
                var a = ld + 16 * (k if la == 0 else Int(lp[unsafe_offset=k]))
                var b = rd + 16 * (k if ra == 0 else Int(rp[unsafe_offset=k]))
                var result: Bool
                if op == EQ:
                    result = string_equals(a, b)
                elif op == NE:
                    result = not string_equals(a, b)
                else:
                    var c = string_compare(a, b)
                    if op == LT:
                        result = c < 0
                    elif op == LE:
                        result = c <= 0
                    elif op == GT:
                        result = c > 0
                    else:
                        result = c >= 0
                target[unsafe_offset=k] = UInt8(1) if result else UInt8(0)
            return
        if lane == _LANE_FLOAT:
            _compare_lanes(
                op,
                self.lane_floats[left],
                self.lane_floats[right],
                Int(out),
                count,
            )
        elif lane == _LANE_DECIMAL:
            _compare_lanes(
                op,
                self.lane_decimals[left],
                self.lane_decimals[right],
                Int(out),
                count,
            )
        elif lane == _LANE_DECIMAL64:
            # 64-bit decimals of different scales align exactly in 128
            # bits.
            var ls = self.bound.dtypes[left].scale()
            var rs = self.bound.dtypes[right].scale()
            if ls == rs:
                _compare_lanes(
                    op,
                    self.lane_ints[left],
                    self.lane_ints[right],
                    Int(out),
                    count,
                )
            else:
                var common = max(ls, rs)
                var lf = pow10(common - ls)
                var rf = pow10(common - rs)
                var a = List[Int128](unsafe_uninit_length=count)
                var b = List[Int128](unsafe_uninit_length=count)
                for k in range(count):
                    a[k] = Int128(self.lane_ints[left][k]) * lf
                    b[k] = Int128(self.lane_ints[right][k]) * rf
                _compare_lanes(op, a, b, Int(out), count)
        else:
            _compare_lanes(
                op, self.lane_ints[left], self.lane_ints[right], Int(out), count
            )

    def _logic(mut self, index: Int, left: Int, right: Int, count: Int):
        """Kleene AND / OR: a dominant operand decides even when the other
        is null (the column kernels' `_logical`)."""
        var is_and = self.bound.expr._nodes[index].op == AND
        var x = self.lane_bools[left].unsafe_ptr()
        var y = self.lane_bools[right].unsafe_ptr()
        var out = self.lane_bools[index].unsafe_ptr()
        var lw = self.whole[left]
        var rw = self.whole[right]
        if lw and rw:
            self.whole[index] = True
            if is_and:
                _and_bytes(Int(out), Int(x), Int(y), count)
            else:
                for k in range(count):
                    out[unsafe_offset=k] = (
                        x[unsafe_offset=k] | y[unsafe_offset=k]
                    )
            return
        # Branch-free bytes (0 or 1): a side known true is valid and set,
        # one known false is valid and clear.
        var pv = self.valid[left].unsafe_ptr()
        var qv = self.valid[right].unsafe_ptr()
        var o = self.valid[index].unsafe_ptr()
        comptime W = _BYTE_LANES
        var lw8 = SIMD[DType.uint8, W](1 if lw else 0)
        var rw8 = SIMD[DType.uint8, W](1 if rw else 0)
        var one = SIMD[DType.uint8, W](1)
        var all_valid = SIMD[DType.uint8, W](1)
        var k = 0
        while k + W <= count:
            var p = pv.unsafe_load[width=W](k) | lw8
            var q = qv.unsafe_load[width=W](k) | rw8
            var xa = x.unsafe_load[width=W](k) & one
            var yb = y.unsafe_load[width=W](k) & one
            var a = p & xa
            var b = q & yb
            var value: SIMD[DType.uint8, W]
            var valid: SIMD[DType.uint8, W]
            if is_and:
                value = a & b
                valid = (p & q) | (p & (xa ^ one)) | (q & (yb ^ one))
            else:
                value = a | b
                valid = (p & q) | a | b
            out.unsafe_offset(k).unsafe_store(value)
            o.unsafe_offset(k).unsafe_store(valid)
            all_valid &= valid
            k += W
        var tail_valid = UInt8(1)
        while k < count:
            var p = pv[unsafe_offset=k] | (UInt8(1) if lw else UInt8(0))
            var q = qv[unsafe_offset=k] | (UInt8(1) if rw else UInt8(0))
            var xa = x[unsafe_offset=k] & 1
            var yb = y[unsafe_offset=k] & 1
            var a = p & xa
            var b = q & yb
            var value: UInt8
            var valid: UInt8
            if is_and:
                value = a & b
                valid = (p & q) | (p & (xa ^ 1)) | (q & (yb ^ 1))
            else:
                value = a | b
                valid = (p & q) | a | b
            out[unsafe_offset=k] = value
            o[unsafe_offset=k] = valid
            tail_valid &= valid
            k += 1
        self.whole[index] = all_valid.reduce_and() == 1 and tail_valid == 1

    def _when(
        mut self,
        index: Int,
        node: _Shape,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """CASE WHEN (DuckDB's ExpressionExecutor for BoundCaseExpression):
        the predicate over the rows, then each branch over only the rows
        that take it, scattered into place; a null predicate takes the
        otherwise branch, and no otherwise is null. A branch never sees a
        row it does not take, so it cannot raise for one."""
        self._evaluate(node.left, input, arena, rows, count)
        var buffers = Int(self.when_rows[index].unsafe_ptr())
        var then_at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=buffers)
        var then_rows = then_at.unsafe_offset(STANDARD_VECTOR_SIZE)
        var other_at = then_at.unsafe_offset(2 * STANDARD_VECTOR_SIZE)
        var other_rows = then_at.unsafe_offset(3 * STANDARD_VECTOR_SIZE)
        var values = self.lane_bools[node.left].unsafe_ptr()
        var flags = self.valid[node.left].unsafe_ptr()
        var whole = self.whole[node.left]
        var taken = 0
        var others = 0
        for k in range(count):
            var keep = values[unsafe_offset=k]
            if not whole:
                keep &= flags[unsafe_offset=k]
            var row = UInt32(rows.get_index(k))
            then_at[unsafe_offset=taken] = UInt32(k)
            then_rows[unsafe_offset=taken] = row
            other_at[unsafe_offset=others] = UInt32(k)
            other_rows[unsafe_offset=others] = row
            taken += Int(keep)
            others += 1 - Int(keep)
        var any_null = False
        if taken > 0:
            self._evaluate(
                node.right, input, arena, SelectionVector(Int(then_rows)), taken
            )
            any_null |= self._scatter(index, node.right, Int(then_at), taken)
        if others > 0:
            if node.extra >= 0:
                self._evaluate(
                    node.extra,
                    input,
                    arena,
                    SelectionVector(Int(other_rows)),
                    others,
                )
                any_null |= self._scatter(
                    index, node.extra, Int(other_at), others
                )
            else:
                var o = self.valid[index].unsafe_ptr()
                for j in range(others):
                    o[unsafe_offset=Int(other_at[unsafe_offset=j])] = 0
                any_null = True
        self.whole[index] = not any_null

    def _scatter(
        mut self, index: Int, child: Int, positions: Int, count: Int
    ) -> Bool:
        """The child's `count` values (computed over the rows a branch
        takes) into node `index` at `positions`, with validity; whether
        any is null."""
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=positions)
        var lane = self.lanes[index]
        if lane == _LANE_BOOL:
            _scatter_values[DType.uint8](
                Int(self.lane_bools[child].unsafe_ptr()),
                Int(self.lane_bools[index].unsafe_ptr()),
                positions,
                count,
            )
        elif lane == _LANE_FLOAT:
            _scatter_values[DType.float64](
                Int(self.lane_floats[child].unsafe_ptr()),
                Int(self.lane_floats[index].unsafe_ptr()),
                positions,
                count,
            )
        elif lane == _LANE_DECIMAL:
            _scatter_values[DType.int128](
                Int(self.lane_decimals[child].unsafe_ptr()),
                Int(self.lane_decimals[index].unsafe_ptr()),
                positions,
                count,
            )
        else:
            _scatter_values[DType.int64](
                Int(self.lane_ints[child].unsafe_ptr()),
                Int(self.lane_ints[index].unsafe_ptr()),
                positions,
                count,
            )
        var o = self.valid[index].unsafe_ptr()
        if self.whole[child]:
            for j in range(count):
                o[unsafe_offset=Int(at[unsafe_offset=j])] = 1
            return False
        var f = self.valid[child].unsafe_ptr()
        var any_null = False
        for j in range(count):
            var v = f[unsafe_offset=j]
            o[unsafe_offset=Int(at[unsafe_offset=j])] = v
            any_null = any_null or v == 0
        return any_null

    def _string_predicate(
        mut self, index: Int, left: Int, rows: SelectionVector, count: Int
    ):
        """starts_with / ends_with / contains of a literal over string_t
        (the column kernels' byte predicates); null stays null."""
        var op = self.bound.expr._nodes[index].op
        var needle = self.bound.expr._nodes[index].text
        var target = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.lane_bools[index].unsafe_ptr())
        )
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            var a = self.valid[left].unsafe_ptr()
            var o = self.valid[index].unsafe_ptr()
            for k in range(count):
                o[unsafe_offset=k] = a[unsafe_offset=k]
        var entries = self._entries(left, rows, count, 0)
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
        var data = self.formats[left].data
        var n = needle.byte_length()
        var probe = Int(needle.unsafe_ptr())
        for k in range(count):
            var entry = data + 16 * (
                k if entries == 0 else Int(at[unsafe_offset=k])
            )
            var length = Int(
                Pointer[StringT, MutAnyOrigin](
                    unsafe_from_address=entry
                )[].length
            )
            var bytes = string_data(entry)
            var found = False
            if n <= length:
                if op == STR_STARTS_WITH:
                    found = (
                        n == 0
                        or external_call["memcmp", Int32](bytes, probe, n) == 0
                    )
                elif op == STR_ENDS_WITH:
                    found = (
                        n == 0
                        or external_call["memcmp", Int32](
                            bytes + length - n, probe, n
                        )
                        == 0
                    )
                else:
                    found = n == 0 or (
                        external_call["memmem", Int](bytes, length, probe, n)
                        != 0
                    )
            target[unsafe_offset=k] = UInt8(1) if found else UInt8(0)
        _ = needle^

    def _members(mut self, index: Int) raises:
        """An IN list's values, parsed once: integers, or string_t over a
        column this expression holds."""
        var op = self.bound.expr._nodes[index].op
        var text = self.bound.expr._nodes[index].text
        if op == INT_IS_IN:
            var values = List[Int64]()
            for part in text.split(","):
                values.append(Int64(atol(part)))
            self.member_ints[index] = values^
            return
        var values = decode_values(text)
        var held = Series("", StringColumn(values))
        ref strings = held._data[StringColumn]
        var entries = List[StringT](capacity=len(values))
        for i in range(len(values)):
            var text = strings._get(i)
            entries.append(
                StringT.make(Int(text.unsafe_ptr()), text.byte_length())
            )
        self.held.append(held^)
        self.member_strings[index] = entries^

    def _is_in(
        mut self, index: Int, left: Int, rows: SelectionVector, count: Int
    ):
        """Whether each value is one of the IN list's; null stays null (the
        column kernels' `is_in_literals` and `integer_is_in`)."""
        var target = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.lane_bools[index].unsafe_ptr())
        )
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            var a = self.valid[left].unsafe_ptr()
            var o = self.valid[index].unsafe_ptr()
            for k in range(count):
                o[unsafe_offset=k] = a[unsafe_offset=k]
        if self.bound.expr._nodes[index].op == INT_IS_IN:
            var x = self.lane_ints[left].unsafe_ptr()
            var wanted = self.member_ints[index].unsafe_ptr()
            var n = len(self.member_ints[index])
            for k in range(count):
                var value = x[unsafe_offset=k]
                var found = UInt8(0)
                for j in range(n):
                    found |= UInt8(value == wanted[unsafe_offset=j])
                target[unsafe_offset=k] = found
            return
        var entries = self._entries(left, rows, count, 0)
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
        var data = self.formats[left].data
        var members = Int(self.member_strings[index].unsafe_ptr())
        var n = len(self.member_strings[index])
        for k in range(count):
            var found = UInt8(0)
            var entry = data + 16 * (
                k if entries == 0 else Int(at[unsafe_offset=k])
            )
            for j in range(n):
                if string_equals(entry, members + 16 * j):
                    found = 1
                    break
            target[unsafe_offset=k] = found

    def _not(mut self, index: Int, left: Int, count: Int):
        var x = self.lane_bools[left].unsafe_ptr()
        var out = self.lane_bools[index].unsafe_ptr()
        for k in range(count):
            out[unsafe_offset=k] = UInt8(1) - x[unsafe_offset=k]
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            var a = self.valid[left].unsafe_ptr()
            var o = self.valid[index].unsafe_ptr()
            for k in range(count):
                o[unsafe_offset=k] = a[unsafe_offset=k]

    def _null_test(
        mut self,
        index: Int,
        left: Int,
        rows: SelectionVector,
        count: Int,
        is_null: Bool,
    ):
        var out = self.lane_bools[index].unsafe_ptr()
        self.whole[index] = True
        if self.whole[left]:
            for k in range(count):
                out[unsafe_offset=k] = UInt8(1) if not is_null else UInt8(0)
            return
        var a = self.valid[left].unsafe_ptr()
        for k in range(count):
            var valid = a[unsafe_offset=k] != 0
            out[unsafe_offset=k] = UInt8(1) if valid != is_null else UInt8(0)

    def _compute(mut self, index: Int, count: Int) raises:
        """Arithmetic or a cast over the computed rows, in lanes."""
        var node = _Shape(
            self.bound.expr._nodes[index].op,
            self.bound.expr._nodes[index].left,
            self.bound.expr._nodes[index].right,
            self.bound.expr._nodes[index].extra,
        )
        var dtype = self.bound.dtypes[index]
        var left = node.left
        var lane = self.lanes[index]
        var source_lane = self.lanes[left]
        if node.op == CAST:
            var source = self.bound.dtypes[left]
            self.whole[index] = self.whole[left]
            if not self.whole[left]:
                var from_flags = self.valid[left].unsafe_ptr()
                var flags = self.valid[index].unsafe_ptr()
                for k in range(count):
                    flags[unsafe_offset=k] = from_flags[unsafe_offset=k]
            if lane == _LANE_DECIMAL:
                var out = self.lane_decimals[index].unsafe_ptr()
                var limit = precision_limit(dtype.precision())
                var overflow = False
                if source_lane == _LANE_INT:
                    var factor = pow10(dtype.scale())
                    var x = self.lane_ints[left].unsafe_ptr()
                    for k in range(count):
                        var value = Int128(x[unsafe_offset=k]) * factor
                        overflow |= value <= -limit or value >= limit
                        out[unsafe_offset=k] = value
                elif source_lane == _LANE_DECIMAL64:
                    var x = self.lane_ints[left].unsafe_ptr()
                    var shift = dtype.scale() - source.scale()
                    var factor = pow10(abs(shift))
                    for k in range(count):
                        var value = Int128(x[unsafe_offset=k])
                        if shift > 0:
                            value *= factor
                        elif shift < 0:
                            value = divide_half_even(value, factor)
                        overflow |= value <= -limit or value >= limit
                        out[unsafe_offset=k] = value
                else:
                    var x = self.lane_decimals[left].unsafe_ptr()
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
                    self._raise_if_kept(index, count)
                return
            if lane == _LANE_DECIMAL64:
                var out = self.lane_ints[index].unsafe_ptr()
                var x = self.lane_ints[left].unsafe_ptr()
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
                    self._raise_if_kept(index, count)
                return
            if lane == _LANE_FLOAT:
                var out = self.lane_floats[index].unsafe_ptr()
                if source_lane == _LANE_FLOAT:
                    var x = self.lane_floats[left].unsafe_ptr()
                    for k in range(count):
                        out[unsafe_offset=k] = x[unsafe_offset=k]
                elif source_lane == _LANE_INT:
                    var x = self.lane_ints[left].unsafe_ptr()
                    for k in range(count):
                        out[unsafe_offset=k] = Float64(x[unsafe_offset=k])
                elif source_lane == _LANE_DECIMAL64:
                    var divisor = Float64(pow10(source.scale()))
                    var x = self.lane_ints[left].unsafe_ptr()
                    for k in range(count):
                        out[unsafe_offset=k] = (
                            Float64(x[unsafe_offset=k]) / divisor
                        )
                else:
                    # value / 10^scale, as `_decimal_to_float` converts.
                    var divisor = Float64(pow10(source.scale()))
                    var x = self.lane_decimals[left].unsafe_ptr()
                    for k in range(count):
                        out[unsafe_offset=k] = (
                            Float64(x[unsafe_offset=k]) / divisor
                        )
                return
            var out = self.lane_ints[index].unsafe_ptr()
            var x = self.lane_ints[left].unsafe_ptr()
            for k in range(count):
                out[unsafe_offset=k] = x[unsafe_offset=k]
            return
        var right = node.right
        self._combine(left, right, index, count)
        var whole = self.whole[index]
        var flags = self.valid[index].unsafe_ptr()
        if lane == _LANE_FLOAT:
            var x = self.lane_floats[left].unsafe_ptr()
            var y = self.lane_floats[right].unsafe_ptr()
            var out = self.lane_floats[index].unsafe_ptr()
            for k in range(count):
                var a = x[unsafe_offset=k]
                var b = y[unsafe_offset=k]
                var value: Float64
                if node.op == ADD:
                    value = a + b
                elif node.op == SUB:
                    value = a - b
                elif node.op == MUL:
                    value = a * b
                else:
                    value = a / b
                out[unsafe_offset=k] = (
                    value if whole or flags[unsafe_offset=k] != 0 else 0
                )
            return
        if lane == _LANE_INT:
            var x = self.lane_ints[left].unsafe_ptr()
            var y = self.lane_ints[right].unsafe_ptr()
            var out = self.lane_ints[index].unsafe_ptr()
            for k in range(count):
                if not whole and flags[unsafe_offset=k] == 0:
                    out[unsafe_offset=k] = 0
                    continue
                var a = x[unsafe_offset=k]
                var b = y[unsafe_offset=k]
                if node.op == ADD:
                    out[unsafe_offset=k] = checked_add(a, b)
                elif node.op == SUB:
                    out[unsafe_offset=k] = _checked_sub(a, b)
                else:
                    out[unsafe_offset=k] = _checked_mul(a, b)
            return
        # Decimals, as `_decimal_dense` computes them: products at the
        # operands' own scales, rounded to the result's; sums and
        # differences at the larger scale.
        var left_type = self.bound.dtypes[left]
        var right_type = self.bound.dtypes[right]
        var limit = precision_limit(dtype.precision())
        var overflow = False
        if lane == _LANE_DECIMAL64:
            var x = self.lane_ints[left].unsafe_ptr()
            var y = self.lane_ints[right].unsafe_ptr()
            var out = self.lane_ints[index].unsafe_ptr()
            if node.op == MUL:
                var shift = (
                    left_type.scale() + right_type.scale() - dtype.scale()
                )
                var rescale = pow10(max(shift, 0))
                for k in range(count):
                    var value = Int128(x[unsafe_offset=k]) * Int128(
                        y[unsafe_offset=k]
                    )
                    if shift > 0:
                        value = divide_half_even(value, rescale)
                    if not whole and flags[unsafe_offset=k] == 0:
                        value = 0
                    overflow |= value <= -limit or value >= limit
                    out[unsafe_offset=k] = Int64(value)
            else:
                var common = max(left_type.scale(), right_type.scale())
                var lf = pow10(common - left_type.scale())
                var rf = pow10(common - right_type.scale())
                for k in range(count):
                    var a = Int128(x[unsafe_offset=k]) * lf
                    var b = Int128(y[unsafe_offset=k]) * rf
                    var value = a + b if node.op == ADD else a - b
                    if not whole and flags[unsafe_offset=k] == 0:
                        value = 0
                    overflow |= value <= -limit or value >= limit
                    out[unsafe_offset=k] = Int64(value)
            if overflow:
                self._raise_if_kept(index, count)
            return
        var x = self.lane_decimals[left].unsafe_ptr()
        var y = self.lane_decimals[right].unsafe_ptr()
        var out = self.lane_decimals[index].unsafe_ptr()
        if node.op == MUL:
            var shift = left_type.scale() + right_type.scale() - dtype.scale()
            var rescale = pow10(max(shift, 0))
            for k in range(count):
                var value = _decimal_mul(x[unsafe_offset=k], y[unsafe_offset=k])
                if shift > 0:
                    value = divide_half_even(value, rescale)
                if not whole and flags[unsafe_offset=k] == 0:
                    value = 0
                overflow |= value <= -limit or value >= limit
                out[unsafe_offset=k] = value
        else:
            var common = max(left_type.scale(), right_type.scale())
            var left_factor = pow10(common - left_type.scale())
            var right_factor = pow10(common - right_type.scale())
            for k in range(count):
                var a = x[unsafe_offset=k]
                var b = y[unsafe_offset=k]
                if left_factor != 1:
                    a = _decimal_mul(a, left_factor)
                if right_factor != 1:
                    b = _decimal_mul(b, right_factor)
                var value = a + b if node.op == ADD else a - b
                if not whole and flags[unsafe_offset=k] == 0:
                    value = 0
                overflow |= value <= -limit or value >= limit
                out[unsafe_offset=k] = value
        if overflow:
            self._raise_if_kept(index, count)

    def _raise_if_kept(self, index: Int, count: Int) raises:
        """A decimal result past its precision: the column kernels' error.
        Every computed row is a kept row, so it is raised."""
        raise Error("value exceeds decimal precision")

    # --- results ----------------------------------------------------------

    def select(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
    ) raises -> Int:
        """ExpressionExecutor::Select for a Boolean expression: the chunk
        rows among `rows` it is true (and not null) for, written to
        `target` in order; how many."""
        self.evaluate(input, arena, rows, count)
        var root = self.root
        var kept = 0
        if self.kinds[root] == _KIND_CONSTANT:
            if self.constant_valid[root] and self.ints[root] != 0:
                for k in range(count):
                    target.set_index(k, rows.get_index(k))
                return count
            return 0
        var values = self.lane_bools[root].unsafe_ptr()
        var flags = self.valid[root].unsafe_ptr()
        var whole = self.whole[root]
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=target.data)
        # Branch-free: every row is written, and the next slot advances by
        # whether it is kept (DuckDB's SelectGenericLoop with a true_sel).
        if rows.is_set():
            var from_rows = Pointer[UInt32, MutAnyOrigin](
                unsafe_from_address=rows.data
            )
            for k in range(count):
                var keep = values[unsafe_offset=k]
                if not whole:
                    keep &= flags[unsafe_offset=k]
                out[unsafe_offset=kept] = from_rows[unsafe_offset=k]
                kept += Int(keep)
            return kept
        for k in range(count):
            var keep = values[unsafe_offset=k]
            if not whole:
                keep &= flags[unsafe_offset=k]
            out[unsafe_offset=kept] = UInt32(k)
            kept += Int(keep)
        return kept

    def write(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        mut output: DataChunk,
        column: Int,
    ) raises:
        """The expression over every row of `input`, as column `column` of
        `output`: a column it reads as it is (Vector::Reference), a
        constant as a constant vector, anything computed into `output`'s
        cache for that column."""
        var root = self.root
        var kind = self.kinds[root]
        if kind == _KIND_COLUMN:
            output.reference(column, input.data[self.bound.sources[root]])
            return
        if kind == _KIND_CONSTANT:
            output.reference(column, self.constant_vectors[root])
            return
        var count = input.count
        self.evaluate(input, arena, SelectionVector(), count)
        if kind == _KIND_FALLBACK:
            output.reference(column, self.constant_vectors[root])
            return
        var lane = self.lanes[root]
        var dtype = self.bound.dtypes[root]
        var physical = _physical(dtype)
        var vector = output.writable(column, physical)
        if lane == _LANE_BOOL:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=vector.data
                ),
                src=self.lane_bools[root].unsafe_ptr(),
                count=count,
            )
        elif lane == _LANE_FLOAT:
            _store[DType.float64](
                physical, self.lane_floats[root], vector.data, count
            )
        elif lane == _LANE_DECIMAL:
            _store[DType.int128](
                physical, self.lane_decimals[root], vector.data, count
            )
        else:
            _store[DType.int64](
                physical, self.lane_ints[root], vector.data, count
            )
        if not self.whole[root]:
            var bits = output.validity_bits(column)
            var flags = self.valid[root].unsafe_ptr()
            for k in range(count):
                if flags[unsafe_offset=k] == 0:
                    set_invalid(bits, k)


def _is_string_predicate(op: Int) -> Bool:
    return op == STR_STARTS_WITH or op == STR_ENDS_WITH or op == STR_CONTAINS


def _scatter_values[
    T: DType
](source: Int, target: Int, positions: Int, count: Int):
    var x = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=source)
    var o = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=positions)
    for j in range(count):
        o[unsafe_offset=Int(at[unsafe_offset=j])] = x[unsafe_offset=j]


def _gather[
    D: DType, T: DType
](data: Int, entries: Int, count: Int, target: Int):
    """`count` values of type D at `data`, at the UInt32 indices at
    `entries` (or rows 0, 1, 2, ... when 0), cast to T into `target`."""
    var source = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=data)
    var out = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    if entries == 0:
        for k in range(count):
            out[unsafe_offset=k] = source[unsafe_offset=k].cast[T]()
        return
    var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
    for k in range(count):
        out[unsafe_offset=k] = source[
            unsafe_offset=Int(at[unsafe_offset=k])
        ].cast[T]()


comptime _LANE_STORAGE = [
    DType.int64,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.float32,
    DType.int128,
]


def _physical(dtype: DataType) -> Int:
    """The physical type a computed value of `dtype` is stored as."""
    if dtype == DataType.BOOL:
        return PHYSICAL_BOOL
    var storage = dtype.physical().storage()
    if not storage:
        return PHYSICAL_INT64
    comptime for k in range(len(_LANE_STORAGE)):
        comptime D = _LANE_STORAGE[k]
        if storage.value() == D:
            return physical_of[D]()
    return PHYSICAL_INT64


def _store[
    T: DType
](physical: Int, values: List[Scalar[T]], address: Int, count: Int):
    """Lane values stored as `physical`."""
    var source = values.unsafe_ptr()
    comptime for k in range(len(_LANE_STORAGE)):
        comptime D = _LANE_STORAGE[k]
        if physical == physical_of[D]():
            var out = Pointer[Scalar[D], MutAnyOrigin](
                unsafe_from_address=address
            )
            for i in range(count):
                out[unsafe_offset=i] = source[unsafe_offset=i].cast[D]()
            return


def _compare_lanes[
    T: DType
](op: Int, a: List[Scalar[T]], b: List[Scalar[T]], address: Int, count: Int):
    """`count` comparisons of two lanes into bytes, a SIMD vector at a
    time (DuckDB's BinaryExecutor over flat data)."""
    if op == EQ:
        _compare_simd[T, EQ](a, b, address, count)
    elif op == NE:
        _compare_simd[T, NE](a, b, address, count)
    elif op == LT:
        _compare_simd[T, LT](a, b, address, count)
    elif op == LE:
        _compare_simd[T, LE](a, b, address, count)
    elif op == GT:
        _compare_simd[T, GT](a, b, address, count)
    else:
        _compare_simd[T, GE](a, b, address, count)


def _compare_simd[
    T: DType, op: Int
](a: List[Scalar[T]], b: List[Scalar[T]], address: Int, count: Int):
    comptime W = 4 if T == DType.int128 else 8
    var target = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var x = a.unsafe_ptr()
    var y = b.unsafe_ptr()
    var k = 0
    while k + W <= count:
        var u = x.unsafe_load[width=W](k)
        var v = y.unsafe_load[width=W](k)
        var m: SIMD[DType.bool, W]
        comptime if op == EQ:
            m = u.eq(v)
        elif op == NE:
            # An ordered compare is false for NaN; NaN != NaN is true.
            m = ~u.eq(v)
        elif op == LT:
            m = u.lt(v)
        elif op == LE:
            m = u.le(v)
        elif op == GT:
            m = u.gt(v)
        else:
            m = u.ge(v)
        target.unsafe_offset(k).unsafe_store(m.cast[DType.uint8]())
        k += W
    while k < count:
        var u = x[unsafe_offset=k]
        var v = y[unsafe_offset=k]
        var r: Bool
        comptime if op == EQ:
            r = u == v
        elif op == NE:
            r = u != v
        elif op == LT:
            r = u < v
        elif op == LE:
            r = u <= v
        elif op == GT:
            r = u > v
        else:
            r = u >= v
        target[unsafe_offset=k] = UInt8(r)
        k += 1


comptime _BYTE_LANES = 32


def _and_bytes(target: Int, a: Int, b: Int, count: Int):
    """target[k] = a[k] & b[k], 32 bytes at a time."""
    var o = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target)
    var x = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a)
    var y = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=b)
    var k = 0
    while k + _BYTE_LANES <= count:
        o.unsafe_offset(k).unsafe_store(
            x.unsafe_load[width=_BYTE_LANES](k)
            & y.unsafe_load[width=_BYTE_LANES](k)
        )
        k += _BYTE_LANES
    while k < count:
        o[unsafe_offset=k] = x[unsafe_offset=k] & y[unsafe_offset=k]
        k += 1
