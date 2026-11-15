"""Evaluate bound expressions over execution chunks.

Inputs use a common selection/data/validity view, allowing flat, constant
and dictionary vectors to share kernels. Reuse per-node output buffers
between chunks. Filters write row selections rather than copying values;
conditional branches evaluate only the rows that observe them, preserving
null behavior and checked-expression errors.
"""
from std.bit import count_trailing_zeros, pop_count
from std.math import ceil, floor
from std.memory import (
    Pointer,
    unsafe_memcpy,
    unsafe_memcmp,
    unsafe_memset_zero,
)
from unicode_mojo.transform import codepoint_slice_bounds

from std.sys import stderr
from std.os import getenv
from dataframe.binding import BoundExpr, bind, op_name
from dataframe.decimal import divide_half_even, pow10, precision_limit
from dataframe.dtype import DataType
from dataframe.execution import evaluate
from dataframe.expr import (
    ABS,
    ADD,
    AND,
    CAST,
    CEIL,
    COL,
    DIV,
    DT_DAY,
    DT_MONTH,
    DT_YEAR,
    EQ,
    FILL_NULL,
    FLOOR,
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
    ROUND,
    STR_CONCAT,
    STR_CONTAINS,
    STR_ENDS_WITH,
    STR_IS_IN,
    STR_LIKE,
    STR_SLICE,
    STR_STARTS_WITH,
    SUB,
    WHEN,
    Expr,
    node_children,
    subtree,
)
from dataframe.kernels.binary import (
    _decimal_mul,
    _round_div,
    _round_half_away,
)
from dataframe.time.calendar import civil_from_days
from dataframe.series import Series
from dataframe.text.bytes import _find_bytes
from dataframe.text.predicates import decode_values
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.vector.adaptive_filter import AdaptiveFilter
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
        if _lane(source) == _LANE_BOOL:
            # A Boolean casts exactly to integer or floating-point 1 or 0.
            return dtype == DataType.INT64 or dtype == DataType.FLOAT64
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
    per-node buffers it computes into, reused for every chunk."""

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
    # Computed strings (substrings): string_t into their input's bytes.
    var lane_strings: List[List[StringT]]
    # The bytes of computed strings made here (concatenations).
    var heaps: List[List[UInt8]]
    var valid: List[List[UInt8]]
    var whole: List[Bool]
    var formats: List[UnifiedVectorFormat]
    var by_chunk_row: List[Bool]
    var ready: Bool
    # A column read by several nodes is loaded once an evaluation: the
    # evaluation that last loaded each column node, and for which rows.
    var stamp: List[Int]
    var stamp_rows: List[Int]
    var stamp_count: List[Int]
    var generation: Int
    # Per column node: the generation its values were last read into a
    # lane (a node read only by kernels over its vector is never read
    # into one), and, when its vector already holds the lane's type for
    # the computed rows in order, the vector's data, read in place (0:
    # the node's own lane buffer).
    var loaded: List[Int]
    var alias: List[Int]
    # Per constant node: whether its lane holds the value on every row.
    # A kernel that reads a constant as one value reads only the first,
    # so the lane is filled only for one that reads it row by row.
    var filled: List[Bool]
    # IN lists, per node: integer values, or string_t over `held`.
    var member_ints: List[List[Int64]]
    # WHEN nodes: the computed rows each branch takes and their chunk rows
    # (then positions, then rows, otherwise positions, otherwise rows).
    var when_rows: List[List[UInt32]]
    var member_strings: List[List[StringT]]
    # The composed selection `_entries` writes.
    var composed: List[UInt32]
    # Select: per comparison node, the column child it compares with a
    # constant directly at the column's own type (-1: evaluated in lanes);
    # per AND / OR node, flatten children and retain two selection buffers.
    var direct: List[Int]
    # Per AND node of two such comparisons on one column: that column
    # (a range selected in one pass), else -1.
    var ranges: List[Int]
    var select_rows: List[List[UInt32]]
    var children: List[List[Int]]
    # Per AND / OR node, adapt child order to measured evaluation time;
    # disabled for other node kinds.
    var adaptive: List[AdaptiveFilter]
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
        self.lane_strings = List[List[StringT]](length=n, fill=List[StringT]())
        self.heaps = List[List[UInt8]](length=n, fill=List[UInt8]())
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
        self.composed = List[UInt32]()
        self.direct = List[Int](length=n, fill=-1)
        self.ranges = List[Int](length=n, fill=-1)
        self.select_rows = List[List[UInt32]](length=n, fill=List[UInt32]())
        self.children = List[List[Int]](length=n, fill=List[Int]())
        self.adaptive = List[AdaptiveFilter](
            length=n, fill=AdaptiveFilter(0, True)
        )
        self.scratch = DataChunk(n)
        # Lane buffers are made by each worker's copy when it first
        # evaluates (`_allocate`): the compiled template stays small to copy.
        self.ready = False
        self.stamp = List[Int](length=n, fill=-1)
        self.stamp_rows = List[Int](length=n, fill=0)
        self.stamp_count = List[Int](length=n, fill=0)
        self.generation = 0
        self.loaded = List[Int](length=n, fill=-1)
        self.alias = List[Int](length=n, fill=0)
        self.filled = List[Bool](length=n, fill=False)
        self._plan(self.root)
        self._narrow_decimals()
        self._share_columns()
        self._plan_direct()

    def _share_columns(mut self):
        """Point every node reading a column at one node per column and
        lane, so the column is loaded once. Not under CASE WHEN, whose branches load their own
        rows."""
        for node in self.bound.expr._nodes:
            if node.op == WHEN:
                return
        var n = len(self.kinds)
        var canonical = List[Int](length=n, fill=-1)
        for i in range(n):
            if self.kinds[i] != _KIND_COLUMN:
                continue
            canonical[i] = i
            for j in range(i):
                if (
                    self.kinds[j] == _KIND_COLUMN
                    and canonical[j] == j
                    and self.bound.sources[j] == self.bound.sources[i]
                    and self.lanes[j] == self.lanes[i]
                ):
                    canonical[i] = j
                    break
        for k in range(n):
            if self.kinds[k] != _KIND_LANE:
                continue
            ref node = self.bound.expr._nodes[k]
            if node.left >= 0 and canonical[node.left] >= 0:
                node.left = canonical[node.left]
            if node.right >= 0 and canonical[node.right] >= 0:
                node.right = canonical[node.right]
            if node.extra >= 0 and canonical[node.extra] >= 0:
                node.extra = canonical[node.extra]
        for i in range(n):
            if canonical[i] >= 0 and canonical[i] != i:
                self.kinds[i] = _KIND_UNUSED

    def _plan_direct(mut self):
        """The comparisons Select runs straight over a column's vector:
        a column of a numeric type against a constant of the same type."""
        for i in range(len(self.kinds)):
            if self.kinds[i] != _KIND_LANE:
                continue
            ref node = self.bound.expr._nodes[i]
            if not _is_compare(node.op) or node.left < 0 or node.right < 0:
                continue
            var l = node.left
            var r = node.right
            if self.bound.dtypes[l] != self.bound.dtypes[r]:
                continue
            var column = -1
            if (
                self.kinds[l] == _KIND_COLUMN
                and self.kinds[r] == _KIND_CONSTANT
            ):
                column = l
            elif (
                self.kinds[r] == _KIND_COLUMN
                and self.kinds[l] == _KIND_CONSTANT
            ):
                column = r
            if column < 0:
                continue
            var lane = self.lanes[column]
            if (
                lane == _LANE_INT
                or lane == _LANE_FLOAT
                or lane == _LANE_DECIMAL
                or lane == _LANE_DECIMAL64
            ):
                self.direct[i] = column
        for i in range(len(self.kinds)):
            ref node = self.bound.expr._nodes[i]
            if self.kinds[i] != _KIND_LANE or node.op != AND:
                continue
            if node.left < 0 or node.right < 0:
                continue
            var a = self.direct[node.left]
            var b = self.direct[node.right]
            if a < 0 or b < 0:
                continue
            if (
                self.bound.sources[a] == self.bound.sources[b]
                and self.bound.dtypes[a] == self.bound.dtypes[b]
            ):
                self.ranges[i] = a
        for i in range(len(self.kinds)):
            var op = self.bound.expr._nodes[i].op
            if self.kinds[i] != _KIND_LANE or (op != AND and op != OR):
                continue
            var flat = List[Int]()
            self._flatten(i, op, flat)
            # AdaptiveFilter(const Expression &): permutations are disabled
            # when any child can throw.
            var can_throw = False
            for c in flat:
                can_throw = can_throw or self._can_throw(c)
            self.adaptive[i] = AdaptiveFilter(len(flat), can_throw)
            self.children[i] = flat^

    def _flatten(self, index: Int, op: Int, mut out: List[Int]):
        """The children of a chain of `op` conjunctions, left to right. A
        lower and an upper bound on one column stay one child, a range
        selected in one pass over the column (`_select_range_of`)."""
        for child in [
            self.bound.expr._nodes[index].left,
            self.bound.expr._nodes[index].right,
        ]:
            if (
                self.kinds[child] == _KIND_LANE
                and self.bound.expr._nodes[child].op == op
                and not (op == AND and self.ranges[child] >= 0)
            ):
                self._flatten(child, op, out)
            else:
                out.append(child)

    def _allocate(mut self):
        """The buffers every node computed in lanes or read in one writes,
        constants filled in."""
        var n = len(self.kinds)
        self.composed = List[UInt32](
            unsafe_uninit_length=2 * STANDARD_VECTOR_SIZE
        )
        for i in range(n):
            var kind = self.kinds[i]
            if kind == _KIND_UNUSED:
                continue
            var constant = kind == _KIND_CONSTANT
            # Buffers are left unwritten until a load or a kernel writes
            # them. A constant holds its value in the first row (the rest
            # are filled when a kernel first reads it row by row, `_fill`)
            # and, when null, null validity on every row.
            if constant and not self.constant_valid[i]:
                self.valid[i] = List[UInt8](length=STANDARD_VECTOR_SIZE, fill=0)
            else:
                self.valid[i] = List[UInt8](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
            var lane = self.lanes[i]
            if lane == _LANE_INT or lane == _LANE_DECIMAL64:
                self.lane_ints[i] = List[Int64](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
                self.lane_ints[i][0] = self.ints[i]
            elif lane == _LANE_DECIMAL:
                self.lane_decimals[i] = List[Int128](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
                self.lane_decimals[i][0] = self.decimals[i]
            elif lane == _LANE_FLOAT:
                self.lane_floats[i] = List[Float64](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
                self.lane_floats[i][0] = self.floats[i]
            elif lane == _LANE_BOOL:
                self.lane_bools[i] = List[UInt8](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
                self.lane_bools[i][0] = UInt8(1) if self.ints[
                    i
                ] != 0 else UInt8(0)
            elif lane == _LANE_STRING and kind == _KIND_LANE:
                self.lane_strings[i] = List[StringT](
                    unsafe_uninit_length=STANDARD_VECTOR_SIZE
                )
            if constant:
                self.whole[i] = self.constant_valid[i]
            if self.bound.expr._nodes[i].op == WHEN:
                self.when_rows[i] = List[UInt32](
                    unsafe_uninit_length=4 * STANDARD_VECTOR_SIZE
                )
        self.ready = True

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
            # Only a vector of a type with a lane carries its validity
            # (`flat_over`): a nested or other opaque column is tested
            # through the at-rest evaluator.
            return (
                node.left >= 0
                and node.right < 0
                and _lane(self.bound.dtypes[node.left]) != _LANE_NONE
            )
        if _is_string_predicate(node.op):
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == DataType.STRING
                and (
                    node.op != STR_CONTAINS
                    or self.bound.expr._nodes[index].integer == 0
                )
                and (
                    node.op != STR_LIKE
                    or Bool(_like_segments(self.bound.expr._nodes[index].text))
                )
            )
        if node.op == WHEN:
            if node.left < 0 or node.right < 0:
                return False
            if self.bound.dtypes[node.left] != DataType.BOOL:
                return False
            if _lane(dtype) == _LANE_NONE:
                return False
            if self.bound.dtypes[node.right] != dtype:
                return False
            return node.extra < 0 or self.bound.dtypes[node.extra] == dtype
        if node.op == FILL_NULL:
            # Both operands of the result's type: a decimal fill of
            # another scale widens first, at rest.
            if node.left < 0 or node.right < 0:
                return False
            var lane = _lane(dtype)
            if lane == _LANE_NONE or lane == _LANE_STRING:
                return False
            return (
                self.bound.dtypes[node.left] == dtype
                and self.bound.dtypes[node.right] == dtype
            )
        if node.op == ABS:
            if node.left < 0 or node.right >= 0:
                return False
            if self.bound.dtypes[node.left] != dtype or dtype.is_temporal():
                return False
            var lane = _lane(dtype)
            return (
                lane == _LANE_INT
                or lane == _LANE_FLOAT
                or lane == _LANE_DECIMAL
            )
        if node.op == STR_CONCAT:
            return (
                node.left >= 0
                and node.right >= 0
                and self.bound.dtypes[node.left] == DataType.STRING
                and self.bound.dtypes[node.right] == DataType.STRING
                and dtype == DataType.STRING
            )
        if node.op == STR_SLICE:
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == DataType.STRING
                and dtype == DataType.STRING
            )
        if node.op == ROUND or node.op == FLOOR or node.op == CEIL:
            # Floats only: an integer is unchanged and a decimal rounds at
            # rest.
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left] == dtype
                and _lane(dtype) == _LANE_FLOAT
            )
        if node.op == DT_YEAR or node.op == DT_MONTH or node.op == DT_DAY:
            return (
                node.left >= 0
                and node.right < 0
                and self.bound.dtypes[node.left].is_date()
                and dtype == DataType.INT64
            )
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
                file=stderr,
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

    def _can_throw(self, index: Int) -> Bool:
        """Whether evaluating node `index`'s subtree may raise: arithmetic and
        casts check overflow, and the at-rest evaluator may raise for anything;
        comparisons, logic, null tests and IN lists never do, and a constant was
        computed when compiling."""
        var kind = self.kinds[index]
        if kind == _KIND_FALLBACK:
            return True
        if kind != _KIND_LANE:
            return False
        var op = self.bound.expr._nodes[index].op
        if op == ADD or op == SUB or op == MUL or op == DIV or op == CAST:
            return True
        if op == ABS:
            return True
        for child in node_children(self.bound.expr._nodes[index]):
            if self._can_throw(child):
                return True
        return False

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
        if not self.ready:
            self._allocate()
        self.scratch.reset()
        self.generation += 1
        self._evaluate(self.root, input, arena, rows, count)

    def _evaluate(
        mut self,
        index: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        in_place: Bool = False,
    ) raises:
        """Compute node `index` for the rows. With `in_place`, a column
        node only takes its vector's view: the caller reads the vector at
        its own physical type, or reads it into the lane later
        (`_evaluate` again without `in_place`)."""
        var kind = self.kinds[index]
        if kind == _KIND_CONSTANT:
            if not in_place and not self.filled[index]:
                self._fill(index)
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
            if (
                self.stamp[index] == self.generation
                and self.stamp_rows[index] == rows.data
                and self.stamp_count[index] == count
            ):
                if not in_place and self.loaded[index] != self.generation:
                    self._load(index, rows, count)
                    self.loaded[index] = self.generation
                return
            self.stamp[index] = self.generation
            self.stamp_rows[index] = rows.data
            self.stamp_count[index] = count
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
            self.loaded[index] = -1
            if not in_place:
                self._load(index, rows, count)
                self.loaded[index] = self.generation
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
            if self._blends(index, node):
                self._blend(index, node, input, arena, rows, count)
            else:
                self._when(index, node, input, arena, rows, count)
            return
        var op = node.op
        if _is_compare(op) and self._compare_vector(
            index, node, input, arena, rows, count
        ):
            return
        if (
            op == AND
            and self.ranges[index] >= 0
            and self._range_vector(index, node, input, arena, rows, count)
        ):
            return
        if op == CAST and self._cast_vector(
            index, left, input, arena, rows, count
        ):
            return
        # A comparison reads a constant side as one value.
        var compare = _is_compare(op)
        self._evaluate(
            left,
            input,
            arena,
            rows,
            count,
            in_place=compare and self.kinds[left] == _KIND_CONSTANT,
        )
        if right >= 0:
            self._evaluate(
                right,
                input,
                arena,
                rows,
                count,
                in_place=compare and self.kinds[right] == _KIND_CONSTANT,
            )
        if compare:
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
        elif op == FILL_NULL:
            self._fill_null(index, left, right, count)
        elif op == ABS:
            self._abs(index, left, count)
        elif op == ROUND or op == FLOOR or op == CEIL:
            self._round(index, left, count, op)
        elif op == DT_YEAR or op == DT_MONTH or op == DT_DAY:
            self._date_part(index, left, count, op)
        elif op == STR_SLICE:
            self._slice(index, left, rows, count)
        elif op == STR_CONCAT:
            self._concat(index, left, right, rows, count)
        else:
            self._compute(index, count)

    def _fill(mut self, index: Int):
        """Constant node `index`'s value on every row of its lane, for a
        kernel that reads it row by row."""
        self.filled[index] = True
        var lane = self.lanes[index]
        if lane == _LANE_INT or lane == _LANE_DECIMAL64:
            _broadcast[DType.int64](
                Int(self.lane_ints[index].unsafe_ptr()), self.ints[index]
            )
        elif lane == _LANE_DECIMAL:
            _broadcast[DType.int128](
                Int(self.lane_decimals[index].unsafe_ptr()),
                self.decimals[index],
            )
        elif lane == _LANE_FLOAT:
            _broadcast[DType.float64](
                Int(self.lane_floats[index].unsafe_ptr()), self.floats[index]
            )
        elif lane == _LANE_BOOL:
            _broadcast[DType.uint8](
                Int(self.lane_bools[index].unsafe_ptr()),
                UInt8(1) if self.ints[index] != 0 else UInt8(0),
            )

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
        """The entries of node `index`'s unified format the computed rows read,
        as the address of `count` UInt32 indices, or 0 when computed row k reads
        entry k: the format's selection composed with the rows' when both are
        set, else whichever is."""
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
        self.alias[index] = 0
        # Validity first.
        self.whole[index] = _validity_bytes(
            self.formats[index].validity,
            entries,
            count,
            Int(self.valid[index].unsafe_ptr()),
        )
        if lane == _LANE_STRING or lane == _LANE_ANY:
            return
        var physical = self.formats[index].physical
        var data = self.formats[index].data
        if entries == 0 and physical == _lane_physical(lane):
            # The vector holds the lane's values for rows 0, 1, 2, ...:
            # read in place, no copy.
            self.alias[index] = data
            return
        if lane == _LANE_BOOL:
            _gather[DType.uint8, DType.uint8](
                data, entries, count, Int(self._u8(index))
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
                        Int(self._f64(index)),
                    )
                elif lane == _LANE_DECIMAL:
                    _gather[D, DType.int128](
                        data,
                        entries,
                        count,
                        Int(self._i128(index)),
                    )
                else:
                    _gather[D, DType.int64](
                        data,
                        entries,
                        count,
                        Int(self._i64(index)),
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
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(o)),
                src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(one)),
                count=count,
            )
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
        var out = self._u8(index)
        var lane = self.lanes[left]
        if lane == _LANE_STRING:
            var la = self._entries(left, rows, count, 0)
            var ra = self._entries(right, rows, count, 1)
            var target = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(self._u8(index))
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
        # A constant side is read as one value, not a lane of copies.
        var lc = self.kinds[left] == _KIND_CONSTANT
        var rc = self.kinds[right] == _KIND_CONSTANT
        if lane == _LANE_FLOAT:
            _compare_lanes[DType.float64](
                op,
                Int(self._f64(left)),
                lc,
                Int(self._f64(right)),
                rc,
                Int(out),
                count,
            )
        elif lane == _LANE_DECIMAL:
            _compare_lanes[DType.int128](
                op,
                Int(self._i128(left)),
                lc,
                Int(self._i128(right)),
                rc,
                Int(out),
                count,
            )
        elif lane == _LANE_DECIMAL64 and (
            self.bound.dtypes[left].scale() != self.bound.dtypes[right].scale()
        ):
            # 64-bit decimals of different scales align exactly in 128
            # bits.
            var ls = self.bound.dtypes[left].scale()
            var rs = self.bound.dtypes[right].scale()
            var common = max(ls, rs)
            var lf = pow10(common - ls)
            var rf = pow10(common - rs)
            var a = List[Int128](unsafe_uninit_length=count)
            var b = List[Int128](unsafe_uninit_length=count)
            var x = self._i64(left)
            var y = self._i64(right)
            for k in range(count):
                a[k] = Int128(x[unsafe_offset=0 if lc else k]) * lf
                b[k] = Int128(y[unsafe_offset=0 if rc else k]) * rf
            _compare_lanes[DType.int128](
                op,
                Int(a.unsafe_ptr()),
                False,
                Int(b.unsafe_ptr()),
                False,
                Int(out),
                count,
            )
        else:
            _compare_lanes[DType.int64](
                op,
                Int(self._i64(left)),
                lc,
                Int(self._i64(right)),
                rc,
                Int(out),
                count,
            )

    def _scalar_of[
        D: DType
    ](self, constant: Int, column: Int) -> Optional[Scalar[D]]:
        """Constant node `constant` as a value of the column's physical
        type D when that is exact (a comparison at D then equals one in
        the column's lane); None when it is not."""
        var lane = self.lanes[column]
        comptime if D.is_floating_point():
            if lane != _LANE_FLOAT:
                return None
            var value = self.floats[constant].cast[D]()
            if value.cast[DType.float64]() != self.floats[constant]:
                return None
            return value
        else:
            if lane == _LANE_DECIMAL:
                var value = self.decimals[constant].cast[D]()
                if value.cast[DType.int128]() != self.decimals[constant]:
                    return None
                return value
            if lane == _LANE_FLOAT:
                return None
            var value = self.ints[constant].cast[D]()
            if value.cast[DType.int64]() != self.ints[constant]:
                return None
            return value

    def _null_bools(mut self, index: Int, count: Int):
        """Every row of Boolean node `index` null."""
        self.whole[index] = False
        unsafe_memset_zero(
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(self.valid[index].unsafe_ptr())
            ),
            count,
        )
        unsafe_memset_zero(
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(self._u8(index))
            ),
            count,
        )

    def _compare_vector(
        mut self,
        index: Int,
        node: _Shape,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises -> Bool:
        """A comparison of a column with a constant, or of two columns
        stored alike, computed over the columns' vectors at their own
        physical type: no value is copied or widened into a lane, and a
        constant is one value. False (nothing computed) when the operands
        are not such, or a constant is not exact at the column's type."""
        var l = node.left
        var r = node.right
        var lk = self.kinds[l]
        var rk = self.kinds[r]
        if not _numeric_lane(self.lanes[l]) or not _numeric_lane(self.lanes[r]):
            return False
        var op = node.op
        if lk == _KIND_COLUMN and rk == _KIND_COLUMN:
            if self.bound.dtypes[l] != self.bound.dtypes[r]:
                return False
            self._evaluate(l, input, arena, rows, count, in_place=True)
            self._evaluate(r, input, arena, rows, count, in_place=True)
            var physical = self.formats[l].physical
            if self.formats[r].physical != physical:
                return False
            var a = self._entries(l, rows, count, 0)
            var b = self._entries(r, rows, count, 1)
            var out = Int(self._u8(index))
            comptime for s in range(len(_LANE_STORAGE)):
                comptime D = _LANE_STORAGE[s]
                if physical == physical_of[D]():
                    _compare_pair[D](
                        op,
                        self.formats[l].data,
                        a,
                        self.formats[r].data,
                        b,
                        out,
                        count,
                    )
                    var flags = Int(self.valid[index].unsafe_ptr())
                    var lw = _validity_bytes(
                        self.formats[l].validity, a, count, flags
                    )
                    var other = Int(self.valid[r].unsafe_ptr())
                    var rw = _validity_bytes(
                        self.formats[r].validity, b, count, other
                    )
                    if lw and not rw:
                        _copy_bytes(flags, other, count)
                    elif not lw and not rw:
                        _and_bytes(flags, flags, other, count)
                    self.whole[index] = lw and rw
                    return True
            return False
        var column: Int
        var constant: Int
        if lk == _KIND_COLUMN and rk == _KIND_CONSTANT:
            column = l
            constant = r
        elif rk == _KIND_COLUMN and lk == _KIND_CONSTANT:
            column = r
            constant = l
            op = _flipped(op)
        else:
            return False
        self._evaluate(column, input, arena, rows, count, in_place=True)
        var physical = self.formats[column].physical
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if physical == physical_of[D]():
                if not self.constant_valid[constant]:
                    self._null_bools(index, count)
                    return True
                var value = self._scalar_of[D](constant, column)
                if not value:
                    return False
                var entries = self._entries(column, rows, count, 0)
                _compare_bytes[D](
                    op,
                    -1,
                    self.formats[column].data,
                    entries,
                    count,
                    value.value(),
                    value.value(),
                    Int(self._u8(index)),
                )
                self.whole[index] = _validity_bytes(
                    self.formats[column].validity,
                    entries,
                    count,
                    Int(self.valid[index].unsafe_ptr()),
                )
                return True
        return False

    def _range_vector(
        mut self,
        index: Int,
        node: _Shape,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises -> Bool:
        """An AND of a lower and an upper bound on one column, both
        constants valid, as one pass over the column's vector at its own
        type (a null row is null, as either bound alone makes it). False
        when the bounds are not such a pair."""
        var a = self._bound_of(node.left)
        var b = self._bound_of(node.right)
        var low = a
        var high = b
        if (a[0] == LT or a[0] == LE) and (b[0] == GT or b[0] == GE):
            low = b
            high = a
        if not (low[0] == GT or low[0] == GE) or not (
            high[0] == LT or high[0] == LE
        ):
            return False
        if not self.constant_valid[low[1]] or not self.constant_valid[high[1]]:
            return False
        var column = self.ranges[index]
        self._evaluate(column, input, arena, rows, count, in_place=True)
        var physical = self.formats[column].physical
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if physical == physical_of[D]():
                var lo = self._scalar_of[D](low[1], column)
                var hi = self._scalar_of[D](high[1], column)
                if not lo or not hi:
                    return False
                var entries = self._entries(column, rows, count, 0)
                _compare_bytes[D](
                    low[0],
                    high[0],
                    self.formats[column].data,
                    entries,
                    count,
                    lo.value(),
                    hi.value(),
                    Int(self._u8(index)),
                )
                self.whole[index] = _validity_bytes(
                    self.formats[column].validity,
                    entries,
                    count,
                    Int(self.valid[index].unsafe_ptr()),
                )
                return True
        return False

    def _cast_vector(
        mut self,
        index: Int,
        left: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises -> Bool:
        """A cast of a column to a float or an integer read straight from
        the column's vector into this node's lane, converting each value
        from its own physical type once (a decimal divided by 10^scale).
        These casts cannot overflow. False for any other cast."""
        var lane = self.lanes[index]
        var source_lane = self.lanes[left]
        if self.kinds[left] != _KIND_COLUMN:
            return False
        if lane == _LANE_INT:
            if source_lane != _LANE_INT:
                return False
        elif lane != _LANE_FLOAT or not _numeric_lane(source_lane):
            return False
        self._evaluate(left, input, arena, rows, count, in_place=True)
        var physical = self.formats[left].physical
        var entries = self._entries(left, rows, count, 0)
        var data = self.formats[left].data
        var source = self.bound.dtypes[left]
        var divisor = Float64(1)
        if source_lane == _LANE_DECIMAL or source_lane == _LANE_DECIMAL64:
            divisor = Float64(pow10(source.scale()))
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if physical == physical_of[D]():
                if lane == _LANE_FLOAT:
                    _convert[D, DType.float64](
                        data,
                        entries,
                        count,
                        divisor,
                        Int(self._f64(index)),
                    )
                else:
                    comptime if D.is_floating_point():
                        return False
                    else:
                        _gather[D, DType.int64](
                            data,
                            entries,
                            count,
                            Int(self._i64(index)),
                        )
                self.whole[index] = _validity_bytes(
                    self.formats[left].validity,
                    entries,
                    count,
                    Int(self.valid[index].unsafe_ptr()),
                )
                return True
        return False

    def _logic(mut self, index: Int, left: Int, right: Int, count: Int):
        """Kleene AND / OR: a dominant operand decides even when the other
        is null (the column kernels' `_logical`)."""
        var is_and = self.bound.expr._nodes[index].op == AND
        var x = self._u8(left)
        var y = self._u8(right)
        var out = self._u8(index)
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

    def _blends(self, index: Int, node: _Shape) -> Bool:
        """Whether a CASE WHEN is computed by blending: a result of numbers
        or Booleans, and branches that cannot raise, so computing each on
        every row observes nothing the rows' own branch would not."""
        var lane = self.lanes[index]
        if lane == _LANE_STRING or lane == _LANE_ANY or lane == _LANE_NONE:
            return False
        if self._can_throw(node.right):
            return False
        return node.extra < 0 or not self._can_throw(node.extra)

    def _blend(
        mut self,
        index: Int,
        node: _Shape,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """CASE WHEN by blending: the predicate and each branch over every
        row, then each row's value and validity taken from the branch its
        predicate picks (a null predicate picks the otherwise, and no
        otherwise is null), in one pass with no split of the rows."""
        self._evaluate(node.left, input, arena, rows, count)
        self._evaluate(node.right, input, arena, rows, count)
        var other = node.extra >= 0
        if other:
            self._evaluate(node.extra, input, arena, rows, count)
        var take = Int(self._u8(node.left))
        var take_valid = 0 if self.whole[node.left] else Int(
            self.valid[node.left].unsafe_ptr()
        )
        var lane = self.lanes[index]
        if lane == _LANE_BOOL:
            _blend_values[DType.uint8](
                take,
                take_valid,
                Int(self._u8(node.right)),
                Int(self._u8(node.extra)) if other else 0,
                Int(self._u8(index)),
                count,
            )
        elif lane == _LANE_FLOAT:
            _blend_values[DType.float64](
                take,
                take_valid,
                Int(self._f64(node.right)),
                Int(self._f64(node.extra)) if other else 0,
                Int(self._f64(index)),
                count,
            )
        elif lane == _LANE_DECIMAL:
            _blend_values[DType.int128](
                take,
                take_valid,
                Int(self._i128(node.right)),
                Int(self._i128(node.extra)) if other else 0,
                Int(self._i128(index)),
                count,
            )
        else:
            _blend_values[DType.int64](
                take,
                take_valid,
                Int(self._i64(node.right)),
                Int(self._i64(node.extra)) if other else 0,
                Int(self._i64(index)),
                count,
            )
        # Validity: the picked branch's, or null where no otherwise is.
        var then_valid = 0 if self.whole[node.right] else Int(
            self.valid[node.right].unsafe_ptr()
        )
        var other_valid = -1
        if other:
            other_valid = 0 if self.whole[node.extra] else Int(
                self.valid[node.extra].unsafe_ptr()
            )
        self.whole[index] = _blend_validity(
            take,
            take_valid,
            then_valid,
            other_valid,
            Int(self.valid[index].unsafe_ptr()),
            count,
        )

    def _when(
        mut self,
        index: Int,
        node: _Shape,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """CASE WHEN:
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
        var values = self._u8(node.left)
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
            any_null |= self._scatter(
                index,
                node.right,
                Int(then_at),
                taken,
                SelectionVector(Int(then_rows)),
            )
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
                    index,
                    node.extra,
                    Int(other_at),
                    others,
                    SelectionVector(Int(other_rows)),
                )
            else:
                var o = self.valid[index].unsafe_ptr()
                for j in range(others):
                    o[unsafe_offset=Int(other_at[unsafe_offset=j])] = 0
                any_null = True
        self.whole[index] = not any_null
        if self.lanes[index] == _LANE_STRING:
            self.formats[index] = UnifiedVectorFormat(
                SelectionVector(),
                Int(self.lane_strings[index].unsafe_ptr()),
                ValidityMask(),
                PHYSICAL_VARCHAR,
            )
            self.by_chunk_row[index] = False

    def _scatter(
        mut self,
        index: Int,
        child: Int,
        positions: Int,
        count: Int,
        rows: SelectionVector,
    ) -> Bool:
        """The child's `count` values (computed over the rows a branch
        takes) into node `index` at `positions`, with validity; whether
        any is null."""
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=positions)
        var lane = self.lanes[index]
        if lane == _LANE_STRING:
            # A string branch is read where it is (a column's vector, a
            # constant, a computed string): its string_t are copied.
            var entries = self._entries(child, rows, count, 1)
            var data = self.formats[child].data
            var o = Pointer[StringT, MutAnyOrigin](
                unsafe_from_address=Int(self.lane_strings[index].unsafe_ptr())
            )
            var source = Pointer[StringT, MutAnyOrigin](
                unsafe_from_address=data
            )
            var from_at = Pointer[UInt32, MutAnyOrigin](
                unsafe_from_address=entries
            )
            for j in range(count):
                var e = j if entries == 0 else Int(from_at[unsafe_offset=j])
                o[unsafe_offset=Int(at[unsafe_offset=j])] = source[
                    unsafe_offset=e
                ]
        elif lane == _LANE_BOOL:
            _scatter_values[DType.uint8](
                Int(self._u8(child)),
                Int(self._u8(index)),
                positions,
                count,
            )
        elif lane == _LANE_FLOAT:
            _scatter_values[DType.float64](
                Int(self._f64(child)),
                Int(self._f64(index)),
                positions,
                count,
            )
        elif lane == _LANE_DECIMAL:
            _scatter_values[DType.int128](
                Int(self._i128(child)),
                Int(self._i128(index)),
                positions,
                count,
            )
        else:
            _scatter_values[DType.int64](
                Int(self._i64(child)),
                Int(self._i64(index)),
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
            unsafe_from_address=Int(self._u8(index))
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
        var segments = List[String]()
        if op == STR_LIKE:
            segments = _like_segments(needle).value().copy()
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
            if op == STR_LIKE:
                found = _like_match(bytes, length, segments)
            elif n <= length:
                if op == STR_STARTS_WITH:
                    found = (
                        n == 0
                        or unsafe_memcmp(
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=bytes
                            ),
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=probe
                            ),
                            n,
                        )
                        == 0
                    )
                elif op == STR_ENDS_WITH:
                    found = (
                        n == 0
                        or unsafe_memcmp(
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=bytes + length - n
                            ),
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=probe
                            ),
                            n,
                        )
                        == 0
                    )
                else:
                    found = (
                        _find_bytes(
                            Span[UInt8, ImmutAnyOrigin](
                                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                                    unsafe_from_address=bytes
                                ),
                                length=length,
                            ),
                            Span[UInt8, ImmutAnyOrigin](
                                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                                    unsafe_from_address=probe
                                ),
                                length=n,
                            ),
                            0,
                        )
                        >= 0
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
            unsafe_from_address=Int(self._u8(index))
        )
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            var a = self.valid[left].unsafe_ptr()
            var o = self.valid[index].unsafe_ptr()
            for k in range(count):
                o[unsafe_offset=k] = a[unsafe_offset=k]
        if self.bound.expr._nodes[index].op == INT_IS_IN:
            var x = self._i64(left)
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
        var x = self._u8(left)
        var out = self._u8(index)
        for k in range(count):
            out[unsafe_offset=k] = UInt8(1) - x[unsafe_offset=k]
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(self.valid[index].unsafe_ptr())
                ),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(self.valid[left].unsafe_ptr())
                ),
                count=count,
            )

    def _lane_address(self, index: Int) -> Int:
        """The address of node `index`'s lane values: its own buffer, or
        the vector a column's values are read in place from."""
        if self.alias[index] != 0:
            return self.alias[index]
        var lane = self.lanes[index]
        if lane == _LANE_DECIMAL:
            return Int(self.lane_decimals[index].unsafe_ptr())
        if lane == _LANE_FLOAT:
            return Int(self.lane_floats[index].unsafe_ptr())
        if lane == _LANE_BOOL:
            return Int(self.lane_bools[index].unsafe_ptr())
        return Int(self.lane_ints[index].unsafe_ptr())

    @always_inline
    def _i64(self, index: Int) -> Pointer[Int64, MutAnyOrigin]:
        return Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self._lane_address(index)
        )

    @always_inline
    def _f64(self, index: Int) -> Pointer[Float64, MutAnyOrigin]:
        return Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=self._lane_address(index)
        )

    @always_inline
    def _i128(self, index: Int) -> Pointer[Int128, MutAnyOrigin]:
        return Pointer[Int128, MutAnyOrigin](
            unsafe_from_address=self._lane_address(index)
        )

    @always_inline
    def _u8(self, index: Int) -> Pointer[UInt8, MutAnyOrigin]:
        return Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self._lane_address(index)
        )

    def _fill_null(mut self, index: Int, left: Int, right: Int, count: Int):
        """fill_null: the left value where it is valid, else the right, as the column kernel's
        `(x & av) | (y & ~av)`."""
        var lane = self.lanes[index]
        var target = self._lane_address(index)
        var x = self._lane_address(left)
        var y = self._lane_address(right)
        var size = 16 if lane == _LANE_DECIMAL else (
            1 if lane == _LANE_BOOL else 8
        )
        if self.whole[left]:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target),
                src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=x),
                count=count * size,
            )
            self.whole[index] = True
            return
        var a = self.valid[left].unsafe_ptr()
        var flags = self.valid[index].unsafe_ptr()
        var right_whole = self.whole[right]
        var b = self.valid[right].unsafe_ptr()
        for k in range(count):
            flags[unsafe_offset=k] = a[unsafe_offset=k] | (
                UInt8(1) if right_whole else b[unsafe_offset=k]
            )
        self.whole[index] = right_whole
        if lane == _LANE_DECIMAL:
            _pick_values[DType.int128](x, y, Int(a), target, count)
        elif lane == _LANE_FLOAT:
            _pick_values[DType.float64](x, y, Int(a), target, count)
        elif lane == _LANE_BOOL:
            _pick_values[DType.uint8](x, y, Int(a), target, count)
        else:
            _pick_values[DType.int64](x, y, Int(a), target, count)

    def _abs(mut self, index: Int, left: Int, count: Int) raises:
        """abs, as the column kernel computes it: a signed integer's
        minimum raises (\"Int64 abs overflow\"), unsigned values and
        decimals within their precision keep their magnitude."""
        var lane = self.lanes[index]
        var dtype = self.bound.dtypes[index]
        var whole = self.whole[left]
        self.whole[index] = whole
        var a = self.valid[left].unsafe_ptr()
        if not whole:
            var flags = self.valid[index].unsafe_ptr()
            for k in range(count):
                flags[unsafe_offset=k] = a[unsafe_offset=k]
        if lane == _LANE_FLOAT:
            var x = self._f64(left)
            var out = self._f64(index)
            for k in range(count):
                out[unsafe_offset=k] = abs(x[unsafe_offset=k])
            return
        if lane == _LANE_DECIMAL:
            var x = self._i128(left)
            var out = self._i128(index)
            for k in range(count):
                var v = x[unsafe_offset=k]
                out[unsafe_offset=k] = -v if v < 0 else v
            return
        var x = self._i64(left)
        var out = self._i64(index)
        var low = Int64.MIN
        var name = String("Int64")
        if dtype == DataType.INT32:
            low = Int64(Int32.MIN)
            name = String(DType.int32)
        elif dtype == DataType.INT16:
            low = Int64(Int16.MIN)
            name = String(DType.int16)
        elif dtype == DataType.INT8:
            low = Int64(Int8.MIN)
            name = String(DType.int8)
        var signed = lane == _LANE_DECIMAL64 or (
            dtype == DataType.INT64
            or dtype == DataType.INT32
            or dtype == DataType.INT16
            or dtype == DataType.INT8
        )
        var overflow = False
        for k in range(count):
            var v = x[unsafe_offset=k]
            if not whole and a[unsafe_offset=k] == 0:
                v = 0
            overflow |= signed and lane != _LANE_DECIMAL64 and v == low
            out[unsafe_offset=k] = -v if signed and v < 0 else v
        if overflow:
            raise Error(name + " abs overflow")

    def _copy_validity(mut self, index: Int, left: Int, count: Int):
        """Node `index` is valid where its only child is."""
        self.whole[index] = self.whole[left]
        if not self.whole[left]:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(self.valid[index].unsafe_ptr())
                ),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(self.valid[left].unsafe_ptr())
                ),
                count=count,
            )

    def _round(mut self, index: Int, left: Int, count: Int, op: Int):
        """round (half away from zero, at `integer` decimals), floor and
        ceil of floats, as the column kernels compute them."""
        self._copy_validity(index, left, count)
        var x = self._f64(left)
        var out = self._f64(index)
        if op == FLOOR:
            for k in range(count):
                out[unsafe_offset=k] = floor(x[unsafe_offset=k])
        elif op == CEIL:
            for k in range(count):
                out[unsafe_offset=k] = ceil(x[unsafe_offset=k])
        else:
            var decimals = Int(self.bound.expr._nodes[index].integer)
            for k in range(count):
                out[unsafe_offset=k] = _round_half_away(
                    x[unsafe_offset=k], decimals
                )

    def _slice(
        mut self, index: Int, left: Int, rows: SelectionVector, count: Int
    ):
        """Clipped code points [offset, offset + length) without copying:
        a long result points into its input's bytes, a short one is inlined."""
        self._copy_validity(index, left, count)
        var offset = Int(self.bound.expr._nodes[index].integer)
        var length = self.bound.expr._nodes[index].min_count
        var entries = self._entries(left, rows, count)
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
        var data = self.formats[left].data
        var out = self.lane_strings[index].unsafe_ptr()
        var flags = self.valid[left].unsafe_ptr()
        var whole = self.whole[left]
        for k in range(count):
            if not whole and flags[unsafe_offset=k] == 0:
                out[unsafe_offset=k] = StringT(0, 0, 0)
                continue
            var entry = data + 16 * (
                k if entries == 0 else Int(at[unsafe_offset=k])
            )
            var size = Int(
                Pointer[StringT, MutAnyOrigin](
                    unsafe_from_address=entry
                )[].length
            )
            var bytes = string_data(entry)
            var text = StringSpan(
                unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                        unsafe_from_address=bytes
                    ),
                    length=size,
                )
            )
            var start, end = codepoint_slice_bounds(text, offset, length)
            out[unsafe_offset=k] = StringT.make(bytes + start, end - start)
        self.formats[index] = UnifiedVectorFormat(
            SelectionVector(),
            Int(self.lane_strings[index].unsafe_ptr()),
            ValidityMask(),
            PHYSICAL_VARCHAR,
        )
        self.by_chunk_row[index] = False

    def _concat(
        mut self,
        index: Int,
        left: Int,
        right: Int,
        rows: SelectionVector,
        count: Int,
    ):
        """concat_str of two strings with a separator, as
        `concat_strings`: a null side makes the row null, or with
        ignore_nulls the valid side is the row (null when both are). The
        joined bytes are written to this node's heap, sized for the chunk
        first so the string_t pointing into it stay put."""
        var separator = self.bound.expr._nodes[index].text
        var skip_nulls = self.bound.expr._nodes[index].integer == 1
        var gap = separator.byte_length()
        var gap_at = Int(separator.unsafe_ptr())
        var a_entries = self._entries(left, rows, count, 0)
        var b_entries = self._entries(right, rows, count, 1)
        var a_at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=a_entries)
        var b_at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=b_entries)
        var a_data = self.formats[left].data
        var b_data = self.formats[right].data
        var a_whole = self.whole[left]
        var b_whole = self.whole[right]
        var a_valid = self.valid[left].unsafe_ptr()
        var b_valid = self.valid[right].unsafe_ptr()
        var flags = self.valid[index].unsafe_ptr()
        var out = self.lane_strings[index].unsafe_ptr()
        var total = 0
        for k in range(count):
            var a = a_data + 16 * (
                k if a_entries == 0 else Int(a_at[unsafe_offset=k])
            )
            var b = b_data + 16 * (
                k if b_entries == 0 else Int(b_at[unsafe_offset=k])
            )
            total += _length_at(a) + gap + _length_at(b)
        if len(self.heaps[index]) < total:
            self.heaps[index] = List[UInt8](
                unsafe_uninit_length=max(total, 2 * len(self.heaps[index]))
            )
        var heap = Int(self.heaps[index].unsafe_ptr())
        var used = 0
        var whole = True
        for k in range(count):
            var a = a_data + 16 * (
                k if a_entries == 0 else Int(a_at[unsafe_offset=k])
            )
            var b = b_data + 16 * (
                k if b_entries == 0 else Int(b_at[unsafe_offset=k])
            )
            var x = a_whole or a_valid[unsafe_offset=k] != 0
            var y = b_whole or b_valid[unsafe_offset=k] != 0
            if x and y:
                var a_len = _length_at(a)
                var b_len = _length_at(b)
                var at = heap + used
                _copy_bytes(at, string_data(a), a_len)
                _copy_bytes(at + a_len, gap_at, gap)
                _copy_bytes(at + a_len + gap, string_data(b), b_len)
                var size = a_len + gap + b_len
                used += size
                out[unsafe_offset=k] = StringT.make(at, size)
                flags[unsafe_offset=k] = 1
            elif skip_nulls and (x or y):
                out[unsafe_offset=k] = Pointer[StringT, MutAnyOrigin](
                    unsafe_from_address=a if x else b
                )[]
                flags[unsafe_offset=k] = 1
            else:
                out[unsafe_offset=k] = StringT(0, 0, 0)
                flags[unsafe_offset=k] = 0
                whole = False
        _ = separator^
        self.whole[index] = whole
        self.formats[index] = UnifiedVectorFormat(
            SelectionVector(),
            Int(self.lane_strings[index].unsafe_ptr()),
            ValidityMask(),
            PHYSICAL_VARCHAR,
        )
        self.by_chunk_row[index] = False

    def _date_part(mut self, index: Int, left: Int, count: Int, op: Int):
        """The year, month or day of dates (`_fields`' civil calendar)."""
        self._copy_validity(index, left, count)
        var x = self._i64(left)
        var out = self._i64(index)
        var flags = self.valid[left].unsafe_ptr()
        var whole = self.whole[left]
        var part = 0 if op == DT_YEAR else (1 if op == DT_MONTH else 2)
        for k in range(count):
            if not whole and flags[unsafe_offset=k] == 0:
                out[unsafe_offset=k] = 0
                continue
            var ymd = civil_from_days(x[unsafe_offset=k])
            out[unsafe_offset=k] = ymd[0] if part == 0 else (
                ymd[1] if part == 1 else ymd[2]
            )

    def _null_test(
        mut self,
        index: Int,
        left: Int,
        rows: SelectionVector,
        count: Int,
        is_null: Bool,
    ):
        var out = self._u8(index)
        self.whole[index] = True
        if self.whole[left]:
            for k in range(count):
                out[unsafe_offset=k] = UInt8(1) if not is_null else UInt8(0)
            return
        var a = self.valid[left].unsafe_ptr()
        # Validity bytes are 0 or 1: a null test is the byte or its flip.
        var flip = UInt8(1) if is_null else UInt8(0)
        for k in range(count):
            out[unsafe_offset=k] = a[unsafe_offset=k] ^ flip

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
                var out = self._i128(index)
                var limit = precision_limit(dtype.precision())
                var overflow = False
                if source_lane == _LANE_INT:
                    var factor = pow10(dtype.scale())
                    var x = self._i64(left)
                    for k in range(count):
                        var value = Int128(x[unsafe_offset=k]) * factor
                        overflow |= value <= -limit or value >= limit
                        out[unsafe_offset=k] = value
                elif source_lane == _LANE_DECIMAL64:
                    var x = self._i64(left)
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
                    var x = self._i128(left)
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
                var out = self._i64(index)
                var x = self._i64(left)
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
                var out = self._f64(index)
                if source_lane == _LANE_FLOAT:
                    var x = self._f64(left)
                    for k in range(count):
                        out[unsafe_offset=k] = x[unsafe_offset=k]
                elif source_lane == _LANE_BOOL:
                    var x = self._u8(left)
                    for k in range(count):
                        out[unsafe_offset=k] = Float64(x[unsafe_offset=k] & 1)
                elif source_lane == _LANE_INT:
                    var x = self._i64(left)
                    for k in range(count):
                        out[unsafe_offset=k] = Float64(x[unsafe_offset=k])
                elif source_lane == _LANE_DECIMAL64:
                    var divisor = Float64(pow10(source.scale()))
                    var x = self._i64(left)
                    for k in range(count):
                        out[unsafe_offset=k] = (
                            Float64(x[unsafe_offset=k]) / divisor
                        )
                else:
                    # value / 10^scale, as `_decimal_to_float` converts.
                    var divisor = Float64(pow10(source.scale()))
                    var x = self._i128(left)
                    for k in range(count):
                        out[unsafe_offset=k] = (
                            Float64(x[unsafe_offset=k]) / divisor
                        )
                return
            var out = self._i64(index)
            if source_lane == _LANE_BOOL:
                var b = self._u8(left)
                for k in range(count):
                    out[unsafe_offset=k] = Int64(b[unsafe_offset=k] & 1)
                return
            var x = self._i64(left)
            for k in range(count):
                out[unsafe_offset=k] = x[unsafe_offset=k]
            return
        var right = node.right
        self._combine(left, right, index, count)
        var whole = self.whole[index]
        var flags = self.valid[index].unsafe_ptr()
        if lane == _LANE_FLOAT:
            var x = self._f64(left)
            var y = self._f64(right)
            var out = self._f64(index)
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
            var x = self._i64(left)
            var y = self._i64(right)
            var out = self._i64(index)
            # Compute both the result and an overflow flag per row using
            # the arithmetic overflow intrinsics,
            # no division and no branch; the error is raised after the
            # loop when a valid row overflowed.
            var overflow = False
            if node.op == ADD:
                for k in range(count):
                    var a = x[unsafe_offset=k]
                    var b = y[unsafe_offset=k]
                    var value = a + b
                    var keep = whole or flags[unsafe_offset=k] != 0
                    overflow |= keep and ((a ^ value) & (b ^ value)) < 0
                    out[unsafe_offset=k] = value if keep else 0
                if overflow:
                    raise Error("Int64 sum overflow")
            elif node.op == SUB:
                for k in range(count):
                    var a = x[unsafe_offset=k]
                    var b = y[unsafe_offset=k]
                    var value = a - b
                    var keep = whole or flags[unsafe_offset=k] != 0
                    overflow |= keep and ((a ^ b) & (a ^ value)) < 0
                    out[unsafe_offset=k] = value if keep else 0
                if overflow:
                    raise Error("Int64 subtraction overflow")
            else:
                for k in range(count):
                    var wide = Int128(x[unsafe_offset=k]) * Int128(
                        y[unsafe_offset=k]
                    )
                    var value = Int64(wide)
                    var keep = whole or flags[unsafe_offset=k] != 0
                    overflow |= keep and Int128(value) != wide
                    out[unsafe_offset=k] = value if keep else 0
                if overflow:
                    raise Error("Int64 multiplication overflow")
            return
        # Decimals, as `_decimal_dense` computes them: products at the
        # operands' own scales, rounded to the result's; sums and
        # differences at the larger scale.
        var left_type = self.bound.dtypes[left]
        var right_type = self.bound.dtypes[right]
        var limit = precision_limit(dtype.precision())
        var overflow = False
        if lane == _LANE_DECIMAL64:
            var x = self._i64(left)
            var y = self._i64(right)
            var out = self._i64(index)
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
        var x = self._i128(left)
        var y = self._i128(right)
        var out = self._i128(index)
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
        raise Error(
            "value exceeds decimal precision "
            + String(self.bound.dtypes[index].precision())
        )

    # --- results ----------------------------------------------------------

    def select(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
    ) raises -> Int:
        """For a Boolean expression: the chunk rows among `rows` it is true (and
        not null) for, written to `target` in order; how many."""
        if not self.ready:
            self._allocate()
        self.scratch.reset()
        self.generation += 1
        return self._select(self.root, input, arena, rows, count, target)

    def _select(
        mut self,
        index: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
        false_target: Int = 0,
    ) raises -> Int:
        """For node `index`: the rows among `rows` it is true for into `target`
        in row order, and when `false_target` is set the others there.

        AND runs each child on the rows the ones before kept, so later children
        see fewer rows; an AND of a lower and an upper bound on one column runs
        as one BETWEEN pass. OR runs each child on the rows the ones before
        rejected, appends each child's true rows, then sorts them back into row
        order. Both run their children in the order the node's AdaptiveFilter
        keeps, timed per call, so the cheapest, most selective child tends to
        run first. A comparison of a column with a constant selects over the
        column's vector; anything else is evaluated in lanes and its true rows
        kept."""
        if count == 0:
            return 0
        var op = self.bound.expr._nodes[index].op
        if self.kinds[index] == _KIND_LANE and (op == AND or op == OR):
            if op == AND and self.ranges[index] >= 0:
                var ranged = self._select_range_of(
                    index, input, arena, rows, count, target, false_target
                )
                if ranged >= 0:
                    return ranged
            if len(self.select_rows[index]) == 0:
                self.select_rows[index] = List[UInt32](
                    unsafe_uninit_length=2 * STANDARD_VECTOR_SIZE
                )
            var own = Int(self.select_rows[index].unsafe_ptr())
            var own2 = own + 4 * STANDARD_VECTOR_SIZE
            if op == AND:
                var current = rows
                var current_count = count
                var false_count = 0
                var temp_false = own if false_target != 0 else 0
                var started = self.adaptive[index].begin()
                for ci in range(len(self.children[index])):
                    var c = self.children[index][
                        self.adaptive[index].permutation[ci]
                    ]
                    var kept = self._select(
                        c,
                        input,
                        arena,
                        current,
                        current_count,
                        target,
                        temp_false,
                    )
                    if false_target != 0:
                        _copy_rows(
                            temp_false,
                            false_target + 4 * false_count,
                            current_count - kept,
                        )
                        false_count += current_count - kept
                    current_count = kept
                    if current_count == 0:
                        break
                    current = target
                self.adaptive[index].end(started)
                return current_count
            var false_rows = false_target if false_target != 0 else own2
            var current = rows
            var current_count = count
            var result = 0
            var started = self.adaptive[index].begin()
            for ci in range(len(self.children[index])):
                var c = self.children[index][
                    self.adaptive[index].permutation[ci]
                ]
                var kept = self._select(
                    c,
                    input,
                    arena,
                    current,
                    current_count,
                    SelectionVector(own),
                    false_rows,
                )
                if kept > 0:
                    _copy_rows(own, target.data + 4 * result, kept)
                    result += kept
                    current_count -= kept
                    current = SelectionVector(false_rows)
                if current_count == 0:
                    break
            if result > 0:
                sort(
                    Span[UInt32, MutAnyOrigin](
                        unsafe_ptr=Pointer[UInt32, MutAnyOrigin](
                            unsafe_from_address=target.data
                        ),
                        length=result,
                    )
                )
            self.adaptive[index].end(started)
            return result
        var column = self.direct[index]
        if column >= 0:
            return self._select_direct(
                index, column, input, arena, rows, count, target, false_target
            )
        # A fresh generation: `rows` may be a buffer reused with other rows.
        self.generation += 1
        self._evaluate(index, input, arena, rows, count)
        return self._true_rows(index, rows, count, target, false_target)

    def _bound_of(self, index: Int) -> Tuple[Int, Int]:
        """A direct comparison as (op with the column on the left, the
        constant node)."""
        ref node = self.bound.expr._nodes[index]
        var column = self.direct[index]
        var constant = node.right if column == node.left else node.left
        var op = node.op
        if column != node.left:
            if op == LT:
                op = GT
            elif op == LE:
                op = GE
            elif op == GT:
                op = LT
            elif op == GE:
                op = LE
        return (op, constant)

    def _select_range_of(
        mut self,
        index: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
        false_target: Int = 0,
    ) raises -> Int:
        """An AND of a lower and an upper bound on one column in one pass;
        -1 when the two are not such a pair (the AND then runs them one
        after the other)."""
        var a = self._bound_of(self.bound.expr._nodes[index].left)
        var b = self._bound_of(self.bound.expr._nodes[index].right)
        var low = a
        var high = b
        if (a[0] == LT or a[0] == LE) and (b[0] == GT or b[0] == GE):
            low = b
            high = a
        if not (low[0] == GT or low[0] == GE) or not (
            high[0] == LT or high[0] == LE
        ):
            return -1
        if not self.constant_valid[low[1]] or not self.constant_valid[high[1]]:
            _write_rows(rows, count, false_target)
            return 0
        var column = self.ranges[index]
        var format = input.unified(self.bound.sources[column], arena)
        var lane = self.lanes[column]
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if format.physical == physical_of[D]():
                var lo: Scalar[D]
                var hi: Scalar[D]
                var exact: Bool
                comptime if D.is_floating_point():
                    lo = self.floats[low[1]].cast[D]()
                    hi = self.floats[high[1]].cast[D]()
                    exact = lane == _LANE_FLOAT
                else:
                    if lane == _LANE_DECIMAL:
                        lo = self.decimals[low[1]].cast[D]()
                        hi = self.decimals[high[1]].cast[D]()
                        exact = (
                            lo.cast[DType.int128]() == self.decimals[low[1]]
                            and hi.cast[DType.int128]()
                            == self.decimals[high[1]]
                        )
                    else:
                        lo = self.ints[low[1]].cast[D]()
                        hi = self.ints[high[1]].cast[D]()
                        exact = (
                            lane != _LANE_FLOAT
                            and lo.cast[DType.int64]() == self.ints[low[1]]
                            and hi.cast[DType.int64]() == self.ints[high[1]]
                        )
                if exact:
                    return _select_range[D](
                        low[0],
                        high[0],
                        format,
                        rows,
                        count,
                        lo,
                        hi,
                        target,
                        false_target,
                    )
        return -1

    def _select_direct(
        mut self,
        index: Int,
        column: Int,
        mut input: DataChunk,
        mut arena: ColumnArena,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
        false_target: Int = 0,
    ) raises -> Int:
        """A column compared with a constant over the column's own vector
        and type; a null constant keeps nothing."""
        ref node = self.bound.expr._nodes[index]
        var constant = node.right if column == node.left else node.left
        if not self.constant_valid[constant]:
            _write_rows(rows, count, false_target)
            return 0
        var op = node.op
        if column != node.left:
            # constant OP column: the column on the left.
            if op == LT:
                op = GT
            elif op == LE:
                op = GE
            elif op == GT:
                op = LT
            elif op == GE:
                op = LE
        var format = input.unified(self.bound.sources[column], arena)
        var lane = self.lanes[column]
        comptime for s in range(len(_LANE_STORAGE)):
            comptime D = _LANE_STORAGE[s]
            if format.physical == physical_of[D]():
                var value: Scalar[D]
                var exact: Bool
                comptime if D.is_floating_point():
                    value = self.floats[constant].cast[D]()
                    exact = lane == _LANE_FLOAT
                else:
                    if lane == _LANE_DECIMAL:
                        value = self.decimals[constant].cast[D]()
                        exact = (
                            value.cast[DType.int128]()
                            == self.decimals[constant]
                        )
                    else:
                        value = self.ints[constant].cast[D]()
                        exact = (
                            lane != _LANE_FLOAT
                            and value.cast[DType.int64]() == self.ints[constant]
                        )
                if exact:
                    return _select_compare[D](
                        op, format, rows, count, value, target, false_target
                    )
        # A constant past the column's type: evaluated in lanes.
        self.generation += 1
        self._evaluate(index, input, arena, rows, count)
        return self._true_rows(index, rows, count, target, false_target)

    def _true_rows(
        mut self,
        index: Int,
        rows: SelectionVector,
        count: Int,
        target: SelectionVector,
        false_target: Int = 0,
    ) -> Int:
        """The rows node `index` (evaluated) is true and valid for, and the
        others into `false_target` when set: branch-free, every row
        written to both and each slot advancing by the row's outcome."""
        var kept = 0
        if self.kinds[index] == _KIND_CONSTANT:
            if self.constant_valid[index] and self.ints[index] != 0:
                _write_rows(rows, count, target.data)
                return count
            _write_rows(rows, count, false_target)
            return 0
        var values = self._u8(index)
        var flags = self.valid[index].unsafe_ptr()
        var whole = self.whole[index]
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=target.data)
        var from_rows = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=rows.data
        )
        var through = rows.is_set()
        if false_target == 0:
            for k in range(count):
                var keep = values[unsafe_offset=k]
                if not whole:
                    keep &= flags[unsafe_offset=k]
                out[unsafe_offset=kept] = from_rows[
                    unsafe_offset=k
                ] if through else UInt32(k)
                kept += Int(keep)
            return kept
        var other = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=false_target
        )
        var dropped = 0
        for k in range(count):
            var keep = values[unsafe_offset=k]
            if not whole:
                keep &= flags[unsafe_offset=k]
            var row = from_rows[unsafe_offset=k] if through else UInt32(k)
            out[unsafe_offset=kept] = row
            other[unsafe_offset=dropped] = row
            kept += Int(keep)
            dropped += 1 - Int(keep)
        return kept

    def write(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        mut output: DataChunk,
        column: Int,
    ) raises:
        """The expression over every row of `input`, as column `column` of
        `output`: a column it reads as it is, with no copy, a constant as a
        constant vector, anything computed into `output`'s cache for that
        column."""
        var root = self.root
        var kind = self.kinds[root]
        if kind == _KIND_COLUMN:
            output.reference(column, input.data[self.bound.sources[root]])
            return
        if kind == _KIND_CONSTANT:
            output.reference(column, self.constant_vectors[root])
            return
        var count = input.count
        if kind == _KIND_FALLBACK:
            self.evaluate(input, arena, SelectionVector(), count)
            output.reference(column, self.constant_vectors[root])
            return
        self._into_output(input, output, column)
        try:
            self.evaluate(input, arena, SelectionVector(), count)
        except e:
            self.alias[root] = 0
            raise e^
        self._store_root(output, column, count)

    def computes_rows(self) -> Bool:
        """Whether the root is computed in lanes, so `write_rows` can compute
        it over some rows only; a column, a constant or an at-rest
        evaluation is read whole."""
        return self.kinds[self.root] == _KIND_LANE

    def write_rows(
        mut self,
        mut input: DataChunk,
        mut arena: ColumnArena,
        mut output: DataChunk,
        column: Int,
        rows: SelectionVector,
        count: Int,
    ) raises:
        """The expression over the `count` chunk rows `rows` selects, as the
        first `count` values of column `column` of `output`, in selection
        order. The root must compute in lanes (`computes_rows`). No other
        row is evaluated, so none can raise."""
        self._into_output(input, output, column)
        try:
            self.evaluate(input, arena, rows, count)
        except e:
            self.alias[self.root] = 0
            raise e^
        self._store_root(output, column, count)

    def _into_output(
        mut self, input: DataChunk, mut output: DataChunk, column: Int
    ):
        """Point the root's values at `output`'s vector for `column` when
        that holds the root's lane type, so the root is computed in place
        there and not copied after (`_store_root` clears the pointer). Not
        when an input vector reads that buffer."""
        var root = self.root
        var lane = self.lanes[root]
        var physical = _physical(self.bound.dtypes[root])
        if self.kinds[root] != _KIND_LANE or _lane_physical(lane) != physical:
            return
        var vector = output.writable(column, physical)
        for v in input.data:
            if v.data == vector.data:
                return
        self.alias[root] = vector.data

    def _store_root(mut self, mut output: DataChunk, column: Int, count: Int):
        """The root's first `count` lane values into `output`'s column."""
        var root = self.root
        var lane = self.lanes[root]
        var dtype = self.bound.dtypes[root]
        var physical = _physical(dtype)
        if lane == _LANE_STRING:
            physical = PHYSICAL_VARCHAR
        var vector = output.writable(column, physical)
        var computed = self._lane_address(root)
        self.alias[root] = 0
        if computed == vector.data:
            # Computed in place (`_into_output`).
            pass
        elif lane == _LANE_STRING:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=vector.data
                ),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(
                        self.lane_strings[root].unsafe_ptr()
                    )
                ),
                count=16 * count,
            )
        elif lane == _LANE_BOOL:
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=vector.data
                ),
                src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=computed),
                count=count,
            )
        elif lane == _LANE_FLOAT:
            _store[DType.float64](physical, computed, vector.data, count)
        elif lane == _LANE_DECIMAL:
            _store[DType.int128](physical, computed, vector.data, count)
        else:
            _store[DType.int64](physical, computed, vector.data, count)
        if not self.whole[root]:
            var bits = output.validity_bits(column)
            var flags = self.valid[root].unsafe_ptr()
            # Eight validity bytes to one bitmap byte at a time.
            var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
            var at = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits)
            var whole_bytes = count >> 3
            for j in range(whole_bytes):
                var x = flags.unsafe_offset(8 * j).unsafe_load[width=8]()
                at[unsafe_offset=j] = (
                    x.ne(0).cast[DType.uint8]() * weights
                ).reduce_add()
            for k in range(8 * whole_bytes, count):
                if flags[unsafe_offset=k] == 0:
                    set_invalid(bits, k)


def _is_string_predicate(op: Int) -> Bool:
    return (
        op == STR_STARTS_WITH
        or op == STR_ENDS_WITH
        or op == STR_CONTAINS
        or op == STR_LIKE
    )


def _like_segments(pattern: String) -> Optional[List[String]]:
    """A LIKE pattern without '_' split at its '%' wildcards (the first and
    last segment are anchored, empty when the pattern starts or ends with
    '%'); None when it has a '_', which matches one character and is left
    to the column kernels."""
    var segments = List[String]()
    var bytes = pattern.as_bytes()
    var start = 0
    for i in range(len(bytes)):
        if bytes[i] == 95:
            return None
        if bytes[i] == 37:
            segments.append(String(pattern[byte=start:i]))
            start = i + 1
    segments.append(String(pattern[byte = start : len(bytes)]))
    return segments^


def _like_match(data: Int, length: Int, segments: List[String]) -> Bool:
    """Whether `length` bytes at `data` match LIKE segments: the first anchored at the start,
    the last at the end, the middle ones found in order between."""
    var n = len(segments)
    var head = segments[0].byte_length()
    if n == 1:
        return head == length and (
            head == 0
            or unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data),
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(segments[0].unsafe_ptr())
                ),
                head,
            )
            == 0
        )
    var tail = segments[n - 1].byte_length()
    if head + tail > length:
        return False
    if head > 0 and (
        unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data),
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(segments[0].unsafe_ptr())
            ),
            head,
        )
        != 0
    ):
        return False
    if tail > 0 and (
        unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=data + length - tail
            ),
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(segments[n - 1].unsafe_ptr())
            ),
            tail,
        )
        != 0
    ):
        return False
    var at = head
    var end = length - tail
    for k in range(1, n - 1):
        var size = segments[k].byte_length()
        if size == 0:
            continue
        if at + size > end:
            return False
        var found = _find_bytes(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=data
                ),
                length=end,
            ),
            segments[k].as_bytes(),
            at,
        )
        if found < 0:
            return False
        at = found + size
    return True


@always_inline
def _picks[W: Int](take: Int, take_valid: Int, k: Int) -> SIMD[DType.bool, W]:
    """Which of rows k..k+W take the then branch: predicate true and
    valid (`take_valid` 0 when every row is)."""
    var t = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=take)
    var picked = t.unsafe_offset(k).unsafe_load[width=W]().ne(0)
    if take_valid != 0:
        var v = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=take_valid)
        picked = picked & v.unsafe_offset(k).unsafe_load[width=W]().ne(0)
    return picked


def _blend_values[
    T: DType
](take: Int, take_valid: Int, then: Int, other: Int, target: Int, count: Int):
    """Each row's value from `then` where it takes the then branch, else
    from `other` (0 when there is none: the row is null)."""
    comptime W = 8
    var a = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=then)
    var b = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=other)
    var o = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    var k = 0
    while k + W <= count:
        var x = a.unsafe_offset(k).unsafe_load[width=W]()
        var y = SIMD[T, W](0)
        if other != 0:
            y = b.unsafe_offset(k).unsafe_load[width=W]()
        o.unsafe_offset(k).unsafe_store(
            _picks[W](take, take_valid, k).select(x, y)
        )
        k += W
    while k < count:
        var y = b[unsafe_offset=k] if other != 0 else Scalar[T](0)
        o[unsafe_offset=k] = a[unsafe_offset=k] if _picks[1](
            take, take_valid, k
        )[0] else y
        k += 1


def _blend_validity(
    take: Int,
    take_valid: Int,
    then_valid: Int,
    other_valid: Int,
    target: Int,
    count: Int,
) -> Bool:
    """Each row's validity byte from its picked branch (`then_valid` /
    `other_valid` 0 when every row of that branch is valid, `other_valid`
    -1 when there is no otherwise, which is null); whether every row is
    valid."""
    comptime W = 8
    var tv = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=then_valid)
    var ov = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=other_valid)
    var o = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target)
    var one = SIMD[DType.uint8, W](1)
    var zero = SIMD[DType.uint8, W](0)
    var nulls = SIMD[DType.uint8, W](0)
    var k = 0
    while k + W <= count:
        var x = one if then_valid == 0 else tv.unsafe_offset(k).unsafe_load[
            width=W
        ]().ne(0).select(one, zero)
        var y = zero
        if other_valid == 0:
            y = one
        elif other_valid > 0:
            y = (
                ov.unsafe_offset(k)
                .unsafe_load[width=W]()
                .ne(0)
                .select(one, zero)
            )
        var v = _picks[W](take, take_valid, k).select(x, y)
        o.unsafe_offset(k).unsafe_store(v)
        nulls |= one - v
        k += W
    var whole = nulls.reduce_or() == 0
    while k < count:
        var x = UInt8(1) if then_valid == 0 else UInt8(
            Int(tv[unsafe_offset=k] != 0)
        )
        var y = UInt8(0)
        if other_valid == 0:
            y = 1
        elif other_valid > 0:
            y = UInt8(Int(ov[unsafe_offset=k] != 0))
        var v = x if _picks[1](take, take_valid, k)[0] else y
        o[unsafe_offset=k] = v
        whole = whole and v != 0
        k += 1
    return whole


def _scatter_values[
    T: DType
](source: Int, target: Int, positions: Int, count: Int):
    var x = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=source)
    var o = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=positions)
    for j in range(count):
        o[unsafe_offset=Int(at[unsafe_offset=j])] = x[unsafe_offset=j]


@always_inline
def _length_at(entry: Int) -> Int:
    """The byte length of the string_t stored at `entry`."""
    return Int(
        Pointer[StringT, MutAnyOrigin](unsafe_from_address=entry)[].length
    )


@always_inline
def _copy_bytes(target: Int, source: Int, count: Int):
    if count > 0:
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target),
            src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=source),
            count=count,
        )


def _select_compare[
    D: DType
](
    op: Int,
    format: UnifiedVectorFormat,
    rows: SelectionVector,
    count: Int,
    value: Scalar[D],
    target: SelectionVector,
    false_target: Int = 0,
) -> Int:
    """The rows among `rows` whose value OP `value` holds, valid only,
    written to `target` in order, branch-free."""
    if op == EQ:
        return _select_compare_op[D, EQ](
            format, rows, count, value, target, value, false_target
        )
    if op == NE:
        return _select_compare_op[D, NE](
            format, rows, count, value, target, value, false_target
        )
    if op == LT:
        return _select_compare_op[D, LT](
            format, rows, count, value, target, value, false_target
        )
    if op == LE:
        return _select_compare_op[D, LE](
            format, rows, count, value, target, value, false_target
        )
    if op == GT:
        return _select_compare_op[D, GT](
            format, rows, count, value, target, value, false_target
        )
    return _select_compare_op[D, GE](
        format, rows, count, value, target, value, false_target
    )


def _flipped(op: Int) -> Int:
    """The comparison with its operands swapped: a < b is b > a."""
    if op == LT:
        return GT
    if op == LE:
        return GE
    if op == GT:
        return LT
    if op == GE:
        return LE
    return op


def _compare_bytes[
    D: DType
](
    op: Int,
    op2: Int,
    data: Int,
    entries: Int,
    count: Int,
    value: Scalar[D],
    value2: Scalar[D],
    target: Int,
):
    """One byte a row, 1 where the value at its entry OP `value` (and
    OP2 `value2` when `op2` is set: a range, `op` GT or GE and `op2` LT
    or LE)."""
    if op2 >= 0:
        if op == GE and op2 == LE:
            _compare_bytes_op[D, GE, LE](
                data, entries, count, value, value2, target
            )
        elif op == GE:
            _compare_bytes_op[D, GE, LT](
                data, entries, count, value, value2, target
            )
        elif op2 == LE:
            _compare_bytes_op[D, GT, LE](
                data, entries, count, value, value2, target
            )
        else:
            _compare_bytes_op[D, GT, LT](
                data, entries, count, value, value2, target
            )
    elif op == EQ:
        _compare_bytes_op[D, EQ](data, entries, count, value, value2, target)
    elif op == NE:
        _compare_bytes_op[D, NE](data, entries, count, value, value2, target)
    elif op == LT:
        _compare_bytes_op[D, LT](data, entries, count, value, value2, target)
    elif op == LE:
        _compare_bytes_op[D, LE](data, entries, count, value, value2, target)
    elif op == GT:
        _compare_bytes_op[D, GT](data, entries, count, value, value2, target)
    else:
        _compare_bytes_op[D, GE](data, entries, count, value, value2, target)


def _compare_bytes_op[
    D: DType, OP: Int, OP2: Int = -1
](
    data: Int,
    entries: Int,
    count: Int,
    value: Scalar[D],
    value2: Scalar[D],
    target: Int,
):
    var x = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=data)
    var o = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target)
    comptime W = 8
    var k = 0
    if entries == 0:
        while k + W <= count:
            o.unsafe_offset(k).unsafe_store(
                _test[D, OP, OP2, W](
                    x.unsafe_offset(k).unsafe_load[width=W](), value, value2
                )
            )
            k += W
        while k < count:
            o[unsafe_offset=k] = _test[D, OP, OP2, 1](
                x[unsafe_offset=k], value, value2
            )[0]
            k += 1
        return
    var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
    while k + W <= count:
        var v = SIMD[D, W]()
        comptime for l in range(W):
            v[l] = x[unsafe_offset=Int(at[unsafe_offset=k + l])]
        o.unsafe_offset(k).unsafe_store(_test[D, OP, OP2, W](v, value, value2))
        k += W
    while k < count:
        o[unsafe_offset=k] = _test[D, OP, OP2, 1](
            x[unsafe_offset=Int(at[unsafe_offset=k])], value, value2
        )[0]
        k += 1


def _compare_pair[
    D: DType
](
    op: Int,
    a: Int,
    a_entries: Int,
    b: Int,
    b_entries: Int,
    target: Int,
    count: Int,
):
    """One byte a row, 1 where a OP b, both read at their entries."""
    if op == EQ:
        _compare_pair_op[D, EQ](a, a_entries, b, b_entries, target, count)
    elif op == NE:
        _compare_pair_op[D, NE](a, a_entries, b, b_entries, target, count)
    elif op == LT:
        _compare_pair_op[D, LT](a, a_entries, b, b_entries, target, count)
    elif op == LE:
        _compare_pair_op[D, LE](a, a_entries, b, b_entries, target, count)
    elif op == GT:
        _compare_pair_op[D, GT](a, a_entries, b, b_entries, target, count)
    else:
        _compare_pair_op[D, GE](a, a_entries, b, b_entries, target, count)


def _compare_pair_op[
    D: DType, OP: Int
](a: Int, a_entries: Int, b: Int, b_entries: Int, target: Int, count: Int):
    var x = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=a)
    var y = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=b)
    var o = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target)
    comptime W = 8
    var k = 0
    if a_entries == 0 and b_entries == 0:
        while k + W <= count:
            o.unsafe_offset(k).unsafe_store(
                _holds2[D, OP, W](
                    x.unsafe_offset(k).unsafe_load[width=W](),
                    y.unsafe_offset(k).unsafe_load[width=W](),
                ).cast[DType.uint8]()
            )
            k += W
    var p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=a_entries)
    var q = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=b_entries)
    while k < count:
        var i = k if a_entries == 0 else Int(p[unsafe_offset=k])
        var j = k if b_entries == 0 else Int(q[unsafe_offset=k])
        o[unsafe_offset=k] = _holds2[D, OP, 1](
            SIMD[D, 1](x[unsafe_offset=i]), SIMD[D, 1](y[unsafe_offset=j])
        ).cast[DType.uint8]()[0]
        k += 1


def _select_range[
    D: DType
](
    low_op: Int,
    high_op: Int,
    format: UnifiedVectorFormat,
    rows: SelectionVector,
    count: Int,
    low: Scalar[D],
    high: Scalar[D],
    target: SelectionVector,
    false_target: Int = 0,
) -> Int:
    """The rows whose value is within a range (low_op GT or GE, high_op LT
    or LE) in one pass, avoiding a separate selection for each bound."""
    if low_op == GE and high_op == LE:
        return _select_compare_op[D, GE, LE](
            format, rows, count, low, target, high, false_target
        )
    if low_op == GE:
        return _select_compare_op[D, GE, LT](
            format, rows, count, low, target, high, false_target
        )
    if high_op == LE:
        return _select_compare_op[D, GT, LE](
            format, rows, count, low, target, high, false_target
        )
    return _select_compare_op[D, GT, LT](
        format, rows, count, low, target, high, false_target
    )


@always_inline
def _holds[
    D: DType, OP: Int, W: Int
](x: SIMD[D, W], value: Scalar[D]) -> SIMD[DType.bool, W]:
    comptime if OP == EQ:
        return x.eq(SIMD[D, W](value))
    elif OP == NE:
        # NaN != NaN is true: the negated equality, as the lanes compare.
        return ~x.eq(SIMD[D, W](value))
    elif OP == LT:
        return x.lt(SIMD[D, W](value))
    elif OP == LE:
        return x.le(SIMD[D, W](value))
    elif OP == GT:
        return x.gt(SIMD[D, W](value))
    else:
        return x.ge(SIMD[D, W](value))


@always_inline
def _test[
    D: DType, OP: Int, OP2: Int, W: Int
](x: SIMD[D, W], value: Scalar[D], value2: Scalar[D]) -> SIMD[DType.uint8, W]:
    """1 where x OP value (and x OP2 value2, for a range), else 0."""
    var held = _holds[D, OP, W](x, value).cast[DType.uint8]()
    comptime if OP2 >= 0:
        held &= _holds[D, OP2, W](x, value2).cast[DType.uint8]()
    return held


def _select_compare_op[
    D: DType, OP: Int, OP2: Int = -1
](
    format: UnifiedVectorFormat,
    rows: SelectionVector,
    count: Int,
    value: Scalar[D],
    target: SelectionVector,
    value2: Scalar[D] = Scalar[D](0),
    false_target: Int = 0,
) -> Int:
    """64 rows at a time: the comparison as a bit mask (eight lanes a
    compare when the rows are contiguous), validity bits ANDed in, then
    the set bits written out as rows."""
    var data = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=format.data)
    var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=target.data)
    var sel = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=format.sel.data)
    var from_rows = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows.data)
    var through_rows = rows.is_set()
    var through_sel = format.sel.is_set()
    var contiguous = not through_rows and not through_sel
    var bits = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=format.validity.bits
    )
    var nullable = not format.validity.all_valid()
    var offset = format.validity.offset
    var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var other = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=false_target)
    var kept = 0
    var dropped = 0
    var k = 0
    while k < count:
        var n = min(64, count - k)
        var mask = UInt64(0)
        if contiguous and n == 64:
            comptime for g in range(8):
                var x = data.unsafe_offset(k + 8 * g).unsafe_load[width=8]()
                var byte = (
                    _test[D, OP, OP2, 8](x, value, value2) * weights
                ).reduce_add()
                mask |= UInt64(byte) << UInt64(8 * g)
            if nullable:
                var b = offset + k
                if b & 7 == 0:
                    var v = UInt64(0)
                    comptime for g in range(8):
                        v |= UInt64(bits[unsafe_offset=(b >> 3) + g]) << UInt64(
                            8 * g
                        )
                    mask &= v
                else:
                    for j in range(64):
                        var c = b + j
                        mask &= ~(
                            UInt64(
                                1
                                - (
                                    (bits[unsafe_offset=c >> 3] >> UInt8(c & 7))
                                    & 1
                                )
                            )
                            << UInt64(j)
                        )
        elif n == 64:
            # Through a selection: eight values gathered, compared at once.
            comptime for g in range(8):
                var x = SIMD[D, 8]()
                var valid = SIMD[DType.uint8, 8](1)
                comptime for l in range(8):
                    var j = 8 * g + l
                    var row = (
                        Int(
                            from_rows[unsafe_offset=k + j]
                        ) if through_rows else k
                        + j
                    )
                    var entry = Int(
                        sel[unsafe_offset=row]
                    ) if through_sel else row
                    x[l] = data[unsafe_offset=entry]
                    if nullable:
                        var c = offset + entry
                        valid[l] = (
                            bits[unsafe_offset=c >> 3] >> UInt8(c & 7)
                        ) & 1
                var byte = (
                    (_test[D, OP, OP2, 8](x, value, value2) & valid) * weights
                ).reduce_add()
                mask |= UInt64(byte) << UInt64(8 * g)
        else:
            for j in range(n):
                var row = (
                    Int(from_rows[unsafe_offset=k + j]) if through_rows else k
                    + j
                )
                var entry = Int(sel[unsafe_offset=row]) if through_sel else row
                # Bitwise, not `and`: a short-circuit is a branch on the
                # comparison, which a filter's data makes unpredictable.
                var keep = UInt64(
                    _test[D, OP, OP2, 1](
                        data[unsafe_offset=entry], value, value2
                    )[0]
                )
                if nullable:
                    var c = offset + entry
                    keep &= UInt64(
                        (bits[unsafe_offset=c >> 3] >> UInt8(c & 7)) & 1
                    )
                mask |= keep << UInt64(j)
        if false_target == 0:
            kept = _emit(mask, n, k, rows, out, kept)
        else:
            # Both outputs in one pass over the block: either may be the
            # rows read (an AND narrows its rows in place, an OR its false
            # rows), and every row is read before either writes over it.
            for j in range(n):
                var row = from_rows[
                    unsafe_offset=k + j
                ] if through_rows else UInt32(k + j)
                var keep = Int((mask >> UInt64(j)) & 1)
                out[unsafe_offset=kept] = row
                other[unsafe_offset=dropped] = row
                kept += keep
                dropped += 1 - keep
        k += 64
    return kept


@always_inline
def _emit(
    mask: UInt64,
    n: Int,
    k: Int,
    rows: SelectionVector,
    dest: Pointer[UInt32, MutAnyOrigin],
    start: Int,
) -> Int:
    """The rows of a 64-row block (from `k`) whose bits are set in `mask`,
    appended to `dest` from `start`; where the next one goes. A dense mask
    writes every row branch-free, a sparse one walks its set bits."""
    var at = start
    var from_rows = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows.data)
    var through = rows.is_set()
    if pop_count(mask) >= 16:
        for j in range(n):
            dest[unsafe_offset=at] = from_rows[
                unsafe_offset=k + j
            ] if through else UInt32(k + j)
            at += Int((mask >> UInt64(j)) & 1)
        return at
    var m = mask
    while m != 0:
        var j = Int(count_trailing_zeros(m))
        dest[unsafe_offset=at] = from_rows[
            unsafe_offset=k + j
        ] if through else UInt32(k + j)
        at += 1
        m &= m - 1
    return at


def select_true(format: UnifiedVectorFormat, count: Int, target: Int) -> Int:
    """The positions among `count` rows of a Boolean vector whose value is
    true and valid, written to `target` in order; how many. A flat vector
    is read 64 rows at a time: its bytes compared at once into a bit
    mask, the validity word ANDed in, the set bits written out."""
    var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
    var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=target)
    var nullable = not format.validity.all_valid()
    var bits = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=format.validity.bits
    )
    var offset = format.validity.offset
    var kept = 0
    if format.sel.is_set():
        for k in range(count):
            var r = format.sel.get_index(k)
            var keep = flags[unsafe_offset=r] != 0
            if nullable:
                var c = offset + r
                keep = (
                    keep
                    and (bits[unsafe_offset=c >> 3] >> UInt8(c & 7)) & 1 != 0
                )
            out[unsafe_offset=kept] = UInt32(k)
            kept += Int(keep)
        return kept
    var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var k = 0
    while k < count:
        var n = min(64, count - k)
        var mask = UInt64(0)
        if n == 64:
            comptime for g in range(8):
                var x = flags.unsafe_offset(k + 8 * g).unsafe_load[width=8]()
                var byte = (x.ne(0).cast[DType.uint8]() * weights).reduce_add()
                mask |= UInt64(byte) << UInt64(8 * g)
        else:
            for j in range(n):
                mask |= UInt64(Int(flags[unsafe_offset=k + j] != 0)) << UInt64(
                    j
                )
        if nullable:
            var b = offset + k
            if b & 7 == 0 and n == 64:
                var v = UInt64(0)
                comptime for g in range(8):
                    v |= UInt64(bits[unsafe_offset=(b >> 3) + g]) << UInt64(
                        8 * g
                    )
                mask &= v
            else:
                for j in range(n):
                    var c = b + j
                    mask &= ~(
                        UInt64(
                            1
                            - ((bits[unsafe_offset=c >> 3] >> UInt8(c & 7)) & 1)
                        )
                        << UInt64(j)
                    )
        kept = _emit(mask, n, k, SelectionVector(), out, kept)
        k += 64
    return kept


def _validity_bytes(
    validity: ValidityMask, entries: Int, count: Int, target: Int
) -> Bool:
    """The validity of `count` rows (entries 0, 1, 2, ... when `entries`
    is 0, else the UInt32 indices there) as one byte a row at `target`;
    whether every row is valid (the bytes are then left unwritten)."""
    if validity.all_valid():
        return True
    var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target)
    var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=validity.bits)
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
            flags.unsafe_offset(8 * j).unsafe_store(spread.cast[DType.uint8]())
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
        var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
        for k in range(count):
            var b = offset + Int(at[unsafe_offset=k])
            flags[unsafe_offset=k] = (
                bits[unsafe_offset=b >> 3] >> UInt8(b & 7)
            ) & 1
    return False


def _lane_physical(lane: Int) -> Int:
    """The physical type a lane's values are held as."""
    if lane == _LANE_INT or lane == _LANE_DECIMAL64:
        return PHYSICAL_INT64
    if lane == _LANE_FLOAT:
        return PHYSICAL_DOUBLE
    if lane == _LANE_DECIMAL:
        return PHYSICAL_INT128
    if lane == _LANE_BOOL:
        return PHYSICAL_BOOL
    return -1


def _write_rows(rows: SelectionVector, count: Int, target: Int):
    """Every row of `rows` into `target` (none when it is unset)."""
    if target == 0:
        return
    var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=target)
    for k in range(count):
        out[unsafe_offset=k] = UInt32(rows.get_index(k))


def _copy_rows(source: Int, target: Int, count: Int):
    if count > 0 and source != target:
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=target),
            src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=source),
            count=4 * count,
        )


def _pick_values[
    T: DType
](left: Int, right: Int, valid: Int, target: Int, count: Int):
    """Per row, the value at `left` where the validity byte is set, else
    the value at `right`, into `target`."""
    var x = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=left)
    var y = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=right)
    var a = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid)
    var o = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    for k in range(count):
        o[unsafe_offset=k] = (
            x[unsafe_offset=k] if a[unsafe_offset=k]
            != 0 else y[unsafe_offset=k]
        )


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


def _broadcast[T: DType](target: Int, value: Scalar[T]):
    """`value` on every row of a lane at `target`."""
    var out = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    for k in range(STANDARD_VECTOR_SIZE):
        out[unsafe_offset=k] = value


def _convert[
    D: DType, T: DType
](data: Int, entries: Int, count: Int, divisor: Float64, target: Int):
    """`count` values of type D (at the UInt32 indices at `entries`, or
    rows 0, 1, 2, ... when 0) converted to the float type T, each divided
    by `divisor` unless it is 1."""
    var source = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=data)
    var out = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=target)
    var at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=entries)
    var d = divisor.cast[T]()
    if divisor == 1:
        if entries == 0:
            for k in range(count):
                out[unsafe_offset=k] = source[unsafe_offset=k].cast[T]()
        else:
            for k in range(count):
                out[unsafe_offset=k] = source[
                    unsafe_offset=Int(at[unsafe_offset=k])
                ].cast[T]()
        return
    if entries == 0:
        for k in range(count):
            out[unsafe_offset=k] = source[unsafe_offset=k].cast[T]() / d
    else:
        for k in range(count):
            out[unsafe_offset=k] = (
                source[unsafe_offset=Int(at[unsafe_offset=k])].cast[T]() / d
            )


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


def _store[T: DType](physical: Int, values: Int, address: Int, count: Int):
    """Lane values (at `values`) stored as `physical`."""
    var source = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=values)
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
](
    op: Int,
    a: Int,
    a_constant: Bool,
    b: Int,
    b_constant: Bool,
    address: Int,
    count: Int,
):
    """`count` comparisons of two lanes into bytes, a SIMD vector at a
    time; a constant side is one value at its address, broadcast."""
    if op == EQ:
        _compare_modes[T, EQ](a, a_constant, b, b_constant, address, count)
    elif op == NE:
        _compare_modes[T, NE](a, a_constant, b, b_constant, address, count)
    elif op == LT:
        _compare_modes[T, LT](a, a_constant, b, b_constant, address, count)
    elif op == LE:
        _compare_modes[T, LE](a, a_constant, b, b_constant, address, count)
    elif op == GT:
        _compare_modes[T, GT](a, a_constant, b, b_constant, address, count)
    else:
        _compare_modes[T, GE](a, a_constant, b, b_constant, address, count)


@always_inline
def _compare_modes[
    T: DType, op: Int
](a: Int, a_constant: Bool, b: Int, b_constant: Bool, address: Int, count: Int):
    if a_constant:
        _compare_simd[T, op, True, False](a, b, address, count)
    elif b_constant:
        _compare_simd[T, op, False, True](a, b, address, count)
    else:
        _compare_simd[T, op, False, False](a, b, address, count)


@always_inline
def _holds2[
    T: DType, op: Int, W: Int
](u: SIMD[T, W], v: SIMD[T, W]) -> SIMD[DType.bool, W]:
    """u OP v per element; an ordered compare is false for NaN, and
    NaN != NaN is true."""
    comptime if op == EQ:
        return u.eq(v)
    elif op == NE:
        return ~u.eq(v)
    elif op == LT:
        return u.lt(v)
    elif op == LE:
        return u.le(v)
    elif op == GT:
        return u.gt(v)
    else:
        return u.ge(v)


def _compare_simd[
    T: DType, op: Int, A_CONSTANT: Bool, B_CONSTANT: Bool
](a: Int, b: Int, address: Int, count: Int):
    if count == 0:
        return
    comptime W = 4 if T == DType.int128 else 8
    var target = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var x = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=a)
    var y = Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=b)
    var xs = SIMD[T, W](x[unsafe_offset=0])
    var ys = SIMD[T, W](y[unsafe_offset=0])
    var k = 0
    while k + W <= count:
        var u = xs
        var v = ys
        comptime if not A_CONSTANT:
            u = x.unsafe_offset(k).unsafe_load[width=W]()
        comptime if not B_CONSTANT:
            v = y.unsafe_offset(k).unsafe_load[width=W]()
        target.unsafe_offset(k).unsafe_store(
            _holds2[T, op, W](u, v).cast[DType.uint8]()
        )
        k += W
    while k < count:
        var u = xs[0] if A_CONSTANT else x[unsafe_offset=k]
        var v = ys[0] if B_CONSTANT else y[unsafe_offset=k]
        target[unsafe_offset=k] = _holds2[T, op, 1](
            SIMD[T, 1](u), SIMD[T, 1](v)
        ).cast[DType.uint8]()[0]
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
