"""Immutable-by-convention expression IR. No data access or execution here."""

comptime COL = 0
comptime LIT_INT = 1
comptime LIT_FLOAT = 2
comptime LIT_BOOL = 3
comptime LIT_STRING = 4
comptime ADD = 5
comptime SUB = 6
comptime MUL = 7
comptime GT = 8
comptime EQ = 9
comptime SUM = 10
comptime COUNT = 11
# A typed null literal; the dtype name is kept in `text`.
comptime LIT_NULL = 12

# Binary operations occupy 20..49.
comptime LT = 20
comptime GE = 21
comptime LE = 22
comptime NE = 23
comptime DIV = 24
comptime FLOORDIV = 25
comptime MOD = 26
comptime POW = 27
comptime CLIP_LOW = 28
comptime CLIP_HIGH = 29
comptime AND = 30
comptime OR = 31
comptime XOR = 32
comptime FILL_NULL = 33
comptime FILL_NAN = 34
# Right's values where left is valid, null where left is null.
comptime KEEP_NULLS = 35

# Unary operations occupy 50..79; ROUND keeps its decimals in `integer`.
comptime NEG = 50
comptime ABS = 51
comptime SQRT = 52
comptime EXP = 53
comptime LOG = 54
comptime FLOOR = 55
comptime CEIL = 56
comptime ROUND = 57
comptime NOT = 60
comptime IS_NULL = 61
comptime IS_NOT_NULL = 62
comptime IS_NAN = 63
comptime IS_NOT_NAN = 64
comptime IS_FINITE = 65
comptime IS_INFINITE = 66

# Reductions occupy 80..99; ANY/ALL keep ignore_nulls in `integer`.
comptime MIN = 80
comptime MAX = 81
comptime MEAN = 82
comptime FIRST = 83
comptime LAST = 84
comptime N_UNIQUE = 85
comptime STD = 86
comptime VAR = 87
comptime MEDIAN = 88
comptime QUANTILE = 89
comptime LEN = 90
comptime ANY = 91
comptime ALL = 92
comptime NULL_COUNT = 93


def is_binary(op: Int) -> Bool:
    return (op >= ADD and op <= EQ) or (op >= 20 and op < 50)


def is_unary(op: Int) -> Bool:
    return op >= 50 and op < 80


def is_reduction(op: Int) -> Bool:
    return op == SUM or op == COUNT or (op >= 80 and op < 100)


def is_comparison(op: Int) -> Bool:
    return op == GT or op == EQ or (op >= LT and op <= NE)


def is_logical(op: Int) -> Bool:
    return op == AND or op == OR or op == XOR


@fieldwise_init
struct Node(Copyable):
    var op: Int
    var left: Int
    var right: Int
    var text: String
    var integer: Int64
    var floating: Float64
    var min_count: Int
    var extra: Int


def _node(
    op: Int,
    left: Int = -1,
    right: Int = -1,
    text: String = "",
    integer: Int64 = 0,
    floating: Float64 = 0,
    min_count: Int = 0,
    extra: Int = -1,
) -> Node:
    return Node(op, left, right, text, integer, floating, min_count, extra)


@fieldwise_init
struct Expr(Copyable):
    """A flat, topologically ordered tree; composition never evaluates data."""

    var _nodes: List[Node]
    var _name: String

    def alias(self, name: String) -> Self:
        var result = self.copy()
        result._name = name
        return result^

    def _binary(self, other: Self, op: Int) -> Self:
        var nodes = self._nodes.copy()
        var offset = len(nodes)
        _append_shifted(nodes, other._nodes, offset)
        nodes.append(_node(op, offset - 1, len(nodes) - 1))
        return Self(nodes^, self._name)

    def _unary(self, op: Int, integer: Int64 = 0) -> Self:
        var nodes = self._nodes.copy()
        nodes.append(_node(op, len(nodes) - 1, integer=integer))
        return Self(nodes^, self._name)

    def __add__(self, other: Self) -> Self:
        return self._binary(other, ADD)

    def __sub__(self, other: Self) -> Self:
        return self._binary(other, SUB)

    def __mul__(self, other: Self) -> Self:
        return self._binary(other, MUL)

    def __truediv__(self, other: Self) -> Self:
        """True division; always Float64, including for Int64 operands."""
        return self._binary(other, DIV)

    def __floordiv__(self, other: Self) -> Self:
        """Floor division; Int64 division by zero yields null."""
        return self._binary(other, FLOORDIV)

    def __mod__(self, other: Self) -> Self:
        """Remainder with the divisor's sign; Int64 modulo zero yields null."""
        return self._binary(other, MOD)

    def __pow__(self, other: Self) -> Self:
        return self._binary(other, POW)

    def pow(self, exponent: Self) -> Self:
        return self._binary(exponent, POW)

    def __neg__(self) -> Self:
        return self._unary(NEG)

    def __gt__(self, other: Self) -> Self:
        return self._binary(other, GT)

    def __lt__(self, other: Self) -> Self:
        return self._binary(other, LT)

    def __ge__(self, other: Self) -> Self:
        return self._binary(other, GE)

    def __le__(self, other: Self) -> Self:
        return self._binary(other, LE)

    def eq(self, other: Self) -> Self:
        return self._binary(other, EQ)

    def ne(self, other: Self) -> Self:
        return self._binary(other, NE)

    def abs(self) -> Self:
        return self._unary(ABS)

    def sqrt(self) -> Self:
        return self._unary(SQRT)

    def exp(self) -> Self:
        return self._unary(EXP)

    def log(self) -> Self:
        """Natural logarithm."""
        return self._unary(LOG)

    def floor(self) -> Self:
        return self._unary(FLOOR)

    def ceil(self) -> Self:
        return self._unary(CEIL)

    def round(self, decimals: Int = 0) -> Self:
        """Round half away from zero to `decimals` places."""
        return self._unary(ROUND, Int64(decimals))

    def clip(self, lower: Self, upper: Self) -> Self:
        """Bound values to [lower, upper]; nulls and NaN pass through."""
        return self._binary(lower, CLIP_LOW)._binary(upper, CLIP_HIGH)

    def clip_min(self, lower: Self) -> Self:
        return self._binary(lower, CLIP_LOW)

    def clip_max(self, upper: Self) -> Self:
        return self._binary(upper, CLIP_HIGH)

    def __and__(self, other: Self) -> Self:
        """Kleene AND: false wins over null; otherwise null propagates."""
        return self._binary(other, AND)

    def __or__(self, other: Self) -> Self:
        """Kleene OR: true wins over null; otherwise null propagates."""
        return self._binary(other, OR)

    def __xor__(self, other: Self) -> Self:
        return self._binary(other, XOR)

    def __invert__(self) -> Self:
        return self._unary(NOT)

    def and_(self, other: Self) -> Self:
        return self._binary(other, AND)

    def or_(self, other: Self) -> Self:
        return self._binary(other, OR)

    def xor(self, other: Self) -> Self:
        return self._binary(other, XOR)

    def not_(self) -> Self:
        return self._unary(NOT)

    def is_null(self) -> Self:
        return self._unary(IS_NULL)

    def is_not_null(self) -> Self:
        return self._unary(IS_NOT_NULL)

    def is_nan(self) -> Self:
        return self._unary(IS_NAN)

    def is_not_nan(self) -> Self:
        return self._unary(IS_NOT_NAN)

    def is_finite(self) -> Self:
        return self._unary(IS_FINITE)

    def is_infinite(self) -> Self:
        return self._unary(IS_INFINITE)

    def fill_null(self, value: Self) -> Self:
        """Replace nulls with value; the dtypes must match."""
        return self._binary(value, FILL_NULL)

    def fill_nan(self, value: Self) -> Self:
        """Replace valid NaNs in a Float64 expression with value."""
        return self._binary(value, FILL_NAN)

    def is_in(self, values: List[Self]) -> Self:
        """True when equal to any value; null input stays null.

        A null in `values` never matches. An empty list yields false.
        """
        var found = lit(False)
        for value in values:
            found = found._binary(
                self._binary(value, EQ)._binary(lit(False), FILL_NULL), OR
            )
        return self._binary(found, KEEP_NULLS)

    def is_between(
        self, lower: Self, upper: Self, closed: String = "both"
    ) raises -> Self:
        """lower <= x <= upper; `closed` is both, left, right, or none."""
        var low: Self
        var high: Self
        if closed == "both":
            low = self._binary(lower, GE)
            high = self._binary(upper, LE)
        elif closed == "left":
            low = self._binary(lower, GE)
            high = self._binary(upper, LT)
        elif closed == "right":
            low = self._binary(lower, GT)
            high = self._binary(upper, LE)
        elif closed == "none":
            low = self._binary(lower, GT)
            high = self._binary(upper, LT)
        else:
            raise Error("closed must be 'both', 'left', 'right', or 'none'")
        return low._binary(high, AND)

    def any(self, ignore_nulls: Bool = True) -> Self:
        """Any true value. With ignore_nulls=False, Kleene: null if no true
        and some null."""
        return self._unary(ANY, Int64(ignore_nulls))

    def all(self, ignore_nulls: Bool = True) -> Self:
        """All values true; empty is true. With ignore_nulls=False, Kleene:
        null if no false and some null."""
        return self._unary(ALL, Int64(ignore_nulls))

    def null_count(self) -> Self:
        """Number of null values, as Int64."""
        return self._unary(NULL_COUNT)

    def min(self) -> Self:
        """Smallest non-null value. NaN sorts above every number, so it is
        the minimum only when every valid value is NaN."""
        return self._unary(MIN)

    def max(self) -> Self:
        """Largest non-null value; any valid NaN makes the maximum NaN."""
        return self._unary(MAX)

    def mean(self) -> Self:
        """Arithmetic mean of non-null values as Float64; null when empty."""
        return self._unary(MEAN)

    def first(self) -> Self:
        """The first row's value, which may be null. Order-dependent."""
        return self._unary(FIRST)

    def last(self) -> Self:
        """The last row's value, which may be null. Order-dependent."""
        return self._unary(LAST)

    def n_unique(self) -> Self:
        """Distinct values, counting null once. NaNs are one value and
        -0.0 equals 0.0."""
        return self._unary(N_UNIQUE)

    def std(self, ddof: Int = 1) -> Self:
        """Standard deviation; null when fewer than ddof + 1 values."""
        return self._unary(STD, Int64(ddof))

    def var(self, ddof: Int = 1) -> Self:
        """Variance; null when fewer than ddof + 1 values."""
        return self._unary(VAR, Int64(ddof))

    def median(self) -> Self:
        return self._quantile(MEDIAN, 0.5, "linear")

    def quantile(
        self, quantile: Float64, interpolation: String = "linear"
    ) -> Self:
        """Interpolation: nearest, lower, higher, midpoint, or linear."""
        return self._quantile(QUANTILE, quantile, interpolation)

    def _quantile(
        self, op: Int, quantile: Float64, interpolation: String
    ) -> Self:
        var nodes = self._nodes.copy()
        nodes.append(
            _node(
                op,
                len(nodes) - 1,
                text=interpolation,
                floating=quantile,
            )
        )
        return Self(nodes^, self._name)

    def len(self) -> Self:
        """Number of rows including nulls, as Int64."""
        return self._unary(LEN)

    def sum(self, min_count: Int = 0) -> Self:
        """Skip nulls; zero when empty unless fewer than min_count are valid."""
        var nodes = self._nodes.copy()
        nodes.append(_node(SUM, len(nodes) - 1, min_count=min_count))
        return Self(nodes^, self._name)

    def count(self) -> Self:
        """Number of non-null values, as Int64."""
        return self._unary(COUNT)


def _append_shifted(mut nodes: List[Node], other: List[Node], offset: Int):
    for node in other:
        var copied = node.copy()
        if copied.left >= 0:
            copied.left += offset
        if copied.right >= 0:
            copied.right += offset
        if copied.extra >= 0:
            copied.extra += offset
        nodes.append(copied^)


def col(name: String) -> Expr:
    return Expr([_node(COL, text=name)], name)


def lit(value: Int64) -> Expr:
    return Expr([_node(LIT_INT, integer=value)], "literal")


def lit(value: Float64) -> Expr:
    return Expr([_node(LIT_FLOAT, floating=value)], "literal")


def lit(value: Bool) -> Expr:
    return Expr([_node(LIT_BOOL, integer=Int64(value))], "literal")


def lit(value: String) -> Expr:
    return Expr([_node(LIT_STRING, text=value)], "literal")


def null(dtype: String) -> Expr:
    """A typed null literal: int64, float64, bool, or string."""
    return Expr([_node(LIT_NULL, text=dtype)], "literal")


def coalesce(exprs: List[Expr]) raises -> Expr:
    """First non-null value per row, left to right; dtypes must match."""
    if len(exprs) == 0:
        raise Error("coalesce requires at least one expression")
    var result = exprs[0].copy()
    for i in range(1, len(exprs)):
        result = result.fill_null(exprs[i])
    return result^
