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

# Unary operations occupy 50..79; ROUND keeps its decimals in `integer`.
comptime NEG = 50
comptime ABS = 51
comptime SQRT = 52
comptime EXP = 53
comptime LOG = 54
comptime FLOOR = 55
comptime CEIL = 56
comptime ROUND = 57


def is_binary(op: Int) -> Bool:
    return (op >= ADD and op <= EQ) or (op >= 20 and op < 50)


def is_unary(op: Int) -> Bool:
    return op >= 50 and op < 80


def is_reduction(op: Int) -> Bool:
    return op == SUM or op == COUNT or (op >= 80 and op < 100)


def is_comparison(op: Int) -> Bool:
    return op == GT or op == EQ or (op >= LT and op <= NE)


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
