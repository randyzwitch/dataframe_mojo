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


@fieldwise_init
struct Node(Copyable):
    var op: Int
    var left: Int
    var right: Int
    var text: String
    var integer: Int64
    var floating: Float64
    var min_count: Int


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
        for node in other._nodes:
            var copied = node.copy()
            if copied.left >= 0:
                copied.left += offset
            if copied.right >= 0:
                copied.right += offset
            nodes.append(copied^)
        nodes.append(Node(op, offset - 1, len(nodes) - 1, "", 0, 0, 0))
        return Self(nodes^, self._name)

    def __add__(self, other: Self) -> Self:
        return self._binary(other, ADD)

    def __sub__(self, other: Self) -> Self:
        return self._binary(other, SUB)

    def __mul__(self, other: Self) -> Self:
        return self._binary(other, MUL)

    def __gt__(self, other: Self) -> Self:
        return self._binary(other, GT)

    def eq(self, other: Self) -> Self:
        return self._binary(other, EQ)

    def sum(self, min_count: Int = 0) -> Self:
        """Skip nulls; zero when empty unless fewer than min_count are valid."""
        var nodes = self._nodes.copy()
        nodes.append(Node(SUM, len(nodes) - 1, -1, "", 0, 0, min_count))
        return Self(nodes^, self._name)

    def count(self) -> Self:
        """Number of non-null values, as Int64."""
        var nodes = self._nodes.copy()
        nodes.append(Node(COUNT, len(nodes) - 1, -1, "", 0, 0, 0))
        return Self(nodes^, self._name)


def col(name: String) -> Expr:
    return Expr([Node(COL, -1, -1, name, 0, 0, 0)], name)


def lit(value: Int64) -> Expr:
    return Expr([Node(LIT_INT, -1, -1, "", value, 0, 0)], "literal")


def lit(value: Float64) -> Expr:
    return Expr([Node(LIT_FLOAT, -1, -1, "", 0, value, 0)], "literal")


def lit(value: Bool) -> Expr:
    return Expr([Node(LIT_BOOL, -1, -1, "", Int64(value), 0, 0)], "literal")


def lit(value: String) -> Expr:
    return Expr([Node(LIT_STRING, -1, -1, value, 0, 0, 0)], "literal")
