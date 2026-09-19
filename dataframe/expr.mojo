"""Immutable-by-convention expression IR. No data access or execution here."""
from std.collections import Optional
from .dtype import DataType

# Node text of an untyped numeric literal; the binder replaces it with the
# adopted dtype name.
comptime UNTYPED = "?"
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
# A column selector leaf, expanded into one expression per matched column
# before binding. text: kind, prefix, suffix joined by SEP; text2: SEP-joined
# names or dtypes; integer: nth index.
comptime SELECTOR = 13
comptime SEP = "\x1f"

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
# String concatenation with the separator in `text`; nulls propagate.
comptime STR_CONCAT = 36

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

# String operations (unary, parameters in text/text2/integer/min_count).
comptime STR_LEN_CHARS = 67
comptime STR_LEN_BYTES = 68
comptime STR_UPPER = 69
comptime STR_LOWER = 70
# integer: 0 both ends, 1 start, 2 end; text: characters ("" = whitespace).
comptime STR_STRIP = 71
comptime STR_STARTS_WITH = 72
comptime STR_ENDS_WITH = 73
comptime STR_CONTAINS = 74
# text: pattern, text2: replacement, integer: 1 replaces every occurrence.
comptime STR_REPLACE = 75
# integer: code point offset (negative from end), min_count: length (-1 all).
comptime STR_SLICE = 76
comptime STR_REVERSE = 77
# integer: 0 pad start, 1 pad end, 2 zfill; min_count: width; text: fill.
comptime STR_PAD = 78
# Cast: text holds the target dtype, integer is 1 for strict.
comptime CAST = 79

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

# Conditional: left=predicate, right=then, extra=otherwise (-1 means null).
comptime WHEN = 100

# Order-dependent window operations occupy 110..121. They are computed over
# the whole input column (per partition) before batches are evaluated.
# min_count: 1 reverses cumulative ops; integer: shift/limit/window size;
# floating: rolling min_samples; text: rank method.
comptime CUM_SUM = 110
comptime CUM_MIN = 111
comptime CUM_MAX = 112
comptime CUM_COUNT = 113
comptime SHIFT = 114
comptime RANK = 115
comptime ROLLING_SUM = 116
comptime ROLLING_MEAN = 117
comptime ROLLING_MIN = 118
comptime ROLLING_MAX = 119
comptime FORWARD_FILL = 120
comptime BACKWARD_FILL = 121
# Evaluate the child per partition of the SEP-joined key names in text2.
comptime OVER = 122

# Temporal field and conversion operations occupy 130..149 (text holds an
# interval, format, or unit; text2 a target dtype name).
comptime DT_YEAR = 130
comptime DT_MONTH = 131
comptime DT_DAY = 132
comptime DT_HOUR = 133
comptime DT_MINUTE = 134
comptime DT_SECOND = 135
comptime DT_NANOSECOND = 136
comptime DT_WEEKDAY = 137
comptime DT_ORDINAL_DAY = 138
comptime DT_DATE = 139
comptime DT_TIME = 140
comptime DT_TRUNCATE = 141
comptime DT_OFFSET_BY = 142
comptime DT_TOTAL = 143
comptime DT_STRFTIME = 144
comptime DT_STRPTIME = 145


def is_binary(op: Int) -> Bool:
    return (op >= ADD and op <= EQ) or (op >= 20 and op < 50)


def is_unary(op: Int) -> Bool:
    return op >= 50 and op < 80


def is_reduction(op: Int) -> Bool:
    return op == SUM or op == COUNT or (op >= 80 and op < 100)


def is_comparison(op: Int) -> Bool:
    return op == GT or op == EQ or (op >= LT and op <= NE)


def is_string_op(op: Int) -> Bool:
    return op >= STR_LEN_CHARS and op <= STR_PAD


def is_dt_op(op: Int) -> Bool:
    return op >= DT_YEAR and op <= DT_STRPTIME


def is_window(op: Int) -> Bool:
    return op >= CUM_SUM and op <= BACKWARD_FILL


def is_conditional(op: Int) -> Bool:
    return op == WHEN


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
    var text2: String


def _node(
    op: Int,
    left: Int = -1,
    right: Int = -1,
    text: String = "",
    integer: Int64 = 0,
    floating: Float64 = 0,
    min_count: Int = 0,
    extra: Int = -1,
    text2: String = "",
) -> Node:
    return Node(
        op, left, right, text, integer, floating, min_count, extra, text2
    )


@fieldwise_init
struct Expr(Copyable):
    """A flat, topologically ordered tree; composition never evaluates data."""

    var _nodes: List[Node]
    var _name: String

    @implicit
    def __init__(out self, then: Then):
        """A when/then chain without otherwise yields null for unmatched rows."""
        self = then.end()

    # Bare numbers and Bools convert to literals wherever an Expr is expected:
    # `col("x") > 0`, `col("x") * 2.5`, `.fill_null(0)`, `when(...).then(1)`,
    # `1 + col("x")`. Strings are not implicit (a list literal of strings
    # would be ambiguous between List[String] and List[Expr] overloads); the
    # comparison, fill_null, is_in, and then/otherwise methods take String
    # overloads instead, so `col("k") == "a"` still works. Numbers are *untyped*: at bind time
    # an integer adopts the other operand's numeric dtype (range-checked) and
    # a float adopts Float32/Float64; alone they default to Int64/Float64.
    # Use lit(...) to fix a type explicitly.

    @implicit
    def __init__(out self, value: Int):
        self = Expr(
            [_node(LIT_INT, integer=Int64(value), text=UNTYPED)], "literal"
        )

    @implicit
    def __init__(out self, value: Float64):
        self = Expr([_node(LIT_FLOAT, floating=value, text=UNTYPED)], "literal")

    @implicit
    def __init__(out self, value: Bool):
        self = lit(value)

    def alias(self, name: String) -> Self:
        var result = self.copy()
        result._name = name
        return result^

    def _rename_selector(self, prefix: String, suffix: String) -> Self:
        var result = self.copy()
        for i in range(len(result._nodes)):
            if result._nodes[i].op == SELECTOR:
                var parts = result._nodes[i].text.split(SEP)
                result._nodes[i].text = (
                    String(parts[0])
                    + SEP
                    + prefix
                    + String(parts[1])
                    + SEP
                    + String(parts[2])
                    + suffix
                )
                return result^
        result._name = prefix + result._name + suffix
        return result^

    def name_prefix(self, prefix: String) -> Self:
        """Prefix the output name; for selectors, every expanded name."""
        return self._rename_selector(prefix, "")

    def name_suffix(self, suffix: String) -> Self:
        """Suffix the output name; for selectors, every expanded name."""
        return self._rename_selector("", suffix)

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

    # Reflected forms, so a bare number can come first: `1 + col("x")`,
    # `10 - col("x")`, `2 ** col("n")`. The literal is the left operand.

    def __radd__(self, other: Self) -> Self:
        return other._binary(self, ADD)

    def __rsub__(self, other: Self) -> Self:
        return other._binary(self, SUB)

    def __rmul__(self, other: Self) -> Self:
        return other._binary(self, MUL)

    def __rtruediv__(self, other: Self) -> Self:
        return other._binary(self, DIV)

    def __rfloordiv__(self, other: Self) -> Self:
        return other._binary(self, FLOORDIV)

    def __rmod__(self, other: Self) -> Self:
        return other._binary(self, MOD)

    def __rpow__(self, other: Self) -> Self:
        return other._binary(self, POW)

    def __rand__(self, other: Self) -> Self:
        return other._binary(self, AND)

    def __ror__(self, other: Self) -> Self:
        return other._binary(self, OR)

    def __rxor__(self, other: Self) -> Self:
        return other._binary(self, XOR)

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

    def __eq__(self, other: Self) -> Self:
        """Elementwise equality (an expression, not a Bool); same as eq."""
        return self.eq(other)

    def __ne__(self, other: Self) -> Self:
        """Elementwise inequality (an expression, not a Bool); same as ne."""
        return self.ne(other)

    def __eq__(self, other: String) -> Self:
        return self.eq(lit(other))

    def __ne__(self, other: String) -> Self:
        return self.ne(lit(other))

    def __lt__(self, other: String) -> Self:
        return self < lit(other)

    def __le__(self, other: String) -> Self:
        return self <= lit(other)

    def __gt__(self, other: String) -> Self:
        return self > lit(other)

    def __ge__(self, other: String) -> Self:
        return self >= lit(other)

    def eq(self, other: String) -> Self:
        return self.eq(lit(other))

    def ne(self, other: String) -> Self:
        return self.ne(lit(other))

    def fill_null(self, value: String) -> Self:
        return self.fill_null(lit(value))

    def is_in(self, values: List[String]) -> Self:
        var literals = List[Expr](capacity=len(values))
        for value in values:
            literals.append(lit(value))
        return self.is_in(literals)

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
        """True when lower <= x <= upper; `closed` is both, left, right, or none."""
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

    def cast(self, dtype: DataType, strict: Bool = True) -> Self:
        """Convert to dtype; see the String overload."""
        return self.cast(dtype.name(), strict)

    def cast(self, dtype: String, strict: Bool = True) -> Self:
        """Convert to any dtype by name (numeric, bool, string, temporal).

        Integer targets are range-checked and floats truncate toward zero.

        Values that cannot convert raise (with the row and value) when
        strict, or become null otherwise. Nulls stay null.
        """
        var nodes = self._nodes.copy()
        nodes.append(
            _node(CAST, len(nodes) - 1, text=dtype, integer=Int64(strict))
        )
        return Self(nodes^, self._name)

    def _window(
        self,
        op: Int,
        integer: Int64 = 0,
        reverse: Bool = False,
        text: String = "",
        floating: Float64 = 0,
    ) -> Self:
        var nodes = self._nodes.copy()
        nodes.append(
            _node(
                op,
                len(nodes) - 1,
                text=text,
                integer=integer,
                floating=floating,
                min_count=Int(reverse),
            )
        )
        return Self(nodes^, self._name)

    def cum_sum(self, reverse: Bool = False) -> Self:
        """Running sum of non-null values; null rows stay null. Int64 is
        checked for overflow."""
        return self._window(CUM_SUM, reverse=reverse)

    def cum_min(self, reverse: Bool = False) -> Self:
        return self._window(CUM_MIN, reverse=reverse)

    def cum_max(self, reverse: Bool = False) -> Self:
        return self._window(CUM_MAX, reverse=reverse)

    def cum_count(self, reverse: Bool = False) -> Self:
        """Running count of non-null values, as Int64 (never null)."""
        return self._window(CUM_COUNT, reverse=reverse)

    def shift(self, n: Int = 1) -> Self:
        """Move values n rows later (earlier when negative); vacated rows
        are null."""
        return self._window(SHIFT, Int64(n))

    def diff(self, n: Int = 1) -> Self:
        """Difference from the value n rows earlier."""
        return self - self.shift(n)

    def pct_change(self, n: Int = 1) -> Self:
        """Relative change from the value n rows earlier, as Float64."""
        var previous = self.shift(n)
        return (self - previous) / previous

    def rank(
        self, method: String = "average", descending: Bool = False
    ) -> Self:
        """Rank non-null values: average, min, max, dense, or ordinal.

        Ranks start at 1; average gives Float64, the rest Int64. Ordinal
        breaks ties by row order. NaN ranks after numbers, as in sort.
        """
        return self._window(RANK, reverse=descending, text=method)

    def rolling_sum(self, window_size: Int, min_samples: Int = -1) -> Self:
        """Sum over the current row and the window_size - 1 rows before it.

        Nulls are skipped; fewer than min_samples valid values (default
        window_size) give null.
        """
        return self._window(
            ROLLING_SUM, Int64(window_size), floating=Float64(min_samples)
        )

    def rolling_mean(self, window_size: Int, min_samples: Int = -1) -> Self:
        return self._window(
            ROLLING_MEAN, Int64(window_size), floating=Float64(min_samples)
        )

    def rolling_min(self, window_size: Int, min_samples: Int = -1) -> Self:
        return self._window(
            ROLLING_MIN, Int64(window_size), floating=Float64(min_samples)
        )

    def rolling_max(self, window_size: Int, min_samples: Int = -1) -> Self:
        return self._window(
            ROLLING_MAX, Int64(window_size), floating=Float64(min_samples)
        )

    def forward_fill(self, limit: Int = -1) -> Self:
        """Fill nulls with the last valid value, at most limit rows ahead
        (-1 means unlimited)."""
        return self._window(FORWARD_FILL, Int64(limit))

    def backward_fill(self, limit: Int = -1) -> Self:
        return self._window(BACKWARD_FILL, Int64(limit))

    def over(self, partition_by: String) -> Self:
        return self.over([partition_by])

    def over(self, partition_by: List[String]) -> Self:
        """Evaluate within partitions of the key columns, keeping row order.

        Aggregates broadcast back to every row of their partition; window
        operations restart in each partition. Nulls form their own key.
        """
        var nodes = self._nodes.copy()
        nodes.append(_node(OVER, len(nodes) - 1, text2=_joined(partition_by)))
        return Self(nodes^, self._name)

    def str(self) -> StrNamespace:
        """String operations: col("name").str().to_uppercase()."""
        return StrNamespace(self.copy())

    def dt(self) -> DtNamespace:
        """Temporal operations: col("when").dt().year()."""
        return DtNamespace(self.copy())

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


def _selector(kind: String, items: String = "", index: Int = 0) -> Expr:
    """Selector expressions start unnamed; expansion names each output."""
    return Expr(
        [
            _node(
                SELECTOR,
                text=kind + SEP + SEP,
                text2=items,
                integer=Int64(index),
            )
        ],
        "",
    )


def _joined(items: List[String]) -> String:
    var out = String()
    for i in range(len(items)):
        if i > 0:
            out += SEP
        out += items[i]
    return out^


def col(names: List[String]) -> Expr:
    """Select several columns; operations apply to each one."""
    return _selector("cols", _joined(names))


def all() -> Expr:
    """Every column, in schema order."""
    return _selector("all")


def exclude(names: List[String]) -> Expr:
    """Every column except the listed ones, in schema order."""
    return _selector("exclude", _joined(names))


def by_dtype(dtypes: List[String]) -> Expr:
    """Columns whose dtype is listed, in schema order."""
    return _selector("dtype", _joined(dtypes))


def nth(index: Int) -> Expr:
    """The column at a position; negative positions count from the end."""
    return _selector("nth", "", index)


def first() -> Expr:
    """The first column in the schema."""
    return nth(0)


def last() -> Expr:
    """The last column in the schema."""
    return nth(-1)


def lit(value: Int64) -> Expr:
    """A typed scalar literal; there is no implicit numeric promotion."""
    return Expr([_node(LIT_INT, integer=value)], "literal")


def lit(value: Float64) -> Expr:
    return Expr([_node(LIT_FLOAT, floating=value)], "literal")


def _numeric_lit[D: DType](value: Scalar[D]) -> Expr:
    """A literal of a non-default numeric type. The node's text names the
    type; integers are held in `integer` (UInt64 by bit pattern) and floats
    in `floating` (exactly)."""
    comptime if D.is_floating_point():
        return Expr(
            [
                _node(
                    LIT_FLOAT,
                    floating=value.cast[DType.float64](),
                    text=String(D),
                )
            ],
            "literal",
        )
    else:
        return Expr(
            [_node(LIT_INT, integer=value.cast[DType.int64](), text=String(D))],
            "literal",
        )


def lit[D: DType](value: Scalar[D]) -> Expr:
    """A literal of any numeric type, for code generic over DType."""
    return _numeric_lit(value)


def lit(value: Int8) -> Expr:
    return _numeric_lit(value)


def lit(value: Int16) -> Expr:
    return _numeric_lit(value)


def lit(value: Int32) -> Expr:
    return _numeric_lit(value)


def lit(value: UInt8) -> Expr:
    return _numeric_lit(value)


def lit(value: UInt16) -> Expr:
    return _numeric_lit(value)


def lit(value: UInt32) -> Expr:
    return _numeric_lit(value)


def lit(value: UInt64) -> Expr:
    return _numeric_lit(value)


def lit(value: Float32) -> Expr:
    return _numeric_lit(value)


def lit(value: Bool) -> Expr:
    return Expr([_node(LIT_BOOL, integer=Int64(value))], "literal")


def lit(value: String) -> Expr:
    return Expr([_node(LIT_STRING, text=value)], "literal")


def null(dtype: String) -> Expr:
    """A typed null literal of any dtype name (see DataType.parse)."""
    return Expr([_node(LIT_NULL, text=dtype)], "literal")


def coalesce(exprs: List[Expr]) raises -> Expr:
    """First non-null value per row, left to right; dtypes must match."""
    if len(exprs) == 0:
        raise Error("coalesce requires at least one expression")
    var result = exprs[0].copy()
    for i in range(1, len(exprs)):
        result = result.fill_null(exprs[i])
    return result^


def _conditional(
    condition: Expr, value: Expr, otherwise: Optional[Expr]
) -> Expr:
    var nodes = condition._nodes.copy()
    var predicate = len(nodes) - 1
    _append_shifted(nodes, value._nodes, len(nodes))
    var then = len(nodes) - 1
    var other = -1
    if otherwise:
        _append_shifted(nodes, otherwise.value()._nodes, len(nodes))
        other = len(nodes) - 1
    nodes.append(_node(WHEN, predicate, then, extra=other))
    return Expr(nodes^, value._name)


@fieldwise_init
struct When(Copyable):
    """A pending condition; call `then` to supply its value."""

    var _conditions: List[Expr]
    var _values: List[Expr]

    def then(self, value: String) -> Then:
        return self.then(lit(value))

    def then(self, value: Expr) -> Then:
        var values = self._values.copy()
        values.append(value.copy())
        return Then(self._conditions.copy(), values^)


@fieldwise_init
struct Then(Copyable):
    """A when/then chain; add branches with `when` or finish with `otherwise`.

    It converts implicitly to an Expr whose unmatched rows are null.
    """

    var _conditions: List[Expr]
    var _values: List[Expr]

    def when(self, condition: Expr) -> When:
        var conditions = self._conditions.copy()
        conditions.append(condition.copy())
        return When(conditions^, self._values.copy())

    def otherwise(self, value: String) -> Expr:
        return self.otherwise(lit(value))

    def otherwise(self, value: Expr) -> Expr:
        return self._build(Optional[Expr](value.copy()))

    def end(self) -> Expr:
        return self._build(Optional[Expr]())

    def alias(self, name: String) -> Expr:
        return self.end().alias(name)

    def _build(self, otherwise: Optional[Expr]) -> Expr:
        var last = len(self._conditions) - 1
        var result = _conditional(
            self._conditions[last], self._values[last], otherwise
        )
        for i in range(last - 1, -1, -1):
            result = _conditional(
                self._conditions[i],
                self._values[i],
                Optional[Expr](result^),
            )
        # The chain is named after its first value, as in Polars.
        result._name = self._values[0]._name
        return result^


def when(condition: Expr) -> When:
    """Start a conditional: when(p).then(a).when(q).then(b).otherwise(c).

    Predicates must be Bool; a null predicate falls through to the next
    branch. Every branch value must share one dtype.
    """
    return When([condition.copy()], List[Expr]())


@fieldwise_init
struct StrNamespace(Copyable):
    """String expressions. Character operations work on Unicode code points;
    there is no grapheme clustering or locale-specific case mapping. Nulls
    propagate. Patterns are literal text; regular expressions are not
    supported."""

    var _expr: Expr

    def _op(
        self,
        op: Int,
        text: String = "",
        integer: Int64 = 0,
        min_count: Int = 0,
        text2: String = "",
    ) -> Expr:
        var nodes = self._expr._nodes.copy()
        nodes.append(
            _node(
                op,
                len(nodes) - 1,
                text=text,
                integer=integer,
                min_count=min_count,
                text2=text2,
            )
        )
        return Expr(nodes^, self._expr._name)

    def len_chars(self) -> Expr:
        """Number of Unicode code points, as Int64."""
        return self._op(STR_LEN_CHARS)

    def len_bytes(self) -> Expr:
        """Number of UTF-8 bytes, as Int64."""
        return self._op(STR_LEN_BYTES)

    def to_uppercase(self) -> Expr:
        return self._op(STR_UPPER)

    def to_lowercase(self) -> Expr:
        return self._op(STR_LOWER)

    def strip_chars(self, characters: String = "") -> Expr:
        """Strip characters from both ends; empty means ASCII whitespace."""
        return self._op(STR_STRIP, characters, 0)

    def strip_chars_start(self, characters: String = "") -> Expr:
        return self._op(STR_STRIP, characters, 1)

    def strip_chars_end(self, characters: String = "") -> Expr:
        return self._op(STR_STRIP, characters, 2)

    def starts_with(self, prefix: String) -> Expr:
        return self._op(STR_STARTS_WITH, prefix)

    def ends_with(self, suffix: String) -> Expr:
        return self._op(STR_ENDS_WITH, suffix)

    def contains(self, literal: String) -> Expr:
        """Literal substring test; regular expressions are not supported."""
        return self._op(STR_CONTAINS, literal)

    def replace(self, pattern: String, value: String) -> Expr:
        """Replace the first occurrence of a literal pattern."""
        return self._op(STR_REPLACE, pattern, 0, text2=value)

    def replace_all(self, pattern: String, value: String) -> Expr:
        return self._op(STR_REPLACE, pattern, 1, text2=value)

    def slice(self, offset: Int, length: Int = -1) -> Expr:
        """Code points [offset, offset + length); a negative offset counts
        from the end and length=-1 takes the rest. Out-of-range parts clip."""
        return self._op(STR_SLICE, "", Int64(offset), length)

    def head(self, n: Int) -> Expr:
        return self.slice(0, n)

    def tail(self, n: Int) -> Expr:
        return self.slice(-n) if n > 0 else self.slice(0, 0)

    def reverse(self) -> Expr:
        """Reverse code point order."""
        return self._op(STR_REVERSE)

    def pad_start(self, width: Int, fill_char: String = " ") -> Expr:
        """Left-pad to width code points; longer strings are unchanged."""
        return self._op(STR_PAD, fill_char, 0, width)

    def pad_end(self, width: Int, fill_char: String = " ") -> Expr:
        return self._op(STR_PAD, fill_char, 1, width)

    def strptime(
        self, dtype: String, format: String = "", strict: Bool = True
    ) -> Expr:
        """Parse text as "date", "datetime[unit]", or "time" using a
        strftime-style format (ISO 8601 when empty); unparseable text raises
        when strict, or is null otherwise."""
        return self._op(DT_STRPTIME, format, Int64(strict), text2=dtype)

    def to_date(self, format: String = "") -> Expr:
        return self.strptime("date", format)

    def to_datetime(self, format: String = "", unit: String = "us") -> Expr:
        return self.strptime("datetime[" + unit + "]", format)

    def zfill(self, width: Int) -> Expr:
        """Left-pad with zeros, after a leading + or - sign."""
        return self._op(STR_PAD, "0", 2, width)


def concat_str(exprs: List[Expr], separator: String = "") raises -> Expr:
    """Join String expressions row-wise; any null input makes the row null."""
    if len(exprs) == 0:
        raise Error("concat_str requires at least one expression")
    var result = exprs[0].copy()
    for i in range(1, len(exprs)):
        var nodes = result._nodes.copy()
        var offset = len(nodes)
        _append_shifted(nodes, exprs[i]._nodes, offset)
        nodes.append(
            _node(STR_CONCAT, offset - 1, len(nodes) - 1, text=separator)
        )
        result = Expr(nodes^, result._name)
    return result^


def subtree(expr: Expr, root: Int) -> Expr:
    """The nodes reachable from root, renumbered, as a standalone Expr."""
    var keep = List[Bool](length=root + 1, fill=False)
    keep[root] = True
    for reverse in range(root + 1):
        var i = root - reverse
        if not keep[i]:
            continue
        ref node = expr._nodes[i]
        if node.left >= 0:
            keep[node.left] = True
        if node.right >= 0:
            keep[node.right] = True
        if node.extra >= 0:
            keep[node.extra] = True
    var position = List[Int](length=root + 1, fill=-1)
    var nodes = List[Node]()
    for i in range(root + 1):
        if keep[i]:
            position[i] = len(nodes)
            var copied = expr._nodes[i].copy()
            if copied.left >= 0:
                copied.left = position[copied.left]
            if copied.right >= 0:
                copied.right = position[copied.right]
            if copied.extra >= 0:
                copied.extra = position[copied.extra]
            nodes.append(copied^)
    return Expr(nodes^, expr._name)


@fieldwise_init
struct DtNamespace(Copyable):
    """Temporal operations on date, datetime, time, and duration expressions.

    Fields use the proleptic Gregorian calendar with no time zones. Nulls
    propagate.
    """

    var _expr: Expr

    def _op(self, op: Int, text: String = "", text2: String = "") -> Expr:
        var nodes = self._expr._nodes.copy()
        nodes.append(_node(op, len(nodes) - 1, text=text, text2=text2))
        return Expr(nodes^, self._expr._name)

    def year(self) -> Expr:
        return self._op(DT_YEAR)

    def month(self) -> Expr:
        return self._op(DT_MONTH)

    def day(self) -> Expr:
        return self._op(DT_DAY)

    def hour(self) -> Expr:
        return self._op(DT_HOUR)

    def minute(self) -> Expr:
        return self._op(DT_MINUTE)

    def second(self) -> Expr:
        return self._op(DT_SECOND)

    def nanosecond(self) -> Expr:
        """Nanoseconds within the second."""
        return self._op(DT_NANOSECOND)

    def weekday(self) -> Expr:
        """ISO weekday: Monday is 1, Sunday is 7."""
        return self._op(DT_WEEKDAY)

    def ordinal_day(self) -> Expr:
        """Day of the year, starting at 1."""
        return self._op(DT_ORDINAL_DAY)

    def date(self) -> Expr:
        """The calendar date of a datetime."""
        return self._op(DT_DATE)

    def time(self) -> Expr:
        """The time of day of a datetime."""
        return self._op(DT_TIME)

    def truncate(self, every: String) -> Expr:
        """Round down to a multiple of every, such as "1d", "15m", "1w"
        (Monday-aligned), "1mo", "3mo", or "1y" (calendar-aligned)."""
        return self._op(DT_TRUNCATE, every)

    def offset_by(self, by: String) -> Expr:
        """Shift by an interval such as "2d", "-3h", "1mo" or "1y2mo";
        month shifts clamp to the last day of the target month."""
        return self._op(DT_OFFSET_BY, by)

    def total(self, unit: String) -> Expr:
        """A duration as a whole number of days, hours, minutes, seconds,
        milliseconds, microseconds, or nanoseconds (truncated), as Int64."""
        return self._op(DT_TOTAL, unit)

    def total_days(self) -> Expr:
        return self.total("days")

    def total_hours(self) -> Expr:
        return self.total("hours")

    def total_minutes(self) -> Expr:
        return self.total("minutes")

    def total_seconds(self) -> Expr:
        return self.total("seconds")

    def total_milliseconds(self) -> Expr:
        return self.total("milliseconds")

    def strftime(self, format: String) -> Expr:
        """Format as text; see dataframe/temporal.mojo for directives."""
        return self._op(DT_STRFTIME, format)
