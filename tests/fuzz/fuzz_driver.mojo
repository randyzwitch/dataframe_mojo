"""Run one differential fuzz case: read a plan, execute it every way.

Usage: fuzz_driver CASE_FILE OUTPUT_DIR [MODE ...]

The case file is one s-expression written by `tests/fuzz/fuzz.py`:

    (case (inputs (in FORMAT PATH (schema (f NAME DTYPE) ...)) ...) PLAN)

Atoms are bare symbols, `i:<int>`, `f:<float>`, `b:0|1` and `s:<hex utf-8>`
(strings are hex so no quoting is needed). PLAN is a tree of frame
operations whose leaves are `(src INDEX)`; expressions are trees too. See
fuzz.py for the grammar it emits.

Each mode writes `OUTPUT_DIR/<mode>.arrow` (the result as Arrow IPC) or
`OUTPUT_DIR/<mode>.err` (the raised error's text):

- eager: inputs read eagerly, every operation run on DataFrame.
- lazy: inputs scanned from their files, one lazy plan, collect().
- lazy_noopt: the same plan, collect(optimize=False).
- lazy_mem: inputs read eagerly, then `.lazy()`, collect().

One program serves every case, so a fuzz run compiles Mojo once.
"""

from std.sys import argv
from std.collections import Optional

from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    col,
    concat,
    concat_list,
    element,
    len as row_len,
    lit,
    null,
    read_csv,
    read_ipc,
    read_parquet,
    scan_csv,
    scan_ipc,
    scan_parquet,
    when,
    write_ipc,
)

comptime LIST = 0
comptime SYM = 1
comptime INT = 2
comptime FLOAT = 3
comptime STR = 4
comptime BOOL = 5


@fieldwise_init
struct Node(Copyable, Movable):
    var tag: Int
    var text: String
    var ival: Int
    var fval: Float64
    var kids: List[Int]


def _hex_value(c: Int) raises -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    raise Error("bad hex digit")


def _unhex(text: String) raises -> String:
    var bytes = text.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i + 1 < len(bytes):
        out.append(
            UInt8(
                _hex_value(Int(bytes[i])) * 16 + _hex_value(Int(bytes[i + 1]))
            )
        )
        i += 2
    return String(unsafe_from_utf8=out)


struct Tree(Movable):
    """A parsed s-expression, stored flat so no type contains itself."""

    var nodes: List[Node]

    def __init__(out self, source: String) raises:
        self.nodes = List[Node]()
        var tokens = List[String]()
        var current = String("")
        for cp in source.codepoints():
            var ch = String(cp)
            if ch == "(" or ch == ")":
                if current.byte_length() > 0:
                    tokens.append(current)
                    current = String("")
                tokens.append(ch)
            elif ch == " " or ch == "\n" or ch == "\t" or ch == "\r":
                if current.byte_length() > 0:
                    tokens.append(current)
                    current = String("")
            else:
                current += ch
        if current.byte_length() > 0:
            tokens.append(current)
        var stack = List[Int]()
        for t in tokens:
            if t == "(":
                self.nodes.append(Node(LIST, "", 0, 0.0, List[Int]()))
                var id = len(self.nodes) - 1
                if len(stack) > 0:
                    self.nodes[stack[len(stack) - 1]].kids.append(id)
                stack.append(id)
            elif t == ")":
                _ = stack.pop()
            else:
                var node: Node
                if t.startswith("i:"):
                    node = Node(
                        INT, t, Int(String(t[byte=2:])), 0.0, List[Int]()
                    )
                elif t.startswith("f:"):
                    node = Node(
                        FLOAT, t, 0, Float64(String(t[byte=2:])), List[Int]()
                    )
                elif t.startswith("s:"):
                    node = Node(
                        STR, _unhex(String(t[byte=2:])), 0, 0.0, List[Int]()
                    )
                elif t.startswith("b:"):
                    node = Node(
                        BOOL, t, 1 if t == "b:1" else 0, 0.0, List[Int]()
                    )
                else:
                    node = Node(SYM, t, 0, 0.0, List[Int]())
                self.nodes.append(node^)
                if len(stack) > 0:
                    self.nodes[stack[len(stack) - 1]].kids.append(
                        len(self.nodes) - 1
                    )

    def head(self, i: Int) -> String:
        """The symbol naming a list node (its first child)."""
        if self.nodes[i].tag != LIST or len(self.nodes[i].kids) == 0:
            return ""
        return self.nodes[self.nodes[i].kids[0]].text

    def kid(self, i: Int, k: Int) -> Int:
        return self.nodes[i].kids[k]

    def nkids(self, i: Int) -> Int:
        return len(self.nodes[i].kids)

    def text(self, i: Int) -> String:
        return self.nodes[i].text

    def int(self, i: Int) -> Int:
        return self.nodes[i].ival

    def flag(self, i: Int) -> Bool:
        return self.nodes[i].ival != 0

    def strings(self, i: Int) -> List[String]:
        """The string atoms of a `(list ...)` node."""
        var out = List[String]()
        for k in range(1, self.nkids(i)):
            out.append(self.text(self.kid(i, k)))
        return out^

    def floats(self, i: Int) -> List[Float64]:
        """The float atoms of a `(list ...)` node."""
        var out = List[Float64]()
        for k in range(1, self.nkids(i)):
            out.append(self.nodes[self.kid(i, k)].fval)
        return out^

    def flags(self, i: Int) -> List[Bool]:
        var out = List[Bool]()
        for k in range(1, self.nkids(i)):
            out.append(self.flag(self.kid(i, k)))
        return out^


def ea(t: Tree, i: Int, k: Int) raises -> Expr:
    """The expression in child k of node i."""
    return build_expr(t, t.kid(i, k))


def build_expr(t: Tree, i: Int) raises -> Expr:
    var node = t.nodes[i].copy()
    if node.tag == INT:
        return Expr(node.ival)
    if node.tag == FLOAT:
        return Expr(node.fval)
    if node.tag == BOOL:
        return Expr(node.ival != 0)
    if node.tag == STR:
        return lit(node.text)
    var op = t.head(i)
    var n = t.nkids(i)

    if op == "col":
        return col(t.text(t.kid(i, 1)))
    if op == "lit":
        # A typed literal: (lit DTYPE VALUE).
        var dt = DataType.parse(t.text(t.kid(i, 1)))
        var v = t.kid(i, 2)
        if t.nodes[v].tag == FLOAT:
            if dt == DataType.FLOAT32:
                return lit(Float32(t.nodes[v].fval))
            return lit(t.nodes[v].fval)
        if t.nodes[v].tag == STR:
            return lit(t.text(v))
        if t.nodes[v].tag == BOOL:
            return lit(t.flag(v))
        return lit(Int64(t.int(v)), dt)
    if op == "null":
        return null(t.text(t.kid(i, 1)))
    if op == "len":
        return row_len()
    if op == "alias":
        return ea(t, i, 1).alias(t.text(t.kid(i, 2)))
    if op == "cast":
        return ea(t, i, 1).cast(t.text(t.kid(i, 2)), strict=t.flag(t.kid(i, 3)))
    # Binary operators.
    if op == "add":
        return ea(t, i, 1) + ea(t, i, 2)
    if op == "sub":
        return ea(t, i, 1) - ea(t, i, 2)
    if op == "mul":
        return ea(t, i, 1) * ea(t, i, 2)
    if op == "truediv":
        return ea(t, i, 1) / ea(t, i, 2)
    if op == "floordiv":
        return ea(t, i, 1) // ea(t, i, 2)
    if op == "mod":
        return ea(t, i, 1) % ea(t, i, 2)
    if op == "eq":
        return ea(t, i, 1).eq(ea(t, i, 2))
    if op == "ne":
        return ea(t, i, 1).ne(ea(t, i, 2))
    if op == "lt":
        return ea(t, i, 1) < ea(t, i, 2)
    if op == "le":
        return ea(t, i, 1) <= ea(t, i, 2)
    if op == "gt":
        return ea(t, i, 1) > ea(t, i, 2)
    if op == "ge":
        return ea(t, i, 1) >= ea(t, i, 2)
    if op == "eq_missing":
        return ea(t, i, 1).eq_missing(ea(t, i, 2))
    if op == "and":
        return ea(t, i, 1) & ea(t, i, 2)
    if op == "or":
        return ea(t, i, 1) | ea(t, i, 2)
    if op == "xor":
        return ea(t, i, 1) ^ ea(t, i, 2)
    if op == "fill_null":
        return ea(t, i, 1).fill_null(ea(t, i, 2))
    if op == "fill_nan":
        return ea(t, i, 1).fill_nan(ea(t, i, 2))
    if op == "clip":
        return ea(t, i, 1).clip(ea(t, i, 2), build_expr(t, t.kid(i, 3)))
    if op == "is_between":
        return ea(t, i, 1).is_between(ea(t, i, 2), build_expr(t, t.kid(i, 3)))
    if op == "when":
        # (when COND THEN [COND THEN]... OTHERWISE)
        var chain = when(build_expr(t, t.kid(i, 1))).then(
            build_expr(t, t.kid(i, 2))
        )
        var k = 3
        while k + 1 < n:
            chain = chain.when(build_expr(t, t.kid(i, k))).then(
                build_expr(t, t.kid(i, k + 1))
            )
            k += 2
        return chain.otherwise(build_expr(t, t.kid(i, n - 1)))
    if op == "is_in":
        var values = List[Expr]()
        var lst = t.kid(i, 2)
        for k in range(1, t.nkids(lst)):
            values.append(build_expr(t, t.kid(lst, k)))
        return ea(t, i, 1).is_in(values)
    # Unary operators.
    if op == "neg":
        return -ea(t, i, 1)
    if op == "not":
        return ~ea(t, i, 1)
    if op == "abs":
        return ea(t, i, 1).abs()
    if op == "is_null":
        return ea(t, i, 1).is_null()
    if op == "is_not_null":
        return ea(t, i, 1).is_not_null()
    if op == "is_nan":
        return ea(t, i, 1).is_nan()
    if op == "is_not_nan":
        return ea(t, i, 1).is_not_nan()
    if op == "sqrt":
        return ea(t, i, 1).sqrt()
    if op == "floor":
        return ea(t, i, 1).floor()
    if op == "ceil":
        return ea(t, i, 1).ceil()
    if op == "round":
        return ea(t, i, 1).round(t.int(t.kid(i, 2)))
    if op == "sign":
        return ea(t, i, 1).sign()
    # Reductions.
    if op == "sum":
        return ea(t, i, 1).sum()
    if op == "mean":
        return ea(t, i, 1).mean()
    if op == "min":
        return ea(t, i, 1).min()
    if op == "max":
        return ea(t, i, 1).max()
    if op == "median":
        return ea(t, i, 1).median()
    if op == "count":
        return ea(t, i, 1).count()
    if op == "null_count":
        return ea(t, i, 1).null_count()
    if op == "n_unique":
        return ea(t, i, 1).n_unique()
    if op == "first":
        return ea(t, i, 1).first()
    if op == "last":
        return ea(t, i, 1).last()
    if op == "std":
        return ea(t, i, 1).std(t.int(t.kid(i, 2)))
    if op == "var":
        return ea(t, i, 1).var(t.int(t.kid(i, 2)))
    if op == "elen":
        return ea(t, i, 1).len()
    if op == "any":
        return ea(t, i, 1).any()
    if op == "all":
        return ea(t, i, 1).all()
    if op == "arg_min":
        return ea(t, i, 1).arg_min()
    if op == "arg_max":
        return ea(t, i, 1).arg_max()
    if op == "quantile":
        return ea(t, i, 1).quantile(
            t.nodes[t.kid(i, 2)].fval, t.text(t.kid(i, 3))
        )
    # Order-dependent and window expressions.
    if op == "cum_sum":
        return ea(t, i, 1).cum_sum(t.flag(t.kid(i, 2)))
    if op == "cum_max":
        return ea(t, i, 1).cum_max(t.flag(t.kid(i, 2)))
    if op == "cum_min":
        return ea(t, i, 1).cum_min(t.flag(t.kid(i, 2)))
    if op == "cum_count":
        return ea(t, i, 1).cum_count(t.flag(t.kid(i, 2)))
    if op == "shift":
        return ea(t, i, 1).shift(t.int(t.kid(i, 2)))
    if op == "diff":
        return ea(t, i, 1).diff(t.int(t.kid(i, 2)))
    if op == "rank":
        return ea(t, i, 1).rank(t.text(t.kid(i, 2)), t.flag(t.kid(i, 3)))
    if op == "over":
        # (over EXPR (list KEYS...) (list ORDER...) DESC NULLS_LAST)
        var keys = t.strings(t.kid(i, 2))
        var order = t.strings(t.kid(i, 3))
        if len(order) == 0:
            return ea(t, i, 1).over(keys)
        return ea(t, i, 1).over(
            keys,
            order,
            descending=t.flag(t.kid(i, 4)),
            nulls_last=t.flag(t.kid(i, 5)),
        )
    # Namespaces.
    if op == "str.len_chars":
        return ea(t, i, 1).str().len_chars()
    if op == "str.len_bytes":
        return ea(t, i, 1).str().len_bytes()
    if op == "str.to_uppercase":
        return ea(t, i, 1).str().to_uppercase()
    if op == "str.to_lowercase":
        return ea(t, i, 1).str().to_lowercase()
    if op == "str.starts_with":
        return ea(t, i, 1).str().starts_with(t.text(t.kid(i, 2)))
    if op == "str.ends_with":
        return ea(t, i, 1).str().ends_with(t.text(t.kid(i, 2)))
    if op == "str.contains":
        return ea(t, i, 1).str().contains(t.text(t.kid(i, 2)), literal=True)
    if op == "str.slice":
        return ea(t, i, 1).str().slice(t.int(t.kid(i, 2)), t.int(t.kid(i, 3)))
    if op == "str.replace_all":
        return (
            ea(t, i, 1)
            .str()
            .replace_all(t.text(t.kid(i, 2)), t.text(t.kid(i, 3)), literal=True)
        )
    if op == "str.strip_chars":
        return ea(t, i, 1).str().strip_chars()
    if op == "str.reverse":
        return ea(t, i, 1).str().reverse()
    if op == "str.pad_start":
        return (
            ea(t, i, 1).str().pad_start(t.int(t.kid(i, 2)), t.text(t.kid(i, 3)))
        )
    if op == "dt.year":
        return ea(t, i, 1).dt().year()
    if op == "dt.month":
        return ea(t, i, 1).dt().month()
    if op == "dt.day":
        return ea(t, i, 1).dt().day()
    if op == "dt.weekday":
        return ea(t, i, 1).dt().weekday()
    if op == "dt.hour":
        return ea(t, i, 1).dt().hour()
    if op == "dt.ordinal_day":
        return ea(t, i, 1).dt().ordinal_day()
    if op == "dt.date":
        return ea(t, i, 1).dt().date()
    if op == "list.len":
        return ea(t, i, 1).list().len()
    if op == "list.sum":
        return ea(t, i, 1).list().sum()
    if op == "list.min":
        return ea(t, i, 1).list().min()
    if op == "list.max":
        return ea(t, i, 1).list().max()
    if op == "list.first":
        return ea(t, i, 1).list().first()
    if op == "interpolate":
        return ea(t, i, 1).interpolate(t.text(t.kid(i, 2)))
    if op == "interpolate_by":
        return ea(t, i, 1).interpolate_by(ea(t, i, 2))
    if op == "rolling_agg":
        # (rolling_agg EXPR FN WINDOW MIN_SAMPLES CENTER (list WEIGHTS))
        var kind = t.text(t.kid(i, 2))
        var w = t.int(t.kid(i, 3))
        var ms = t.int(t.kid(i, 4))
        var center = t.flag(t.kid(i, 5))
        var weights = t.floats(t.kid(i, 6))
        var x = ea(t, i, 1)
        if kind == "sum":
            return x.rolling_sum(w, ms, center=center, weights=weights)
        if kind == "mean":
            return x.rolling_mean(w, ms, center=center, weights=weights)
        if kind == "min":
            return x.rolling_min(w, ms, center=center, weights=weights)
        if kind == "max":
            return x.rolling_max(w, ms, center=center, weights=weights)
        if kind == "std":
            return x.rolling_std(w, ms, center=center, weights=weights)
        return x.rolling_var(w, ms, center=center, weights=weights)
    if op == "rolling_quantile":
        return ea(t, i, 1).rolling_quantile(
            t.nodes[t.kid(i, 2)].fval,
            t.text(t.kid(i, 3)),
            t.int(t.kid(i, 4)),
            t.int(t.kid(i, 5)),
            center=t.flag(t.kid(i, 6)),
            weights=t.floats(t.kid(i, 7)),
        )
    if op == "rolling_median":
        return ea(t, i, 1).rolling_median(
            t.int(t.kid(i, 2)),
            t.int(t.kid(i, 3)),
            center=t.flag(t.kid(i, 4)),
            weights=t.floats(t.kid(i, 5)),
        )
    if op == "ewm":
        # (ewm EXPR FN ALPHA ADJUST BIAS MIN_SAMPLES IGNORE_NULLS)
        var kind = t.text(t.kid(i, 2))
        var a = t.nodes[t.kid(i, 3)].fval
        var adjust = t.flag(t.kid(i, 4))
        var bias = t.flag(t.kid(i, 5))
        var ms = t.int(t.kid(i, 6))
        var ignore = t.flag(t.kid(i, 7))
        var x = ea(t, i, 1)
        if kind == "mean":
            return x.ewm_mean(
                alpha=a, adjust=adjust, min_samples=ms, ignore_nulls=ignore
            )
        if kind == "var":
            return x.ewm_var(
                alpha=a,
                adjust=adjust,
                bias=bias,
                min_samples=ms,
                ignore_nulls=ignore,
            )
        if kind == "std":
            return x.ewm_std(
                alpha=a,
                adjust=adjust,
                bias=bias,
                min_samples=ms,
                ignore_nulls=ignore,
            )
        return x.ewm_sum(alpha=a, min_samples=ms, ignore_nulls=ignore)
    if op == "ewm_by":
        # (ewm_by EXPR BY FN HALF_LIFE)
        if t.text(t.kid(i, 3)) == "mean":
            return ea(t, i, 1).ewm_mean_by(ea(t, i, 2), t.text(t.kid(i, 4)))
        return ea(t, i, 1).ewm_sum_by(ea(t, i, 2), t.text(t.kid(i, 4)))
    if op == "rolling_agg_by":
        # (rolling_agg_by EXPR BY FN PERIOD CLOSED MIN_SAMPLES)
        var kind = t.text(t.kid(i, 3))
        var by = ea(t, i, 2)
        var period = t.text(t.kid(i, 4))
        var closed = t.text(t.kid(i, 5))
        var ms = t.int(t.kid(i, 6))
        var x = ea(t, i, 1)
        if kind == "sum":
            return x.rolling_sum_by(by, period, closed, ms)
        if kind == "mean":
            return x.rolling_mean_by(by, period, closed, ms)
        if kind == "min":
            return x.rolling_min_by(by, period, closed, ms)
        if kind == "max":
            return x.rolling_max_by(by, period, closed, ms)
        if kind == "std":
            return x.rolling_std_by(by, period, closed, ms)
        return x.rolling_var_by(by, period, closed, ms)
    if op == "rolling_quantile_by":
        # (rolling_quantile_by EXPR BY PERIOD Q METHOD CLOSED MIN_SAMPLES)
        return ea(t, i, 1).rolling_quantile_by(
            ea(t, i, 2),
            t.text(t.kid(i, 3)),
            t.nodes[t.kid(i, 4)].fval,
            t.text(t.kid(i, 5)),
            t.text(t.kid(i, 6)),
            t.int(t.kid(i, 7)),
        )
    if op == "list.get":
        return ea(t, i, 1).list().get(t.int(t.kid(i, 2)))
    if op == "list.median":
        return ea(t, i, 1).list().median()
    if op == "list.std":
        return ea(t, i, 1).list().std(t.int(t.kid(i, 2)))
    if op == "list.var":
        return ea(t, i, 1).list().var(t.int(t.kid(i, 2)))
    if op == "list.arg_min":
        return ea(t, i, 1).list().arg_min()
    if op == "list.arg_max":
        return ea(t, i, 1).list().arg_max()
    if op == "list.count_matches" or op == "list.contains":
        var item: Expr
        var kid = t.kid(i, 2)
        if t.nodes[kid].tag == STR:
            item = lit(t.text(kid))
        else:
            item = lit(Int64(t.int(kid)))
        if op == "list.count_matches":
            return ea(t, i, 1).list().count_matches(item)
        return ea(t, i, 1).list().contains(item)
    if op == "list.explode":
        return (
            ea(t, i, 1)
            .list()
            .explode(
                empty_as_null=t.flag(t.kid(i, 2)),
                keep_nulls=t.flag(t.kid(i, 3)),
            )
        )
    if op == "list.to_struct":
        var names = List[String]()
        for f in range(t.int(t.kid(i, 2))):
            names.append("f" + String(f))
        return ea(t, i, 1).list().to_struct(names)
    if op == "list.eval":
        var body = t.text(t.kid(i, 2))
        var el = element()
        if body == "gt":
            return ea(t, i, 1).list().filter(el > 0)
        if body == "mul2":
            return ea(t, i, 1).list().eval(el * 2)
        if body == "cum_sum":
            return ea(t, i, 1).list().eval(el.cum_sum())
        if body == "rank":
            return ea(t, i, 1).list().eval(el.rank("dense"))
        return ea(t, i, 1).list().eval(el - el.min())
    if op == "list.agg":
        var body = t.text(t.kid(i, 2))
        var el = element()
        if body == "sum":
            return ea(t, i, 1).list().agg(el.sum())
        if body == "max":
            return ea(t, i, 1).list().agg(el.max())
        if body == "null_count":
            return ea(t, i, 1).list().agg(el.null_count())
        return ea(t, i, 1).list().agg(el.n_unique())
    if op == "list.set_union":
        return ea(t, i, 1).list().set_union(ea(t, i, 2))
    if op == "list.set_intersection":
        return ea(t, i, 1).list().set_intersection(ea(t, i, 2))
    if op == "list.set_difference":
        return ea(t, i, 1).list().set_difference(ea(t, i, 2))
    if op == "list.set_symmetric_difference":
        return ea(t, i, 1).list().set_symmetric_difference(ea(t, i, 2))
    if op == "list.n_unique":
        return ea(t, i, 1).list().n_unique()
    if op == "list.unique":
        return ea(t, i, 1).list().unique(t.flag(t.kid(i, 2)))
    if op == "list.sort":
        return ea(t, i, 1).list().sort(t.flag(t.kid(i, 2)), t.flag(t.kid(i, 3)))
    if op == "list.reverse":
        return ea(t, i, 1).list().reverse()
    if op == "list.slice":
        return ea(t, i, 1).list().slice(t.int(t.kid(i, 2)), t.int(t.kid(i, 3)))
    if op == "list.head":
        return ea(t, i, 1).list().head(t.int(t.kid(i, 2)))
    if op == "list.tail":
        return ea(t, i, 1).list().tail(t.int(t.kid(i, 2)))
    if op == "list.gather":
        return (
            ea(t, i, 1)
            .list()
            .gather([t.int(t.kid(i, 2)), t.int(t.kid(i, 3))], null_on_oob=True)
        )
    if op == "list.gather_every":
        return (
            ea(t, i, 1)
            .list()
            .gather_every(t.int(t.kid(i, 2)), t.int(t.kid(i, 3)))
        )
    if op == "list.drop_nulls":
        return ea(t, i, 1).list().drop_nulls()
    if op == "list.shift":
        return ea(t, i, 1).list().shift(t.int(t.kid(i, 2)))
    if op == "list.diff":
        return ea(t, i, 1).list().diff(t.int(t.kid(i, 2)))
    if op == "list.concat":
        return ea(t, i, 1).list().concat(ea(t, i, 2))
    if op == "concat_list":
        return concat_list([ea(t, i, 1), ea(t, i, 2)])
    if op == "field":
        return ea(t, i, 1).field(t.text(t.kid(i, 2)))
    raise Error("fuzz driver: unknown expression " + op)


def optional_text(t: Tree, i: Int) -> Optional[String]:
    """An optional string argument: the empty string means none."""
    var text = t.text(i)
    if text.byte_length() == 0:
        return None
    return text


def build_exprs(t: Tree, i: Int) raises -> List[Expr]:
    var out = List[Expr]()
    for k in range(1, t.nkids(i)):
        out.append(build_expr(t, t.kid(i, k)))
    return out^


struct Inputs(Movable):
    var formats: List[String]
    var paths: List[String]
    var schemas: List[List[Tuple[String, String]]]
    var eager_cache: List[Optional[DataFrame]]

    def __init__(out self, t: Tree, i: Int) raises:
        self.formats = List[String]()
        self.paths = List[String]()
        self.schemas = List[List[Tuple[String, String]]]()
        self.eager_cache = List[Optional[DataFrame]]()
        for k in range(1, t.nkids(i)):
            var node = t.kid(i, k)
            self.formats.append(t.text(t.kid(node, 1)))
            self.paths.append(t.text(t.kid(node, 2)))
            var fields = List[Tuple[String, String]]()
            var schema = t.kid(node, 3)
            for f in range(1, t.nkids(schema)):
                var field = t.kid(schema, f)
                fields.append(
                    (t.text(t.kid(field, 1)), t.text(t.kid(field, 2)))
                )
            self.schemas.append(fields^)
            self.eager_cache.append(None)

    def csv_schema(self, index: Int) raises -> CsvSchema:
        var fields = List[CsvField]()
        for f in self.schemas[index]:
            if f[1] == "categorical":
                fields.append(CsvField.categorical(f[0]))
            else:
                fields.append(CsvField(f[0], DataType.parse(f[1])))
        return CsvSchema(fields^)

    def read(mut self, index: Int) raises -> DataFrame:
        if self.eager_cache[index]:
            return self.eager_cache[index].value().copy()
        var frame: DataFrame
        var format = self.formats[index]
        if format == "parquet":
            frame = read_parquet(self.paths[index])
        elif format == "ipc":
            frame = read_ipc(self.paths[index])
        elif format == "csv":
            frame = read_csv(self.paths[index], self.csv_schema(index))
        else:
            raise Error("unknown input format " + format)
        self.eager_cache[index] = frame.copy()
        return frame^

    def scan(self, index: Int) raises -> LazyFrame:
        var format = self.formats[index]
        if format == "parquet":
            return scan_parquet(self.paths[index])
        if format == "ipc":
            return scan_ipc(self.paths[index])
        if format == "csv":
            return scan_csv(self.paths[index], self.csv_schema(index))
        raise Error("unknown input format " + format)


def run_eager(t: Tree, i: Int, mut inputs: Inputs) raises -> DataFrame:
    var op = t.head(i)
    if op == "src":
        return inputs.read(t.int(t.kid(i, 1)))
    if op == "concat":
        var frames = List[DataFrame]()
        for k in range(2, t.nkids(i)):
            frames.append(run_eager(t, t.kid(i, k), inputs))
        return concat(frames, how=t.text(t.kid(i, 1)))
    var df = run_eager(t, t.kid(i, 1), inputs)
    if op == "filter":
        return df.filter(build_expr(t, t.kid(i, 2)))
    if op == "select":
        return df.select_exprs(build_exprs(t, t.kid(i, 2)))
    if op == "with_columns":
        return df.with_columns(build_exprs(t, t.kid(i, 2)))
    if op == "group_by":
        # (group_by P (list KEYS) MAINTAIN (list AGGS))
        return df.group_by(
            t.strings(t.kid(i, 2)), maintain_order=t.flag(t.kid(i, 3))
        ).agg(build_exprs(t, t.kid(i, 4)))
    if op == "join":
        # (join P Q (list LEFT_ON) (list RIGHT_ON) HOW COALESCE)
        var right = run_eager(t, t.kid(i, 2), inputs)
        return df.join(
            right,
            left_on=t.strings(t.kid(i, 3)),
            right_on=t.strings(t.kid(i, 4)),
            how=t.text(t.kid(i, 5)),
            coalesce=t.flag(t.kid(i, 6)),
        )
    if op == "cross":
        var right = run_eager(t, t.kid(i, 2), inputs)
        return df.join(right, how="cross")
    if op == "sort":
        # (sort P (list KEYS) (list DESC) (list NULLS_LAST) MAINTAIN)
        return df.sort(
            t.strings(t.kid(i, 2)),
            descending=t.flags(t.kid(i, 3)),
            nulls_last=t.flags(t.kid(i, 4)),
            maintain_order=t.flag(t.kid(i, 5)),
        )
    if op == "unique":
        return df.unique(
            t.strings(t.kid(i, 2)),
            keep=t.text(t.kid(i, 3)),
            maintain_order=t.flag(t.kid(i, 4)),
        )
    if op == "head":
        return df.head(t.int(t.kid(i, 2)))
    if op == "tail":
        return df.tail(t.int(t.kid(i, 2)))
    if op == "slice":
        return df.slice(t.int(t.kid(i, 2)), t.int(t.kid(i, 3)))
    if op == "explode":
        return df.explode(t.text(t.kid(i, 2)))
    if op == "drop_nulls":
        return df.drop_nulls(t.strings(t.kid(i, 2)))
    if op == "drop":
        return df.drop(t.strings(t.kid(i, 2)))
    if op == "reverse":
        return df.reverse()
    if op == "with_row_index":
        return df.with_row_index(t.text(t.kid(i, 2)))
    if op == "unnest":
        return df.unnest(t.text(t.kid(i, 2)))
    if op == "rolling":
        # (rolling P INDEX PERIOD OFFSET CLOSED (list KEYS) (list AGGS))
        return df.rolling(
            t.text(t.kid(i, 2)),
            t.text(t.kid(i, 3)),
            optional_text(t, t.kid(i, 4)),
            t.text(t.kid(i, 5)),
            t.strings(t.kid(i, 6)),
        ).agg(build_exprs(t, t.kid(i, 7)))
    if op == "dynamic":
        # (dynamic P INDEX EVERY PERIOD OFFSET CLOSED LABEL (list KEYS)
        #  (list AGGS))
        return df.group_by_dynamic(
            t.text(t.kid(i, 2)),
            t.text(t.kid(i, 3)),
            optional_text(t, t.kid(i, 4)),
            optional_text(t, t.kid(i, 5)),
            t.text(t.kid(i, 6)),
            t.text(t.kid(i, 7)),
            t.strings(t.kid(i, 8)),
        ).agg(build_exprs(t, t.kid(i, 9)))
    raise Error("fuzz driver: unknown plan op " + op)


def run_lazy(
    t: Tree, i: Int, mut inputs: Inputs, from_memory: Bool
) raises -> LazyFrame:
    var op = t.head(i)
    if op == "src":
        var index = t.int(t.kid(i, 1))
        if from_memory:
            return inputs.read(index).lazy()
        return inputs.scan(index)
    if op == "concat":
        var frames = List[LazyFrame]()
        for k in range(2, t.nkids(i)):
            frames.append(run_lazy(t, t.kid(i, k), inputs, from_memory))
        return concat(frames, how=t.text(t.kid(i, 1)))
    var lf = run_lazy(t, t.kid(i, 1), inputs, from_memory)
    if op == "filter":
        return lf.filter(build_expr(t, t.kid(i, 2)))
    if op == "select":
        return lf.select_exprs(build_exprs(t, t.kid(i, 2)))
    if op == "with_columns":
        return lf.with_columns(build_exprs(t, t.kid(i, 2)))
    if op == "group_by":
        return lf.group_by(
            t.strings(t.kid(i, 2)), maintain_order=t.flag(t.kid(i, 3))
        ).agg(build_exprs(t, t.kid(i, 4)))
    if op == "join":
        var right = run_lazy(t, t.kid(i, 2), inputs, from_memory)
        return lf.join(
            right,
            left_on=t.strings(t.kid(i, 3)),
            right_on=t.strings(t.kid(i, 4)),
            how=t.text(t.kid(i, 5)),
            coalesce=t.flag(t.kid(i, 6)),
        )
    if op == "cross":
        var right = run_lazy(t, t.kid(i, 2), inputs, from_memory)
        return lf.join(right, how="cross")
    if op == "sort":
        return lf.sort(
            t.strings(t.kid(i, 2)),
            descending=t.flags(t.kid(i, 3)),
            nulls_last=t.flags(t.kid(i, 4)),
            maintain_order=t.flag(t.kid(i, 5)),
        )
    if op == "unique":
        return lf.unique(
            t.strings(t.kid(i, 2)),
            keep=t.text(t.kid(i, 3)),
            maintain_order=t.flag(t.kid(i, 4)),
        )
    if op == "head":
        return lf.head(t.int(t.kid(i, 2)))
    if op == "tail":
        return lf.tail(t.int(t.kid(i, 2)))
    if op == "slice":
        return lf.slice(t.int(t.kid(i, 2)), t.int(t.kid(i, 3)))
    if op == "explode":
        return lf.explode(t.text(t.kid(i, 2)))
    if op == "drop_nulls":
        return lf.drop_nulls(t.strings(t.kid(i, 2)))
    if op == "drop":
        return lf.drop(t.strings(t.kid(i, 2)))
    if op == "reverse":
        return lf.reverse()
    if op == "with_row_index":
        return lf.with_row_index(t.text(t.kid(i, 2)))
    if op == "unnest":
        return lf.unnest(t.text(t.kid(i, 2)))
    if op == "rolling":
        return lf.rolling(
            t.text(t.kid(i, 2)),
            t.text(t.kid(i, 3)),
            optional_text(t, t.kid(i, 4)),
            t.text(t.kid(i, 5)),
            t.strings(t.kid(i, 6)),
        ).agg(build_exprs(t, t.kid(i, 7)))
    if op == "dynamic":
        return lf.group_by_dynamic(
            t.text(t.kid(i, 2)),
            t.text(t.kid(i, 3)),
            optional_text(t, t.kid(i, 4)),
            optional_text(t, t.kid(i, 5)),
            t.text(t.kid(i, 6)),
            t.text(t.kid(i, 7)),
            t.strings(t.kid(i, 8)),
        ).agg(build_exprs(t, t.kid(i, 9)))
    raise Error("fuzz driver: unknown plan op " + op)


def write_text(path: String, text: String) raises:
    with open(path, "w") as file:
        file.write(text)


def run_mode(
    t: Tree, plan: Int, mut inputs: Inputs, mode: String, out_dir: String
) raises:
    try:
        var result: DataFrame
        if mode == "eager":
            result = run_eager(t, plan, inputs)
        elif mode == "lazy":
            result = run_lazy(t, plan, inputs, False).collect()
        elif mode == "lazy_noopt":
            result = run_lazy(t, plan, inputs, False).collect(optimize=False)
        elif mode == "lazy_mem":
            result = run_lazy(t, plan, inputs, True).collect()
        else:
            raise Error("unknown mode " + mode)
        write_ipc(result, out_dir + "/" + mode + ".arrow")
    except e:
        write_text(out_dir + "/" + mode + ".err", String(e))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: fuzz_driver CASE_FILE OUTPUT_DIR [MODE ...]")
    var source: String
    with open(String(args[1]), "r") as file:
        source = file.read()
    var t = Tree(source)
    var root = 0
    if t.head(root) != "case":
        raise Error("case file must start with (case")
    var inputs = Inputs(t, t.kid(root, 1))
    var plan = t.kid(root, 2)
    var out_dir = String(args[2])
    var modes = List[String]()
    for k in range(3, len(args)):
        modes.append(String(args[k]))
    if len(modes) == 0:
        modes = ["eager", "lazy", "lazy_noopt", "lazy_mem"]
    for mode in modes:
        # Mark the mode as started so a crash names the mode it died in.
        write_text(out_dir + "/" + mode + ".started", "")
        run_mode(t, plan, inputs, mode, out_dir)
