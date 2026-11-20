#!/usr/bin/env python3
"""Differential fuzzer: random operation chains, our engine against Polars.

Each case is a few random input tables (written as Parquet, Arrow IPC or
CSV) and a random plan of frame operations. The Mojo driver
(tests/fuzz/fuzz_driver.mojo, compiled once) runs the plan eagerly and
lazily; Polars runs the same plan as the reference. Results are compared
with float tolerance, ignoring row order unless the plan fixes it.

A failing case is shrunk (operations, expressions, rows and columns are
dropped while it still fails the same way) and saved under the output
directory as a replayable case.

    pixi run -e oracle fuzz --cases 500 --seed 1
    pixi run -e oracle python3 tests/fuzz/fuzz.py replay build/fuzz/failures/<case>

See tests/fuzz/README.md.
"""

import argparse
import datetime as dt
import decimal
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[2]
MODES = ["eager", "lazy", "lazy_noopt", "lazy_mem"]
RTOL = 1e-6
MAX_ROWS = 600_000
ATOL = 1e-8

# Our errors that are documented choices rather than bugs: integer
# arithmetic raises on overflow where Polars wraps.
ALLOWED_ERRORS = ["overflow"]

# ---------------------------------------------------------------------------
# Types. A dtype is a short spec string shared by the generator, the driver
# (as a DataType.parse name) and the Polars side.

INTS = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
FLOATS = ["float32", "float64"]
SCALAR_TYPES = INTS + FLOATS + [
    "bool",
    "string",
    "categorical",
    "date",
    "datetime",
    "decimal[10,2]",
]
NESTED_TYPES = ["list[int64]", "list[string]", "struct"]


def pl_dtype(spec):
    m = {
        "int8": pl.Int8,
        "int16": pl.Int16,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "uint8": pl.UInt8,
        "uint16": pl.UInt16,
        "uint32": pl.UInt32,
        "uint64": pl.UInt64,
        "float32": pl.Float32,
        "float64": pl.Float64,
        "bool": pl.Boolean,
        "string": pl.String,
        "categorical": pl.Categorical,
        "date": pl.Date,
        "datetime": pl.Datetime("us"),
        "decimal[10,2]": pl.Decimal(10, 2),
        "list[int64]": pl.List(pl.Int64),
        "list[string]": pl.List(pl.String),
        "struct": pl.Struct({"x": pl.Int64, "y": pl.String}),
    }
    return m[spec]


def spec_of(dtype):
    """Our spec for a Polars dtype, or None when the generator should not
    build expressions on it."""
    if dtype == pl.Categorical:
        return "categorical"
    for s in SCALAR_TYPES + NESTED_TYPES:
        if s == "categorical":
            continue
        if dtype == pl_dtype(s):
            return s
    if isinstance(dtype, pl.Datetime):
        return "datetime_other"
    if isinstance(dtype, pl.Decimal):
        return "decimal_other"
    if isinstance(dtype, pl.List):
        return "list_other"
    if isinstance(dtype, pl.Struct):
        return "struct_other"
    if dtype == pl.Null:
        return "null"
    if isinstance(dtype, pl.Duration):
        return "duration"
    return None


def is_int(s):
    return s in INTS


def is_float(s):
    return s in FLOATS


def is_numeric(s):
    return is_int(s) or is_float(s)


def is_signed(s):
    return s in ("int8", "int16", "int32", "int64")


def is_orderable(s):
    return s in SCALAR_TYPES


def is_key(s):
    return s in SCALAR_TYPES and s != "float32"


# ---------------------------------------------------------------------------
# Random tables.

WORDS = ["", "a", "b", "ab", "B", "x y", "héllo", "日本", "🙂", "zz", "a,b", "Ab"]
CATS = ["red", "green", "blue", "", "ünï", "black"]


def gen_values(rng, spec, n, card):
    """n values of spec (None for null); card bounds distinct values so
    keys repeat."""
    null_p = rng.choice([0.0, 0.0, 0.1, 0.3, 1.0 if rng.random() < 0.03 else 0.2])

    def one():
        if rng.random() < null_p:
            return None
        if spec in ("int8", "int16", "int32", "int64"):
            lim = {"int8": 20, "int16": 300, "int32": 5000, "int64": 100000}[spec]
            lim = min(lim, card) if rng.random() < 0.8 else lim
            return rng.randint(-lim, lim)
        if spec in ("uint8", "uint16", "uint32", "uint64"):
            lim = {"uint8": 40, "uint16": 600, "uint32": 9000, "uint64": 200000}[spec]
            lim = min(lim, card) if rng.random() < 0.8 else lim
            return rng.randint(0, lim)
        if spec in FLOATS:
            r = rng.random()
            if r < 0.05:
                return float("nan")
            if r < 0.07:
                return rng.choice([float("inf"), float("-inf"), -0.0])
            if r < 0.6:
                return float(rng.randint(-card, card)) / 2.0
            v = rng.uniform(-1000, 1000)
            return float(pl.Series([v], dtype=pl_dtype(spec))[0]) if spec == "float32" else v
        if spec == "bool":
            return rng.random() < 0.5
        if spec == "string":
            if rng.random() < 0.6:
                return rng.choice(WORDS[: max(2, min(len(WORDS), card))])
            return "".join(rng.choice("abcxyzé日 ") for _ in range(rng.randint(0, 6)))
        if spec == "categorical":
            return rng.choice(CATS[: max(2, min(len(CATS), card))])
        if spec == "date":
            return dt.date(2000, 1, 1) + dt.timedelta(days=rng.randint(-card * 30, card * 30))
        if spec == "datetime":
            return dt.datetime(2020, 1, 1) + dt.timedelta(
                seconds=rng.randint(-card * 5000, card * 5000),
                microseconds=rng.choice([0, 0, rng.randint(0, 999999)]),
            )
        if spec == "decimal[10,2]":
            return decimal.Decimal(rng.randint(-card * 100, card * 100)).scaleb(-2)
        if spec == "list[int64]":
            return [
                None if rng.random() < 0.1 else rng.randint(-card, card)
                for _ in range(rng.choice([0, 1, 1, 2, 3, 5]))
            ]
        if spec == "list[string]":
            return [
                None if rng.random() < 0.1 else rng.choice(WORDS)
                for _ in range(rng.choice([0, 1, 2, 3]))
            ]
        if spec == "struct":
            return {
                "x": None if rng.random() < 0.1 else rng.randint(-card, card),
                "y": None if rng.random() < 0.1 else rng.choice(WORDS[:4]),
            }
        raise ValueError(spec)

    return [one() for _ in range(n)]


def gen_table(rng, n, prefix, shared_keys):
    """A table of n rows. shared_keys (name -> spec) are columns every table
    of the case has, so joins and concats find matching keys."""
    cols = {}
    specs = {}
    card = rng.choice([2, 3, 5, 10, 50, 1000])
    for name, spec in shared_keys.items():
        cols[name] = gen_values(rng, spec, n, card)
        specs[name] = spec
    for i in range(rng.randint(1, 5)):
        pool = SCALAR_TYPES if rng.random() < 0.85 else NESTED_TYPES
        spec = rng.choice(pool)
        name = f"{prefix}{i}_{spec.split('[')[0]}"
        cols[name] = gen_values(rng, spec, n, card)
        specs[name] = spec
    df = pl.DataFrame(
        {k: pl.Series(k, v, dtype=pl_dtype(specs[k])) for k, v in cols.items()}
    )
    return df, specs


def csv_ok(specs):
    return all(s not in NESTED_TYPES for s in specs.values())


# ---------------------------------------------------------------------------
# S-expression serialization for the driver.


def atom(v):
    if isinstance(v, bool):
        return "b:1" if v else "b:0"
    if isinstance(v, int):
        return f"i:{v}"
    if isinstance(v, float):
        if math.isnan(v):
            return "f:nan"
        if math.isinf(v):
            return "f:inf" if v > 0 else "f:-inf"
        return f"f:{v!r}"
    if isinstance(v, str):
        return "s:" + v.encode("utf-8").hex()
    raise TypeError(v)


def sexp(node):
    if isinstance(node, (tuple, list)):
        return "(" + " ".join([node[0]] + [sexp(k) for k in node[1:]]) + ")"
    return atom(node)


# ---------------------------------------------------------------------------
# Polars translation of expressions and plans.


def pl_lit_typed(spec, v):
    return pl.lit(v, dtype=pl_dtype(spec))


def ieee_compare(op, a, b, a_float):
    """Our comparisons follow IEEE for floats (NaN compares false, and
    NaN != NaN); Polars orders NaN as the largest value. Rewrite so the
    oracle gives the IEEE answer."""
    base = {
        "eq": a.eq(b),
        "ne": a.ne(b),
        "lt": a < b,
        "le": a <= b,
        "gt": a > b,
        "ge": a >= b,
    }[op]
    if not a_float:
        return base
    nan = a.is_nan() | b.is_nan()
    return pl.when(nan).then(pl.lit(op == "ne")).otherwise(base)


def to_pl_expr(e, types):
    """Translate an expression tree. types maps column names to specs."""
    if isinstance(e, bool):
        return pl.lit(e)
    if isinstance(e, int):
        return pl.lit(e)
    if isinstance(e, float):
        return pl.lit(e)
    if isinstance(e, str):
        return pl.lit(e)
    op = e[0]
    E = lambda k: to_pl_expr(e[k], types)
    if op == "col":
        return pl.col(e[1])
    if op == "lit":
        return pl_lit_typed(e[1], e[2])
    if op == "null":
        return pl.lit(None, dtype=pl_dtype(e[1]))
    if op == "len":
        return pl.len()
    if op == "alias":
        return E(1).alias(e[2])
    if op == "cast":
        return E(1).cast(pl_dtype(e[2]), strict=e[3])
    binops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "truediv": lambda a, b: a / b,
        "floordiv": lambda a, b: a // b,
        "mod": lambda a, b: a % b,
        "and": lambda a, b: a & b,
        "or": lambda a, b: a | b,
        "xor": lambda a, b: a ^ b,
        "eq_missing": lambda a, b: a.eq_missing(b),
        "fill_null": lambda a, b: a.fill_null(b),
        "fill_nan": lambda a, b: a.fill_nan(b),
    }
    if op in binops:
        return binops[op](E(1), E(2))
    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
        return ieee_compare(op, E(1), E(2), expr_is_float(e[1], types))
    if op == "clip":
        return E(1).clip(E(2), E(3))
    if op == "is_between":
        return E(1).is_between(E(2), E(3))
    if op == "when":
        chain = pl.when(E(1)).then(E(2))
        k = 3
        while k + 1 < len(e) - 0 and k + 1 <= len(e) - 2:
            chain = chain.when(E(k)).then(E(k + 1))
            k += 2
        return chain.otherwise(E(len(e) - 1))
    if op == "is_in":
        vals = [v for v in e[2][1:]]
        return E(1).is_in(vals)
    unary = {
        "neg": lambda a: -a,
        "not": lambda a: ~a,
        "abs": lambda a: a.abs(),
        "is_null": lambda a: a.is_null(),
        "is_not_null": lambda a: a.is_not_null(),
        "is_nan": lambda a: a.is_nan(),
        "is_not_nan": lambda a: a.is_not_nan(),
        "sqrt": lambda a: a.sqrt(),
        "floor": lambda a: a.floor(),
        "ceil": lambda a: a.ceil(),
        "sign": lambda a: a.sign(),
        "sum": lambda a: a.sum(),
        "mean": lambda a: a.mean(),
        "min": lambda a: a.min(),
        "max": lambda a: a.max(),
        "median": lambda a: a.median(),
        "count": lambda a: a.count(),
        "null_count": lambda a: a.null_count(),
        "n_unique": lambda a: a.n_unique(),
        "first": lambda a: a.first(),
        "last": lambda a: a.last(),
        "elen": lambda a: a.len(),
        "any": lambda a: a.any(),
        "all": lambda a: a.all(),
        "arg_min": lambda a: a.arg_min(),
        "arg_max": lambda a: a.arg_max(),
        "str.len_chars": lambda a: a.str.len_chars(),
        "str.len_bytes": lambda a: a.str.len_bytes(),
        "str.to_uppercase": lambda a: a.str.to_uppercase(),
        "str.to_lowercase": lambda a: a.str.to_lowercase(),
        "str.strip_chars": lambda a: a.str.strip_chars(),
        "str.reverse": lambda a: a.str.reverse(),
        "dt.year": lambda a: a.dt.year(),
        "dt.month": lambda a: a.dt.month(),
        "dt.day": lambda a: a.dt.day(),
        "dt.weekday": lambda a: a.dt.weekday(),
        "dt.hour": lambda a: a.dt.hour(),
        "dt.ordinal_day": lambda a: a.dt.ordinal_day(),
        "dt.date": lambda a: a.dt.date(),
        "list.len": lambda a: a.list.len(),
        "list.sum": lambda a: a.list.sum(),
        "list.min": lambda a: a.list.min(),
        "list.max": lambda a: a.list.max(),
        "list.first": lambda a: a.list.first(),
    }
    if op in unary:
        a = E(1)
        if op.startswith("str."):
            # Our string functions read a categorical's labels.
            a = a.cast(pl.String)
        return unary[op](a)
    if op == "round":
        return E(1).round(e[2])
    if op == "std":
        return E(1).std(e[2])
    if op == "var":
        return E(1).var(e[2])
    if op == "quantile":
        return E(1).quantile(e[2], e[3])
    if op in ("cum_sum", "cum_max", "cum_min", "cum_count"):
        return getattr(E(1), op)(reverse=e[2])
    if op == "shift":
        return E(1).shift(e[2])
    if op == "diff":
        return E(1).diff(e[2])
    if op == "rank":
        return E(1).rank(e[2], descending=e[3])
    if op == "over":
        keys = list(e[2][1:])
        order = list(e[3][1:])
        if not order:
            return E(1).over(keys)
        return E(1).over(
            keys or None, order_by=order, descending=e[4], nulls_last=e[5]
        )
    if op == "str.starts_with":
        return E(1).cast(pl.String).str.starts_with(e[2])
    if op == "str.ends_with":
        return E(1).cast(pl.String).str.ends_with(e[2])
    if op == "str.contains":
        return E(1).cast(pl.String).str.contains(e[2], literal=True)
    if op == "str.slice":
        return E(1).cast(pl.String).str.slice(e[2], None if e[3] < 0 else e[3])
    if op == "str.replace_all":
        return E(1).cast(pl.String).str.replace_all(e[2], e[3], literal=True)
    if op == "str.pad_start":
        return E(1).cast(pl.String).str.pad_start(e[2], e[3])
    if op == "list.get":
        return E(1).list.get(e[2], null_on_oob=True)
    if op == "field":
        return E(1).struct.field(e[2])
    raise ValueError("unknown expression " + op)


def expr_is_float(e, types):
    """Whether an expression's value is a float (for IEEE comparisons)."""
    return expr_type(e, types) in FLOATS


def expr_type(e, types):
    """The spec of an expression's result, as far as the generator tracks
    it (None when unknown)."""
    if isinstance(e, bool):
        return "bool"
    if isinstance(e, int):
        return "int_lit"
    if isinstance(e, float):
        return "float_lit"
    if isinstance(e, str):
        return "string"
    op = e[0]
    if op == "col":
        return types.get(e[1])
    if op in ("lit", "null"):
        return e[1]
    if op == "cast":
        return e[2]
    if op in ("alias", "fill_null", "fill_nan", "clip", "abs", "neg", "floor",
              "ceil", "round", "sign", "min", "max", "first", "last",
              "shift", "cum_max", "cum_min", "over"):
        t = expr_type(e[1], types)
        if t in ("int_lit", "float_lit") and op == "fill_null":
            return t
        return t
    if op in ("add", "sub", "mul", "floordiv", "mod"):
        a = expr_type(e[1], types)
        b = expr_type(e[2], types)
        if a in ("int_lit", "float_lit"):
            return b if b not in ("int_lit", "float_lit") else a
        return a
    if op in ("truediv", "sqrt", "mean", "median", "std", "var", "quantile"):
        a = expr_type(e[1], types)
        return "float32" if a == "float32" else "float64"
    if op == "when":
        return expr_type(e[2], types)
    return None


def nan_last_sort(lf, keys, desc, nulls_last, schema, maintain):
    """Sort as our engine does: NaN after every number in both directions
    (Polars puts NaN first when descending)."""
    by, d, nl = [], [], []
    for k, de, n in zip(keys, desc, nulls_last):
        if spec_of(schema[k]) in FLOATS:
            by.append(pl.col(k).is_nan())
            d.append(False)
            nl.append(n)
        by.append(pl.col(k))
        d.append(de)
        nl.append(n)
    return lf.sort(by, descending=d, nulls_last=nl, maintain_order=maintain)


def to_pl_plan(p, srcs):
    """Translate a plan tree into a Polars LazyFrame. srcs are the input
    frames as Polars read them."""
    op = p[0]
    if op == "src":
        return srcs[p[1]].lazy()
    if op == "concat":
        return pl.concat([to_pl_plan(k, srcs) for k in p[2:]], how=p[1])
    lf = to_pl_plan(p[1], srcs)
    schema = lf.collect_schema()
    types = {k: spec_of(v) for k, v in schema.items()}
    X = lambda e: to_pl_expr(e, types)
    if op == "filter":
        return lf.filter(X(p[2]))
    if op == "select":
        return lf.select([X(e) for e in p[2][1:]])
    if op == "with_columns":
        return lf.with_columns([X(e) for e in p[2][1:]])
    if op == "group_by":
        return lf.group_by(list(p[2][1:]), maintain_order=p[3]).agg(
            [X(e) for e in p[4][1:]]
        )
    if op == "join":
        right = to_pl_plan(p[2], srcs)
        how = p[5]
        kw = {}
        if how not in ("semi", "anti"):
            kw["coalesce"] = p[6]
        return lf.join(
            right, left_on=list(p[3][1:]), right_on=list(p[4][1:]), how=how, **kw
        )
    if op == "cross":
        return lf.join(to_pl_plan(p[2], srcs), how="cross")
    if op == "sort":
        return nan_last_sort(
            lf, list(p[2][1:]), list(p[3][1:]), list(p[4][1:]), schema, p[5]
        )
    if op == "unique":
        subset = list(p[2][1:]) or None
        return lf.unique(subset, keep=p[3], maintain_order=p[4])
    if op == "head":
        return lf.head(p[2])
    if op == "tail":
        return lf.tail(p[2])
    if op == "slice":
        return lf.slice(p[2], None if p[3] < 0 else p[3])
    if op == "explode":
        return lf.explode(p[2])
    if op == "drop_nulls":
        return lf.drop_nulls(list(p[2][1:]) or None)
    if op == "drop":
        return lf.drop(list(p[2][1:]))
    if op == "reverse":
        return lf.reverse()
    if op == "with_row_index":
        return lf.with_row_index(p[2])
    if op == "unnest":
        return lf.unnest(p[2])
    raise ValueError("unknown plan op " + op)


# ---------------------------------------------------------------------------
# Random expressions and plans. The generator keeps operand types equal
# (our engine has no implicit numeric promotion) and only uses
# order-dependent operations where the input order is defined.


class Gen:
    def __init__(self, rng, schema, ordered):
        self.rng = rng
        self.schema = schema  # name -> spec
        self.ordered = ordered
        self.counter = 0

    def cols(self, pred):
        return [k for k, v in self.schema.items() if v is not None and pred(v)]

    def name(self, base="e"):
        self.counter += 1
        return f"{base}{self.counter}_{self.rng.randint(0, 999)}"

    def small_int(self, spec):
        if spec.startswith("u"):
            return self.rng.randint(0, 5)
        return self.rng.randint(-5, 5)

    def literal_for(self, spec):
        """A literal comparable with a column of spec."""
        r = self.rng
        if is_int(spec):
            return self.small_int(spec) * r.choice([1, 1, 2, 10])
        if is_float(spec):
            return r.choice([0.0, 0.5, -1.5, 2.0, 10.0])
        if spec in ("string", "categorical"):
            return r.choice(WORDS + CATS)
        if spec == "bool":
            return r.random() < 0.5
        if spec == "date":
            return ("lit", "date", r.randint(10000, 12000))
        if spec == "datetime":
            return ("lit", "datetime", 1577836800000000 + r.randint(-10**11, 10**11))
        return None

    def numeric(self, depth=0):
        """(expr, spec) of a row-valued numeric expression."""
        r = self.rng
        cs = self.cols(is_numeric)
        if not cs:
            return None
        c = r.choice(cs)
        t = self.schema[c]
        e = ("col", c)
        if depth > 1 or r.random() < 0.35:
            return e, t
        choice = r.randint(0, 11)
        if choice == 0:
            return ("add", e, self.small_int(t)), t
        if choice == 1:
            same = [k for k in cs if self.schema[k] == t]
            return (r.choice(["add", "sub", "mul"]), e, ("col", r.choice(same))), t
        if choice == 2 and is_int(t):
            return (r.choice(["floordiv", "mod"]), e, r.choice([2, 3, 0, 7])), t
        if choice == 3:
            return ("truediv", e, ("col", r.choice([k for k in cs if self.schema[k] == t]))), (
                "float32" if t == "float32" else "float64"
            )
        if choice == 4 and is_signed(t) or is_float(t) and choice == 4:
            return ("abs", e), t
        if choice == 5:
            v = self.literal_for(t)
            return ("fill_null", e, v), t
        if choice == 6 and is_float(t):
            return (r.choice(["floor", "ceil", "sqrt"]), e), t
        if choice == 7 and is_float(t):
            return ("round", e, r.randint(0, 2)), t
        if choice == 8:
            p = self.predicate(depth + 1)
            if p is not None:
                other = self.numeric(depth + 1)
                if other is not None and other[1] == t:
                    return ("when", p, e, other[0]), t
                return ("when", p, e, ("null", t)), t
        if choice == 9:
            target = r.choice(["int64", "float64", "int32"])
            if is_float(t) and target.startswith("int"):
                return e, t
            return ("cast", e, target, False), target
        if choice == 10 and is_float(t):
            return ("fill_nan", e, 0.0), t
        if choice == 11:
            return ("mul", e, self.small_int(t) if is_int(t) else 1.5), t
        return e, t

    def string(self, depth=0):
        r = self.rng
        cs = self.cols(lambda s: s in ("string", "categorical"))
        if not cs:
            return None
        c = r.choice(cs)
        e = ("col", c)
        if r.random() < 0.5:
            return e, self.schema[c]
        choice = r.randint(0, 6)
        if choice == 0:
            return ("str.to_uppercase", e), "string"
        if choice == 1:
            return ("str.slice", e, r.randint(0, 2), r.choice([-1, 1, 2])), "string"
        if choice == 2:
            return ("str.replace_all", e, r.choice(["a", "é", "b"]), r.choice(["", "Z"])), "string"
        if choice == 3:
            return ("fill_null", e, "N"), "string"
        if choice == 4:
            ints = self.cols(lambda s: s in ("int32", "int64"))
            if ints:
                return ("cast", ("col", r.choice(ints)), "string", True), "string"
        if choice == 5:
            return ("str.pad_start", e, r.randint(0, 4), "*"), "string"
        return ("str.to_lowercase", e), "string"

    def predicate(self, depth=0):
        r = self.rng
        choice = r.randint(0, 10)
        if depth > 2:
            choice = r.choice([0, 1, 2])
        if choice in (0, 1):
            cs = self.cols(lambda s: is_orderable(s) and s != "decimal[10,2]")
            if not cs:
                return None
            c = r.choice(cs)
            t = self.schema[c]
            lit = self.literal_for(t)
            if lit is None:
                return None
            ops = ["eq", "ne"] if t in ("bool",) else ["eq", "ne", "lt", "le", "gt", "ge"]
            if t == "categorical":
                ops = ["eq", "ne"]
            return (r.choice(ops), ("col", c), lit)
        if choice == 2:
            cs = self.cols(lambda s: True)
            return (r.choice(["is_null", "is_not_null"]), ("col", r.choice(cs)))
        if choice == 3:
            n = self.numeric(depth + 1)
            if n is None:
                return None
            e, t = n
            same = [k for k in self.cols(is_numeric) if self.schema[k] == t]
            if t in self.schema.values() and same:
                return (r.choice(["lt", "gt", "eq", "le"]), e, ("col", r.choice(same)))
            return ("gt", e, 0 if is_int(t) else 0.0)
        if choice == 4:
            a = self.predicate(depth + 1)
            b = self.predicate(depth + 1)
            if a is None or b is None:
                return a or b
            return (r.choice(["and", "or", "xor"]), a, b)
        if choice == 5:
            a = self.predicate(depth + 1)
            return ("not", a) if a is not None else None
        if choice == 6:
            cs = self.cols(lambda s: s in ("string", "categorical") or s in ("int32", "int64"))
            if not cs:
                return None
            c = r.choice(cs)
            t = self.schema[c]
            vals = [self.literal_for(t) for _ in range(r.randint(1, 3))]
            return ("is_in", ("col", c), ("list",) + tuple(vals))
        if choice == 7:
            cs = self.cols(lambda s: s == "string")
            if not cs:
                return None
            f = r.choice(["str.starts_with", "str.ends_with", "str.contains"])
            return (f, ("col", r.choice(cs)), r.choice(["a", "", "é", "b", "日"]))
        if choice == 8:
            cs = self.cols(lambda s: s in ("date", "datetime"))
            if not cs:
                return None
            return ("gt", (r.choice(["dt.year", "dt.month", "dt.day"]), ("col", r.choice(cs))), r.randint(1, 2000))
        if choice == 9:
            cs = self.cols(is_float)
            if not cs:
                return None
            return (r.choice(["is_nan", "is_not_nan"]), ("col", r.choice(cs)))
        if choice == 10:
            cs = self.cols(lambda s: s == "bool")
            if not cs:
                return None
            return ("col", r.choice(cs))
        return None

    def row_expr(self):
        """A named row-valued expression for select / with_columns."""
        r = self.rng
        k = r.randint(0, 9)
        if k <= 3:
            x = self.numeric()
        elif k <= 5:
            x = self.string()
        elif k == 6:
            p = self.predicate()
            x = (p, "bool") if p is not None else None
        elif k == 7:
            x = self.window()
        elif k == 8:
            x = self.temporal_or_nested()
        else:
            cs = list(self.schema)
            x = (("col", r.choice(cs)), None) if cs else None
        if x is None:
            return None
        return ("alias", x[0], self.name())

    def temporal_or_nested(self):
        r = self.rng
        cs = self.cols(lambda s: s in ("date", "datetime", "list[int64]", "list[string]", "struct"))
        if not cs:
            return None
        c = r.choice(cs)
        t = self.schema[c]
        e = ("col", c)
        if t == "date":
            return (r.choice(["dt.year", "dt.month", "dt.day", "dt.weekday", "dt.ordinal_day"]), e), "int"
        if t == "datetime":
            return (r.choice(["dt.year", "dt.hour", "dt.date", "dt.day"]), e), "int"
        if t == "list[int64]":
            return (r.choice(["list.len", "list.sum", "list.min", "list.max", "list.first"]), e), "int"
        if t == "list[string]":
            return (r.choice(["list.len", "list.first"]), e), "int"
        if t == "struct":
            return ("field", e, r.choice(["x", "y"])), None
        return None

    def agg(self, allow_order):
        """An aggregation expression for group_by().agg or a scalar select."""
        r = self.rng
        cs = self.cols(lambda s: s is not None and s not in NESTED_TYPES)
        if not cs:
            return None
        c = r.choice(cs)
        t = self.schema[c]
        if r.random() < 0.3 and is_numeric(t):
            n = self.numeric(1)
            base = n[0] if n else ("col", c)
        else:
            base = ("col", c)
        if r.random() < 0.15:
            p = self.predicate(2)
            # A filtered aggregate: sum over rows meeting a condition.
            if p is not None and is_numeric(t):
                base = ("when", p, base, ("null", t))
        fns = ["count", "n_unique", "null_count", "elen"]
        if is_numeric(t) or t == "decimal[10,2]":
            fns += ["sum", "mean", "min", "max", "std", "var", "median"]
        elif t in ("string", "date", "datetime", "categorical"):
            fns += ["min", "max"]
        elif t == "bool":
            fns += ["sum", "any", "all"]
        if allow_order:
            fns += ["first", "last"]
            if is_numeric(t):
                fns += ["arg_min", "arg_max"]
        f = r.choice(fns)
        if f in ("std", "var"):
            e = (f, base, r.choice([0, 1]))
        elif f == "len_rows":
            e = ("len",)
        else:
            e = (f, base)
        if r.random() < 0.08:
            e = ("len",)
        return ("alias", e, self.name("a"))

    def window(self):
        r = self.rng
        keys = self.cols(is_key)
        if not keys:
            return None
        part = r.sample(keys, k=min(len(keys), r.choice([1, 1, 2])))
        a = self.agg(allow_order=self.ordered)
        if a is None:
            return None
        inner = a[1]
        if inner[0] in ("arg_min", "arg_max"):
            return None
        if self.ordered and r.random() < 0.3:
            nums = self.cols(lambda s: s in ("int32", "int64", "float64"))
            if nums:
                fn = r.choice(["cum_sum", "cum_max", "cum_count", "shift", "diff"])
                c = ("col", r.choice(nums))
                if fn in ("shift", "diff"):
                    inner = (fn, c, r.choice([1, -1, 2]))
                else:
                    inner = (fn, c, r.random() < 0.3)
        return ("over", inner, ("list",) + tuple(part), ("list",), False, False), None


def plan_schema(plan, srcs):
    lf = to_pl_plan(plan, srcs)
    return {k: spec_of(v) for k, v in lf.collect_schema().items()}


class PlanGen:
    def __init__(self, rng, tables):
        self.rng = rng
        self.tables = tables  # list of (polars df, specs)

    def source(self, i):
        return ("src", i), {k: v for k, v in self.tables[i][1].items()}, True

    def step(self, plan, schema, ordered, srcs, depth):
        """One random operation on top of plan; returns (plan, ordered)."""
        r = self.rng
        g = Gen(r, schema, ordered)
        names = list(schema)
        kind = r.choices(
            ["filter", "with_columns", "select", "group_by", "join", "sort",
             "unique", "head", "explode", "concat", "drop_nulls", "agg_select",
             "with_row_index", "reverse", "unnest", "drop"],
            weights=[14, 12, 6, 14, 12, 9, 6, 5, 3, 3, 2, 3, 1, 1, 2, 1],
        )[0]
        if kind == "filter":
            p = g.predicate()
            if p is None:
                return None
            return ("filter", plan, p), ordered
        if kind == "with_columns":
            es = [g.row_expr() for _ in range(r.randint(1, 3))]
            es = [e for e in es if e is not None]
            if not es:
                return None
            return ("with_columns", plan, ("list",) + tuple(es)), ordered
        if kind == "select":
            keep = r.sample(names, k=r.randint(1, len(names)))
            es = [("col", n) for n in keep]
            extra = g.row_expr()
            if extra is not None and r.random() < 0.5:
                es.append(extra)
            return ("select", plan, ("list",) + tuple(es)), ordered
        if kind == "agg_select":
            es = [g.agg(allow_order=ordered) for _ in range(r.randint(1, 3))]
            es = [e for e in es if e is not None]
            if not es:
                return None
            return ("select", plan, ("list",) + tuple(es)), True
        if kind == "group_by":
            keys = g.cols(lambda s: is_key(s) or s in ("list[int64]", "struct") and r.random() < 0.3)
            if not keys:
                return None
            ks = r.sample(keys, k=min(len(keys), r.choice([1, 1, 2, 3])))
            maintain = ordered and r.random() < 0.4
            sub = Gen(r, {k: v for k, v in schema.items() if k not in ks}, ordered)
            aggs = [sub.agg(allow_order=ordered) for _ in range(r.randint(1, 4))]
            aggs = [a for a in aggs if a is not None]
            if not aggs:
                return None
            return ("group_by", plan, ("list",) + tuple(ks), maintain, ("list",) + tuple(aggs)), maintain
        if kind == "join":
            j = r.randrange(len(self.tables))
            right, rschema, _ = self.source(j)
            if r.random() < 0.4:
                # A filtered or projected right side.
                rg = Gen(r, rschema, True)
                p = rg.predicate()
                if p is not None:
                    right = ("filter", right, p)
            how = r.choice(["inner", "inner", "left", "left", "full", "semi", "anti", "right", "cross"])
            if how == "cross":
                return ("cross", plan, right), False
            pairs = [
                (a, b)
                for a in schema
                for b in rschema
                if schema[a] == rschema[b] and is_key(schema[a]) and (schema[a] != "float64" or r.random() < 0.2)
            ]
            if not pairs:
                return None
            r.shuffle(pairs)
            chosen, used_l, used_r = [], set(), set()
            for a, b in pairs:
                if a not in used_l and b not in used_r:
                    chosen.append((a, b))
                    used_l.add(a)
                    used_r.add(b)
                if len(chosen) >= r.choice([1, 1, 2, 3]):
                    break
            return (
                "join",
                plan,
                right,
                ("list",) + tuple(a for a, _ in chosen),
                ("list",) + tuple(b for _, b in chosen),
                how,
                r.random() < 0.8,
            ), False
        if kind == "sort":
            cand = [n for n in names if is_orderable(schema[n])]
            if not cand:
                return None
            if r.random() < 0.5:
                keys = cand[:]
                r.shuffle(keys)
                total = len(keys) == len(names)
            else:
                keys = r.sample(cand, k=r.randint(1, min(3, len(cand))))
                total = False
            maintain = r.random() < 0.3
            desc = [r.random() < 0.4 for _ in keys]
            nl = [r.random() < 0.5 for _ in keys]
            plan = ("sort", plan, ("list",) + tuple(keys), ("list",) + tuple(desc), ("list",) + tuple(nl), maintain)
            return plan, total or (maintain and ordered)
        if kind == "unique":
            cand = [n for n in names if schema[n] is not None]
            if r.random() < 0.5 or not ordered:
                return ("unique", plan, ("list",), r.choice(["any", "first", "none"]) if ordered else r.choice(["any", "none"]), ordered and r.random() < 0.5), False if not ordered else None
            subset = r.sample(cand, k=r.randint(1, len(cand)))
            maintain = r.random() < 0.6
            return ("unique", plan, ("list",) + tuple(subset), r.choice(["first", "last", "none"]), maintain), maintain
        if kind == "head":
            if not ordered:
                return None
            c = r.randint(0, 2)
            if c == 0:
                return ("head", plan, r.choice([0, 1, 3, 10, 100, 3000])), True
            if c == 1:
                return ("tail", plan, r.choice([0, 1, 5, 100, 2500])), True
            return ("slice", plan, r.choice([0, 1, 7, -5, 2049]), r.choice([-1, 0, 3, 50, 4000])), True
        if kind == "explode":
            cs = g.cols(lambda s: s in ("list[int64]", "list[string]"))
            if not cs:
                return None
            return ("explode", plan, r.choice(cs)), ordered
        if kind == "unnest":
            cs = g.cols(lambda s: s == "struct")
            if not cs or "x" in schema or "y" in schema:
                return None
            return ("unnest", plan, r.choice(cs)), ordered
        if kind == "concat":
            # The same source again, filtered differently, so schemas match.
            other = plan
            if r.random() < 0.5:
                p = g.predicate()
                if p is not None:
                    other = ("filter", plan, p)
            return ("concat", "vertical", plan, other), ordered
        if kind == "drop_nulls":
            subset = r.sample(names, k=r.randint(0, min(2, len(names))))
            return ("drop_nulls", plan, ("list",) + tuple(subset)), ordered
        if kind == "with_row_index":
            if not ordered or "idx" in schema:
                return None
            return ("with_row_index", plan, "idx"), ordered
        if kind == "reverse":
            return ("reverse", plan), ordered
        if kind == "drop":
            if len(names) < 2:
                return None
            return ("drop", plan, ("list",) + tuple(r.sample(names, k=1))), ordered
        return None


def fix_unique_order(plan, ordered):
    if ordered is None:
        return plan[4]
    return ordered


def gen_case(rng):
    """A random case: tables, formats and a validated plan."""
    ntables = rng.choice([1, 1, 2, 2, 3])
    shared = {}
    for _ in range(rng.choice([1, 1, 2])):
        spec = rng.choice(["int64", "int32", "string", "categorical", "uint8", "date", "int16", "bool", "float64", "decimal[10,2]"])
        shared[f"k_{spec.split('[')[0]}"] = spec
    tables, formats = [], []
    for t in range(ntables):
        size_class = rng.random()
        if size_class < 0.35:
            n = rng.randint(0, 20)
        elif size_class < 0.7:
            n = rng.randint(20, 400)
        elif size_class < 0.92:
            n = rng.randint(400, 6000)
        elif size_class < 0.97:
            n = rng.randint(6000, 70000)
        else:
            # More than one morsel.
            n = rng.randint(100_001, 240_000)
        df, specs = gen_table(rng, n, f"t{t}c", shared)
        tables.append((df, specs))
        fmts = ["parquet", "parquet", "ipc"]
        if csv_ok(specs):
            fmts.append("csv")
        formats.append(rng.choice(fmts))
    case = {"tables": tables, "formats": formats}
    srcs = oracle_sources(case, None)
    pg = PlanGen(rng, tables)
    plan, schema, ordered = pg.source(0)
    nsteps = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
    for depth in range(nsteps):
        for _attempt in range(6):
            got = pg.step(plan, schema, ordered, srcs, depth)
            if got is None:
                continue
            new_plan, new_ordered = got
            if new_plan[0] == "unique" and new_ordered is None:
                new_ordered = new_plan[4]
            try:
                result = to_pl_plan(new_plan, srcs).collect()
            except Exception:
                continue
            # Keep results small: joins on low-cardinality keys multiply.
            if result.width == 0 or result.height > MAX_ROWS:
                continue
            new_schema = {k: spec_of(v) for k, v in result.schema.items()}
            plan, schema, ordered = new_plan, new_schema, new_ordered
            break
    case["plan"] = plan
    case["ordered"] = ordered
    # Some failures only show when the plan runs (not in its schema).
    to_pl_plan(plan, srcs).collect()
    return case


# ---------------------------------------------------------------------------
# Writing, running and comparing.


def write_inputs(case, d):
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ((df, specs), fmt) in enumerate(zip(case["tables"], case["formats"])):
        if fmt == "parquet":
            p = d / f"t{i}.parquet"
            df.write_parquet(p, row_group_size=max(1, rng_row_group(df)))
        elif fmt == "ipc":
            p = d / f"t{i}.arrow"
            df.write_ipc(p, compat_level=pl.CompatLevel.oldest())
        else:
            p = d / f"t{i}.csv"
            df.write_csv(p)
        paths.append(str(p))
    return paths


def rng_row_group(df):
    """Several row groups for larger tables, so scans see more than one."""
    n = df.height
    return n if n < 3000 else max(1000, n // 3)


def oracle_sources(case, paths):
    """The frames Polars computes on: the tables themselves, or for CSV
    what Polars reads back from the same file."""
    out = []
    for i, ((df, specs), fmt) in enumerate(zip(case["tables"], case["formats"])):
        if fmt == "csv" and paths is not None:
            schema = {k: pl_dtype(v) for k, v in specs.items()}
            out.append(pl.read_csv(paths[i], schema=schema))
        else:
            out.append(df)
    return out


def case_sexp(case, paths):
    ins = []
    for (df, specs), fmt, path in zip(case["tables"], case["formats"], paths):
        fields = [("f", k, v) for k, v in specs.items()]
        ins.append(("in", fmt, path, ("schema",) + tuple(fields)))
    return sexp(("case", ("inputs",) + tuple(ins), case["plan"]))


def driver_path():
    return Path(os.environ.get("FUZZ_DRIVER", ROOT / "build/fuzz/fuzz_driver"))


def run_driver(case_file, out_dir, threads, timeout=120):
    """Run every mode. Returns mode -> ("ok", frame) | ("error", text) |
    ("crash", detail) | ("hang", detail)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DATAFRAME_THREADS"] = str(threads)
    remaining = list(MODES)
    results = {}
    while remaining:
        for f in out_dir.glob("*"):
            f.unlink()
        try:
            proc = subprocess.run(
                [str(driver_path()), str(case_file), str(out_dir)] + remaining,
                env=env,
                capture_output=True,
                timeout=timeout,
            )
            status, detail = proc.returncode, proc.stderr.decode(errors="replace")[-2000:]
        except subprocess.TimeoutExpired:
            status, detail = "hang", f"timeout after {timeout}s"
        done = []
        for m in remaining:
            arrow = out_dir / f"{m}.arrow"
            err = out_dir / f"{m}.err"
            started = (out_dir / f"{m}.started").exists()
            if arrow.exists():
                try:
                    with pa.memory_map(str(arrow)) as src:
                        table = ipc.open_file(src).read_all()
                    results[m] = ("ok", pl.from_arrow(table, rechunk=True))
                except Exception as exc:
                    results[m] = ("error", f"unreadable output: {exc}")
                done.append(m)
            elif err.exists():
                results[m] = ("error", err.read_text(errors="replace"))
                done.append(m)
            elif started:
                kind = "hang" if status == "hang" else "crash"
                results[m] = (kind, f"exit {status}: {detail}")
                done.append(m)
                break
            else:
                break
        if not done:
            for m in remaining:
                results[m] = ("crash", f"exit {status} before any mode: {detail}")
            break
        remaining = [m for m in remaining if m not in results]
    return results


def norm_value(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, list):
        return tuple(norm_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(norm_value(x) for x in v.values())
    return v


def dtype_class(dtype):
    if dtype in (pl.Categorical, pl.String) or isinstance(dtype, pl.Enum):
        return "string"
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if isinstance(dtype, pl.Decimal):
        return "decimal"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    if isinstance(dtype, pl.List):
        return "list[" + dtype_class(dtype.inner) + "]"
    if isinstance(dtype, pl.Struct):
        return "struct"
    return str(dtype)


def comparable(df):
    """Categoricals as strings; everything else as is."""
    out = []
    for name, dtype in df.schema.items():
        if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            out.append(pl.col(name).cast(pl.String))
        else:
            out.append(pl.col(name))
    return df.select(out) if out else df


def sort_key(row):
    key = []
    for v in row:
        if v is None:
            key.append((0, 0))
        elif isinstance(v, float):
            if math.isnan(v):
                key.append((2, 0))
            else:
                key.append((1, float(f"{v:.6g}")))
        elif isinstance(v, bool):
            key.append((1, int(v)))
        elif isinstance(v, (int,)):
            key.append((1, v))
        elif isinstance(v, tuple):
            key.append((3, repr(sort_key(v))))
        else:
            key.append((4, str(v)))
    return key


def values_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= ATOL + RTOL * max(abs(fa), abs(fb))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def compare(expected, got, ordered, sort_keys=None):
    """None when equal, else a short description of the first difference."""
    if expected.columns != got.columns:
        return f"columns differ: expected {expected.columns}, got {got.columns}"
    for name in expected.columns:
        ce, cg = dtype_class(expected.schema[name]), dtype_class(got.schema[name])
        if ce != cg and {ce, cg} != {"float", "decimal"} and not (ce == "null" or cg == "null"):
            return f"dtype of {name!r}: expected {expected.schema[name]}, got {got.schema[name]}"
    if expected.height != got.height:
        return f"height: expected {expected.height}, got {got.height}"
    e_rows = [tuple(norm_value(v) for v in r) for r in comparable(expected).rows()]
    g_rows = [tuple(norm_value(v) for v in r) for r in comparable(got).rows()]
    if sort_keys:
        # A sort with ties: the key columns are in a fixed order even when
        # the tied rows are not.
        idx = [expected.columns.index(k) for k in sort_keys]
        for i, (er, gr) in enumerate(zip(e_rows, g_rows)):
            for j in idx:
                if not values_equal(er[j], gr[j]):
                    return f"sort key {expected.columns[j]!r} row {i}: expected {er[j]!r}, got {gr[j]!r}"
    if not ordered:
        e_rows.sort(key=sort_key)
        g_rows.sort(key=sort_key)
    for i, (er, gr) in enumerate(zip(e_rows, g_rows)):
        for j, (a, b) in enumerate(zip(er, gr)):
            if not values_equal(a, b):
                return (
                    f"row {i}{'' if ordered else ' (sorted)'} column {expected.columns[j]!r}: "
                    f"expected {a!r}, got {b!r}\n  expected row {er!r}\n  got row      {gr!r}"
                )
    return None


def final_sort_keys(plan):
    if plan[0] == "sort":
        return list(plan[2][1:])
    return None


def allowed_error(text):
    t = text.lower()
    return any(a in t for a in ALLOWED_ERRORS)


def evaluate(case, work, threads_list=(1, 8)):
    """Run one case. Returns a list of failures: (kind, mode, threads,
    detail), and the expected frame (or None if Polars failed)."""
    if work.exists():
        shutil.rmtree(work)
    paths = write_inputs(case, work / "in")
    srcs = oracle_sources(case, paths)
    try:
        expected = to_pl_plan(case["plan"], srcs).collect()
    except Exception as exc:
        return [("oracle-error", "-", 0, str(exc)[:300])], None
    case_file = work / "case.sexp"
    case_file.write_text(case_sexp(case, paths))
    failures = []
    for threads in threads_list:
        results = run_driver(case_file, work / f"out{threads}", threads)
        for mode in MODES:
            kind, payload = results.get(mode, ("crash", "no result"))
            if kind == "ok":
                diff = compare(expected, payload, case["ordered"], final_sort_keys(case["plan"]))
                if diff:
                    failures.append(("mismatch", mode, threads, diff))
            elif kind == "error":
                if not allowed_error(payload):
                    failures.append(("error", mode, threads, payload.strip()[:500]))
            else:
                failures.append((kind, mode, threads, payload[-500:]))
    return failures, expected


def signature(failures):
    """What must stay true while shrinking: the same kinds of failure."""
    return {(k, first_line(d) if k in ("error",) else "") for k, _, _, d in failures}


def first_line(text):
    line = text.strip().splitlines()[0] if text.strip() else ""
    # Drop numbers so shrinking can change them.
    return "".join(c for c in line if not c.isdigit())[:120]


def matches(sig, failures):
    if not failures:
        return False
    got = signature(failures)
    return bool(sig & got)


# ---------------------------------------------------------------------------
# Shrinking.


def plan_children(p):
    """Positions of child plans in a plan node."""
    op = p[0]
    if op == "src":
        return []
    if op == "concat":
        return list(range(2, len(p)))
    if op in ("join", "cross"):
        return [1, 2]
    return [1]


def plan_variants(p):
    """Smaller plans: an operation dropped, a list entry dropped, an
    expression simplified."""
    op = p[0]
    kids = plan_children(p)
    for k in kids:
        yield p[k]
    for k in kids:
        for v in plan_variants(p[k]):
            yield p[:k] + (v,) + p[k + 1 :]
    for pos in range(1, len(p)):
        item = p[pos]
        if pos in kids:
            continue
        if isinstance(item, tuple) and item and item[0] == "list" and len(item) > 2:
            for j in range(1, len(item)):
                shorter = item[:j] + item[j + 1 :]
                # Join key lists must stay paired.
                if op == "join" and pos in (3, 4):
                    other = 7 - pos
                    o = p[other]
                    shorter_o = o[:j] + o[j + 1 :]
                    q = list(p)
                    q[pos] = shorter
                    q[other] = shorter_o
                    yield tuple(q)
                else:
                    yield p[:pos] + (shorter,) + p[pos + 1 :]
        if isinstance(item, tuple) and item and item[0] != "list":
            for v in expr_variants(item):
                yield p[:pos] + (v,) + p[pos + 1 :]
        if isinstance(item, tuple) and item and item[0] == "list":
            for j in range(1, len(item)):
                if isinstance(item[j], tuple):
                    for v in expr_variants(item[j]):
                        yield p[:pos] + (item[:j] + (v,) + item[j + 1 :],) + p[pos + 1 :]
    if op == "join" and p[5] != "inner":
        yield p[:5] + ("inner",) + p[6:]
    if op in ("group_by",) and p[3]:
        yield p[:3] + (False,) + p[4:]


def expr_variants(e):
    if not isinstance(e, tuple) or not e or e[0] in ("col", "lit", "null", "len", "list"):
        return
    if e[0] == "alias":
        for v in expr_variants(e[1]):
            yield ("alias", v, e[2])
        return
    for k in range(1, len(e)):
        if isinstance(e[k], tuple) and e[k] and e[k][0] not in ("list",):
            yield e[k]
    for k in range(1, len(e)):
        if isinstance(e[k], tuple):
            for v in expr_variants(e[k]):
                yield e[:k] + (v,) + e[k + 1 :]


def with_tables(case, tables):
    c = dict(case)
    c["tables"] = tables
    return c


def table_variants(case):
    """Fewer rows or columns in one input table."""
    for i, (df, specs) in enumerate(case["tables"]):
        n = df.height
        if n > 1:
            for lo, hi in ((0, n // 2), (n // 2, n), (0, n - n // 4), (n // 4, n)):
                if hi - lo < n:
                    t = list(case["tables"])
                    t[i] = (df.slice(lo, hi - lo), specs)
                    yield with_tables(case, t)
        if 1 <= n <= 24:
            for r in range(n):
                t = list(case["tables"])
                t[i] = (pl.concat([df.slice(0, r), df.slice(r + 1)]), specs)
                yield with_tables(case, t)
        if len(specs) > 1:
            for name in list(specs):
                t = list(case["tables"])
                s2 = {k: v for k, v in specs.items() if k != name}
                t[i] = (df.drop(name), s2)
                yield with_tables(case, t)
        if case["formats"][i] != "parquet" and (case["formats"][i] != "csv" or csv_ok(specs)):
            c = dict(case)
            c["formats"] = list(case["formats"])
            c["formats"][i] = "parquet"
            yield c


def still_valid(case):
    try:
        srcs = oracle_sources(case, None)
        to_pl_plan(case["plan"], srcs).collect_schema()
        return True
    except Exception:
        return False


def recompute_ordered(case):
    """Order flags of a shrunk plan: conservative, recomputed from scratch."""
    return plan_ordered(case["plan"], case)


def plan_ordered(p, case):
    op = p[0]
    if op == "src":
        return True
    if op in ("join", "cross"):
        return False
    if op == "concat":
        return all(plan_ordered(k, case) for k in p[2:])
    inner = plan_ordered(p[1], case)
    if op == "group_by":
        return inner and p[3]
    if op == "sort":
        srcs = oracle_sources(case, None)
        names = list(to_pl_plan(p[1], srcs).collect_schema())
        return set(p[2][1:]) == set(names) or (inner and p[5])
    if op == "unique":
        return inner and p[4]
    if op == "select":
        aggs = all(is_scalar_agg(e) for e in p[2][1:])
        return True if aggs else inner
    return inner


def is_scalar_agg(e):
    if not isinstance(e, tuple):
        return False
    if e[0] == "alias":
        return is_scalar_agg(e[1])
    return e[0] in ("sum", "mean", "min", "max", "median", "count", "null_count",
                    "n_unique", "first", "last", "std", "var", "elen", "any",
                    "all", "arg_min", "arg_max", "len", "quantile")


def order_safe(case):
    """A shrunk plan must not use order-dependent steps on unordered input."""
    return check_order(case["plan"], case)


def check_order(p, case):
    op = p[0]
    if op == "src":
        return True
    for k in plan_children(p):
        if not check_order(p[k], case):
            return False
    if op in ("head", "tail", "slice", "with_row_index"):
        return plan_ordered(p[1], case)
    if op == "unique" and (len(p[2]) > 1 or p[3] in ("first", "last")):
        return plan_ordered(p[1], case)
    if op in ("group_by", "with_columns", "select", "filter"):
        if uses_order(p) and not plan_ordered(p[1], case):
            return False
    return True


def uses_order(node):
    if isinstance(node, tuple) and node:
        if node[0] in ("first", "last", "arg_min", "arg_max", "cum_sum", "cum_max",
                       "cum_min", "cum_count", "shift", "diff"):
            return True
        return any(uses_order(k) for k in node[1:] if isinstance(k, tuple))
    return False


def shrink(case, sig, work, budget=400):
    """Greedy shrinking: take the first smaller variant that still fails."""
    tries = 0
    improved = True
    while improved and tries < budget:
        improved = False
        candidates = []
        for v in plan_variants(case["plan"]):
            c = dict(case)
            c["plan"] = v
            candidates.append(c)
        candidates += list(table_variants(case))
        for c in candidates:
            if tries >= budget:
                break
            if not still_valid(c):
                continue
            c["ordered"] = recompute_ordered(c)
            if not order_safe(c):
                continue
            tries += 1
            failures, _ = evaluate(c, work)
            if matches(sig, failures):
                case = c
                improved = True
                break
    return case


# ---------------------------------------------------------------------------
# Saving and replaying cases.


def save_case(case, failures, expected, d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, (df, specs) in enumerate(case["tables"]):
        df.write_parquet(d / f"table{i}.parquet")
    meta = {
        "formats": case["formats"],
        "specs": [specs for _, specs in case["tables"]],
        "plan": case["plan"],
        "ordered": case["ordered"],
    }
    (d / "case.json").write_text(json.dumps(meta, indent=1))
    lines = [f"plan: {case['plan']!r}", f"ordered: {case['ordered']}", ""]
    for i, ((df, specs), fmt) in enumerate(zip(case["tables"], case["formats"])):
        lines.append(f"table {i} ({fmt}): {specs}")
        lines.append(str(df))
    lines.append("")
    for k, m, t, detail in failures:
        lines.append(f"{k} [{m}, threads={t}]: {detail}")
    if expected is not None:
        lines += ["", "expected (Polars):", str(expected)]
    (d / "report.txt").write_text("\n".join(lines))


def tuplify(x):
    if isinstance(x, list):
        return tuple(tuplify(v) for v in x)
    return x


def load_case(d):
    meta = json.loads((d / "case.json").read_text())
    tables = []
    for i, specs in enumerate(meta["specs"]):
        df = pl.read_parquet(d / f"table{i}.parquet")
        df = df.cast({k: pl_dtype(v) for k, v in specs.items()})
        tables.append((df, specs))
    return {
        "tables": tables,
        "formats": meta["formats"],
        "plan": tuplify(meta["plan"]),
        "ordered": meta["ordered"],
    }


# ---------------------------------------------------------------------------
# Main loop.


def bucket(failures):
    """A coarse grouping key for the summary."""
    k, m, t, d = failures[0]
    if k == "error":
        return f"error: {first_line(d)}"
    if k == "mismatch":
        return "mismatch: " + first_line(d).split(":")[0]
    return k


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fail_dir = out / "failures"
    work = out / "work"
    rng = random.Random(args.seed)
    stats = {"cases": 0, "failed": 0, "oracle_error": 0}
    buckets = {}
    log = open(out / "log.txt", "a")
    t0 = time.time()
    for n in range(args.cases):
        seed = rng.randrange(1 << 30)
        crng = random.Random(seed)
        try:
            case = gen_case(crng)
        except Exception:
            traceback.print_exc(file=log)
            continue
        failures, expected = evaluate(case, work, args.threads)
        stats["cases"] += 1
        if failures and failures[0][0] == "oracle-error":
            stats["oracle_error"] += 1
            continue
        if not failures:
            continue
        stats["failed"] += 1
        b = bucket(failures)
        buckets[b] = buckets.get(b, 0) + 1
        print(f"[{n}] seed {seed}: {b}", flush=True)
        log.write(f"[{n}] seed {seed}: {b}\n")
        if buckets[b] > args.max_per_bucket:
            continue
        sig = signature(failures)
        small = shrink(case, sig, work, args.shrink_budget) if args.shrink else case
        small_failures, small_expected = evaluate(small, work, args.threads)
        if not small_failures:
            small, small_failures, small_expected = case, failures, expected
        save_case(small, small_failures, small_expected, fail_dir / f"seed{seed}")
        log.flush()
    elapsed = time.time() - t0
    summary = {"stats": stats, "buckets": buckets, "seconds": round(elapsed)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def replay(args):
    d = Path(args.case)
    case = load_case(d)
    failures, expected = evaluate(case, Path(args.out) / "replay", args.threads)
    print(f"plan: {case['plan']!r}")
    print("expected (Polars):")
    print(expected)
    if not failures:
        print("PASS: every mode matches")
        return 0
    for k, m, t, detail in failures:
        print(f"{k} [{m}, threads={t}]: {detail}")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="generate and check random cases (default)")
    r.add_argument("--cases", type=int, default=200)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", default=str(ROOT / "build/fuzz"))
    r.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    r.add_argument("--no-shrink", dest="shrink", action="store_false")
    r.add_argument("--shrink-budget", type=int, default=300)
    r.add_argument("--max-per-bucket", type=int, default=3, help="shrink and save at most this many failures per bucket")
    p = sub.add_parser("replay", help="rerun a saved case")
    p.add_argument("case")
    p.add_argument("--out", default=str(ROOT / "build/fuzz"))
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("run", "replay", "-h", "--help"):
        argv = ["run"] + argv
    args = ap.parse_args(argv)
    if args.cmd == "replay":
        sys.exit(replay(args))
    run(args)


if __name__ == "__main__":
    main()
