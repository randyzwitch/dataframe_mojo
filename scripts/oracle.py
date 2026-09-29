"""Differential testing against Polars (a development-only oracle).

For each seed: generate random inputs, pick an operation from the registry,
compute the expected result with Polars, run the same operation through the
Mojo runner, and compare. Both results are read back through Polars' CSV
reader with the same schema, so formatting differences cannot cause false
failures; floats compare with a relative tolerance.

    pixi run -e oracle oracle --cases 200 --seed 1
    pixi run -e oracle oracle --mutation-check

Failures print the seed, the operation, and minimized inputs (rows are
removed while the mismatch persists) so they can be reproduced.
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parent.parent
LEFT = {"k": pl.String, "g": pl.Int64, "x": pl.Float64, "y": pl.Float64, "n": pl.Int64, "b": pl.Boolean}
RIGHT = {"k": pl.String, "r": pl.Int64}
WORDS = ["a", "b", "c", "", "é", "x y", 'q"u', "c,d"]


def maybe(rng: random.Random, value, null_rate: float = 0.15):
    return None if rng.random() < null_rate else value


def gen_left(rng: random.Random, rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "k": [maybe(rng, rng.choice(WORDS[:4])) for _ in range(rows)],
            "g": [maybe(rng, rng.randint(-3, 3)) for _ in range(rows)],
            "x": [maybe(rng, rng.choice([-2.5, -0.0, 0.0, 0.1, 0.2, 0.5, 1.25, 3.0, 1e10])) for _ in range(rows)],
            "y": [maybe(rng, rng.choice([-1.0, 0.25, 2.0, 7.5])) for _ in range(rows)],
            "n": [maybe(rng, rng.randint(-1000, 1000)) for _ in range(rows)],
            "b": [maybe(rng, rng.random() < 0.5) for _ in range(rows)],
        },
        schema=LEFT,
    )


def gen_right(rng: random.Random, rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "k": [maybe(rng, rng.choice(WORDS)) for _ in range(rows)],
            "r": [rng.randint(0, 99) for _ in range(rows)],
        },
        schema=RIGHT,
    )


KINDS = ["filter", "arith", "agg", "sort", "join", "unique", "cum_sum", "cast", "stat", "describe", "value_counts", "decimal"]


def gen_op(rng: random.Random, kinds: list[str] = KINDS) -> list[str]:
    kind = rng.choice(kinds)
    if kind == "filter":
        column = rng.choice(["g", "x", "n", "k", "b"])
        op = rng.choice(["gt", "lt", "ge", "le", "eq", "ne"])
        if column == "b":
            op = rng.choice(["eq", "ne"])
        value = {
            "g": str(rng.randint(-2, 2)),
            "x": rng.choice(["0.5", "0.0", "-1.0"]),
            "n": str(rng.randint(-500, 500)),
            "k": rng.choice(["a", "b", ""]) or "a",
            "b": rng.choice(["true", "false"]),
        }[column]
        return ["filter", column, op, value]
    if kind == "decimal":
        return ["decimal", rng.choice(["add", "sub", "mul", "sum", "mean", "cast"])]
    if kind == "arith":
        pair = rng.choice([("x", "y"), ("g", "n"), ("n", "g")])
        op = rng.choice(["add", "sub", "mul", "div"])
        return ["arith", op, *pair]
    if kind == "agg":
        fn = rng.choice(["sum", "count", "min", "max", "mean", "n_unique", "first", "last"])
        column = rng.choice(["g", "x", "n"] + (["k", "b"] if fn in ("count", "n_unique", "first", "last", "min", "max") else []))
        return ["agg", fn, column]
    if kind == "sort":
        names = rng.sample(["k", "g", "x", "n", "b"], rng.randint(1, 3))
        return ["sort", ",".join(names), ",".join(rng.choice("01") for _ in names)]
    if kind == "join":
        return ["join", rng.choice(["inner", "left", "semi", "anti"])]
    if kind == "unique":
        return ["unique", ",".join(rng.sample(["k", "g", "b"], rng.randint(1, 2)))]
    if kind == "cum_sum":
        return ["cum_sum", rng.choice(["g", "x", "n"])]
    if kind == "stat":
        return gen_stat(rng)
    if kind == "describe":
        # Numeric and Bool columns; "xn" adds NaNs (see STAT_FNS).
        names = rng.sample(["g", "x", "y", "n", "b", "xn"], rng.randint(1, 4))
        qs = rng.sample(["0.1", "0.25", "0.5", "0.75", "0.9", "0.333", "0", "1"], rng.randint(0, 3))
        return ["describe", ",".join(names), ",".join(qs) or "none"]
    if kind == "value_counts":
        context = rng.choice(["global", "group"])
        column = rng.choice(["g", "n", "b", "x", "y"] + (["k"] if context == "global" else []))
        return ["value_counts", context, column, rng.choice("01"), rng.choice("01")]
    return ["cast", rng.choice(["x", "g", "b"]), rng.choice(["int64", "float64", "string"])]


# Statistics from #226. "xn" is x * (x / x): x, but NaN where x is zero, so
# NaN handling is covered without changing the shared inputs.
STAT_FNS = {
    "arg_min": ["g", "x", "n", "k", "b", "xn"],
    "arg_max": ["g", "x", "n", "k", "b", "xn"],
    "mode": ["g", "n", "k", "b", "y"],
    "skew": ["g", "x", "n", "y", "xn"],
    "skew_unbiased": ["x", "n", "y"],
    "kurtosis": ["g", "x", "n", "y", "xn"],
    "kurtosis_unbiased": ["x", "n", "y"],
    "kurtosis_pearson": ["x", "n", "y"],
    "corr": ["x,y", "g,n", "n,y"],
    "corr_spearman": ["x,y", "g,n", "n,y"],
    "cov": ["x,y", "g,n", "n,y"],
    "cov_ddof0": ["x,y", "g,n"],
}


def gen_stat(rng: random.Random) -> list[str]:
    fn = rng.choice(sorted(STAT_FNS))
    context = rng.choice(["global", "group"] + ([] if fn == "mode" else ["over"]))
    return ["stat", context, fn, rng.choice(STAT_FNS[fn])]


def stat_expr(fn: str, column: str) -> pl.Expr:
    if fn.startswith("corr") or fn.startswith("cov"):
        a, b = column.split(",")
        if fn == "corr":
            return pl.corr(a, b)
        if fn == "corr_spearman":
            return pl.corr(a, b, method="spearman")
        return pl.cov(a, b, ddof=0 if fn == "cov_ddof0" else 1)
    c = pl.col("x") * (pl.col("x") / pl.col("x")) if column == "xn" else pl.col(column)
    return {
        "arg_min": c.arg_min(), "arg_max": c.arg_max(), "mode": c.mode().sort(nulls_last=True),
        "skew": c.skew(), "skew_unbiased": c.skew(bias=False), "kurtosis": c.kurtosis(),
        "kurtosis_unbiased": c.kurtosis(bias=False), "kurtosis_pearson": c.kurtosis(fisher=False),
    }[fn]


def stat_input(left: pl.DataFrame, fn: str, column: str) -> pl.DataFrame:
    """corr/cov drop rows where either side is null (the documented
    semantics); Polars' grouped path disagrees with its global path at one
    or zero pairs, so both engines get the complete pairs."""
    if "," in column:
        a, b = column.split(",")
        return left.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
    return left


def expected(left: pl.DataFrame, right: pl.DataFrame, spec: list[str]) -> pl.DataFrame:
    op = spec[0]
    if op == "filter":
        column, cmp, raw = spec[1], spec[2], spec[3]
        value = {"g": int, "n": int, "x": float, "k": str, "b": lambda t: t == "true"}[column](raw)
        c = pl.col(column)
        pred = {"gt": c > value, "lt": c < value, "ge": c >= value, "le": c <= value, "eq": c == value, "ne": c != value}[cmp]
        return left.filter(pred)
    if op == "decimal":
        dtype = pl.Decimal(15, 2)
        x, y = pl.col("x").cast(dtype), pl.col("y").cast(dtype)
        if spec[1] == "cast":
            return left.select(x.cast(pl.String).alias("out"))
        if spec[1] == "mean":
            # Float64 in both engines (#341); compared as numbers, since
            # the two round the last digit of a long quotient differently.
            return left.select(x.mean().alias("out"))
        if spec[1] == "sum":
            return left.select(x.sum().cast(pl.String).alias("out"))
        e = {"add": x + y, "sub": x - y, "mul": x * y}[spec[1]]
        return left.select(e.cast(pl.String).alias("out"))
    if op == "arith":
        a, b = pl.col(spec[2]), pl.col(spec[3])
        e = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[spec[1]]
        return left.with_columns(e.alias("out"))
    if op == "agg":
        c = pl.col(spec[2])
        e = {
            "sum": c.sum(), "count": c.count(), "min": c.min(), "max": c.max(),
            "mean": c.mean(), "n_unique": c.n_unique(), "first": c.first(), "last": c.last(),
        }[spec[1]]
        return left.group_by("k", maintain_order=True).agg(e.alias("out"))
    if op == "sort":
        names = spec[1].split(",")
        flags = [f == "1" for f in spec[2].split(",")]
        return left.sort(names, descending=flags, nulls_last=True, maintain_order=True)
    if op == "join":
        how = spec[1]
        order = "left_right" if how in ("inner", "left") else "left"
        return left.join(right, on="k", how=how, maintain_order=order)
    if op == "unique":
        return left.unique(subset=spec[1].split(","), keep="first", maintain_order=True)
    if op == "cum_sum":
        return left.with_columns(pl.col(spec[1]).cum_sum().alias("out"))
    if op == "stat":
        context, fn, column = spec[1], spec[2], spec[3]
        frame = stat_input(left, fn, column)
        e = stat_expr(fn, column).alias("out")
        if context == "global" and fn == "arg_max" and column == "k":
            # Polars' global string arg_max returns the last tied index,
            # unlike its grouped/over paths and every other dtype, which
            # return the first; the contract is the first.
            best = frame.select(pl.col("k").max()).item()
            if best is None:
                return pl.DataFrame({"out": [None]}, schema={"out": pl.UInt32})
            first = frame.with_row_index("i").filter(pl.col("k") == best)["i"][0]
            return pl.DataFrame({"out": [first]}, schema={"out": pl.UInt32})
        if context == "global":
            return frame.select(e)
        if context == "over":
            return frame.with_columns(e.over("k"))
        grouped = frame.group_by("k", maintain_order=True).agg(e)
        return grouped.explode("out", empty_as_null=True) if fn == "mode" else grouped
    if op == "describe":
        frame = left.with_columns((pl.col("x") * (pl.col("x") / pl.col("x"))).alias("xn")).select(spec[1].split(","))
        qs = None if spec[2] == "none" else [float(q) for q in spec[2].split(",")]
        got = frame.describe(percentiles=qs)
        # Polars gives max null (but min NaN) for a column whose valid values
        # are all NaN; the contract is NaN for both.
        mins = got.row(4, named=True)
        return got.with_columns(
            pl.when((pl.col("statistic") == "max") & pl.lit(mins[c] is not None and math.isnan(mins[c])))
            .then(float("nan")).otherwise(pl.col(c)).alias(c)
            for c in frame.columns
        )
    if op == "value_counts":
        context, column, by_count, share = spec[1], spec[2], spec[3] == "1", spec[4] == "1"
        field = "proportion" if share else "count"
        e = pl.col(column).value_counts(sort=by_count, normalize=share).alias("out")
        # Polars' order is unspecified without sort and among count ties:
        # the contract is ascending by value, null last, after count.
        keys = ([field] if by_count else []) + [column]
        desc = ([True] if by_count else []) + [False]
        if context == "global":
            return left.select(e).unnest("out").sort(keys, descending=desc, nulls_last=True)
        grouped = left.group_by("k", maintain_order=True).agg(e).with_row_index("gi")
        flat = grouped.explode("out", empty_as_null=True).unnest("out")
        return flat.sort(["gi"] + keys, descending=[False] + desc, nulls_last=True).drop("gi")
    return left.with_columns(pl.col(spec[1]).cast({"int64": pl.Int64, "float64": pl.Float64, "string": pl.String}[spec[2]], strict=False).alias("out"))


def normalize(frame: pl.DataFrame) -> pl.DataFrame:
    # Mojo prints floats in shortest round-trip form and bools as true/false;
    # string casts are compared on the numeric value they represent.
    return frame.with_columns(pl.col(pl.Boolean).cast(pl.String))


def run_case(runner: Path, left: pl.DataFrame, right: pl.DataFrame, spec: list[str], env: dict) -> str | None:
    """Return None when Mojo matches Polars, else a failure description."""
    with tempfile.TemporaryDirectory() as tmp:
        lp, rp, out, exp = (Path(tmp) / n for n in ("l.csv", "r.csv", "o.csv", "e.csv"))
        left.write_csv(lp)
        right.write_csv(rp)
        want = expected(left, right, spec)
        want.write_csv(exp)
        proc = subprocess.run([str(runner), str(lp), str(rp), str(out), *spec], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            return "runner failed: " + (proc.stderr.strip().splitlines() or ["?"])[-1]
        schema = want.schema
        if spec[0] == "cast" and spec[2] == "string":
            schema = {**dict(schema), "out": pl.String}
        got = pl.read_csv(out, schema=schema) if out.stat().st_size else pl.DataFrame(schema=schema)
        ref = pl.read_csv(exp, schema=schema) if want.height or want.width else want
        if spec[0] == "cast" and spec[2] == "string":
            # Float text differs in form (1e10 vs 10000000000.0); compare values.
            source = want.schema[spec[1]]
            if source != pl.Boolean:
                got = got.with_columns(pl.col("out").cast(source, strict=False))
                ref = ref.with_columns(pl.col("out").cast(source, strict=False))
        try:
            # Moment statistics are computed by different algorithms (online
            # updates here, two-pass in Polars): compare them more loosely.
            statistic = spec[0] == "describe" or (spec[0] == "stat" and spec[2] not in ("arg_min", "arg_max", "mode"))
            assert_frame_equal(
                normalize(got), normalize(ref), check_dtypes=False,
                rel_tol=1e-6 if statistic else 1e-9, abs_tol=1e-9 if statistic else 1e-12,
            )
        except AssertionError as error:
            return str(error).splitlines()[0]
    return None


def minimize(runner, left, right, spec, env):
    """Drop rows one at a time while the case keeps failing."""
    changed = True
    while changed:
        changed = False
        for frame_name in ("left", "right"):
            frame = left if frame_name == "left" else right
            for i in range(frame.height):
                smaller = frame.with_row_index("_i").filter(pl.col("_i") != i).drop("_i")
                args = (smaller, right) if frame_name == "left" else (left, smaller)
                if run_case(runner, *args, spec, env):
                    left, right = args
                    changed = True
                    break
            if changed:
                break
    return left, right


def build_runner() -> Path:
    out = ROOT / "build" / "oracle_runner"
    out.parent.mkdir(exist_ok=True)
    subprocess.run(["mojo", "build", "-I", str(ROOT), str(ROOT / "tests" / "oracle" / "runner.mojo"), "-o", str(out)], check=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mutation-check", action="store_true")
    parser.add_argument("--kinds", default=",".join(KINDS), help="comma-separated operation kinds to generate")
    args = parser.parse_args()
    runner = build_runner()
    env = dict(os.environ)
    if args.mutation_check:
        env["DATAFRAME_ORACLE_INJECT"] = "1"
    failures = 0
    detected = 0
    for case in range(args.cases):
        seed = args.seed + case
        rng = random.Random(seed)
        left, right = gen_left(rng, rng.randint(0, 12)), gen_right(rng, rng.randint(0, 6))
        spec = gen_op(rng, args.kinds.split(","))
        problem = run_case(runner, left, right, spec, env)
        if args.mutation_check:
            expect_nonempty = expected(left, right, spec).height > 0
            detected += bool(problem) or not expect_nonempty
            continue
        if problem:
            failures += 1
            left, right = minimize(runner, left, right, spec, env)
            print(f"FAIL seed={seed} spec={' '.join(spec)}: {problem}")
            print("left:\n" + left.write_csv() + "right:\n" + right.write_csv())
    if args.mutation_check:
        print(f"mutation check: detected {detected}/{args.cases} injected bugs")
        return 0 if detected == args.cases else 1
    print(f"{args.cases - failures}/{args.cases} cases match Polars (seeds {args.seed}..{args.seed + args.cases - 1})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
