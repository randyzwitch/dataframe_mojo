import sys, time, polars as pl
path, reps = sys.argv[1], int(sys.argv[2])
f = pl.read_csv(path, schema_overrides={"key_str": pl.String})
shapes = {
    "sum_count": [pl.col("x").sum().alias("s"), pl.col("n").count().alias("c")],
    "sum_count_mean": [pl.col("x").sum().alias("s"), pl.col("n").count().alias("c"), pl.col("y").mean().alias("m")],
    "sum": [pl.col("x").sum().alias("s")],
    "min_max_int": [pl.col("n").min().alias("lo"), pl.col("n").max().alias("hi")],
}
for shape, exprs in shapes.items():
    for key in ["key_low", "key_high", "key_skew"]:
        warm = f.group_by(key).agg(exprs)
        best = min((lambda: (lambda t: (f.group_by(key).agg(exprs), time.perf_counter_ns() - t)[1])(time.perf_counter_ns()))() for _ in range(reps))
        print(shape, key, best, warm.height, sep="\t")
