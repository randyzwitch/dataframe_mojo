"""Development controls for mixed aggregation and general row sorting.

FALLBACK_CASE=plain|mixed|computed|sort|packed|prefix|wide
FALLBACK_GROUPS=100000|1000000; DATAFRAME_THREADS=8.
One warmup followed by three timings, reporting the best milliseconds.
Use /usr/bin/time -f %M for process peak RSS (includes input construction).
"""
from std.os import getenv
from std.time import monotonic
from std.testing import assert_equal
from dataframe import DataFrame, Series, Column, Expr, col


def main() raises:
    var mode = getenv("FALLBACK_CASE", "mixed")
    var n = 1_000_000
    var groups = Int(getenv("FALLBACK_GROUPS", "100000"))
    var keys = List[Int64](capacity=n)
    var x = List[Int64](capacity=n)
    var y = List[Float64](capacity=n)
    var z = List[Float64](capacity=n)
    for i in range(n):
        keys.append(Int64((i * 37) % groups))
        x.append(Int64(i % 101 - 50))
        y.append(Float64(i % 97))
        z.append(Float64(i % 89))
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Int64](x^)),
            Series("y", Column[Float64](y^)),
            Series("z", Column[Float64](z^)),
        ]
    )
    var sorting = (
        mode == "sort" or mode == "packed" or mode == "prefix" or mode == "wide"
    )
    if sorting:
        var a = List[String](capacity=n)
        var b = List[String](capacity=n)
        var last = List[Int32](capacity=n)
        var first = List[Int64](capacity=n)
        for i in range(n):
            var text = String()
            for j in range(10 + i % 31):
                text += "abcdefghijklmnopqrstuvwxy"[
                    codepoint=(i * 13 + j * 7) % 25
                ]
            a.append(
                "abcdefghijklmnopqrstuvwx" + String(i % 2003) if mode
                == "prefix" else text.copy()
            )
            var other = String()
            for j in range(10 + (i * 7) % 31):
                other += "abcdefghijklmnopqrstuvwxy"[
                    codepoint=(i * 17 + j * 3) % 25
                ]
            b.append(other^)
            first.append(Int64(i % 997))
            last.append(Int32(i % 37))
        frame = DataFrame(
            [
                Series("k", Column[Int64](first^)),
                Series("a", Column[String](a^)),
                Series("b", Column[String](b^)),
                Series("last", Column[Int32](last^)),
            ]
        )
    var expressions: List[Expr] = [
        col("x").sum().alias("sx"),
        col("y").sum().alias("sy"),
        col("z").mean().alias("mz"),
    ]
    if mode == "mixed":
        expressions.append(col("z").median().alias("median"))
    if mode == "computed":
        expressions.append((col("y") + col("z")).sum().alias("computed"))
        expressions.append(col("x").first().alias("first"))
    var by: List[String] = ["k", "a", "b", "last"]
    var descending: List[Bool] = [True, False, False, False]
    var nulls: List[Bool] = [True, True, True, True]
    if mode == "packed":
        by = ["k", "last"]
        descending = [True, False]
        nulls = [True, True]
    if mode == "wide":
        frame = frame.with_columns(
            [
                (col("k") % 17).alias("extra_i"),
                col("a").str().slice(3, 20).alias("extra_a"),
                col("b").str().to_uppercase().alias("extra_b"),
                (col("last").cast("int64") * 7).alias("extra_last"),
            ]
        )
        by = [
            "k",
            "a",
            "b",
            "last",
            "extra_i",
            "extra_a",
            "extra_b",
            "extra_last",
        ]
        descending = [True, False, False, False, False, True, False, True]
        nulls = List[Bool](length=8, fill=True)
    var best = Float64(1e30)
    var checksum = Int64(0)
    for rep in range(4):
        var start = monotonic()
        if sorting:
            var order = frame.arg_sort(
                by, descending=descending, nulls_last=nulls
            )
            var ms = Float64(monotonic() - start) / 1e6
            if rep > 0:
                best = min(best, ms)
            assert_equal(len(order), n)
            checksum = Int64(0)
            for i in range(n):
                checksum += Int64((i + 1) * order[i])
        else:
            var result = frame.group_by("k").agg(expressions)
            var ms = Float64(monotonic() - start) / 1e6
            if rep > 0:
                best = min(best, ms)
            assert_equal(result.height(), groups)
            checksum = result.select(col("sx").sum()).item(0, "sx").int64()
    print(
        mode, "rows", n, "groups", groups, "best_ms", best, "checksum", checksum
    )
