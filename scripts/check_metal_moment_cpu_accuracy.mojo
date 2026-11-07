from std.math import sqrt
from dataframe import Column, DataFrame, Expr, Series, col
from tests.support.coverage_generators import chunk_layout, offset_layout


def main() raises:
    var failures = 0
    for n in [2, 128, 65536]:
        var values = List[Int64]()
        var valid = List[Bool]()
        for i in range(n):
            values.append((Int64(1) << 60) + Int64(256 * (i % 2)))
            valid.append(i % 4 < 2)
        for nullable in [False, True]:
            var frame = DataFrame(
                [
                    Series(
                        "v", Column[Int64](values.copy(), valid.copy())
                    ) if nullable else Series(
                        "v", Column[Int64](values.copy())
                    ),
                    Series("key", Column[Int32](List[Int32](length=n, fill=1))),
                ]
            )
            var layouts = List[DataFrame](
                [
                    frame.copy(),
                    offset_layout(frame),
                    chunk_layout(frame),
                ]
            )
            var count = n // 2 if nullable and n >= 4 else n
            for layout_id in range(len(layouts)):
                for grouped in [False, True]:
                    for ddof in [0, 1]:
                        var expressions = List[Expr](
                            [
                                col("v").var(ddof).alias("variance"),
                                col("v").std(ddof).alias("deviation"),
                            ]
                        )
                        var actual = (
                            layouts[layout_id]
                            .lazy()
                            .group_by("key")
                            .agg(expressions)
                            .collect(engine="cpu") if grouped else layouts[
                                layout_id
                            ]
                            .lazy()
                            .select_exprs(expressions)
                            .collect(engine="cpu")
                        )
                        # Each retained pair is [2**60, 2**60+256], exactly
                        # representable in binary64. Centered squares are
                        # exactly 128**2 per value; no floating oracle sum.
                        var variance = (
                            Float64(16384)
                            * Float64(count)
                            / Float64(count - ddof)
                        )
                        var deviation = sqrt(variance)
                        var got_v = actual.item(0, "variance").float64()
                        var got_s = actual.item(0, "deviation").float64()
                        var v_error = abs(got_v - variance) / variance
                        var s_error = abs(got_s - deviation) / deviation
                        var passed = v_error <= 1e-10 and s_error <= 1e-10
                        if not passed:
                            failures += 1
                        print(
                            "CASE",
                            n,
                            nullable,
                            layout_id,
                            grouped,
                            ddof,
                            "actual",
                            got_v,
                            got_s,
                            "reference",
                            variance,
                            deviation,
                            "relative_error",
                            v_error,
                            s_error,
                            "PASS" if passed else "FAIL",
                        )
    print("MATHEMATICAL_REFERENCE_DIFFERENCES", failures)
