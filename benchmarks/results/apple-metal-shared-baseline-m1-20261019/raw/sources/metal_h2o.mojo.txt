"""H2O.ai db-benchmark group-by and join queries (see engines.py for the
reference SQL and the Polars versions). Usage: see suite_common.mojo."""
from metal_coverage_audit import CoverageAuditRuntime as MetalRuntime
from std.collections import Dict
from std.sys import argv

from dataframe import DataFrame, col, corr, lit
from suite_common import run


def groupby(query: String, t: Dict[String, DataFrame]) raises -> DataFrame:
    var x = t["x"].lazy()
    if query == "q1":
        return (x.group_by("id1").agg([col("v1").sum().alias("v1")])).collect(accelerator=MetalRuntime())
    if query == "q2":
        return (x.group_by(["id1", "id2"]).agg([col("v1").sum().alias("v1")])).collect(accelerator=MetalRuntime())
    if query == "q3":
        return (x.group_by("id3").agg(
            [col("v1").sum().alias("v1"), col("v3").mean().alias("v3")]
        )).collect(accelerator=MetalRuntime())
    if query == "q4":
        return (x.group_by("id4").agg(
            [
                col("v1").mean().alias("v1"),
                col("v2").mean().alias("v2"),
                col("v3").mean().alias("v3"),
            ]
        )).collect(accelerator=MetalRuntime())
    if query == "q5":
        return (x.group_by("id6").agg(
            [
                col("v1").sum().alias("v1"),
                col("v2").sum().alias("v2"),
                col("v3").sum().alias("v3"),
            ]
        )).collect(accelerator=MetalRuntime())
    if query == "q6":
        return (x.group_by(["id4", "id5"]).agg(
            [
                col("v3").median().alias("median_v3"),
                col("v3").std().alias("sd_v3"),
            ]
        )).collect(accelerator=MetalRuntime())
    if query == "q7":
        return (x.group_by("id3").agg(
            [(col("v1").max() - col("v2").min()).alias("range_v1_v2")]
        )).collect(accelerator=MetalRuntime())
    if query == "q8":
        # Two largest v3 per id6: an ordinal rank over each group, as
        # DuckDB's row_number() does.
        return ((
            x.filter(col("v3").is_not_null())
            .filter(col("v3").rank("ordinal", descending=True).over("id6") <= 2)
            .select_exprs([col("id6"), col("v3").alias("largest2_v3")])
        )).collect(accelerator=MetalRuntime())
    if query == "q9":
        return (x.group_by(["id2", "id4"]).agg(
            [corr(col("v1"), col("v2")).pow(lit(2.0)).alias("r2")]
        )).collect(accelerator=MetalRuntime())
    if query == "q10":
        return (x.group_by(["id1", "id2", "id3", "id4", "id5", "id6"]).agg(
            [col("v3").sum().alias("v3"), col("v1").len().alias("count")]
        )).collect(accelerator=MetalRuntime())
    raise Error("unknown query " + query)


def join(query: String, t: Dict[String, DataFrame]) raises -> DataFrame:
    var x = t["x"].lazy()
    if query == "q1":
        return (x.join(t["small"].lazy(), "id1")).collect(accelerator=MetalRuntime())
    if query == "q2":
        return (x.join(t["medium"].lazy(), "id2")).collect(accelerator=MetalRuntime())
    if query == "q3":
        return (x.join(t["medium"].lazy(), "id2", how="left")).collect(accelerator=MetalRuntime())
    if query == "q4":
        return (x.join(t["medium"].lazy(), "id5")).collect(accelerator=MetalRuntime())
    if query == "q5":
        return (x.join(t["big"].lazy(), "id3")).collect(accelerator=MetalRuntime())
    raise Error("unknown query " + query)


def main() raises:
    if String(argv()[1]) == "h2o_groupby":
        run[groupby]()
    else:
        run[join]()
