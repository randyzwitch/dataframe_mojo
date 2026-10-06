"""Polars and DuckDB workers for the external benchmark suites.

Usage (oracle environment):

    python engines.py ENGINE SUITE QUERIES REPS name=path [name=path ...]

QUERIES is comma-separated. The worker loads every table into memory once
(untimed), then for each query runs it once to warm up and times REPS runs.
Each timed run materializes the full result: Polars collects a DataFrame and
DuckDB creates a temporary table, as the H2O.ai db-benchmark does. It prints
the lines documented in suite_common.mojo, each tagged with its query, so
the driver checks every engine's answer the same way.

DuckDB runs the reference SQL for each suite: db-benchmark's queries,
`tpch_queries()` for PDS-H, `tpcds_queries()` for TPC-DS, and ClickBench's
`queries.sql`. The Polars versions are idiomatic translations of the same
queries. A TPC-DS query without a translation yet is reported as
unsupported, not dropped.
"""

from datetime import date
import os
import sys
import time

# --- answer summaries ---------------------------------------------------


def summary(frame) -> tuple:
    """Order-insensitive summary of a Polars result: its height and one
    number per column. Numbers and booleans sum (nulls skipped); strings sum
    their UTF-8 byte lengths; dates sum days since 1970-01-01; datetimes sum
    whole seconds. The Mojo runners compute the same values."""
    import polars as pl

    values = []
    for name, dtype in frame.schema.items():
        column = pl.col(name)
        if dtype == pl.String:
            expr = column.str.len_bytes().sum()
        elif dtype == pl.Date:
            expr = column.cast(pl.Int64).sum()
        elif isinstance(dtype, pl.Datetime):
            expr = column.dt.epoch("s").sum()
        elif dtype == pl.Boolean:
            expr = column.cast(pl.Int64).sum()
        elif dtype.is_numeric():
            expr = column.cast(pl.Float64).sum()
        else:
            expr = column.is_not_null().sum()
        value = frame.select(expr).item()
        values.append(float(value or 0))
    return frame.height, values


# --- H2O.ai db-benchmark ------------------------------------------------

H2O_GROUPBY_SQL = {
    "q1": "SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1",
    "q2": "SELECT id1, id2, sum(v1) AS v1 FROM x GROUP BY id1, id2",
    "q3": "SELECT id3, sum(v1) AS v1, avg(v3) AS v3 FROM x GROUP BY id3",
    "q4": "SELECT id4, avg(v1) AS v1, avg(v2) AS v2, avg(v3) AS v3 FROM x GROUP BY id4",
    "q5": "SELECT id6, sum(v1) AS v1, sum(v2) AS v2, sum(v3) AS v3 FROM x GROUP BY id6",
    "q6": "SELECT id4, id5, quantile_cont(v3, 0.5) AS median_v3, stddev(v3) AS sd_v3 FROM x GROUP BY id4, id5",
    "q7": "SELECT id3, max(v1) - min(v2) AS range_v1_v2 FROM x GROUP BY id3",
    "q8": "SELECT id6, largest2_v3 FROM (SELECT id6, v3 AS largest2_v3, row_number() OVER (PARTITION BY id6 ORDER BY v3 DESC) AS order_v3 FROM x WHERE v3 IS NOT NULL) sub_query WHERE order_v3 <= 2",
    "q9": "SELECT id2, id4, pow(corr(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4",
    "q10": "SELECT id1, id2, id3, id4, id5, id6, sum(v3) AS v3, count(*) AS count FROM x GROUP BY id1, id2, id3, id4, id5, id6",
}

H2O_JOIN_SQL = {
    "q1": "SELECT * FROM x JOIN small USING (id1)",
    "q2": "SELECT * FROM x JOIN medium USING (id2)",
    "q3": "SELECT * FROM x LEFT JOIN medium USING (id2)",
    "q4": "SELECT * FROM x JOIN medium USING (id5)",
    "q5": "SELECT * FROM x JOIN big USING (id3)",
}


def h2o_groupby_polars(t, q):
    import polars as pl

    x = t["x"].lazy()
    if q == "q1":
        out = x.group_by("id1").agg(pl.col("v1").sum())
    elif q == "q2":
        out = x.group_by("id1", "id2").agg(pl.col("v1").sum())
    elif q == "q3":
        out = x.group_by("id3").agg(pl.col("v1").sum(), pl.col("v3").mean())
    elif q == "q4":
        out = x.group_by("id4").agg(pl.col("v1", "v2", "v3").mean())
    elif q == "q5":
        out = x.group_by("id6").agg(pl.col("v1", "v2", "v3").sum())
    elif q == "q6":
        out = x.group_by("id4", "id5").agg(
            pl.col("v3").median().alias("median_v3"),
            pl.col("v3").std().alias("sd_v3"),
        )
    elif q == "q7":
        out = x.group_by("id3").agg(
            (pl.col("v1").max() - pl.col("v2").min()).alias("range_v1_v2")
        )
    elif q == "q8":
        out = (
            x.drop_nulls("v3")
            .group_by("id6")
            .agg(pl.col("v3").top_k(2).alias("largest2_v3"))
            .explode("largest2_v3")
        )
    elif q == "q9":
        out = x.group_by("id2", "id4").agg(
            (pl.corr("v1", "v2") ** 2).alias("r2")
        )
    else:
        out = x.group_by("id1", "id2", "id3", "id4", "id5", "id6").agg(
            pl.col("v3").sum(), pl.len().alias("count")
        )
    return out.collect()


def h2o_join_polars(t, q):
    x = t["x"].lazy()
    if q == "q1":
        out = x.join(t["small"].lazy(), on="id1")
    elif q == "q2":
        out = x.join(t["medium"].lazy(), on="id2")
    elif q == "q3":
        out = x.join(t["medium"].lazy(), on="id2", how="left")
    elif q == "q4":
        out = x.join(t["medium"].lazy(), on="id5")
    else:
        out = x.join(t["big"].lazy(), on="id3")
    return out.collect()


# --- PDS-H (TPC-H derived) ------------------------------------------------


def pdsh_polars(t, q):
    import polars as pl

    c = {name: frame.lazy() for name, frame in t.items()}
    line, orders, cust = c["lineitem"], c["orders"], c["customer"]
    part, supp, ps = c["part"], c["supplier"], c["partsupp"]
    nation, region = c["nation"], c["region"]
    disc_price = pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
    d = date
    if q == "q1":
        out = (
            line.filter(pl.col("l_shipdate") <= d(1998, 9, 2))
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.col("l_quantity").sum().alias("sum_qty"),
                pl.col("l_extendedprice").sum().alias("sum_base_price"),
                disc_price.sum().alias("sum_disc_price"),
                (disc_price * (1 + pl.col("l_tax"))).sum().alias("sum_charge"),
                pl.col("l_quantity").mean().alias("avg_qty"),
                pl.col("l_extendedprice").mean().alias("avg_price"),
                pl.col("l_discount").mean().alias("avg_disc"),
                pl.len().alias("count_order"),
            )
            .sort("l_returnflag", "l_linestatus")
        )
    elif q == "q2":
        europe = (
            region.filter(pl.col("r_name") == "EUROPE")
            .join(nation, left_on="r_regionkey", right_on="n_regionkey")
            .join(supp, left_on="n_nationkey", right_on="s_nationkey")
            .join(ps, left_on="s_suppkey", right_on="ps_suppkey")
        )
        brass = part.filter(
            (pl.col("p_size") == 15) & pl.col("p_type").str.ends_with("BRASS")
        ).join(europe, left_on="p_partkey", right_on="ps_partkey")
        out = (
            brass.filter(
                pl.col("ps_supplycost")
                == pl.col("ps_supplycost").min().over("p_partkey")
            )
            .select(
                "s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
                "s_address", "s_phone", "s_comment",
            )
            .sort(
                ["s_acctbal", "n_name", "s_name", "p_partkey"],
                descending=[True, False, False, False],
            )
            .head(100)
        )
    elif q == "q3":
        out = (
            cust.filter(pl.col("c_mktsegment") == "BUILDING")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .filter(pl.col("o_orderdate") < d(1995, 3, 15))
            .filter(pl.col("l_shipdate") > d(1995, 3, 15))
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(disc_price.sum().alias("revenue"))
            .select(
                pl.col("o_orderkey").alias("l_orderkey"),
                "revenue", "o_orderdate", "o_shippriority",
            )
            .sort(["revenue", "o_orderdate"], descending=[True, False])
            .head(10)
        )
    elif q == "q4":
        late = line.filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        out = (
            orders.filter(
                pl.col("o_orderdate").is_between(
                    d(1993, 7, 1), d(1993, 10, 1), closed="left"
                )
            )
            .join(late, left_on="o_orderkey", right_on="l_orderkey", how="semi")
            .group_by("o_orderpriority")
            .agg(pl.len().alias("order_count"))
            .sort("o_orderpriority")
        )
    elif q == "q5":
        out = (
            region.filter(pl.col("r_name") == "ASIA")
            .join(nation, left_on="r_regionkey", right_on="n_regionkey")
            .join(cust, left_on="n_nationkey", right_on="c_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .join(
                supp,
                left_on=["l_suppkey", "n_nationkey"],
                right_on=["s_suppkey", "s_nationkey"],
            )
            .filter(
                pl.col("o_orderdate").is_between(
                    d(1994, 1, 1), d(1995, 1, 1), closed="left"
                )
            )
            .group_by("n_name")
            .agg(disc_price.sum().alias("revenue"))
            .sort("revenue", descending=True)
        )
    elif q == "q6":
        out = line.filter(
            pl.col("l_shipdate").is_between(
                d(1994, 1, 1), d(1995, 1, 1), closed="left"
            )
            & pl.col("l_discount").is_between(0.05, 0.07)
            & (pl.col("l_quantity") < 24)
        ).select(
            (pl.col("l_extendedprice") * pl.col("l_discount"))
            .sum()
            .alias("revenue")
        )
    elif q == "q7":
        n1 = nation.select(
            pl.col("n_nationkey").alias("s_nationkey"),
            pl.col("n_name").alias("supp_nation"),
        )
        n2 = nation.select(
            pl.col("n_nationkey").alias("c_nationkey"),
            pl.col("n_name").alias("cust_nation"),
        )
        out = (
            line.filter(
                pl.col("l_shipdate").is_between(d(1995, 1, 1), d(1996, 12, 31))
            )
            .join(supp, left_on="l_suppkey", right_on="s_suppkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(cust, left_on="o_custkey", right_on="c_custkey")
            .join(n1, on="s_nationkey")
            .join(n2, on="c_nationkey")
            .filter(
                (
                    (pl.col("supp_nation") == "FRANCE")
                    & (pl.col("cust_nation") == "GERMANY")
                )
                | (
                    (pl.col("supp_nation") == "GERMANY")
                    & (pl.col("cust_nation") == "FRANCE")
                )
            )
            .with_columns(pl.col("l_shipdate").dt.year().alias("l_year"))
            .group_by("supp_nation", "cust_nation", "l_year")
            .agg(disc_price.sum().alias("revenue"))
            .sort("supp_nation", "cust_nation", "l_year")
        )
    elif q == "q8":
        n1 = nation.select("n_nationkey", "n_regionkey")
        n2 = nation.select(
            pl.col("n_nationkey").alias("s_nationkey"),
            pl.col("n_name").alias("nation"),
        )
        out = (
            part.filter(pl.col("p_type") == "ECONOMY ANODIZED STEEL")
            .join(line, left_on="p_partkey", right_on="l_partkey")
            .join(supp, left_on="l_suppkey", right_on="s_suppkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(cust, left_on="o_custkey", right_on="c_custkey")
            .join(n1, left_on="c_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(pl.col("r_name") == "AMERICA")
            .join(n2, on="s_nationkey")
            .filter(
                pl.col("o_orderdate").is_between(d(1995, 1, 1), d(1996, 12, 31))
            )
            .with_columns(
                pl.col("o_orderdate").dt.year().alias("o_year"),
                disc_price.alias("volume"),
            )
            .group_by("o_year")
            .agg(
                (
                    pl.when(pl.col("nation") == "BRAZIL")
                    .then(pl.col("volume"))
                    .otherwise(0)
                    .sum()
                    / pl.col("volume").sum()
                ).alias("mkt_share")
            )
            .sort("o_year")
        )
    elif q == "q9":
        out = (
            part.filter(pl.col("p_name").str.contains("green", literal=True))
            .join(line, left_on="p_partkey", right_on="l_partkey")
            .join(supp, left_on="l_suppkey", right_on="s_suppkey")
            .join(
                ps,
                left_on=["l_suppkey", "p_partkey"],
                right_on=["ps_suppkey", "ps_partkey"],
            )
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .with_columns(
                pl.col("n_name").alias("nation"),
                pl.col("o_orderdate").dt.year().alias("o_year"),
                (
                    disc_price
                    - pl.col("ps_supplycost") * pl.col("l_quantity")
                ).alias("amount"),
            )
            .group_by("nation", "o_year")
            .agg(pl.col("amount").sum().alias("sum_profit"))
            .sort(["nation", "o_year"], descending=[False, True])
        )
    elif q == "q10":
        out = (
            cust.join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey")
            .filter(
                pl.col("o_orderdate").is_between(
                    d(1993, 10, 1), d(1994, 1, 1), closed="left"
                )
                & (pl.col("l_returnflag") == "R")
            )
            .group_by(
                "c_custkey", "c_name", "c_acctbal", "c_phone", "n_name",
                "c_address", "c_comment",
            )
            .agg(disc_price.sum().alias("revenue"))
            .select(
                "c_custkey", "c_name", "revenue", "c_acctbal", "n_name",
                "c_address", "c_phone", "c_comment",
            )
            .sort("revenue", descending=True)
            .head(20)
        )
    elif q == "q11":
        german = (
            ps.join(supp, left_on="ps_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("n_name") == "GERMANY")
            .with_columns(
                (pl.col("ps_supplycost") * pl.col("ps_availqty")).alias("v")
            )
        )
        threshold = german.select((pl.col("v").sum() * 0.0001).alias("t"))
        out = (
            german.group_by("ps_partkey")
            .agg(pl.col("v").sum().alias("value"))
            .join(threshold, how="cross")
            .filter(pl.col("value") > pl.col("t"))
            .select("ps_partkey", "value")
            .sort("value", descending=True)
        )
    elif q == "q12":
        high = pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
        out = (
            orders.join(line, left_on="o_orderkey", right_on="l_orderkey")
            .filter(
                pl.col("l_shipmode").is_in(["MAIL", "SHIP"])
                & (pl.col("l_commitdate") < pl.col("l_receiptdate"))
                & (pl.col("l_shipdate") < pl.col("l_commitdate"))
                & pl.col("l_receiptdate").is_between(
                    d(1994, 1, 1), d(1995, 1, 1), closed="left"
                )
            )
            .group_by("l_shipmode")
            .agg(
                high.cast(pl.Int64).sum().alias("high_line_count"),
                (~high).cast(pl.Int64).sum().alias("low_line_count"),
            )
            .sort("l_shipmode")
        )
    elif q == "q13":
        out = (
            cust.join(
                orders.filter(
                    ~pl.col("o_comment").str.contains("special.*requests")
                ),
                left_on="c_custkey",
                right_on="o_custkey",
                how="left",
            )
            .group_by("c_custkey")
            .agg(pl.col("o_orderkey").count().alias("c_count"))
            .group_by("c_count")
            .agg(pl.len().alias("custdist"))
            .sort(["custdist", "c_count"], descending=[True, True])
        )
    elif q == "q14":
        out = (
            line.join(part, left_on="l_partkey", right_on="p_partkey")
            .filter(
                pl.col("l_shipdate").is_between(
                    d(1995, 9, 1), d(1995, 10, 1), closed="left"
                )
            )
            .select(
                (
                    100.0
                    * pl.when(pl.col("p_type").str.starts_with("PROMO"))
                    .then(disc_price)
                    .otherwise(0)
                    .sum()
                    / disc_price.sum()
                ).alias("promo_revenue")
            )
        )
    elif q == "q15":
        revenue = (
            line.filter(
                pl.col("l_shipdate").is_between(
                    d(1996, 1, 1), d(1996, 4, 1), closed="left"
                )
            )
            .group_by("l_suppkey")
            .agg(disc_price.sum().alias("total_revenue"))
        )
        out = (
            supp.join(revenue, left_on="s_suppkey", right_on="l_suppkey")
            .filter(
                pl.col("total_revenue") == pl.col("total_revenue").max()
            )
            .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
            .sort("s_suppkey")
        )
    elif q == "q16":
        complaints = supp.filter(
            pl.col("s_comment").str.contains("Customer.*Complaints")
        ).select(pl.col("s_suppkey").alias("ps_suppkey"))
        out = (
            part.filter(
                (pl.col("p_brand") != "Brand#45")
                & ~pl.col("p_type").str.starts_with("MEDIUM POLISHED")
                & pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9])
            )
            .join(ps, left_on="p_partkey", right_on="ps_partkey")
            .join(complaints, on="ps_suppkey", how="anti")
            .group_by("p_brand", "p_type", "p_size")
            .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort(
                ["supplier_cnt", "p_brand", "p_type", "p_size"],
                descending=[True, False, False, False],
            )
        )
    elif q == "q17":
        chosen = part.filter(
            (pl.col("p_brand") == "Brand#23")
            & (pl.col("p_container") == "MED BOX")
        ).join(line, left_on="p_partkey", right_on="l_partkey")
        out = (
            chosen.filter(
                pl.col("l_quantity")
                < 0.2 * pl.col("l_quantity").mean().over("p_partkey")
            )
            .select((pl.col("l_extendedprice").sum() / 7.0).alias("avg_yearly"))
        )
    elif q == "q18":
        big = (
            line.group_by("l_orderkey")
            .agg(pl.col("l_quantity").sum().alias("q"))
            .filter(pl.col("q") > 300)
            .select("l_orderkey")
        )
        out = (
            orders.join(big, left_on="o_orderkey", right_on="l_orderkey", how="semi")
            .join(cust, left_on="o_custkey", right_on="c_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .group_by(
                "c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice"
            )
            .agg(pl.col("l_quantity").sum().alias("sum_qty"))
            .select(
                "c_name",
                pl.col("o_custkey").alias("c_custkey"),
                "o_orderkey", "o_orderdate", "o_totalprice", "sum_qty",
            )
            .sort(["o_totalprice", "o_orderdate"], descending=[True, False])
            .head(100)
        )
    elif q == "q19":
        def branch(brand, containers, low, size):
            return (
                (pl.col("p_brand") == brand)
                & pl.col("p_container").is_in(containers)
                & pl.col("l_quantity").is_between(low, low + 10)
                & pl.col("p_size").is_between(1, size)
            )

        out = (
            line.join(part, left_on="l_partkey", right_on="p_partkey")
            .filter(
                pl.col("l_shipmode").is_in(["AIR", "AIR REG"])
                & (pl.col("l_shipinstruct") == "DELIVER IN PERSON")
                & (
                    branch("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
                    | branch("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
                    | branch("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
                )
            )
            .select(disc_price.sum().alias("revenue"))
        )
    elif q == "q20":
        shipped = (
            line.filter(
                pl.col("l_shipdate").is_between(
                    d(1994, 1, 1), d(1995, 1, 1), closed="left"
                )
            )
            .group_by("l_partkey", "l_suppkey")
            .agg((0.5 * pl.col("l_quantity").sum()).alias("half"))
        )
        forest = part.filter(pl.col("p_name").str.starts_with("forest"))
        suppliers = (
            ps.join(forest, left_on="ps_partkey", right_on="p_partkey", how="semi")
            .join(
                shipped,
                left_on=["ps_partkey", "ps_suppkey"],
                right_on=["l_partkey", "l_suppkey"],
            )
            .filter(pl.col("ps_availqty") > pl.col("half"))
            .select("ps_suppkey")
        )
        out = (
            supp.join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("n_name") == "CANADA")
            .join(suppliers, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
            .select("s_name", "s_address")
            .sort("s_name")
        )
    elif q == "q21":
        per_order = line.group_by("l_orderkey").agg(
            pl.col("l_suppkey").n_unique().alias("suppliers")
        )
        late = line.filter(pl.col("l_receiptdate") > pl.col("l_commitdate"))
        late_per_order = late.group_by("l_orderkey").agg(
            pl.col("l_suppkey").n_unique().alias("late_suppliers")
        )
        out = (
            late.join(per_order, on="l_orderkey")
            .join(late_per_order, on="l_orderkey")
            .filter((pl.col("suppliers") > 1) & (pl.col("late_suppliers") == 1))
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .filter(pl.col("o_orderstatus") == "F")
            .join(supp, left_on="l_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("n_name") == "SAUDI ARABIA")
            .group_by("s_name")
            .agg(pl.len().alias("numwait"))
            .sort(["numwait", "s_name"], descending=[True, False])
            .head(100)
        )
    else:
        codes = ["13", "31", "23", "29", "30", "18", "17"]
        chosen = cust.with_columns(
            pl.col("c_phone").str.slice(0, 2).alias("cntrycode")
        ).filter(pl.col("cntrycode").is_in(codes))
        average = chosen.filter(pl.col("c_acctbal") > 0.0).select(
            pl.col("c_acctbal").mean().alias("avg_bal")
        )
        out = (
            chosen.join(average, how="cross")
            .filter(pl.col("c_acctbal") > pl.col("avg_bal"))
            .join(orders, left_on="c_custkey", right_on="o_custkey", how="anti")
            .group_by("cntrycode")
            .agg(
                pl.len().alias("numcust"),
                pl.col("c_acctbal").sum().alias("totacctbal"),
            )
            .sort("cntrycode")
        )
    return out.collect()


# --- ClickBench ------------------------------------------------------------

CLICKBENCH_SQL = [
    "SELECT COUNT(*) FROM hits",
    "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0",
    "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits",
    "SELECT AVG(UserID) FROM hits",
    "SELECT COUNT(DISTINCT UserID) FROM hits",
    "SELECT COUNT(DISTINCT SearchPhrase) FROM hits",
    "SELECT MIN(EventDate), MAX(EventDate) FROM hits",
    "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC",
    "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10",
    "SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10",
    "SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10",
    "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10",
    "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
    "SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10",
    "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10",
    "SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10",
    "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10",
    "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10",
    "SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10",
    "SELECT UserID FROM hits WHERE UserID = 435090932899640449",
    "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'",
    "SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
    "SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
    "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10",
    "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10",
    "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10",
    "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10",
    "SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25",
    "SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25",
    "SELECT "
    + ", ".join(
        ["SUM(ResolutionWidth)"]
        + [f"SUM(ResolutionWidth + {i})" for i in range(1, 90)]
    )
    + " FROM hits",
    "SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10",
    "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10",
    "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10",
    "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10",
    "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10",
    "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10",
    "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10",
    "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10",
    "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000",
    "SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000",
    "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100",
    "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000",
    "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000",
]


def clickbench_polars(t, q):
    import polars as pl

    hits = t["hits"].lazy()
    n = int(q[1:])
    col = pl.col
    count = pl.len()
    phrase = col("SearchPhrase") != ""
    july = (
        (col("CounterID") == 62)
        & (col("EventDate") >= date(2013, 7, 1))
        & (col("EventDate") <= date(2013, 7, 31))
    )

    def top(frame, by, k=10, offset=0, descending=True):
        return frame.sort(by, descending=descending).slice(offset, k)

    if n == 0:
        out = hits.select(count)
    elif n == 1:
        out = hits.filter(col("AdvEngineID") != 0).select(count)
    elif n == 2:
        out = hits.select(
            col("AdvEngineID").sum(), count, col("ResolutionWidth").mean()
        )
    elif n == 3:
        out = hits.select(col("UserID").mean())
    elif n == 4:
        out = hits.select(col("UserID").n_unique())
    elif n == 5:
        out = hits.select(col("SearchPhrase").n_unique())
    elif n == 6:
        out = hits.select(
            col("EventDate").min().alias("min"), col("EventDate").max().alias("max")
        )
    elif n == 7:
        out = (
            hits.filter(col("AdvEngineID") != 0)
            .group_by("AdvEngineID")
            .agg(count.alias("c"))
            .sort("c", descending=True)
        )
    elif n == 8:
        out = top(
            hits.group_by("RegionID").agg(col("UserID").n_unique().alias("u")), "u"
        )
    elif n == 9:
        out = top(
            hits.group_by("RegionID").agg(
                col("AdvEngineID").sum(),
                count.alias("c"),
                col("ResolutionWidth").mean(),
                col("UserID").n_unique().alias("users"),
            ),
            "c",
        )
    elif n == 10:
        out = top(
            hits.filter(col("MobilePhoneModel") != "")
            .group_by("MobilePhoneModel")
            .agg(col("UserID").n_unique().alias("u")),
            "u",
        )
    elif n == 11:
        out = top(
            hits.filter(col("MobilePhoneModel") != "")
            .group_by("MobilePhone", "MobilePhoneModel")
            .agg(col("UserID").n_unique().alias("u")),
            "u",
        )
    elif n == 12:
        out = top(
            hits.filter(phrase).group_by("SearchPhrase").agg(count.alias("c")), "c"
        )
    elif n == 13:
        out = top(
            hits.filter(phrase)
            .group_by("SearchPhrase")
            .agg(col("UserID").n_unique().alias("u")),
            "u",
        )
    elif n == 14:
        out = top(
            hits.filter(phrase)
            .group_by("SearchEngineID", "SearchPhrase")
            .agg(count.alias("c")),
            "c",
        )
    elif n == 15:
        out = top(hits.group_by("UserID").agg(count.alias("c")), "c")
    elif n == 16:
        out = top(
            hits.group_by("UserID", "SearchPhrase").agg(count.alias("c")), "c"
        )
    elif n == 17:
        out = hits.group_by("UserID", "SearchPhrase").agg(count.alias("c")).head(10)
    elif n == 18:
        out = top(
            hits.group_by(
                "UserID", col("EventTime").dt.minute().alias("m"), "SearchPhrase"
            ).agg(count.alias("c")),
            "c",
        )
    elif n == 19:
        out = hits.filter(col("UserID") == 435090932899640449).select("UserID")
    elif n == 20:
        out = hits.filter(col("URL").str.contains("google", literal=True)).select(
            count
        )
    elif n == 21:
        out = top(
            hits.filter(col("URL").str.contains("google", literal=True) & phrase)
            .group_by("SearchPhrase")
            .agg(col("URL").min(), count.alias("c")),
            "c",
        )
    elif n == 22:
        out = top(
            hits.filter(
                col("Title").str.contains("Google", literal=True)
                & ~col("URL").str.contains(".google.", literal=True)
                & phrase
            )
            .group_by("SearchPhrase")
            .agg(
                col("URL").min(),
                col("Title").min(),
                count.alias("c"),
                col("UserID").n_unique().alias("users"),
            ),
            "c",
        )
    elif n == 23:
        out = top(
            hits.filter(col("URL").str.contains("google", literal=True)),
            "EventTime",
            descending=False,
        )
    elif n == 24:
        out = top(
            hits.filter(phrase).select("EventTime", "SearchPhrase"),
            "EventTime",
            descending=False,
        ).select("SearchPhrase")
    elif n == 25:
        out = top(
            hits.filter(phrase).select("SearchPhrase"),
            "SearchPhrase",
            descending=False,
        )
    elif n == 26:
        out = top(
            hits.filter(phrase).select("EventTime", "SearchPhrase"),
            ["EventTime", "SearchPhrase"],
            descending=False,
        ).select("SearchPhrase")
    elif n == 27:
        out = top(
            hits.filter(col("URL") != "")
            .group_by("CounterID")
            .agg(col("URL").str.len_chars().mean().alias("l"), count.alias("c"))
            .filter(col("c") > 100000),
            "l",
            k=25,
        )
    elif n == 28:
        out = top(
            hits.filter(col("Referer") != "")
            .with_columns(
                col("Referer")
                .str.replace(r"^https?://(?:www\.)?([^/]+)/.*$", "${1}")
                .alias("k")
            )
            .group_by("k")
            .agg(
                col("Referer").str.len_chars().mean().alias("l"),
                count.alias("c"),
                col("Referer").min(),
            )
            .filter(col("c") > 100000),
            "l",
            k=25,
        )
    elif n == 29:
        out = hits.select(
            [col("ResolutionWidth").sum().alias("s0")]
            + [
                (col("ResolutionWidth").cast(pl.Int64) + i).sum().alias(f"s{i}")
                for i in range(1, 90)
            ]
        )
    elif n in (30, 31, 32):
        source = hits.filter(phrase) if n < 32 else hits
        keys = ["SearchEngineID", "ClientIP"] if n == 30 else ["WatchID", "ClientIP"]
        out = top(
            source.group_by(keys).agg(
                count.alias("c"),
                col("IsRefresh").sum(),
                col("ResolutionWidth").mean(),
            ),
            "c",
        )
    elif n == 33:
        out = top(hits.group_by("URL").agg(count.alias("c")), "c")
    elif n == 34:
        out = top(
            hits.group_by("URL").agg(count.alias("c")).select(
                pl.lit(1).alias("one"), "URL", "c"
            ),
            "c",
        )
    elif n == 35:
        out = top(
            hits.group_by("ClientIP")
            .agg(count.alias("c"))
            .select(
                "ClientIP",
                (col("ClientIP") - 1).alias("m1"),
                (col("ClientIP") - 2).alias("m2"),
                (col("ClientIP") - 3).alias("m3"),
                "c",
            ),
            "c",
        )
    elif n in (36, 37):
        target = "URL" if n == 36 else "Title"
        out = top(
            hits.filter(
                july
                & (col("DontCountHits") == 0)
                & (col("IsRefresh") == 0)
                & (col(target) != "")
            )
            .group_by(target)
            .agg(count.alias("PageViews")),
            "PageViews",
        )
    elif n == 38:
        out = top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & (col("IsLink") != 0)
                & (col("IsDownload") == 0)
            )
            .group_by("URL")
            .agg(count.alias("PageViews")),
            "PageViews",
            offset=1000,
        )
    elif n == 39:
        out = top(
            hits.filter(july & (col("IsRefresh") == 0))
            .group_by(
                "TraficSourceID",
                "SearchEngineID",
                "AdvEngineID",
                pl.when((col("SearchEngineID") == 0) & (col("AdvEngineID") == 0))
                .then(col("Referer"))
                .otherwise(pl.lit(""))
                .alias("Src"),
                col("URL").alias("Dst"),
            )
            .agg(count.alias("PageViews")),
            "PageViews",
            offset=1000,
        )
    elif n == 40:
        out = top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & col("TraficSourceID").is_in([-1, 6])
                & (col("RefererHash") == 3594120000172545465)
            )
            .group_by("URLHash", "EventDate")
            .agg(count.alias("PageViews")),
            "PageViews",
            offset=100,
        )
    elif n == 41:
        out = top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & (col("DontCountHits") == 0)
                & (col("URLHash") == 2868770270353813622)
            )
            .group_by("WindowClientWidth", "WindowClientHeight")
            .agg(count.alias("PageViews")),
            "PageViews",
            offset=10000,
        )
    else:
        out = (
            hits.filter(
                (col("CounterID") == 62)
                & (col("EventDate") >= date(2013, 7, 14))
                & (col("EventDate") <= date(2013, 7, 15))
                & (col("IsRefresh") == 0)
                & (col("DontCountHits") == 0)
            )
            .group_by(col("EventTime").dt.truncate("1m").alias("M"))
            .agg(count.alias("PageViews"))
            .sort("M")
            .slice(1000, 10)
        )
    return out.collect()


# --- TPC-DS ---------------------------------------------------------------


def tpcds_polars(t, q):
    """The translated TPC-DS queries, as tpcds.mojo has them. Any other
    query is reported as not translated."""
    import polars as pl

    c = {name: frame.lazy() for name, frame in t.items()}
    col = pl.col
    d = date

    def dates(keep, key):
        return c["date_dim"].filter(keep).select(col("d_date_sk").alias(key))

    def brand_sales(days, items):
        return (
            c["date_dim"]
            .filter(days)
            .join(c["store_sales"], left_on="d_date_sk", right_on="ss_sold_date_sk")
            .join(c["item"].filter(items), left_on="ss_item_sk", right_on="i_item_sk")
        )

    def promoted_averages(sales, prefix, demographics):
        buyers = c["customer_demographics"].filter(
            (col("cd_gender") == "M")
            & (col("cd_marital_status") == "S")
            & (col("cd_education_status") == "College")
        )
        promotions = c["promotion"].filter(
            (col("p_channel_email") == "N") | (col("p_channel_event") == "N")
        )
        return (
            c[sales]
            .join(buyers, left_on=demographics, right_on="cd_demo_sk")
            .join(
                dates(col("d_year") == 2000, "d_date_sk"),
                left_on=prefix + "_sold_date_sk",
                right_on="d_date_sk",
            )
            .join(c["item"], left_on=prefix + "_item_sk", right_on="i_item_sk")
            .join(promotions, left_on=prefix + "_promo_sk", right_on="p_promo_sk")
            .group_by("i_item_id")
            .agg(
                col(prefix + "_quantity").mean().alias("agg1"),
                col(prefix + "_list_price").mean().alias("agg2"),
                col(prefix + "_coupon_amt").mean().alias("agg3"),
                col(prefix + "_sales_price").mean().alias("agg4"),
            )
            .sort("i_item_id", nulls_last=True)
            .head(100)
        )

    def returned_then_bought(sold, returned, bought):
        return (
            c["store_sales"]
            .join(dates(sold, "d1_sk"), left_on="ss_sold_date_sk", right_on="d1_sk")
            .join(c["item"], left_on="ss_item_sk", right_on="i_item_sk")
            .join(c["store"], left_on="ss_store_sk", right_on="s_store_sk")
            .join(
                c["store_returns"].join(
                    dates(returned, "d2_sk"),
                    left_on="sr_returned_date_sk",
                    right_on="d2_sk",
                ),
                left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
                right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
            )
            .join(
                c["catalog_sales"].join(
                    dates(bought, "d3_sk"),
                    left_on="cs_sold_date_sk",
                    right_on="d3_sk",
                ),
                left_on=["ss_customer_sk", "ss_item_sk"],
                right_on=["cs_bill_customer_sk", "cs_item_sk"],
            )
        )

    def stocked_items(low, high, makers, first_day, last_day, sales, key):
        in_stock = (
            c["inventory"]
            .filter(col("inv_quantity_on_hand").is_between(100, 500))
            .join(
                dates(col("d_date").is_between(first_day, last_day), "d_date_sk"),
                left_on="inv_date_sk",
                right_on="d_date_sk",
            )
        )
        return (
            c["item"]
            .filter(
                col("i_current_price").is_between(low, high)
                & col("i_manufact_id").is_in(makers)
            )
            .join(in_stock, left_on="i_item_sk", right_on="inv_item_sk")
            .join(c[sales], left_on="i_item_sk", right_on=key, how="semi")
            .select("i_item_id", "i_item_desc", "i_current_price")
            .unique()
            .sort("i_item_id", nulls_last=True)
            .head(100)
        )

    def one_if(condition):
        return pl.when(condition).then(1).otherwise(0)

    def sql_sum(value):
        """SQL's sum: null, not zero, when no value is valid."""
        return pl.when(value.count() > 0).then(value.sum())

    def day_buckets(shipped, sold):
        lag = col(shipped) - col(sold)
        return [
            one_if(lag <= 30).sum().alias("30 days"),
            one_if((lag > 30) & (lag <= 60)).sum().alias("31-60 days"),
            one_if((lag > 60) & (lag <= 90)).sum().alias("61-90 days"),
            one_if((lag > 90) & (lag <= 120)).sum().alias("91-120 days"),
            one_if(lag > 120).sum().alias(">120 days"),
        ]

    def shipping_delays(
        sales, prefix, channel, sales_key, channel_key, channel_name
    ):
        return (
            c[sales]
            .join(
                dates(col("d_month_seq").is_between(1200, 1211), "d_sk"),
                left_on=prefix + "_ship_date_sk",
                right_on="d_sk",
            )
            .join(
                c["warehouse"].select(
                    "w_warehouse_sk",
                    col("w_warehouse_name").str.slice(0, 20).alias("w_substr"),
                ),
                left_on=prefix + "_warehouse_sk",
                right_on="w_warehouse_sk",
            )
            .join(
                c["ship_mode"],
                left_on=prefix + "_ship_mode_sk",
                right_on="sm_ship_mode_sk",
            )
            .join(c[channel], left_on=sales_key, right_on=channel_key)
            .group_by("w_substr", "sm_type", channel_name)
            .agg(day_buckets(prefix + "_ship_date_sk", prefix + "_sold_date_sk"))
        )

    def tickets(days, stores, households, keys, sums, bought_city=False):
        sold = (
            c["store_sales"]
            .join(dates(days, "d_sk"), left_on="ss_sold_date_sk", right_on="d_sk")
            .join(
                c["store"].filter(stores),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                c["household_demographics"].filter(households),
                left_on="ss_hdemo_sk",
                right_on="hd_demo_sk",
            )
        )
        if bought_city:
            sold = sold.join(
                c["customer_address"].select(
                    col("ca_address_sk").alias("bought_sk"),
                    col("ca_city").alias("bought_city"),
                ),
                left_on="ss_addr_sk",
                right_on="bought_sk",
            )
        return sold.group_by(keys).agg(sums)

    three_years = col("d_year").is_in([1999, 2000, 2001])

    def excess_discount(sales, prefix, maker):
        amount = prefix + "_ext_discount_amt"
        in_range = c[sales].join(
            dates(col("d_date").is_between(d(2000, 1, 27), d(2000, 4, 26)), "d_sk"),
            left_on=prefix + "_sold_date_sk",
            right_on="d_sk",
        )
        typical = in_range.group_by(prefix + "_item_sk").agg(
            col(amount).cast(pl.Float64).mean().alias("typical")
        ).select(col(prefix + "_item_sk").alias("typical_item"), "typical")
        return (
            in_range.join(
                c["item"].filter(col("i_manufact_id") == maker),
                left_on=prefix + "_item_sk",
                right_on="i_item_sk",
            )
            .join(typical, left_on=prefix + "_item_sk", right_on="typical_item")
            .filter(col(amount).cast(pl.Float64) > 1.3 * col("typical"))
            .select(col(amount).sum().alias("excess"))
        )

    def buyers(sales, date_key, customer_key, days, marker):
        return (
            c[sales]
            .join(dates(days, "d_sk"), left_on=date_key, right_on="d_sk")
            .select(
                col(customer_key).alias(marker + "_sk"),
                col(customer_key).alias(marker),
            )
            .unique()
        )

    def active_customers(days, places, also_elsewhere):
        found = (
            c["customer"]
            .join(
                c["customer_address"].filter(places),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .join(
                c["customer_demographics"],
                left_on="c_current_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(
                buyers("store_sales", "ss_sold_date_sk", "ss_customer_sk", days, "st"),
                left_on="c_customer_sk",
                right_on="st_sk",
                how="semi",
            )
            .join(
                buyers(
                    "web_sales", "ws_sold_date_sk", "ws_bill_customer_sk", days, "web"
                ),
                left_on="c_customer_sk",
                right_on="web_sk",
                how="left",
            )
            .join(
                buyers(
                    "catalog_sales", "cs_sold_date_sk", "cs_ship_customer_sk",
                    days, "cat",
                ),  # fmt: skip
                left_on="c_customer_sk",
                right_on="cat_sk",
                how="left",
            )
        )
        if also_elsewhere:
            return found.filter(col("web").is_not_null() | col("cat").is_not_null())
        return found.filter(col("web").is_null() & col("cat").is_null())

    def split_shipments(
        sales, prefix, returns, returned_order, first_day, last_day, state,
        channel, sales_key, channel_key,
    ):  # fmt: skip
        order = prefix + "_order_number"
        warehouse = prefix + "_warehouse_sk"
        split = (
            c[sales]
            .group_by(order)
            .agg(
                col(warehouse).min().alias("first_warehouse"),
                col(warehouse).max().alias("last_warehouse"),
            )
            .filter(col("first_warehouse") != col("last_warehouse"))
            .select(col(order).alias("split_order"))
        )
        return (
            c[sales]
            .join(
                dates(col("d_date").is_between(first_day, last_day), "d_sk"),
                left_on=prefix + "_ship_date_sk",
                right_on="d_sk",
            )
            .join(
                c["customer_address"].filter(col("ca_state") == state),
                left_on=prefix + "_ship_addr_sk",
                right_on="ca_address_sk",
            )
            .join(channel, left_on=sales_key, right_on=channel_key)
            .join(split, left_on=order, right_on="split_order", how="semi")
            .join(c[returns], left_on=order, right_on=returned_order, how="anti")
            .select(
                col(order).n_unique().alias("order count"),
                sql_sum(col(prefix + "_ext_ship_cost")).alias("total shipping cost"),
                sql_sum(col(prefix + "_net_profit")).alias("total net profit"),
            )
        )

    def returners(returns, date_key, customer_key, address_key, amount, year):
        totals = (
            c[returns]
            .join(dates(col("d_year") == year, "d_sk"), left_on=date_key, right_on="d_sk")
            .join(
                c["customer_address"].select(
                    col("ca_address_sk").alias("return_address"),
                    col("ca_state").alias("ctr_state"),
                ),
                left_on=address_key,
                right_on="return_address",
            )
            .group_by(customer_key, "ctr_state")
            .agg(sql_sum(col(amount)).alias("ctr_total_return"))
            .select(
                col(customer_key).alias("ctr_customer_sk"),
                "ctr_state",
                "ctr_total_return",
            )
        )
        typical = totals.group_by("ctr_state").agg(
            col("ctr_total_return").cast(pl.Float64).mean().alias("typical")
        ).select(col("ctr_state").alias("typical_state"), "typical")
        return (
            totals.join(typical, left_on="ctr_state", right_on="typical_state")
            .filter(col("ctr_total_return").cast(pl.Float64) > 1.2 * col("typical"))
            .join(c["customer"], left_on="ctr_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"].filter(col("ca_state") == "GA"),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
        )

    if q == "q3":
        out = (
            brand_sales(col("d_moy") == 11, col("i_manufact_id") == 128)
            .group_by("d_year", "i_brand", "i_brand_id")
            .agg(col("ss_ext_sales_price").sum().alias("sum_agg"))
            .select(
                "d_year",
                col("i_brand_id").alias("brand_id"),
                col("i_brand").alias("brand"),
                "sum_agg",
            )
            .sort(
                ["d_year", "sum_agg", "brand_id"],
                descending=[False, True, False],
                nulls_last=True,
            )
            .head(100)
        )
    elif q == "q7":
        out = promoted_averages("store_sales", "ss", "ss_cdemo_sk")
    elif q in ("q13", "q48"):
        joined = (
            c["store_sales"]
            .join(
                c["store"].select("s_store_sk"),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                dates(col("d_year") == (2001 if q == "q13" else 2000), "d_sk"),
                left_on="ss_sold_date_sk",
                right_on="d_sk",
            )
            .join(
                c["customer_demographics"],
                left_on="ss_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(
                c["customer_address"].filter(col("ca_country") == "United States"),
                left_on="ss_addr_sk",
                right_on="ca_address_sk",
            )
        )
        price = col("ss_sales_price")
        profit = col("ss_net_profit")
        married = col("cd_marital_status")
        schooling = col("cd_education_status")
        if q == "q13":
            out = (
                joined.join(
                    c["household_demographics"],
                    left_on="ss_hdemo_sk",
                    right_on="hd_demo_sk",
                )
                .filter(
                    (
                        (married == "M")
                        & (schooling == "Advanced Degree")
                        & price.is_between(100, 150)
                        & (col("hd_dep_count") == 3)
                    )
                    | (
                        (married == "S")
                        & (schooling == "College")
                        & price.is_between(50, 100)
                        & (col("hd_dep_count") == 1)
                    )
                    | (
                        (married == "W")
                        & (schooling == "2 yr Degree")
                        & price.is_between(150, 200)
                        & (col("hd_dep_count") == 1)
                    )
                )
                .filter(
                    (col("ca_state").is_in(["TX", "OH"]) & profit.is_between(100, 200))
                    | (
                        col("ca_state").is_in(["OR", "NM", "KY"])
                        & profit.is_between(150, 300)
                    )
                    | (
                        col("ca_state").is_in(["VA", "TX", "MS"])
                        & profit.is_between(50, 250)
                    )
                )
                .select(
                    col("ss_quantity").mean().alias("avg1"),
                    col("ss_ext_sales_price").mean().alias("avg2"),
                    col("ss_ext_wholesale_cost").mean().alias("avg3"),
                    col("ss_ext_wholesale_cost").sum().alias("sum4"),
                )
            )
        else:
            out = (
                joined.filter(
                    (
                        (married == "M")
                        & (schooling == "4 yr Degree")
                        & price.is_between(100, 150)
                    )
                    | (
                        (married == "D")
                        & (schooling == "2 yr Degree")
                        & price.is_between(50, 100)
                    )
                    | (
                        (married == "S")
                        & (schooling == "College")
                        & price.is_between(150, 200)
                    )
                )
                .filter(
                    (
                        col("ca_state").is_in(["CO", "OH", "TX"])
                        & profit.is_between(0, 2000)
                    )
                    | (
                        col("ca_state").is_in(["OR", "MN", "KY"])
                        & profit.is_between(150, 3000)
                    )
                    | (
                        col("ca_state").is_in(["VA", "CA", "MS"])
                        & profit.is_between(50, 25000)
                    )
                )
                .select(col("ss_quantity").sum().alias("total"))
            )
    elif q == "q15":
        zips = [
            "85669", "86197", "88274", "83405", "86475",
            "85392", "85460", "80348", "81792",
        ]  # fmt: skip
        out = (
            c["catalog_sales"]
            .join(
                dates((col("d_qoy") == 2) & (col("d_year") == 2001), "d_sk"),
                left_on="cs_sold_date_sk",
                right_on="d_sk",
            )
            .join(
                c["customer"],
                left_on="cs_bill_customer_sk",
                right_on="c_customer_sk",
            )
            .join(
                c["customer_address"],
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .filter(
                col("ca_zip").str.slice(0, 5).is_in(zips)
                | col("ca_state").is_in(["CA", "WA", "GA"])
                | (col("cs_sales_price") > 500)
            )
            .group_by("ca_zip")
            .agg(col("cs_sales_price").sum().alias("total"))
            .sort("ca_zip", nulls_last=False)
            .head(100)
        )
    elif q == "q17":
        later = ["2001Q1", "2001Q2", "2001Q3"]
        aggregates = []
        for measure, label in (
            ("ss_quantity", "store_sales"),
            ("sr_return_quantity", "store_returns"),
            ("cs_quantity", "catalog_sales"),
        ):
            value = col(measure)
            label += "_quantity"
            aggregates += [
                value.count().alias(label + "count"),
                value.mean().alias(label + "ave"),
                value.std().alias(label + "stdev"),
                (value.std() / value.mean()).alias(label + "cov"),
            ]
        keys = ["i_item_id", "i_item_desc", "s_state"]
        out = (
            returned_then_bought(
                col("d_quarter_name") == "2001Q1",
                col("d_quarter_name").is_in(later),
                col("d_quarter_name").is_in(later),
            )
            .group_by(keys)
            .agg(aggregates)
            .sort(keys, nulls_last=False)
            .head(100)
        )
    elif q == "q19":
        out = (
            brand_sales(
                (col("d_moy") == 11) & (col("d_year") == 1998),
                col("i_manager_id") == 8,
            )
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"],
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .join(c["store"], left_on="ss_store_sk", right_on="s_store_sk")
            .filter(col("ca_zip").str.slice(0, 5) != col("s_zip").str.slice(0, 5))
            .group_by("i_brand", "i_brand_id", "i_manufact_id", "i_manufact")
            .agg(col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                col("i_brand_id").alias("brand_id"),
                col("i_brand").alias("brand"),
                "i_manufact_id",
                "i_manufact",
                "ext_price",
            )
            .sort(
                ["ext_price", "brand", "brand_id", "i_manufact_id", "i_manufact"],
                descending=[True, False, False, False, False],
                nulls_last=True,
            )
            .head(100)
        )
    elif q in ("q25", "q29"):
        keys = ["i_item_id", "i_item_desc", "s_store_id", "s_store_name"]
        if q == "q25":
            months = col("d_moy").is_between(4, 10) & (col("d_year") == 2001)
            linked = returned_then_bought(
                (col("d_moy") == 4) & (col("d_year") == 2001), months, months
            )
            aggregates = [
                col("ss_net_profit").sum().alias("store_sales_profit"),
                col("sr_net_loss").sum().alias("store_returns_loss"),
                col("cs_net_profit").sum().alias("catalog_sales_profit"),
            ]
        else:
            linked = returned_then_bought(
                (col("d_moy") == 9) & (col("d_year") == 1999),
                col("d_moy").is_between(9, 12) & (col("d_year") == 1999),
                col("d_year").is_in([1999, 2000, 2001]),
            )
            aggregates = [
                col("ss_quantity").sum().alias("store_sales_quantity"),
                col("sr_return_quantity").sum().alias("store_returns_quantity"),
                col("cs_quantity").sum().alias("catalog_sales_quantity"),
            ]
        out = (
            linked.group_by(keys)
            .agg(aggregates)
            .sort(keys, nulls_last=True)
            .head(100)
        )
    elif q == "q26":
        out = promoted_averages("catalog_sales", "cs", "cs_bill_cdemo_sk")
    elif q == "q37":
        out = stocked_items(
            68, 98, [677, 940, 694, 808], d(2000, 2, 1), d(2000, 4, 1),
            "catalog_sales", "cs_item_sk",
        )  # fmt: skip
    elif q == "q40":
        net = col("cs_sales_price") - col("cr_refunded_cash").fill_null(0)
        day = d(2000, 3, 11)
        out = (
            c["catalog_sales"]
            .join(
                c["catalog_returns"],
                left_on=["cs_order_number", "cs_item_sk"],
                right_on=["cr_order_number", "cr_item_sk"],
                how="left",
            )
            .join(
                c["warehouse"],
                left_on="cs_warehouse_sk",
                right_on="w_warehouse_sk",
            )
            .join(
                c["item"].filter(col("i_current_price").is_between(0.99, 1.49)),
                left_on="cs_item_sk",
                right_on="i_item_sk",
            )
            .join(
                c["date_dim"].filter(
                    col("d_date").is_between(d(2000, 2, 10), d(2000, 4, 10))
                ),
                left_on="cs_sold_date_sk",
                right_on="d_date_sk",
            )
            .group_by("w_state", "i_item_id")
            .agg(
                pl.when(col("d_date") < day)
                .then(net)
                .otherwise(0)
                .sum()
                .alias("sales_before"),
                pl.when(col("d_date") >= day)
                .then(net)
                .otherwise(0)
                .sum()
                .alias("sales_after"),
            )
            .sort(["w_state", "i_item_id"], nulls_last=True)
            .head(100)
        )
    elif q == "q42":
        out = (
            brand_sales(
                (col("d_moy") == 11) & (col("d_year") == 2000),
                col("i_manager_id") == 1,
            )
            .group_by("d_year", "i_category_id", "i_category")
            .agg(col("ss_ext_sales_price").sum().alias("total"))
            .sort(
                ["total", "d_year", "i_category_id", "i_category"],
                descending=[True, False, False, False],
                nulls_last=True,
            )
            .head(100)
        )
    elif q == "q43":
        days = [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday",
        ]  # fmt: skip
        sums = [
            pl.when(col("d_day_name") == day)
            .then(col("ss_sales_price"))
            .sum()
            .alias(day[:3].lower() + "_sales")
            for day in days
        ]
        out = (
            c["date_dim"]
            .filter(col("d_year") == 2000)
            .join(c["store_sales"], left_on="d_date_sk", right_on="ss_sold_date_sk")
            .join(
                c["store"].filter(col("s_gmt_offset") == -5),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .group_by("s_store_name", "s_store_id")
            .agg(sums)
            .sort(
                ["s_store_name", "s_store_id"]
                + [day[:3].lower() + "_sales" for day in days],
                nulls_last=True,
            )
            .head(100)
        )
    elif q == "q50":
        keys = [
            "s_store_name", "s_company_id", "s_street_number", "s_street_name",
            "s_street_type", "s_suite_number", "s_city", "s_county", "s_state",
            "s_zip",
        ]  # fmt: skip
        out = (
            c["store_sales"]
            .join(
                c["store_returns"].join(
                    dates((col("d_year") == 2001) & (col("d_moy") == 8), "d2_sk"),
                    left_on="sr_returned_date_sk",
                    right_on="d2_sk",
                ),
                left_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
                right_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"],
            )
            .join(
                dates(col("d_date_sk").is_not_null(), "d1_sk"),
                left_on="ss_sold_date_sk",
                right_on="d1_sk",
            )
            .join(c["store"], left_on="ss_store_sk", right_on="s_store_sk")
            .group_by(keys)
            .agg(day_buckets("sr_returned_date_sk", "ss_sold_date_sk"))
            .sort(keys, nulls_last=True)
            .head(100)
        )
    elif q == "q52":
        out = (
            brand_sales(
                (col("d_moy") == 11) & (col("d_year") == 2000),
                col("i_manager_id") == 1,
            )
            .group_by("d_year", "i_brand", "i_brand_id")
            .agg(col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                "d_year",
                col("i_brand_id").alias("brand_id"),
                col("i_brand").alias("brand"),
                "ext_price",
            )
            .sort(
                ["d_year", "ext_price", "brand_id"],
                descending=[False, True, False],
                nulls_last=True,
            )
            .head(100)
        )
    elif q == "q55":
        out = (
            brand_sales(
                (col("d_moy") == 11) & (col("d_year") == 1999),
                col("i_manager_id") == 28,
            )
            .group_by("i_brand", "i_brand_id")
            .agg(col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                col("i_brand_id").alias("brand_id"),
                col("i_brand").alias("brand"),
                "ext_price",
            )
            .sort(
                ["ext_price", "brand_id"], descending=[True, False], nulls_last=True
            )
            .head(100)
        )
    elif q == "q72":
        sold = (
            c["catalog_sales"]
            .join(
                c["date_dim"]
                .filter(col("d_year") == 1999)
                .select(
                    col("d_date_sk").alias("d1_sk"),
                    col("d_date").alias("sold_date"),
                    "d_week_seq",
                ),
                left_on="cs_sold_date_sk",
                right_on="d1_sk",
            )
            .join(
                c["customer_demographics"].filter(col("cd_marital_status") == "D"),
                left_on="cs_bill_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(
                c["household_demographics"].filter(
                    col("hd_buy_potential") == ">10000"
                ),
                left_on="cs_bill_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["date_dim"].select(
                    col("d_date_sk").alias("d3_sk"),
                    col("d_date").alias("ship_date"),
                ),
                left_on="cs_ship_date_sk",
                right_on="d3_sk",
            )
            .filter(col("ship_date") > col("sold_date") + pl.duration(days=5))
        )
        stock = c["inventory"].join(
            c["date_dim"].select(
                col("d_date_sk").alias("d2_sk"),
                col("d_week_seq").alias("inv_week_seq"),
            ),
            left_on="inv_date_sk",
            right_on="d2_sk",
        )
        out = (
            sold.join(
                stock,
                left_on=["cs_item_sk", "d_week_seq"],
                right_on=["inv_item_sk", "inv_week_seq"],
            )
            .filter(col("inv_quantity_on_hand") < col("cs_quantity"))
            .join(
                c["warehouse"],
                left_on="inv_warehouse_sk",
                right_on="w_warehouse_sk",
            )
            .join(c["item"], left_on="cs_item_sk", right_on="i_item_sk")
            .join(
                c["promotion"].select("p_promo_sk"),
                left_on="cs_promo_sk",
                right_on="p_promo_sk",
                how="left",
                coalesce=False,
            )
            .join(
                c["catalog_returns"].select("cr_item_sk", "cr_order_number"),
                left_on=["cs_item_sk", "cs_order_number"],
                right_on=["cr_item_sk", "cr_order_number"],
                how="left",
            )
            .group_by("i_item_desc", "w_warehouse_name", "d_week_seq")
            .agg(
                one_if(col("p_promo_sk").is_null()).sum().alias("no_promo"),
                one_if(col("p_promo_sk").is_not_null()).sum().alias("promo"),
                pl.len().alias("total_cnt"),
            )
            .sort(
                ["total_cnt", "i_item_desc", "w_warehouse_name", "d_week_seq"],
                descending=[True, False, False, False],
                nulls_last=False,
            )
            .head(100)
        )
    elif q == "q82":
        out = stocked_items(
            62, 92, [129, 270, 821, 423], d(2000, 5, 25), d(2000, 7, 24),
            "store_sales", "ss_item_sk",
        )  # fmt: skip
    elif q == "q84":
        out = (
            c["customer"]
            .join(
                c["customer_address"].filter(col("ca_city") == "Edgewood"),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .join(
                c["household_demographics"].join(
                    c["income_band"].filter(
                        (col("ib_lower_bound") >= 38128)
                        & (col("ib_upper_bound") <= 88128)
                    ),
                    left_on="hd_income_band_sk",
                    right_on="ib_income_band_sk",
                ),
                left_on="c_current_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["customer_demographics"].select("cd_demo_sk"),
                left_on="c_current_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(
                c["store_returns"].select("sr_cdemo_sk"),
                left_on="c_current_cdemo_sk",
                right_on="sr_cdemo_sk",
            )
            .select(
                col("c_customer_id").alias("customer_id"),
                pl.concat_str(
                    col("c_last_name").fill_null(""),
                    pl.lit(", "),
                    col("c_first_name").fill_null(""),
                ).alias("customername"),
            )
            .sort("customer_id", nulls_last=False)
            .head(100)
        )
    elif q == "q85":
        people = c["customer_demographics"]
        refunded = people.select(
            col("cd_demo_sk").alias("cd1_sk"),
            col("cd_marital_status").alias("marital"),
            col("cd_education_status").alias("education"),
        )
        returning = people.select(
            col("cd_demo_sk").alias("cd2_sk"),
            col("cd_marital_status").alias("marital2"),
            col("cd_education_status").alias("education2"),
        )
        price = col("ws_sales_price")
        profit = col("ws_net_profit")
        out = (
            c["web_sales"]
            .join(
                c["web_returns"],
                left_on=["ws_item_sk", "ws_order_number"],
                right_on=["wr_item_sk", "wr_order_number"],
            )
            .join(
                c["web_page"].select("wp_web_page_sk"),
                left_on="ws_web_page_sk",
                right_on="wp_web_page_sk",
            )
            .join(
                dates(col("d_year") == 2000, "d_sk"),
                left_on="ws_sold_date_sk",
                right_on="d_sk",
            )
            .join(refunded, left_on="wr_refunded_cdemo_sk", right_on="cd1_sk")
            .join(
                returning,
                left_on=["wr_returning_cdemo_sk", "marital", "education"],
                right_on=["cd2_sk", "marital2", "education2"],
            )
            .join(
                c["customer_address"].filter(col("ca_country") == "United States"),
                left_on="wr_refunded_addr_sk",
                right_on="ca_address_sk",
            )
            .join(c["reason"], left_on="wr_reason_sk", right_on="r_reason_sk")
            .filter(
                (
                    (col("marital") == "M")
                    & (col("education") == "Advanced Degree")
                    & price.is_between(100, 150)
                )
                | (
                    (col("marital") == "S")
                    & (col("education") == "College")
                    & price.is_between(50, 100)
                )
                | (
                    (col("marital") == "W")
                    & (col("education") == "2 yr Degree")
                    & price.is_between(150, 200)
                )
            )
            .filter(
                (col("ca_state").is_in(["IN", "OH", "NJ"]) & profit.is_between(100, 200))
                | (
                    col("ca_state").is_in(["WI", "CT", "KY"])
                    & profit.is_between(150, 300)
                )
                | (
                    col("ca_state").is_in(["LA", "IA", "AR"])
                    & profit.is_between(50, 250)
                )
            )
            .group_by("r_reason_desc")
            .agg(
                col("ws_quantity").mean().alias("avg1"),
                col("wr_refunded_cash").mean().alias("avg2"),
                col("wr_fee").mean().alias("avg3"),
            )
            .select(
                col("r_reason_desc").str.slice(0, 20).alias("reason"),
                "avg1",
                "avg2",
                "avg3",
            )
            .sort(["reason", "avg1", "avg2", "avg3"], nulls_last=True)
            .head(100)
        )
    elif q == "q91":
        buyers = (
            c["customer"]
            .join(
                c["customer_demographics"].filter(
                    (
                        (col("cd_marital_status") == "M")
                        & (col("cd_education_status") == "Unknown")
                    )
                    | (
                        (col("cd_marital_status") == "W")
                        & (col("cd_education_status") == "Advanced Degree")
                    )
                ),
                left_on="c_current_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(
                c["household_demographics"].filter(
                    col("hd_buy_potential").str.starts_with("Unknown")
                ),
                left_on="c_current_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["customer_address"].filter(col("ca_gmt_offset") == -7),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
        )
        out = (
            c["call_center"]
            .join(
                c["catalog_returns"],
                left_on="cc_call_center_sk",
                right_on="cr_call_center_sk",
            )
            .join(
                dates((col("d_year") == 1998) & (col("d_moy") == 11), "d_sk"),
                left_on="cr_returned_date_sk",
                right_on="d_sk",
            )
            .join(
                buyers,
                left_on="cr_returning_customer_sk",
                right_on="c_customer_sk",
            )
            .group_by(
                "cc_call_center_id", "cc_name", "cc_manager",
                "cd_marital_status", "cd_education_status",
            )  # fmt: skip
            .agg(col("cr_net_loss").sum().alias("Returns_Loss"))
            .select(
                col("cc_call_center_id").alias("Call_Center"),
                col("cc_name").alias("Call_Center_Name"),
                col("cc_manager").alias("Manager"),
                "Returns_Loss",
            )
            .sort("Returns_Loss", descending=True)
        )
    elif q == "q96":
        out = (
            c["store_sales"]
            .join(
                c["household_demographics"].filter(col("hd_dep_count") == 7),
                left_on="ss_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["time_dim"].filter(
                    (col("t_hour") == 20) & (col("t_minute") >= 30)
                ),
                left_on="ss_sold_time_sk",
                right_on="t_time_sk",
            )
            .join(
                c["store"].filter(col("s_store_name") == "ese"),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .select(pl.len().alias("count"))
        )
    elif q == "q21":
        day = d(2000, 3, 11)
        out = (
            c["inventory"]
            .join(
                c["warehouse"],
                left_on="inv_warehouse_sk",
                right_on="w_warehouse_sk",
            )
            .join(
                c["item"].filter(col("i_current_price").is_between(0.99, 1.49)),
                left_on="inv_item_sk",
                right_on="i_item_sk",
            )
            .join(
                c["date_dim"].filter(
                    col("d_date").is_between(d(2000, 2, 10), d(2000, 4, 10))
                ),
                left_on="inv_date_sk",
                right_on="d_date_sk",
            )
            .group_by("w_warehouse_name", "i_item_id")
            .agg(
                pl.when(col("d_date") < day)
                .then(col("inv_quantity_on_hand"))
                .otherwise(0)
                .sum()
                .alias("inv_before"),
                pl.when(col("d_date") >= day)
                .then(col("inv_quantity_on_hand"))
                .otherwise(0)
                .sum()
                .alias("inv_after"),
            )
            .filter(
                (col("inv_before") > 0)
                & (col("inv_after") / col("inv_before")).is_between(2 / 3, 1.5)
            )
            .sort(["w_warehouse_name", "i_item_id"], nulls_last=False)
            .head(100)
        )
    elif q == "q32":
        out = excess_discount("catalog_sales", "cs", 977)
    elif q in ("q34", "q73"):
        per_car = col("hd_dep_count") / col("hd_vehicle_count")
        potential = col("hd_buy_potential").is_in([">10000", "Unknown"])
        if q == "q34":
            days = (
                col("d_dom").is_between(1, 3) | col("d_dom").is_between(25, 28)
            ) & three_years
            stores = col("s_county") == "Williamson County"
            low, high, ratio = 15, 20, 1.2
        else:
            days = col("d_dom").is_between(1, 2) & three_years
            stores = col("s_county").is_in(
                [
                    "Orange County", "Bronx County", "Franklin Parish",
                    "Williamson County",
                ]  # fmt: skip
            )
            low, high, ratio = 1, 5, 1.0
        found = (
            tickets(
                days,
                stores,
                potential & (col("hd_vehicle_count") > 0) & (per_car > ratio),
                ["ss_ticket_number", "ss_customer_sk"],
                [pl.len().alias("cnt")],
            )
            .filter(col("cnt").is_between(low, high))
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .select(
                "c_last_name", "c_first_name", "c_salutation",
                "c_preferred_cust_flag", "ss_ticket_number", "cnt",
            )  # fmt: skip
        )
        if q == "q73":
            out = found.sort(["cnt", "c_last_name"], descending=[True, False])
        else:
            out = found.sort(
                [
                    "c_last_name", "c_first_name", "c_salutation",
                    "c_preferred_cust_flag", "ss_ticket_number",
                ],  # fmt: skip
                descending=[False, False, False, True, False],
                nulls_last=False,
            )
    elif q == "q41":
        shapes = [
            ("Women", ["powder", "khaki"], ["Ounce", "Oz"], ["medium", "extra large"]),
            ("Women", ["brown", "honeydew"], ["Bunch", "Ton"], ["N/A", "small"]),
            ("Men", ["floral", "deep"], ["N/A", "Dozen"], ["petite"]),
            ("Men", ["light", "cornflower"], ["Box", "Pound"], ["medium", "extra large"]),
            ("Women", ["midnight", "snow"], ["Pallet", "Gross"], ["medium", "extra large"]),
            ("Women", ["cyan", "papaya"], ["Cup", "Dram"], ["N/A", "small"]),
            ("Men", ["orange", "frosted"], ["Each", "Tbl"], ["petite"]),
            ("Men", ["forest", "ghost"], ["Lb", "Bundle"], ["medium", "extra large"]),
        ]  # fmt: skip
        described = pl.any_horizontal(
            (col("i_category") == category)
            & col("i_color").is_in(colors)
            & col("i_units").is_in(units)
            & col("i_size").is_in(sizes)
            for category, colors, units, sizes in shapes
        )
        makers = (
            c["item"]
            .filter(described)
            .select(col("i_manufact").alias("maker"))
            .unique()
        )
        out = (
            c["item"]
            .filter(col("i_manufact_id").is_between(738, 778))
            .join(makers, left_on="i_manufact", right_on="maker", how="semi")
            .select("i_product_name")
            .unique()
            .sort("i_product_name", nulls_last=True)
            .head(100)
        )
    elif q == "q45":
        zips = [
            "85669", "86197", "88274", "83405", "86475",
            "85392", "85460", "80348", "81792",
        ]  # fmt: skip
        listed = (
            c["item"]
            .filter(col("i_item_sk").is_in([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
            .select(
                col("i_item_id").alias("listed_id"),
                col("i_item_id").alias("listed"),
            )
            .unique()
        )
        out = (
            c["web_sales"]
            .join(
                dates((col("d_qoy") == 2) & (col("d_year") == 2001), "d_sk"),
                left_on="ws_sold_date_sk",
                right_on="d_sk",
            )
            .join(
                c["customer"],
                left_on="ws_bill_customer_sk",
                right_on="c_customer_sk",
            )
            .join(
                c["customer_address"],
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .join(c["item"], left_on="ws_item_sk", right_on="i_item_sk")
            .join(listed, left_on="i_item_id", right_on="listed_id", how="left")
            .filter(
                col("ca_zip").str.slice(0, 5).is_in(zips)
                | col("listed").is_not_null()
            )
            .group_by("ca_zip", "ca_city")
            .agg(col("ws_sales_price").sum().alias("total"))
            .sort(["ca_zip", "ca_city"], nulls_last=True)
            .head(100)
        )
    elif q in ("q46", "q68"):
        keys = ["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "bought_city"]
        if q == "q46":
            days = col("d_dow").is_in([6, 0]) & three_years
            sums = [
                col("ss_coupon_amt").sum().alias("amt"),
                col("ss_net_profit").sum().alias("profit"),
            ]
            shown = [
                "c_last_name", "c_first_name", "ca_city", "bought_city",
                "ss_ticket_number", "amt", "profit",
            ]  # fmt: skip
            order = [
                "c_last_name", "c_first_name", "ca_city", "bought_city",
                "ss_ticket_number",
            ]  # fmt: skip
        else:
            days = col("d_dom").is_between(1, 2) & three_years
            sums = [
                col("ss_ext_sales_price").sum().alias("extended_price"),
                col("ss_ext_list_price").sum().alias("list_price"),
                col("ss_ext_tax").sum().alias("extended_tax"),
            ]
            shown = [
                "c_last_name", "c_first_name", "ca_city", "bought_city",
                "ss_ticket_number", "extended_price", "extended_tax", "list_price",
            ]  # fmt: skip
            order = ["c_last_name", "ss_ticket_number"]
        out = (
            tickets(
                days,
                col("s_city").is_in(["Fairview", "Midway"]),
                (col("hd_dep_count") == 4) | (col("hd_vehicle_count") == 3),
                keys,
                sums,
                bought_city=True,
            )
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"],
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .filter(col("ca_city") != col("bought_city"))
            .select(shown)
            .sort(order, nulls_last=False)
            .head(100)
        )
    elif q == "q62":
        out = (
            shipping_delays(
                "web_sales", "ws", "web_site",
                "ws_web_site_sk", "web_site_sk", "web_name",
            )  # fmt: skip
            .sort(["w_substr", "sm_type", "web_name"], nulls_last=False)
            .head(100)
        )
    elif q == "q79":
        out = (
            tickets(
                (col("d_dow") == 1) & three_years,
                col("s_number_employees").is_between(200, 295),
                (col("hd_dep_count") == 6) | (col("hd_vehicle_count") > 2),
                ["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "s_city"],
                [
                    sql_sum(col("ss_coupon_amt")).alias("amt"),
                    sql_sum(col("ss_net_profit")).alias("profit"),
                ],
            )
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .select(
                "c_last_name",
                "c_first_name",
                col("s_city").str.slice(0, 30).alias("city"),
                "ss_ticket_number",
                "amt",
                "profit",
            )
            .sort(
                ["c_last_name", "c_first_name", "city", "profit", "ss_ticket_number"],
                nulls_last=[False, False, False, False, True],
            )
            .head(100)
        )
    elif q == "q92":
        out = excess_discount("web_sales", "ws", 350)
    elif q == "q93":
        kept = col("ss_quantity") - col("sr_return_quantity")
        out = (
            c["store_sales"]
            .join(
                c["store_returns"].join(
                    c["reason"].filter(col("r_reason_desc") == "reason 28"),
                    left_on="sr_reason_sk",
                    right_on="r_reason_sk",
                ),
                left_on=["ss_item_sk", "ss_ticket_number"],
                right_on=["sr_item_sk", "sr_ticket_number"],
            )
            .with_columns(
                (
                    pl.when(col("sr_return_quantity").is_not_null())
                    .then(kept)
                    .otherwise(col("ss_quantity"))
                    * col("ss_sales_price")
                ).alias("act_sales")
            )
            .group_by("ss_customer_sk")
            .agg(sql_sum(col("act_sales")).alias("sumsales"))
            .sort(["sumsales", "ss_customer_sk"], nulls_last=False)
            .head(100)
        )
    elif q == "q99":
        out = (
            shipping_delays(
                "catalog_sales", "cs", "call_center",
                "cs_call_center_sk", "cc_call_center_sk", "cc_name",
            )  # fmt: skip
            .select(
                "w_substr",
                "sm_type",
                col("cc_name").str.to_lowercase().alias("cc_name_lower"),
                "30 days", "31-60 days", "61-90 days", "91-120 days", ">120 days",
            )  # fmt: skip
            .sort(["w_substr", "sm_type", "cc_name_lower"], nulls_last=False)
            .head(100)
        )
    elif q == "q6":
        month = (
            c["date_dim"]
            .filter((col("d_year") == 2001) & (col("d_moy") == 1))
            .select("d_month_seq")
            .unique()
            .collect()
            .item()
        )
        price = col("i_current_price").cast(pl.Float64)
        # An item without a category has no average to compare with.
        costly = (
            c["item"]
            .filter(col("i_category").is_not_null())
            .filter(price > 1.2 * price.mean().over("i_category"))
            .select("i_item_sk")
        )
        out = (
            c["store_sales"]
            .join(
                dates(col("d_month_seq") == month, "d_sk"),
                left_on="ss_sold_date_sk",
                right_on="d_sk",
            )
            .join(costly, left_on="ss_item_sk", right_on="i_item_sk")
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"],
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .group_by("ca_state")
            .agg(pl.len().alias("cnt"))
            .filter(col("cnt") >= 10)
            .select(col("ca_state").alias("state"), "cnt")
            .sort(["cnt", "state"], nulls_last=False)
            .head(100)
        )
    elif q == "q9":
        buckets = []
        for i, (low, count) in enumerate(
            zip([1, 21, 41, 61, 81], [74129, 122840, 56580, 10097, 165306])
        ):
            band = col("ss_quantity").is_between(low, low + 19)
            buckets.append(
                pl.when(band.sum() > count)
                .then(
                    col("ss_ext_discount_amt").cast(pl.Float64).filter(band).mean()
                )
                .otherwise(col("ss_net_paid").cast(pl.Float64).filter(band).mean())
                .alias(f"bucket{i + 1}")
            )
        out = c["store_sales"].select(buckets)
    elif q == "q28":
        bands = []
        for i, (low, high, price, coupon, cost) in enumerate(
            zip(
                [0, 6, 11, 16, 21, 26],
                [5, 10, 15, 20, 25, 30],
                [8, 90, 142, 135, 122, 154],
                [459, 2323, 12214, 6071, 836, 7326],
                [57, 31, 79, 38, 17, 7],
            )
        ):
            tag = f"B{i + 1}"
            bands.append(
                c["store_sales"]
                .filter(
                    col("ss_quantity").is_between(low, high)
                    & (
                        col("ss_list_price").is_between(price, price + 10)
                        | col("ss_coupon_amt").is_between(coupon, coupon + 1000)
                        | col("ss_wholesale_cost").is_between(cost, cost + 20)
                    )
                    & col("ss_list_price").is_not_null()
                )
                .select(
                    col("ss_list_price").mean().alias(tag + "_LP"),
                    col("ss_list_price").count().alias(tag + "_CNT"),
                    col("ss_list_price").n_unique().alias(tag + "_CNTD"),
                )
            )
        out = pl.concat(bands, how="horizontal")
    elif q == "q61":
        promoted = c["promotion"].filter(
            (col("p_channel_dmail") == "Y")
            | (col("p_channel_email") == "Y")
            | (col("p_channel_tv") == "Y")
        ).select("p_promo_sk", col("p_promo_sk").alias("promoted"))
        out = (
            c["store_sales"]
            .join(
                dates((col("d_year") == 1998) & (col("d_moy") == 11), "d_sk"),
                left_on="ss_sold_date_sk",
                right_on="d_sk",
            )
            .join(
                c["store"].filter(col("s_gmt_offset") == -5),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                c["item"].filter(col("i_category") == "Jewelry"),
                left_on="ss_item_sk",
                right_on="i_item_sk",
            )
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"].filter(col("ca_gmt_offset") == -5),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .join(promoted, left_on="ss_promo_sk", right_on="p_promo_sk", how="left")
            .select(
                sql_sum(
                    pl.when(col("promoted").is_not_null()).then(
                        col("ss_ext_sales_price")
                    )
                ).alias("promotions"),
                sql_sum(col("ss_ext_sales_price")).alias("total"),
            )
            .with_columns(
                (
                    col("promotions").cast(pl.Float64)
                    / col("total").cast(pl.Float64)
                    * 100
                ).alias("share")
            )
        )
    elif q == "q65":
        revenue = (
            c["store_sales"]
            .join(
                dates(col("d_month_seq").is_between(1176, 1187), "d_sk"),
                left_on="ss_sold_date_sk",
                right_on="d_sk",
            )
            .group_by("ss_store_sk", "ss_item_sk")
            .agg(sql_sum(col("ss_sales_price")).alias("revenue"))
        )
        typical = revenue.group_by("ss_store_sk").agg(
            col("revenue").cast(pl.Float64).mean().alias("ave")
        ).select(col("ss_store_sk").alias("ave_store"), "ave")
        out = (
            revenue.join(typical, left_on="ss_store_sk", right_on="ave_store")
            .filter(col("revenue").cast(pl.Float64) <= 0.1 * col("ave"))
            .join(c["store"], left_on="ss_store_sk", right_on="s_store_sk")
            .join(c["item"], left_on="ss_item_sk", right_on="i_item_sk")
            .select(
                "s_store_name", "i_item_desc", "revenue",
                "i_current_price", "i_wholesale_cost", "i_brand",
            )  # fmt: skip
            .sort(["s_store_name", "i_item_desc"], nulls_last=False)
            .head(100)
        )
    elif q == "q88":
        households = (
            ((col("hd_dep_count") == 4) & (col("hd_vehicle_count") <= 6))
            | ((col("hd_dep_count") == 2) & (col("hd_vehicle_count") <= 4))
            | ((col("hd_dep_count") == 0) & (col("hd_vehicle_count") <= 2))
        )
        names = [
            "h8_30_to_9", "h9_to_9_30", "h9_30_to_10", "h10_to_10_30",
            "h10_30_to_11", "h11_to_11_30", "h11_30_to_12", "h12_to_12_30",
        ]  # fmt: skip
        counts = []
        for i, name in enumerate(names):
            hour = 8 + (i + 1) // 2
            half = col("t_minute") >= 30 if i % 2 == 0 else col("t_minute") < 30
            counts.append(one_if((col("t_hour") == hour) & half).sum().alias(name))
        out = (
            c["store_sales"]
            .join(
                c["household_demographics"].filter(households),
                left_on="ss_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["store"].filter(col("s_store_name") == "ese"),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                c["time_dim"].filter(col("t_hour").is_between(8, 12)),
                left_on="ss_sold_time_sk",
                right_on="t_time_sk",
            )
            .select(counts)
        )
    elif q == "q90":
        out = (
            c["web_sales"]
            .join(
                c["household_demographics"].filter(col("hd_dep_count") == 6),
                left_on="ws_ship_hdemo_sk",
                right_on="hd_demo_sk",
            )
            .join(
                c["web_page"].filter(col("wp_char_count").is_between(5000, 5200)),
                left_on="ws_web_page_sk",
                right_on="wp_web_page_sk",
            )
            .join(c["time_dim"], left_on="ws_sold_time_sk", right_on="t_time_sk")
            .select(
                one_if(col("t_hour").is_between(8, 9)).sum().alias("amc"),
                one_if(col("t_hour").is_between(19, 20)).sum().alias("pmc"),
            )
            .select(
                pl.when(col("pmc") != 0)
                .then(col("amc") / col("pmc"))
                .alias("am_pm_ratio")
            )
        )
    elif q in ("q10", "q69"):
        keys = [
            "cd_gender", "cd_marital_status", "cd_education_status",
            "cd_purchase_estimate", "cd_credit_rating",
        ]  # fmt: skip
        if q == "q10":
            keys += ["cd_dep_count", "cd_dep_employed_count", "cd_dep_college_count"]
            customers = active_customers(
                (col("d_year") == 2002) & col("d_moy").is_between(1, 4),
                col("ca_county").is_in(
                    [
                        "Rush County", "Toole County", "Jefferson County",
                        "Dona Ana County", "La Porte County",
                    ]  # fmt: skip
                ),
                True,
            )
        else:
            customers = active_customers(
                (col("d_year") == 2001) & col("d_moy").is_between(4, 6),
                col("ca_state").is_in(["KY", "GA", "NM"]),
                False,
            )
        shown = []
        for i, key in enumerate(keys):
            shown.append(col(key))
            if i >= 2:
                shown.append(col("cnt").alias(f"cnt{i - 1}"))
        out = (
            customers.group_by(keys)
            .agg(pl.len().alias("cnt"))
            .select(shown)
            .sort(keys, nulls_last=True)
            .head(100)
        )
    elif q == "q35":
        keys = [
            "ca_state", "cd_gender", "cd_marital_status", "cd_dep_count",
            "cd_dep_employed_count", "cd_dep_college_count",
        ]  # fmt: skip
        aggregates = [pl.len().alias("cnt")]
        shown = []
        for i, key in enumerate(keys):
            shown.append(col(key))
            if i >= 3:
                n = str(i - 2)
                aggregates += [
                    col(key).min().alias("min" + n),
                    col(key).max().alias("max" + n),
                    col(key).mean().alias("avg" + n),
                ]
                shown += [
                    col("cnt").alias("cnt" + n),
                    col("min" + n), col("max" + n), col("avg" + n),
                ]  # fmt: skip
        out = (
            active_customers(
                (col("d_year") == 2002) & (col("d_qoy") < 4),
                col("ca_address_sk").is_not_null(),
                True,
            )
            .group_by(keys)
            .agg(aggregates)
            .select(shown)
            .sort(keys, nulls_last=False)
            .head(100)
        )
    elif q == "q16":
        out = split_shipments(
            "catalog_sales", "cs", "catalog_returns", "cr_order_number",
            d(2002, 2, 1), d(2002, 4, 2), "GA",
            c["call_center"].filter(col("cc_county") == "Williamson County"),
            "cs_call_center_sk", "cc_call_center_sk",
        )  # fmt: skip
    elif q == "q94":
        out = split_shipments(
            "web_sales", "ws", "web_returns", "wr_order_number",
            d(1999, 2, 1), d(1999, 4, 2), "IL",
            c["web_site"].filter(col("web_company_name") == "pri"),
            "ws_web_site_sk", "web_site_sk",
        )  # fmt: skip
    elif q == "q1":
        totals = (
            c["store_returns"]
            .join(
                dates(col("d_year") == 2000, "d_sk"),
                left_on="sr_returned_date_sk",
                right_on="d_sk",
            )
            .group_by("sr_customer_sk", "sr_store_sk")
            .agg(sql_sum(col("sr_return_amt")).alias("total"))
        )
        typical = totals.group_by("sr_store_sk").agg(
            col("total").cast(pl.Float64).mean().alias("typical")
        ).select(col("sr_store_sk").alias("typical_store"), "typical")
        out = (
            totals.join(typical, left_on="sr_store_sk", right_on="typical_store")
            .filter(col("total").cast(pl.Float64) > 1.2 * col("typical"))
            .join(
                c["store"].filter(col("s_state") == "TN"),
                left_on="sr_store_sk",
                right_on="s_store_sk",
            )
            .join(c["customer"], left_on="sr_customer_sk", right_on="c_customer_sk")
            .select("c_customer_id")
            .sort("c_customer_id", nulls_last=True)
            .head(100)
        )
    elif q == "q24":
        keys = [
            "c_last_name", "c_first_name", "s_store_name", "ca_state", "s_state",
            "i_color", "i_current_price", "i_manager_id", "i_units", "i_size",
        ]  # fmt: skip
        ssales = (
            c["store_sales"]
            .join(
                c["store_returns"].select("sr_ticket_number", "sr_item_sk"),
                left_on=["ss_ticket_number", "ss_item_sk"],
                right_on=["sr_ticket_number", "sr_item_sk"],
            )
            .join(
                c["store"].filter(col("s_market_id") == 8),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(c["item"], left_on="ss_item_sk", right_on="i_item_sk")
            .join(c["customer"], left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                c["customer_address"],
                left_on=["c_current_addr_sk", "s_zip"],
                right_on=["ca_address_sk", "ca_zip"],
            )
            .filter(col("c_birth_country") != col("ca_country").str.to_uppercase())
            .group_by(keys)
            .agg(sql_sum(col("ss_net_paid")).alias("netpaid"))
            .collect()
        )
        # 5% of the mean over every row of ssales; with no rows the mean
        # is null and the comparison keeps nothing.
        average = ssales.select(col("netpaid").cast(pl.Float64).mean()).item()
        above = pl.lit(False)
        if average is not None:
            above = col("paid").cast(pl.Float64) > average * 0.05
        out = (
            ssales.lazy()
            .filter(col("i_color") == "peach")
            .group_by("c_last_name", "c_first_name", "s_store_name")
            .agg(sql_sum(col("netpaid")).alias("paid"))
            .filter(above)
            .sort(["c_last_name", "c_first_name", "s_store_name"], nulls_last=True)
        )
    elif q == "q30":
        shown = [
            "c_customer_id", "c_salutation", "c_first_name", "c_last_name",
            "c_preferred_cust_flag", "c_birth_day", "c_birth_month",
            "c_birth_year", "c_birth_country", "c_login", "c_email_address",
            "c_last_review_date_sk", "ctr_total_return",
        ]  # fmt: skip
        out = (
            returners(
                "web_returns", "wr_returned_date_sk", "wr_returning_customer_sk",
                "wr_returning_addr_sk", "wr_return_amt", 2002,
            )  # fmt: skip
            .select(shown)
            .sort(shown, nulls_last=False)
            .head(100)
        )
    elif q == "q39":
        quantity = col("inv_quantity_on_hand").cast(pl.Float64)
        stock = (
            c["inventory"]
            .join(c["item"].select("i_item_sk"), left_on="inv_item_sk", right_on="i_item_sk")
            .join(
                c["warehouse"].select("w_warehouse_sk", "w_warehouse_name"),
                left_on="inv_warehouse_sk",
                right_on="w_warehouse_sk",
            )
            .join(
                c["date_dim"].filter(col("d_year") == 2001).select("d_date_sk", "d_moy"),
                left_on="inv_date_sk",
                right_on="d_date_sk",
            )
            .group_by("w_warehouse_name", "inv_warehouse_sk", "inv_item_sk", "d_moy")
            .agg(quantity.std().alias("stdev"), quantity.mean().alias("mean"))
            .filter((col("mean") != 0) & (col("stdev") / col("mean") > 1))
            .with_columns((col("stdev") / col("mean")).alias("cov"))
        )
        january = stock.filter(col("d_moy") == 1).select(
            col("inv_warehouse_sk").alias("wsk1"),
            col("inv_item_sk").alias("isk1"),
            col("d_moy").alias("dmoy1"),
            col("mean").alias("mean1"),
            col("cov").alias("cov1"),
        )
        february = stock.filter(col("d_moy") == 2).select(
            col("inv_warehouse_sk").alias("wsk2"),
            col("inv_item_sk").alias("isk2"),
            col("d_moy").alias("dmoy2"),
            col("mean").alias("mean2"),
            col("cov").alias("cov2"),
        )
        out = (
            january.join(february, left_on=["isk1", "wsk1"], right_on=["isk2", "wsk2"])
            .select(
                "wsk1", "isk1", "dmoy1", "mean1", "cov1",
                col("wsk1").alias("wsk2"), col("isk1").alias("isk2"),
                "dmoy2", "mean2", "cov2",
            )  # fmt: skip
            .sort(
                ["wsk1", "isk1", "dmoy1", "mean1", "cov1", "dmoy2", "mean2", "cov2"],
                nulls_last=False,
            )
        )
    elif q == "q59":
        days = [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday",
        ]  # fmt: skip
        short = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
        sums = [
            sql_sum(pl.when(col("d_day_name") == day).then(col("ss_sales_price")))
            .alias(name + "_sales")
            for day, name in zip(days, short)
        ]
        weekly = (
            c["store_sales"]
            .join(
                c["date_dim"].select("d_date_sk", "d_week_seq", "d_day_name"),
                left_on="ss_sold_date_sk",
                right_on="d_date_sk",
            )
            .group_by("d_week_seq", "ss_store_sk")
            .agg(sums)
            .join(
                c["store"].select("s_store_sk", "s_store_name", "s_store_id"),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
        )
        weeks = c["date_dim"].select(col("d_week_seq").alias("week"), "d_month_seq")
        first = weekly.join(
            weeks.filter(col("d_month_seq").is_between(1212, 1223)),
            left_on="d_week_seq",
            right_on="week",
        ).select(
            col("s_store_name").alias("s_store_name1"),
            col("d_week_seq").alias("d_week_seq1"),
            col("s_store_id").alias("s_store_id1"),
            (col("d_week_seq") + 52).alias("next_week"),
            *[col(name + "_sales").alias(name + "_sales1") for name in short],
        )
        second = weekly.join(
            weeks.filter(col("d_month_seq").is_between(1224, 1235)),
            left_on="d_week_seq",
            right_on="week",
        ).select(
            col("d_week_seq").alias("d_week_seq2"),
            col("s_store_id").alias("s_store_id2"),
            *[col(name + "_sales").alias(name + "_sales2") for name in short],
        )
        out = (
            first.join(
                second,
                left_on=["s_store_id1", "next_week"],
                right_on=["s_store_id2", "d_week_seq2"],
            )
            .select(
                "s_store_name1",
                "s_store_id1",
                "d_week_seq1",
                *[
                    (
                        col(name + "_sales1").cast(pl.Float64)
                        / col(name + "_sales2").cast(pl.Float64)
                    ).alias(name + "_sales_ratio")
                    for name in short
                ],
            )
            .sort(["s_store_name1", "s_store_id1", "d_week_seq1"], nulls_last=False)
            .head(100)
        )
    elif q == "q81":
        shown = [
            "c_customer_id", "c_salutation", "c_first_name", "c_last_name",
            "ca_street_number", "ca_street_name", "ca_street_type",
            "ca_suite_number", "ca_city", "ca_county", "ca_state", "ca_zip",
            "ca_country", "ca_gmt_offset", "ca_location_type", "ctr_total_return",
        ]  # fmt: skip
        out = (
            returners(
                "catalog_returns", "cr_returned_date_sk",
                "cr_returning_customer_sk", "cr_returning_addr_sk",
                "cr_return_amt_inc_tax", 2000,
            )  # fmt: skip
            .select(shown)
            .sort(shown, nulls_last=True)
            .head(100)
        )
    elif q == "q95":
        split = (
            c["web_sales"]
            .group_by("ws_order_number")
            .agg(
                col("ws_warehouse_sk").min().alias("first_warehouse"),
                col("ws_warehouse_sk").max().alias("last_warehouse"),
            )
            .filter(col("first_warehouse") != col("last_warehouse"))
            .select(col("ws_order_number").alias("split_order"))
        )
        out = (
            c["web_sales"]
            .join(
                dates(col("d_date").is_between(d(1999, 2, 1), d(1999, 4, 2)), "d_sk"),
                left_on="ws_ship_date_sk",
                right_on="d_sk",
            )
            .join(
                c["customer_address"].filter(col("ca_state") == "IL"),
                left_on="ws_ship_addr_sk",
                right_on="ca_address_sk",
            )
            .join(
                c["web_site"].filter(col("web_company_name") == "pri"),
                left_on="ws_web_site_sk",
                right_on="web_site_sk",
            )
            .join(split, left_on="ws_order_number", right_on="split_order", how="semi")
            .join(
                c["web_returns"].select("wr_order_number"),
                left_on="ws_order_number",
                right_on="wr_order_number",
                how="semi",
            )
            .select(
                col("ws_order_number").n_unique().alias("order count"),
                sql_sum(col("ws_ext_ship_cost")).alias("total shipping cost"),
                sql_sum(col("ws_net_profit")).alias("total net profit"),
            )
        )
    else:
        raise NotImplementedError("unsupported: not translated")
    return out.collect()


# --- workers ---------------------------------------------------------------


def duckdb_sql(suite: str, query: str, con) -> str:
    if suite == "h2o_groupby":
        return H2O_GROUPBY_SQL[query]
    if suite == "h2o_join":
        return H2O_JOIN_SQL[query]
    if suite == "pdsh":
        number = int(query[1:])
        return con.execute(
            "SELECT query FROM tpch_queries() WHERE query_nr = ?", [number]
        ).fetchone()[0]
    if suite == "tpcds":
        number = int(query[1:])
        return con.execute(
            "SELECT query FROM tpcds_queries() WHERE query_nr = ?", [number]
        ).fetchone()[0].rstrip().rstrip(";")
    return CLICKBENCH_SQL[int(query[1:])]


POLARS = {
    "h2o_groupby": h2o_groupby_polars,
    "h2o_join": h2o_join_polars,
    "pdsh": pdsh_polars,
    "tpcds": tpcds_polars,
    "clickbench": clickbench_polars,
}


def _report(query, times, result):
    for ns in times:
        print(f"time\t{query}\t{ns}")
    height, values = summary(result)
    print(
        f"summary\t{query}\t{height}\t"
        + ",".join(repr(v) for v in values)
        + "\t"
        + ",".join(result.columns)
    )


def main():
    engine, suite, queries, reps = sys.argv[1:5]
    reps = int(reps)
    tables = dict(arg.split("=", 1) for arg in sys.argv[5:])
    if engine == "polars":
        import polars as pl

        frames = {name: pl.read_parquet(path) for name, path in tables.items()}
        run = POLARS[suite]

        def execute(query):
            result = run(frames, query)
            times = []
            for _ in range(reps):
                start = time.perf_counter_ns()
                result = run(frames, query)
                times.append(time.perf_counter_ns() - start)
            return times, result

    elif engine == "duckdb":
        import duckdb

        con = duckdb.connect()
        threads = os.environ.get("BENCH_THREADS")
        if threads:
            con.execute(f"SET threads = {int(threads)}")
        if suite == "pdsh":
            con.execute("LOAD tpch")
        if suite == "tpcds":
            con.execute("LOAD tpcds")
        for name, path in tables.items():
            con.execute(
                f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path}')"
            )

        def execute(query):
            sql = duckdb_sql(suite, query, con)
            con.execute(f"CREATE OR REPLACE TEMP TABLE ans AS {sql}")
            times = []
            for _ in range(reps):
                start = time.perf_counter_ns()
                con.execute(f"CREATE OR REPLACE TEMP TABLE ans AS {sql}")
                times.append(time.perf_counter_ns() - start)
            return times, con.execute("SELECT * FROM ans").pl()

    else:
        raise SystemExit(f"unknown engine {engine}")
    for query in queries.split(","):
        try:
            times, result = execute(query)
        except Exception as error:  # report and continue with the next query
            message = " ".join(str(error).split())
            if message.startswith("unsupported:"):
                print(f"unsupported\t{query}\t{message[13:]}")
            else:
                print(f"failed\t{query}\t{message}")
            continue
        _report(query, times, result)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
