"""Polars and DuckDB workers for the external benchmark suites.

Usage (oracle environment):

    python engines.py ENGINE SUITE QUERY REPS name=path [name=path ...]

The worker loads every table into memory (untimed), runs the query once to
warm up, then times REPS runs. Each timed run materializes the full result:
Polars collects a DataFrame and DuckDB creates a temporary table, as the
H2O.ai db-benchmark does. It prints, one per line:

    time<TAB>NANOSECONDS          (one line per timed run)
    summary<TAB>HEIGHT<TAB>V1,V2<TAB>NAME1,NAME2  (see `summary`)

or `unsupported<TAB>REASON`. The Mojo runners print the same protocol, so
the driver checks every engine's answer the same way.

DuckDB runs the reference SQL for each suite: db-benchmark's queries,
`tpch_queries()` for PDS-H, and ClickBench's `queries.sql`. The Polars
versions are idiomatic translations of the same queries.
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
    return CLICKBENCH_SQL[int(query[1:])]


POLARS = {
    "h2o_groupby": h2o_groupby_polars,
    "h2o_join": h2o_join_polars,
    "pdsh": pdsh_polars,
    "clickbench": clickbench_polars,
}


def main():
    engine, suite, query, reps = sys.argv[1:5]
    reps = int(reps)
    tables = dict(arg.split("=", 1) for arg in sys.argv[5:])
    times = []
    if engine == "polars":
        import polars as pl

        frames = {name: pl.read_parquet(path) for name, path in tables.items()}
        run = POLARS[suite]
        result = run(frames, query)
        for _ in range(reps):
            start = time.perf_counter_ns()
            result = run(frames, query)
            times.append(time.perf_counter_ns() - start)
    elif engine == "duckdb":
        import duckdb

        con = duckdb.connect()
        threads = os.environ.get("BENCH_THREADS")
        if threads:
            con.execute(f"SET threads = {int(threads)}")
        if suite == "pdsh":
            con.execute("LOAD tpch")
        for name, path in tables.items():
            con.execute(
                f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path}')"
            )
        sql = duckdb_sql(suite, query, con)
        con.execute(f"CREATE OR REPLACE TEMP TABLE ans AS {sql}")
        for _ in range(reps):
            start = time.perf_counter_ns()
            con.execute(f"CREATE OR REPLACE TEMP TABLE ans AS {sql}")
            times.append(time.perf_counter_ns() - start)
        result = con.execute("SELECT * FROM ans").pl()
    else:
        raise SystemExit(f"unknown engine {engine}")
    for ns in times:
        print(f"time\t{ns}")
    height, values = summary(result)
    print(
        f"summary\t{height}\t"
        + ",".join(repr(v) for v in values)
        + "\t"
        + ",".join(result.columns)
    )


if __name__ == "__main__":
    main()
