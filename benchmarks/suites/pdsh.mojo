"""PDS-H: the 22 TPC-H-derived queries of Polars' benchmark, written with
this library's lazy API, as the Polars versions are: each query builds one
plan over the tables and collects it, so projection and predicate pushdown
decide which columns and rows reach each join (#329). q11 and q22 collect a
scalar subquery first and use its value, as a user would. engines.py holds
the Polars versions; DuckDB runs `tpch_queries()`. The same queries run on both
data variants (bench_suites.py): money columns as Float64, or as
Decimal(15, 2), where literals must be decimals too (`money`) and ratios
and comparisons with averages convert to Float64 (`real`). Usage: see
suite_common.mojo.
"""
from std.collections import Dict

from dataframe import DataFrame, DataType, Expr, LazyFrame, col, lit, when
from suite_common import date_lit, run


def sort_by(
    frame: LazyFrame, names: List[String], descending: List[Bool]
) raises -> LazyFrame:
    var nulls_last = List[Bool](length=len(names), fill=True)
    return frame.sort(names, descending=descending, nulls_last=nulls_last)


def like_two(name: String, first: String, second: String) -> Expr:
    """SQL `name LIKE '%first%second%'` without regular expressions: the
    first occurrence of `first` ends before the last `second` starts."""
    var text = col(name)
    var before = text.str().split(first)
    var after = text.str().split(second)
    var first_end = before.list().first().str().len_chars() + lit(
        Int64(first.byte_length())
    )
    var last_start = (
        text.str().len_chars()
        - after.list().last().str().len_chars()
        - lit(Int64(second.byte_length()))
    )
    return (
        (before.list().len() > 1)
        & (after.list().len() > 1)
        & (first_end <= last_start)
    )


def between_dates(name: String, low: String, high: String) raises -> Expr:
    """Whether low <= name < high."""
    return col(name).is_between(date_lit(low), date_lit(high), closed="left")


def money(decimal: Bool, text: String) raises -> Expr:
    """A literal of the money columns' type: Decimal(15, 2) in the `decimal`
    data variant, where decimals only combine with decimals, else Float64."""
    if decimal:
        return lit(text).cast(DataType.decimal(15, 2))
    return lit(Float64(text))


def real(e: Expr) -> Expr:
    """Float64, for ratios and comparisons with averages. DuckDB divides
    decimals into DOUBLE; a decimal quotient would round to scale 2."""
    return e.cast(DataType.FLOAT64)


def query(q: String, t: Dict[String, DataFrame]) raises -> DataFrame:
    return plan(q, t).collect()


def plan(q: String, t: Dict[String, DataFrame]) raises -> LazyFrame:
    var line = t["lineitem"].lazy()
    var orders = t["orders"].lazy()
    var cust = t["customer"].lazy()
    var part = t["part"].lazy()
    var supp = t["supplier"].lazy()
    var ps = t["partsupp"].lazy()
    var nation = t["nation"].lazy()
    var region = t["region"].lazy()
    var dec = t["lineitem"].column("l_discount").dtype().is_decimal()
    var disc_price = col("l_extendedprice") * (
        money(dec, "1") - col("l_discount")
    )
    if q == "q1":
        return (
            line.filter(col("l_shipdate") <= date_lit("1998-09-02"))
            .group_by(["l_returnflag", "l_linestatus"])
            .agg(
                [
                    col("l_quantity").sum().alias("sum_qty"),
                    col("l_extendedprice").sum().alias("sum_base_price"),
                    disc_price.sum().alias("sum_disc_price"),
                    (disc_price * (money(dec, "1") + col("l_tax")))
                    .sum()
                    .alias("sum_charge"),
                    col("l_quantity").mean().alias("avg_qty"),
                    col("l_extendedprice").mean().alias("avg_price"),
                    col("l_discount").mean().alias("avg_disc"),
                    col("l_quantity").len().alias("count_order"),
                ]
            )
            .sort(["l_returnflag", "l_linestatus"])
        )
    if q == "q2":
        var europe = (
            region.filter(col("r_name") == "EUROPE")
            .join(nation, left_on=["r_regionkey"], right_on=["n_regionkey"])
            .join(supp, left_on=["n_nationkey"], right_on=["s_nationkey"])
            .join(ps, left_on=["s_suppkey"], right_on=["ps_suppkey"])
        )
        var brass = part.filter(
            (col("p_size") == 15) & col("p_type").str().ends_with("BRASS")
        ).join(europe, left_on=["p_partkey"], right_on=["ps_partkey"])
        var cheapest = brass.filter(
            col("ps_supplycost") == col("ps_supplycost").min().over("p_partkey")
        ).select(
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ]
        )
        return sort_by(
            cheapest,
            ["s_acctbal", "n_name", "s_name", "p_partkey"],
            [True, False, False, False],
        ).head(100)
    if q == "q3":
        var grouped = (
            cust.filter(col("c_mktsegment") == "BUILDING")
            .join(
                orders.filter(col("o_orderdate") < date_lit("1995-03-15")),
                left_on=["c_custkey"],
                right_on=["o_custkey"],
            )
            .join(
                line.filter(col("l_shipdate") > date_lit("1995-03-15")),
                left_on=["o_orderkey"],
                right_on=["l_orderkey"],
            )
            .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
            .agg([disc_price.sum().alias("revenue")])
            .select_exprs(
                [
                    col("o_orderkey").alias("l_orderkey"),
                    col("revenue"),
                    col("o_orderdate"),
                    col("o_shippriority"),
                ]
            )
        )
        return sort_by(grouped, ["revenue", "o_orderdate"], [True, False]).head(
            10
        )
    if q == "q4":
        var late = line.filter(col("l_commitdate") < col("l_receiptdate"))
        return (
            orders.filter(
                between_dates("o_orderdate", "1993-07-01", "1993-10-01")
            )
            .join(
                late,
                left_on=["o_orderkey"],
                right_on=["l_orderkey"],
                how="semi",
            )
            .group_by(["o_orderpriority"])
            .agg([col("o_orderkey").len().alias("order_count")])
            .sort(["o_orderpriority"])
        )
    if q == "q5":
        var grouped = (
            region.filter(col("r_name") == "ASIA")
            .join(nation, left_on=["r_regionkey"], right_on=["n_regionkey"])
            .join(cust, left_on=["n_nationkey"], right_on=["c_nationkey"])
            .join(
                orders.filter(
                    between_dates("o_orderdate", "1994-01-01", "1995-01-01")
                ),
                left_on=["c_custkey"],
                right_on=["o_custkey"],
            )
            .join(line, left_on=["o_orderkey"], right_on=["l_orderkey"])
            .join(
                supp,
                left_on=["l_suppkey", "n_nationkey"],
                right_on=["s_suppkey", "s_nationkey"],
            )
            .group_by(["n_name"])
            .agg([disc_price.sum().alias("revenue")])
        )
        return grouped.sort("revenue", descending=True)
    if q == "q6":
        return line.filter(
            between_dates("l_shipdate", "1994-01-01", "1995-01-01")
            & col("l_discount").is_between(
                money(dec, "0.05"), money(dec, "0.07")
            )
            & (col("l_quantity") < money(dec, "24"))
        ).select(
            (col("l_extendedprice") * col("l_discount")).sum().alias("revenue")
        )
    if q == "q7":
        var n1 = nation.select_exprs(
            [
                col("n_nationkey").alias("s_nationkey"),
                col("n_name").alias("supp_nation"),
            ]
        )
        var n2 = nation.select_exprs(
            [
                col("n_nationkey").alias("c_nationkey"),
                col("n_name").alias("cust_nation"),
            ]
        )
        return (
            line.filter(
                col("l_shipdate").is_between(
                    date_lit("1995-01-01"), date_lit("1996-12-31")
                )
            )
            .join(supp, left_on=["l_suppkey"], right_on=["s_suppkey"])
            .join(orders, left_on=["l_orderkey"], right_on=["o_orderkey"])
            .join(cust, left_on=["o_custkey"], right_on=["c_custkey"])
            .join(n1, "s_nationkey")
            .join(n2, "c_nationkey")
            .filter(
                (
                    (col("supp_nation") == "FRANCE")
                    & (col("cust_nation") == "GERMANY")
                )
                | (
                    (col("supp_nation") == "GERMANY")
                    & (col("cust_nation") == "FRANCE")
                )
            )
            .with_columns(
                [
                    col("l_shipdate").dt().year().alias("l_year"),
                    disc_price.alias("volume"),
                ]
            )
            .group_by(["supp_nation", "cust_nation", "l_year"])
            .agg([col("volume").sum().alias("revenue")])
            .sort(["supp_nation", "cust_nation", "l_year"])
        )
    if q == "q8":
        var n1 = nation.select(["n_nationkey", "n_regionkey"])
        var n2 = nation.select_exprs(
            [
                col("n_nationkey").alias("s_nationkey"),
                col("n_name").alias("nation"),
            ]
        )
        return (
            part.filter(col("p_type") == "ECONOMY ANODIZED STEEL")
            .join(line, left_on=["p_partkey"], right_on=["l_partkey"])
            .join(supp, left_on=["l_suppkey"], right_on=["s_suppkey"])
            .join(
                orders.filter(
                    col("o_orderdate").is_between(
                        date_lit("1995-01-01"), date_lit("1996-12-31")
                    )
                ),
                left_on=["l_orderkey"],
                right_on=["o_orderkey"],
            )
            .join(cust, left_on=["o_custkey"], right_on=["c_custkey"])
            .join(n1, left_on=["c_nationkey"], right_on=["n_nationkey"])
            .join(
                region.filter(col("r_name") == "AMERICA"),
                left_on=["n_regionkey"],
                right_on=["r_regionkey"],
            )
            .join(n2, "s_nationkey")
            .with_columns(
                [
                    col("o_orderdate").dt().year().alias("o_year"),
                    disc_price.alias("volume"),
                ]
            )
            .group_by(["o_year"])
            .agg(
                [
                    (
                        when(col("nation") == "BRAZIL")
                        .then(real(col("volume")))
                        .otherwise(lit(0.0))
                        .sum()
                        / real(col("volume").sum())
                    ).alias("mkt_share")
                ]
            )
            .sort(["o_year"])
        )
    if q == "q9":
        var profit = (
            part.filter(col("p_name").str().contains("green"))
            .join(line, left_on=["p_partkey"], right_on=["l_partkey"])
            .join(supp, left_on=["l_suppkey"], right_on=["s_suppkey"])
            .join(
                ps,
                left_on=["l_suppkey", "p_partkey"],
                right_on=["ps_suppkey", "ps_partkey"],
            )
            .join(orders, left_on=["l_orderkey"], right_on=["o_orderkey"])
            .join(nation, left_on=["s_nationkey"], right_on=["n_nationkey"])
            .with_columns(
                [
                    col("n_name").alias("nation"),
                    col("o_orderdate").dt().year().alias("o_year"),
                    (
                        disc_price - col("ps_supplycost") * col("l_quantity")
                    ).alias("amount"),
                ]
            )
            .group_by(["nation", "o_year"])
            .agg([col("amount").sum().alias("sum_profit")])
        )
        return sort_by(profit, ["nation", "o_year"], [False, True])
    if q == "q10":
        var grouped = (
            cust.join(
                orders.filter(
                    between_dates("o_orderdate", "1993-10-01", "1994-01-01")
                ),
                left_on=["c_custkey"],
                right_on=["o_custkey"],
            )
            .join(
                line.filter(col("l_returnflag") == "R"),
                left_on=["o_orderkey"],
                right_on=["l_orderkey"],
            )
            .join(nation, left_on=["c_nationkey"], right_on=["n_nationkey"])
            .group_by(
                [
                    "c_custkey",
                    "c_name",
                    "c_acctbal",
                    "c_phone",
                    "n_name",
                    "c_address",
                    "c_comment",
                ]
            )
            .agg([disc_price.sum().alias("revenue")])
            .select(
                [
                    "c_custkey",
                    "c_name",
                    "revenue",
                    "c_acctbal",
                    "n_name",
                    "c_address",
                    "c_phone",
                    "c_comment",
                ]
            )
        )
        return grouped.sort("revenue", descending=True).head(20)
    if q == "q11":
        var german = (
            nation.filter(col("n_name") == "GERMANY")
            .join(supp, left_on=["n_nationkey"], right_on=["s_nationkey"])
            .join(ps, left_on=["s_suppkey"], right_on=["ps_suppkey"])
            .with_columns(
                [
                    (
                        real(col("ps_supplycost"))
                        * col("ps_availqty").cast(DataType.FLOAT64)
                    ).alias("v")
                ]
            )
        )
        var threshold = (
            german.select(col("v").sum()).collect().item().float64() * 0.0001
        )
        return (
            german.group_by(["ps_partkey"])
            .agg([col("v").sum().alias("value")])
            .filter(col("value") > lit(threshold))
            .sort("value", descending=True)
        )
    if q == "q12":
        var urgent: List[String] = ["1-URGENT", "2-HIGH"]
        var high = col("o_orderpriority").is_in(urgent)
        var modes: List[String] = ["MAIL", "SHIP"]
        return (
            orders.join(
                line.filter(
                    col("l_shipmode").is_in(modes)
                    & (col("l_commitdate") < col("l_receiptdate"))
                    & (col("l_shipdate") < col("l_commitdate"))
                    & between_dates("l_receiptdate", "1994-01-01", "1995-01-01")
                ),
                left_on=["o_orderkey"],
                right_on=["l_orderkey"],
            )
            .group_by(["l_shipmode"])
            .agg(
                [
                    high.cast(DataType.INT64).sum().alias("high_line_count"),
                    (~high).cast(DataType.INT64).sum().alias("low_line_count"),
                ]
            )
            .sort(["l_shipmode"])
        )
    if q == "q13":
        var counts = (
            cust.join(
                orders.filter(~like_two("o_comment", "special", "requests")),
                left_on=["c_custkey"],
                right_on=["o_custkey"],
                how="left",
            )
            .group_by(["c_custkey"])
            .agg([col("o_orderkey").count().alias("c_count")])
            .group_by(["c_count"])
            .agg([col("c_custkey").len().alias("custdist")])
        )
        return sort_by(counts, ["custdist", "c_count"], [True, True])
    if q == "q14":
        return (
            line.filter(between_dates("l_shipdate", "1995-09-01", "1995-10-01"))
            .join(part, left_on=["l_partkey"], right_on=["p_partkey"])
            .select(
                (
                    lit(100.0)
                    * when(col("p_type").str().starts_with("PROMO"))
                    .then(real(disc_price))
                    .otherwise(lit(0.0))
                    .sum()
                    / real(disc_price.sum())
                ).alias("promo_revenue")
            )
        )
    if q == "q15":
        var revenue = (
            line.filter(between_dates("l_shipdate", "1996-01-01", "1996-04-01"))
            .group_by(["l_suppkey"])
            .agg([disc_price.sum().alias("total_revenue")])
        )
        return (
            supp.join(
                revenue.filter(
                    col("total_revenue") == col("total_revenue").max()
                ),
                left_on=["s_suppkey"],
                right_on=["l_suppkey"],
            )
            .select(
                ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
            )
            .sort(["s_suppkey"])
        )
    if q == "q16":
        var complaints = supp.filter(
            like_two("s_comment", "Customer", "Complaints")
        ).select_exprs([col("s_suppkey").alias("ps_suppkey")])
        var sizes: List[Expr] = [
            Expr(49),
            Expr(14),
            Expr(23),
            Expr(45),
            Expr(19),
            Expr(3),
            Expr(36),
            Expr(9),
        ]
        var grouped = (
            part.filter(
                col("p_brand").ne("Brand#45")
                & ~col("p_type").str().starts_with("MEDIUM POLISHED")
                & col("p_size").is_in(sizes)
            )
            .join(ps, left_on=["p_partkey"], right_on=["ps_partkey"])
            .join(complaints, "ps_suppkey", how="anti")
            .group_by(["p_brand", "p_type", "p_size"])
            .agg([col("ps_suppkey").n_unique().alias("supplier_cnt")])
        )
        return sort_by(
            grouped,
            ["supplier_cnt", "p_brand", "p_type", "p_size"],
            [True, False, False, False],
        )
    if q == "q17":
        var chosen = part.filter(
            (col("p_brand") == "Brand#23") & (col("p_container") == "MED BOX")
        ).join(line, left_on=["p_partkey"], right_on=["l_partkey"])
        return chosen.filter(
            real(col("l_quantity"))
            < lit(0.2) * col("l_quantity").mean().over("p_partkey")
        ).select(
            (real(col("l_extendedprice").sum()) / lit(7.0)).alias("avg_yearly")
        )
    if q == "q18":
        var big = (
            line.group_by(["l_orderkey"])
            .agg([col("l_quantity").sum().alias("q")])
            .filter(col("q") > money(dec, "300"))
            .select(["l_orderkey"])
        )
        var grouped = (
            orders.join(
                big, left_on=["o_orderkey"], right_on=["l_orderkey"], how="semi"
            )
            .join(cust, left_on=["o_custkey"], right_on=["c_custkey"])
            .join(line, left_on=["o_orderkey"], right_on=["l_orderkey"])
            .group_by(
                [
                    "c_name",
                    "o_custkey",
                    "o_orderkey",
                    "o_orderdate",
                    "o_totalprice",
                ]
            )
            .agg([col("l_quantity").sum().alias("sum_qty")])
            .select_exprs(
                [
                    col("c_name"),
                    col("o_custkey").alias("c_custkey"),
                    col("o_orderkey"),
                    col("o_orderdate"),
                    col("o_totalprice"),
                    col("sum_qty"),
                ]
            )
        )
        return sort_by(
            grouped, ["o_totalprice", "o_orderdate"], [True, False]
        ).head(100)
    if q == "q19":
        var sm: List[String] = ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
        var med: List[String] = ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
        var lg: List[String] = ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
        var air: List[String] = ["AIR", "AIR REG"]

        def branch(
            brand: String,
            containers: List[String],
            low: Int,
            size: Int,
            decimal: Bool,
        ) raises -> Expr:
            return (
                (col("p_brand") == brand)
                & col("p_container").is_in(containers)
                & col("l_quantity").is_between(
                    money(decimal, String(low)),
                    money(decimal, String(low + 10)),
                )
                & col("p_size").is_between(Expr(1), Expr(size))
            )

        return (
            line.filter(
                col("l_shipmode").is_in(air)
                & (col("l_shipinstruct") == "DELIVER IN PERSON")
            )
            .join(part, left_on=["l_partkey"], right_on=["p_partkey"])
            .filter(
                branch("Brand#12", sm, 1, 5, dec)
                | branch("Brand#23", med, 10, 10, dec)
                | branch("Brand#34", lg, 20, 15, dec)
            )
            .select(disc_price.sum().alias("revenue"))
        )
    if q == "q20":
        var shipped = (
            line.filter(between_dates("l_shipdate", "1994-01-01", "1995-01-01"))
            .group_by(["l_partkey", "l_suppkey"])
            .agg([(lit(0.5) * real(col("l_quantity").sum())).alias("half")])
        )
        var forest = part.filter(col("p_name").str().starts_with("forest"))
        var suppliers = (
            ps.join(
                forest,
                left_on=["ps_partkey"],
                right_on=["p_partkey"],
                how="semi",
            )
            .join(
                shipped,
                left_on=["ps_partkey", "ps_suppkey"],
                right_on=["l_partkey", "l_suppkey"],
            )
            .filter(col("ps_availqty").cast(DataType.FLOAT64) > col("half"))
            .select(["ps_suppkey"])
        )
        return (
            nation.filter(col("n_name") == "CANADA")
            .join(supp, left_on=["n_nationkey"], right_on=["s_nationkey"])
            .join(
                suppliers,
                left_on=["s_suppkey"],
                right_on=["ps_suppkey"],
                how="semi",
            )
            .select(["s_name", "s_address"])
            .sort(["s_name"])
        )
    if q == "q21":
        var per_order = line.group_by(["l_orderkey"]).agg(
            [col("l_suppkey").n_unique().alias("suppliers")]
        )
        var late = line.filter(col("l_receiptdate") > col("l_commitdate"))
        var late_per_order = late.group_by(["l_orderkey"]).agg(
            [col("l_suppkey").n_unique().alias("late_suppliers")]
        )
        var waiting = (
            late.join(per_order, "l_orderkey")
            .join(late_per_order, "l_orderkey")
            .filter((col("suppliers") > 1) & (col("late_suppliers") == 1))
            .join(
                orders.filter(col("o_orderstatus") == "F"),
                left_on=["l_orderkey"],
                right_on=["o_orderkey"],
            )
            .join(supp, left_on=["l_suppkey"], right_on=["s_suppkey"])
            .join(
                nation.filter(col("n_name") == "SAUDI ARABIA"),
                left_on=["s_nationkey"],
                right_on=["n_nationkey"],
            )
            .group_by(["s_name"])
            .agg([col("s_name").len().alias("numwait")])
        )
        return sort_by(waiting, ["numwait", "s_name"], [True, False]).head(100)
    if q == "q22":
        var codes: List[String] = ["13", "31", "23", "29", "30", "18", "17"]
        var chosen = cust.with_columns(
            [col("c_phone").str().slice(0, 2).alias("cntrycode")]
        ).filter(col("cntrycode").is_in(codes))
        var average = (
            chosen.filter(col("c_acctbal") > money(dec, "0"))
            .select(col("c_acctbal").mean())
            .collect()
            .item()
            .float64()
        )
        return (
            chosen.filter(real(col("c_acctbal")) > lit(average))
            .join(
                orders,
                left_on=["c_custkey"],
                right_on=["o_custkey"],
                how="anti",
            )
            .group_by(["cntrycode"])
            .agg(
                [
                    col("c_custkey").len().alias("numcust"),
                    col("c_acctbal").sum().alias("totacctbal"),
                ]
            )
            .sort(["cntrycode"])
        )
    raise Error("unknown query " + q)


def main() raises:
    run[query]()
