"""TPC-DS: queries from DuckDB's `tpcds_queries()`, written with this
library's lazy API. Each query builds one plan over the tables and collects
it, as the PDS-H runner does. engines.py holds the Polars versions; DuckDB
runs the SQL text.

Translated so far: the 23 single-block queries (one SELECT, no window,
rollup or set operation), chosen by that rule and not by their timings. Any
other query reports `unsupported: not translated`, so it is counted in
every report instead of dropped. The same queries run on both data variants
(bench_suites.py): money columns as Float64, or as declared decimals, where
literals compared with them must be decimals too (`like`). Usage: see
suite_common.mojo.

SQL ORDER BY puts nulls last in DuckDB unless the query says NULLS FIRST;
`ordered` takes the placement because a LIMIT follows most sorts.
"""
from std.collections import Dict

from dataframe import (
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    coalesce,
    col,
    concat_str,
    lit,
    when,
)
from suite_common import date_lit, run, unsupported


def ordered(
    frame: LazyFrame,
    names: List[String],
    descending: List[Bool],
    nulls_first: Bool = False,
) raises -> LazyFrame:
    var nulls_last = List[Bool](length=len(names), fill=not nulls_first)
    return frame.sort(names, descending=descending, nulls_last=nulls_last)


def ascending(
    frame: LazyFrame, names: List[String], nulls_first: Bool = False
) raises -> LazyFrame:
    return ordered(
        frame, names, List[Bool](length=len(names), fill=False), nulls_first
    )


def like(frame: DataFrame, column: String, text: String) raises -> Expr:
    """A literal of `column`'s type: a decimal in the `decimal` data variant,
    where decimals only combine with decimals, else Float64."""
    var dtype = frame.column(column).dtype()
    if dtype.is_decimal():
        return lit(text).cast(dtype)
    return lit(Float64(text))


def between(
    frame: DataFrame, column: String, low: String, high: String
) raises -> Expr:
    return col(column).is_between(
        like(frame, column, low), like(frame, column, high)
    )


def ints(values: List[Int]) -> List[Expr]:
    var literals = List[Expr](capacity=len(values))
    for value in values:
        literals.append(lit(Int64(value)))
    return literals^


def dates(
    t: Dict[String, DataFrame], keep: Expr, key: String
) raises -> LazyFrame:
    """The date_dim rows `keep` selects, as one key column named `key`."""
    return (
        t["date_dim"]
        .lazy()
        .filter(keep)
        .select_exprs([col("d_date_sk").alias(key)])
    )


def one_if(condition: Expr) -> Expr:
    return when(condition).then(lit(Int64(1))).otherwise(lit(Int64(0)))


def brand_sales(
    t: Dict[String, DataFrame], days: Expr, items: Expr
) raises -> LazyFrame:
    """Store sales joined to the chosen days and items (q3, q42, q52, q55)."""
    return (
        t["date_dim"]
        .lazy()
        .filter(days)
        .join(
            t["store_sales"].lazy(),
            left_on=["d_date_sk"],
            right_on=["ss_sold_date_sk"],
        )
        .join(
            t["item"].lazy().filter(items),
            left_on=["ss_item_sk"],
            right_on=["i_item_sk"],
        )
    )


def promoted_averages(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    demographics: String,
) raises -> LazyFrame:
    """Average quantity and prices per item for unmarried male college
    customers in 2000, on promotions not sent by email or event (q7, q26)."""
    var buyers = (
        t["customer_demographics"]
        .lazy()
        .filter(
            (col("cd_gender") == "M")
            & (col("cd_marital_status") == "S")
            & (col("cd_education_status") == "College")
        )
    )
    var promotions = (
        t["promotion"]
        .lazy()
        .filter(
            (col("p_channel_email") == "N") | (col("p_channel_event") == "N")
        )
    )
    var grouped = (
        t[sales]
        .lazy()
        .join(buyers, left_on=[demographics], right_on=["cd_demo_sk"])
        .join(
            dates(t, col("d_year") == 2000, "d_date_sk"),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_date_sk"],
        )
        .join(
            t["item"].lazy(),
            left_on=[prefix + "_item_sk"],
            right_on=["i_item_sk"],
        )
        .join(
            promotions, left_on=[prefix + "_promo_sk"], right_on=["p_promo_sk"]
        )
        .group_by(["i_item_id"])
        .agg(
            [
                col(prefix + "_quantity").mean().alias("agg1"),
                col(prefix + "_list_price").mean().alias("agg2"),
                col(prefix + "_coupon_amt").mean().alias("agg3"),
                col(prefix + "_sales_price").mean().alias("agg4"),
            ]
        )
    )
    return ascending(grouped, ["i_item_id"]).head(100)


def returned_then_bought(
    t: Dict[String, DataFrame], sold: Expr, returned: Expr, bought: Expr
) raises -> LazyFrame:
    """Store sales in the `sold` days that were returned in the `returned`
    days by a customer who bought the item from the catalog in the `bought`
    days, with the store and item of each (q17, q25, q29)."""
    return (
        t["store_sales"]
        .lazy()
        .join(
            dates(t, sold, "d1_sk"),
            left_on=["ss_sold_date_sk"],
            right_on=["d1_sk"],
        )
        .join(t["item"].lazy(), left_on=["ss_item_sk"], right_on=["i_item_sk"])
        .join(
            t["store"].lazy(), left_on=["ss_store_sk"], right_on=["s_store_sk"]
        )
        .join(
            t["store_returns"]
            .lazy()
            .join(
                dates(t, returned, "d2_sk"),
                left_on=["sr_returned_date_sk"],
                right_on=["d2_sk"],
            ),
            left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
            right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
        )
        .join(
            t["catalog_sales"]
            .lazy()
            .join(
                dates(t, bought, "d3_sk"),
                left_on=["cs_sold_date_sk"],
                right_on=["d3_sk"],
            ),
            left_on=["ss_customer_sk", "ss_item_sk"],
            right_on=["cs_bill_customer_sk", "cs_item_sk"],
        )
    )


def stocked_items(
    t: Dict[String, DataFrame],
    low: String,
    high: String,
    makers: List[Int],
    first_day: String,
    last_day: String,
    sales: String,
    key: String,
) raises -> LazyFrame:
    """Items in a price band from some makers that were in stock (100 to
    500 on hand) in a date range and sold through `sales` (q37, q82)."""
    var in_stock = (
        t["inventory"]
        .lazy()
        .filter(
            col("inv_quantity_on_hand").is_between(
                lit(Int64(100)), lit(Int64(500))
            )
        )
        .join(
            dates(
                t,
                col("d_date").is_between(
                    date_lit(first_day), date_lit(last_day)
                ),
                "d_date_sk",
            ),
            left_on=["inv_date_sk"],
            right_on=["d_date_sk"],
        )
    )
    var found = (
        t["item"]
        .lazy()
        .filter(
            between(t["item"], "i_current_price", low, high)
            & col("i_manufact_id").is_in(ints(makers))
        )
        .join(in_stock, left_on=["i_item_sk"], right_on=["inv_item_sk"])
        .join(
            t[sales].lazy(),
            left_on=["i_item_sk"],
            right_on=[key],
            how="semi",
        )
        .select(["i_item_id", "i_item_desc", "i_current_price"])
        .unique()
    )
    return ascending(found, ["i_item_id"]).head(100)


def query(q: String, t: Dict[String, DataFrame]) raises -> DataFrame:
    return plan(q, t).collect()


def plan(q: String, t: Dict[String, DataFrame]) raises -> LazyFrame:
    if q == "q3":
        var grouped = (
            brand_sales(t, col("d_moy") == 11, col("i_manufact_id") == 128)
            .group_by(["d_year", "i_brand", "i_brand_id"])
            .agg([col("ss_ext_sales_price").sum().alias("sum_agg")])
            .select_exprs(
                [
                    col("d_year"),
                    col("i_brand_id").alias("brand_id"),
                    col("i_brand").alias("brand"),
                    col("sum_agg"),
                ]
            )
        )
        return ordered(
            grouped, ["d_year", "sum_agg", "brand_id"], [False, True, False]
        ).head(100)
    if q == "q7":
        return promoted_averages(t, "store_sales", "ss", "ss_cdemo_sk")
    if q == "q13" or q == "q48":
        # Each store sale with its buyer's demographics and address; the
        # OR of the three demographic bands and of the three address bands
        # cannot be pushed into one dimension, so it filters the joined rows.
        ref sales = t["store_sales"]
        var joined = (
            sales.lazy()
            .join(
                t["store"].lazy().select(["s_store_sk"]),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                dates(
                    t,
                    col("d_year") == lit(Int64(2001 if q == "q13" else 2000)),
                    "d_sk",
                ),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["customer_demographics"].lazy(),
                left_on=["ss_cdemo_sk"],
                right_on=["cd_demo_sk"],
            )
            .join(
                t["customer_address"]
                .lazy()
                .filter(col("ca_country") == "United States"),
                left_on=["ss_addr_sk"],
                right_on=["ca_address_sk"],
            )
        )
        if q == "q13":
            var by_household = joined.join(
                t["household_demographics"].lazy(),
                left_on=["ss_hdemo_sk"],
                right_on=["hd_demo_sk"],
            ).filter(
                (
                    (col("cd_marital_status") == "M")
                    & (col("cd_education_status") == "Advanced Degree")
                    & between(sales, "ss_sales_price", "100.00", "150.00")
                    & (col("hd_dep_count") == 3)
                )
                | (
                    (col("cd_marital_status") == "S")
                    & (col("cd_education_status") == "College")
                    & between(sales, "ss_sales_price", "50.00", "100.00")
                    & (col("hd_dep_count") == 1)
                )
                | (
                    (col("cd_marital_status") == "W")
                    & (col("cd_education_status") == "2 yr Degree")
                    & between(sales, "ss_sales_price", "150.00", "200.00")
                    & (col("hd_dep_count") == 1)
                )
            )
            return by_household.filter(
                (
                    col("ca_state").is_in(["TX", "OH", "TX"])
                    & between(sales, "ss_net_profit", "100", "200")
                )
                | (
                    col("ca_state").is_in(["OR", "NM", "KY"])
                    & between(sales, "ss_net_profit", "150", "300")
                )
                | (
                    col("ca_state").is_in(["VA", "TX", "MS"])
                    & between(sales, "ss_net_profit", "50", "250")
                )
            ).select_exprs(
                [
                    col("ss_quantity").mean().alias("avg1"),
                    col("ss_ext_sales_price").mean().alias("avg2"),
                    col("ss_ext_wholesale_cost").mean().alias("avg3"),
                    col("ss_ext_wholesale_cost").sum().alias("sum4"),
                ]
            )
        return (
            joined.filter(
                (
                    (col("cd_marital_status") == "M")
                    & (col("cd_education_status") == "4 yr Degree")
                    & between(sales, "ss_sales_price", "100.00", "150.00")
                )
                | (
                    (col("cd_marital_status") == "D")
                    & (col("cd_education_status") == "2 yr Degree")
                    & between(sales, "ss_sales_price", "50.00", "100.00")
                )
                | (
                    (col("cd_marital_status") == "S")
                    & (col("cd_education_status") == "College")
                    & between(sales, "ss_sales_price", "150.00", "200.00")
                )
            )
            .filter(
                (
                    col("ca_state").is_in(["CO", "OH", "TX"])
                    & between(sales, "ss_net_profit", "0", "2000")
                )
                | (
                    col("ca_state").is_in(["OR", "MN", "KY"])
                    & between(sales, "ss_net_profit", "150", "3000")
                )
                | (
                    col("ca_state").is_in(["VA", "CA", "MS"])
                    & between(sales, "ss_net_profit", "50", "25000")
                )
            )
            .select_exprs([col("ss_quantity").sum().alias("total")])
        )
    if q == "q15":
        var zips: List[String] = [
            "85669",
            "86197",
            "88274",
            "83405",
            "86475",
            "85392",
            "85460",
            "80348",
            "81792",
        ]
        var grouped = (
            t["catalog_sales"]
            .lazy()
            .join(
                dates(t, (col("d_qoy") == 2) & (col("d_year") == 2001), "d_sk"),
                left_on=["cs_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["customer"].lazy(),
                left_on=["cs_bill_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_address"].lazy(),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .filter(
                col("ca_zip").str().slice(0, 5).is_in(zips)
                | col("ca_state").is_in(["CA", "WA", "GA"])
                | (
                    col("cs_sales_price")
                    > like(t["catalog_sales"], "cs_sales_price", "500")
                )
            )
            .group_by(["ca_zip"])
            .agg([col("cs_sales_price").sum().alias("total")])
        )
        return ascending(grouped, ["ca_zip"], nulls_first=True).head(100)
    if q == "q17":
        var later: List[String] = ["2001Q1", "2001Q2", "2001Q3"]
        var aggregates = List[Expr]()
        var measures: List[String] = [
            "ss_quantity",
            "sr_return_quantity",
            "cs_quantity",
        ]
        var labels: List[String] = [
            "store_sales",
            "store_returns",
            "catalog_sales",
        ]
        for i in range(3):
            var value = col(measures[i])
            var label = labels[i] + "_quantity"
            aggregates.append(value.count().alias(label + "count"))
            aggregates.append(value.mean().alias(label + "ave"))
            aggregates.append(value.std().alias(label + "stdev"))
            aggregates.append((value.std() / value.mean()).alias(label + "cov"))
        var grouped = (
            returned_then_bought(
                t,
                col("d_quarter_name") == "2001Q1",
                col("d_quarter_name").is_in(later),
                col("d_quarter_name").is_in(later),
            )
            .group_by(["i_item_id", "i_item_desc", "s_state"])
            .agg(aggregates)
        )
        return ascending(
            grouped, ["i_item_id", "i_item_desc", "s_state"], nulls_first=True
        ).head(100)
    if q == "q19":
        var grouped = (
            brand_sales(
                t,
                (col("d_moy") == 11) & (col("d_year") == 1998),
                col("i_manager_id") == 8,
            )
            .join(
                t["customer"].lazy(),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_address"].lazy(),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["store"].lazy(),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .filter(
                col("ca_zip").str().slice(0, 5)
                != col("s_zip").str().slice(0, 5)
            )
            .group_by(["i_brand", "i_brand_id", "i_manufact_id", "i_manufact"])
            .agg([col("ss_ext_sales_price").sum().alias("ext_price")])
            .select_exprs(
                [
                    col("i_brand_id").alias("brand_id"),
                    col("i_brand").alias("brand"),
                    col("i_manufact_id"),
                    col("i_manufact"),
                    col("ext_price"),
                ]
            )
        )
        return ordered(
            grouped,
            ["ext_price", "brand", "brand_id", "i_manufact_id", "i_manufact"],
            [True, False, False, False, False],
        ).head(100)
    if q == "q25" or q == "q29":
        var keys: List[String] = [
            "i_item_id",
            "i_item_desc",
            "s_store_id",
            "s_store_name",
        ]
        var linked: LazyFrame
        var aggregates: List[Expr]
        if q == "q25":
            var months = col("d_moy").is_between(
                lit(Int64(4)), lit(Int64(10))
            ) & (col("d_year") == 2001)
            linked = returned_then_bought(
                t, (col("d_moy") == 4) & (col("d_year") == 2001), months, months
            )
            aggregates = [
                col("ss_net_profit").sum().alias("store_sales_profit"),
                col("sr_net_loss").sum().alias("store_returns_loss"),
                col("cs_net_profit").sum().alias("catalog_sales_profit"),
            ]
        else:
            linked = returned_then_bought(
                t,
                (col("d_moy") == 9) & (col("d_year") == 1999),
                col("d_moy").is_between(lit(Int64(9)), lit(Int64(12)))
                & (col("d_year") == 1999),
                col("d_year").is_in(ints([1999, 2000, 2001])),
            )
            aggregates = [
                col("ss_quantity").sum().alias("store_sales_quantity"),
                col("sr_return_quantity").sum().alias("store_returns_quantity"),
                col("cs_quantity").sum().alias("catalog_sales_quantity"),
            ]
        return ascending(linked.group_by(keys).agg(aggregates), keys).head(100)
    if q == "q26":
        return promoted_averages(t, "catalog_sales", "cs", "cs_bill_cdemo_sk")
    if q == "q37":
        return stocked_items(
            t,
            "68",
            "98",
            [677, 940, 694, 808],
            "2000-02-01",
            "2000-04-01",
            "catalog_sales",
            "cs_item_sk",
        )
    if q == "q40":
        ref sales = t["catalog_sales"]
        var net = col("cs_sales_price") - coalesce(
            [col("cr_refunded_cash"), like(sales, "cs_sales_price", "0")]
        )
        var zero = like(sales, "cs_sales_price", "0")
        var grouped = (
            sales.lazy()
            .join(
                t["catalog_returns"].lazy(),
                left_on=["cs_order_number", "cs_item_sk"],
                right_on=["cr_order_number", "cr_item_sk"],
                how="left",
            )
            .join(
                t["warehouse"].lazy(),
                left_on=["cs_warehouse_sk"],
                right_on=["w_warehouse_sk"],
            )
            .join(
                t["item"]
                .lazy()
                .filter(between(t["item"], "i_current_price", "0.99", "1.49")),
                left_on=["cs_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                t["date_dim"]
                .lazy()
                .filter(
                    col("d_date").is_between(
                        date_lit("2000-02-10"), date_lit("2000-04-10")
                    )
                ),
                left_on=["cs_sold_date_sk"],
                right_on=["d_date_sk"],
            )
            .group_by(["w_state", "i_item_id"])
            .agg(
                [
                    when(col("d_date") < date_lit("2000-03-11"))
                    .then(net)
                    .otherwise(zero)
                    .sum()
                    .alias("sales_before"),
                    when(col("d_date") >= date_lit("2000-03-11"))
                    .then(net)
                    .otherwise(zero)
                    .sum()
                    .alias("sales_after"),
                ]
            )
        )
        return ascending(grouped, ["w_state", "i_item_id"]).head(100)
    if q == "q42":
        var grouped = (
            brand_sales(
                t,
                (col("d_moy") == 11) & (col("d_year") == 2000),
                col("i_manager_id") == 1,
            )
            .group_by(["d_year", "i_category_id", "i_category"])
            .agg([col("ss_ext_sales_price").sum().alias("total")])
        )
        return ordered(
            grouped,
            ["total", "d_year", "i_category_id", "i_category"],
            [True, False, False, False],
        ).head(100)
    if q == "q43":
        var days: List[String] = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        var names: List[String] = [
            "sun",
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
        ]
        var aggregates = List[Expr]()
        var order: List[String] = ["s_store_name", "s_store_id"]
        for i in range(7):
            aggregates.append(
                when(col("d_day_name") == days[i])
                .then(col("ss_sales_price"))
                .end()
                .sum()
                .alias(names[i] + "_sales")
            )
            order.append(names[i] + "_sales")
        var grouped = (
            t["date_dim"]
            .lazy()
            .filter(col("d_year") == 2000)
            .join(
                t["store_sales"].lazy(),
                left_on=["d_date_sk"],
                right_on=["ss_sold_date_sk"],
            )
            .join(
                t["store"]
                .lazy()
                .filter(
                    col("s_gmt_offset")
                    == like(t["store"], "s_gmt_offset", "-5")
                ),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .group_by(["s_store_name", "s_store_id"])
            .agg(aggregates)
        )
        return ascending(grouped, order).head(100)
    if q == "q50":
        var keys: List[String] = [
            "s_store_name",
            "s_company_id",
            "s_street_number",
            "s_street_name",
            "s_street_type",
            "s_suite_number",
            "s_city",
            "s_county",
            "s_state",
            "s_zip",
        ]
        var lag = col("sr_returned_date_sk") - col("ss_sold_date_sk")
        var grouped = (
            t["store_sales"]
            .lazy()
            .join(
                t["store_returns"]
                .lazy()
                .join(
                    dates(
                        t,
                        (col("d_year") == 2001) & (col("d_moy") == 8),
                        "d2_sk",
                    ),
                    left_on=["sr_returned_date_sk"],
                    right_on=["d2_sk"],
                ),
                left_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
                right_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"],
            )
            .join(
                dates(t, col("d_date_sk").is_not_null(), "d1_sk"),
                left_on=["ss_sold_date_sk"],
                right_on=["d1_sk"],
            )
            .join(
                t["store"].lazy(),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .group_by(keys)
            .agg(
                [
                    one_if(lag <= 30).sum().alias("30 days"),
                    one_if((lag > 30) & (lag <= 60)).sum().alias("31-60 days"),
                    one_if((lag > 60) & (lag <= 90)).sum().alias("61-90 days"),
                    one_if((lag > 90) & (lag <= 120))
                    .sum()
                    .alias("91-120 days"),
                    one_if(lag > 120).sum().alias(">120 days"),
                ]
            )
        )
        return ascending(grouped, keys).head(100)
    if q == "q52":
        var grouped = (
            brand_sales(
                t,
                (col("d_moy") == 11) & (col("d_year") == 2000),
                col("i_manager_id") == 1,
            )
            .group_by(["d_year", "i_brand", "i_brand_id"])
            .agg([col("ss_ext_sales_price").sum().alias("ext_price")])
            .select_exprs(
                [
                    col("d_year"),
                    col("i_brand_id").alias("brand_id"),
                    col("i_brand").alias("brand"),
                    col("ext_price"),
                ]
            )
        )
        return ordered(
            grouped, ["d_year", "ext_price", "brand_id"], [False, True, False]
        ).head(100)
    if q == "q55":
        var grouped = (
            brand_sales(
                t,
                (col("d_moy") == 11) & (col("d_year") == 1999),
                col("i_manager_id") == 28,
            )
            .group_by(["i_brand", "i_brand_id"])
            .agg([col("ss_ext_sales_price").sum().alias("ext_price")])
            .select_exprs(
                [
                    col("i_brand_id").alias("brand_id"),
                    col("i_brand").alias("brand"),
                    col("ext_price"),
                ]
            )
        )
        return ordered(grouped, ["ext_price", "brand_id"], [True, False]).head(
            100
        )
    if q == "q72":
        # Catalog orders of 1999 by divorced customers with high buying
        # potential, shipped more than five days after the sale, for items
        # whose stock in the week of the sale was below the ordered quantity.
        var sold = (
            t["catalog_sales"]
            .lazy()
            .join(
                t["date_dim"]
                .lazy()
                .filter(col("d_year") == 1999)
                .select_exprs(
                    [
                        col("d_date_sk").alias("d1_sk"),
                        col("d_date").alias("sold_date"),
                        col("d_week_seq"),
                    ]
                ),
                left_on=["cs_sold_date_sk"],
                right_on=["d1_sk"],
            )
            .join(
                t["customer_demographics"]
                .lazy()
                .filter(col("cd_marital_status") == "D"),
                left_on=["cs_bill_cdemo_sk"],
                right_on=["cd_demo_sk"],
            )
            .join(
                t["household_demographics"]
                .lazy()
                .filter(col("hd_buy_potential") == ">10000"),
                left_on=["cs_bill_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["date_dim"]
                .lazy()
                .select_exprs(
                    [
                        col("d_date_sk").alias("d3_sk"),
                        col("d_date").alias("ship_date"),
                    ]
                ),
                left_on=["cs_ship_date_sk"],
                right_on=["d3_sk"],
            )
            .filter(
                col("ship_date").cast(DataType.INT64)
                > col("sold_date").cast(DataType.INT64) + 5
            )
        )
        var stock = (
            t["inventory"]
            .lazy()
            .join(
                t["date_dim"]
                .lazy()
                .select_exprs(
                    [
                        col("d_date_sk").alias("d2_sk"),
                        col("d_week_seq").alias("inv_week_seq"),
                    ]
                ),
                left_on=["inv_date_sk"],
                right_on=["d2_sk"],
            )
        )
        var grouped = (
            sold.join(
                stock,
                left_on=["cs_item_sk", "d_week_seq"],
                right_on=["inv_item_sk", "inv_week_seq"],
            )
            .filter(col("inv_quantity_on_hand") < col("cs_quantity"))
            .join(
                t["warehouse"].lazy(),
                left_on=["inv_warehouse_sk"],
                right_on=["w_warehouse_sk"],
            )
            .join(
                t["item"].lazy(), left_on=["cs_item_sk"], right_on=["i_item_sk"]
            )
            .join(
                # A left join keeps only the left key (#466), so a second
                # copy of the promotion key says whether one matched.
                t["promotion"]
                .lazy()
                .select_exprs(
                    [col("p_promo_sk"), col("p_promo_sk").alias("promotion")]
                ),
                left_on=["cs_promo_sk"],
                right_on=["p_promo_sk"],
                how="left",
            )
            .join(
                t["catalog_returns"]
                .lazy()
                .select(["cr_item_sk", "cr_order_number"]),
                left_on=["cs_item_sk", "cs_order_number"],
                right_on=["cr_item_sk", "cr_order_number"],
                how="left",
            )
            .group_by(["i_item_desc", "w_warehouse_name", "d_week_seq"])
            .agg(
                [
                    one_if(col("promotion").is_null()).sum().alias("no_promo"),
                    one_if(col("promotion").is_not_null()).sum().alias("promo"),
                    col("cs_item_sk").len().alias("total_cnt"),
                ]
            )
        )
        return ordered(
            grouped,
            ["total_cnt", "i_item_desc", "w_warehouse_name", "d_week_seq"],
            [True, False, False, False],
            nulls_first=True,
        ).head(100)
    if q == "q82":
        return stocked_items(
            t,
            "62",
            "92",
            [129, 270, 821, 423],
            "2000-05-25",
            "2000-07-24",
            "store_sales",
            "ss_item_sk",
        )
    if q == "q84":
        var found = (
            t["customer"]
            .lazy()
            .join(
                t["customer_address"]
                .lazy()
                .filter(col("ca_city") == "Edgewood"),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["household_demographics"]
                .lazy()
                .join(
                    t["income_band"]
                    .lazy()
                    .filter(
                        (col("ib_lower_bound") >= 38128)
                        & (col("ib_upper_bound") <= 88128)
                    ),
                    left_on=["hd_income_band_sk"],
                    right_on=["ib_income_band_sk"],
                ),
                left_on=["c_current_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["customer_demographics"].lazy().select(["cd_demo_sk"]),
                left_on=["c_current_cdemo_sk"],
                right_on=["cd_demo_sk"],
            )
            .join(
                t["store_returns"].lazy().select(["sr_cdemo_sk"]),
                left_on=["c_current_cdemo_sk"],
                right_on=["sr_cdemo_sk"],
            )
            .select_exprs(
                [
                    col("c_customer_id").alias("customer_id"),
                    concat_str(
                        [
                            col("c_last_name").fill_null(""),
                            lit(", "),
                            col("c_first_name").fill_null(""),
                        ]
                    ).alias("customername"),
                ]
            )
        )
        return ascending(found, ["customer_id"], nulls_first=True).head(100)
    if q == "q85":
        ref sales = t["web_sales"]
        var refunded = (
            t["customer_demographics"]
            .lazy()
            .select_exprs(
                [
                    col("cd_demo_sk").alias("cd1_sk"),
                    col("cd_marital_status").alias("marital"),
                    col("cd_education_status").alias("education"),
                ]
            )
        )
        var returning = (
            t["customer_demographics"]
            .lazy()
            .select_exprs(
                [
                    col("cd_demo_sk").alias("cd2_sk"),
                    col("cd_marital_status").alias("marital2"),
                    col("cd_education_status").alias("education2"),
                ]
            )
        )
        var grouped = (
            sales.lazy()
            .join(
                t["web_returns"].lazy(),
                left_on=["ws_item_sk", "ws_order_number"],
                right_on=["wr_item_sk", "wr_order_number"],
            )
            .join(
                t["web_page"].lazy().select(["wp_web_page_sk"]),
                left_on=["ws_web_page_sk"],
                right_on=["wp_web_page_sk"],
            )
            .join(
                dates(t, col("d_year") == 2000, "d_sk"),
                left_on=["ws_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                refunded, left_on=["wr_refunded_cdemo_sk"], right_on=["cd1_sk"]
            )
            .join(
                returning,
                left_on=["wr_returning_cdemo_sk", "marital", "education"],
                right_on=["cd2_sk", "marital2", "education2"],
            )
            .join(
                t["customer_address"]
                .lazy()
                .filter(col("ca_country") == "United States"),
                left_on=["wr_refunded_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["reason"].lazy(),
                left_on=["wr_reason_sk"],
                right_on=["r_reason_sk"],
            )
            .filter(
                (
                    (col("marital") == "M")
                    & (col("education") == "Advanced Degree")
                    & between(sales, "ws_sales_price", "100.00", "150.00")
                )
                | (
                    (col("marital") == "S")
                    & (col("education") == "College")
                    & between(sales, "ws_sales_price", "50.00", "100.00")
                )
                | (
                    (col("marital") == "W")
                    & (col("education") == "2 yr Degree")
                    & between(sales, "ws_sales_price", "150.00", "200.00")
                )
            )
            .filter(
                (
                    col("ca_state").is_in(["IN", "OH", "NJ"])
                    & between(sales, "ws_net_profit", "100", "200")
                )
                | (
                    col("ca_state").is_in(["WI", "CT", "KY"])
                    & between(sales, "ws_net_profit", "150", "300")
                )
                | (
                    col("ca_state").is_in(["LA", "IA", "AR"])
                    & between(sales, "ws_net_profit", "50", "250")
                )
            )
            .group_by(["r_reason_desc"])
            .agg(
                [
                    col("ws_quantity").mean().alias("avg1"),
                    col("wr_refunded_cash").mean().alias("avg2"),
                    col("wr_fee").mean().alias("avg3"),
                ]
            )
            .select_exprs(
                [
                    col("r_reason_desc").str().slice(0, 20).alias("reason"),
                    col("avg1"),
                    col("avg2"),
                    col("avg3"),
                ]
            )
        )
        return ascending(grouped, ["reason", "avg1", "avg2", "avg3"]).head(100)
    if q == "q91":
        var buyers = (
            t["customer"]
            .lazy()
            .join(
                t["customer_demographics"]
                .lazy()
                .filter(
                    (
                        (col("cd_marital_status") == "M")
                        & (col("cd_education_status") == "Unknown")
                    )
                    | (
                        (col("cd_marital_status") == "W")
                        & (col("cd_education_status") == "Advanced Degree")
                    )
                ),
                left_on=["c_current_cdemo_sk"],
                right_on=["cd_demo_sk"],
            )
            .join(
                t["household_demographics"]
                .lazy()
                .filter(col("hd_buy_potential").str().starts_with("Unknown")),
                left_on=["c_current_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["customer_address"]
                .lazy()
                .filter(
                    col("ca_gmt_offset")
                    == like(t["customer_address"], "ca_gmt_offset", "-7")
                ),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
        )
        return (
            t["call_center"]
            .lazy()
            .join(
                t["catalog_returns"].lazy(),
                left_on=["cc_call_center_sk"],
                right_on=["cr_call_center_sk"],
            )
            .join(
                dates(
                    t, (col("d_year") == 1998) & (col("d_moy") == 11), "d_sk"
                ),
                left_on=["cr_returned_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                buyers,
                left_on=["cr_returning_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .group_by(
                [
                    "cc_call_center_id",
                    "cc_name",
                    "cc_manager",
                    "cd_marital_status",
                    "cd_education_status",
                ]
            )
            .agg([col("cr_net_loss").sum().alias("Returns_Loss")])
            .select_exprs(
                [
                    col("cc_call_center_id").alias("Call_Center"),
                    col("cc_name").alias("Call_Center_Name"),
                    col("cc_manager").alias("Manager"),
                    col("Returns_Loss"),
                ]
            )
            .sort("Returns_Loss", descending=True)
        )
    if q == "q96":
        return (
            t["store_sales"]
            .lazy()
            .join(
                t["household_demographics"]
                .lazy()
                .filter(col("hd_dep_count") == 7),
                left_on=["ss_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["time_dim"]
                .lazy()
                .filter((col("t_hour") == 20) & (col("t_minute") >= 30)),
                left_on=["ss_sold_time_sk"],
                right_on=["t_time_sk"],
            )
            .join(
                t["store"].lazy().filter(col("s_store_name") == "ese"),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .select_exprs([col("ss_store_sk").len().alias("count")])
        )
    raise unsupported("not translated")


def main() raises:
    run[query]()
