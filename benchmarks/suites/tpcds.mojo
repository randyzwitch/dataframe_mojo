"""TPC-DS: queries from DuckDB's `tpcds_queries()`, written with this
library's lazy API. Each query builds one plan over the tables and collects
it, as the PDS-H runner does. engines.py holds the Polars versions; DuckDB
runs the SQL text.

Translated so far, each batch chosen by a rule on the SQL text and not by
timings: the 23 single-block queries (one SELECT, no window, rollup or set
operation), then the 13 with exactly one more SELECT (a derived table or
a subquery) and still no window, rollup, set operation, EXISTS or WITH,
then the 7 with several such subqueries under the same exclusions, then
the 5 that add only EXISTS or NOT EXISTS, then the 7 whose WITH clause
defines one table, under the same exclusions.
Any other query reports `unsupported: not translated`, so it is counted in
every report instead of dropped. The same queries run on both data variants
(bench_suites.py): money columns as Float64, or as declared decimals, where
literals compared with them must be decimals too (`like`). Usage: see
suite_common.mojo.

SQL ORDER BY puts nulls last in DuckDB unless the query says NULLS FIRST;
`ordered` takes the placement because a LIMIT follows most sorts.
"""
from std.collections import Dict, Optional

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


def day_buckets(shipped: String, sold: String) -> List[Expr]:
    """Counts of rows by how many days passed between two date keys, in the
    five ranges q50, q62 and q99 report."""
    var lag = col(shipped) - col(sold)
    return [
        one_if(lag <= 30).sum().alias("30 days"),
        one_if((lag > 30) & (lag <= 60)).sum().alias("31-60 days"),
        one_if((lag > 60) & (lag <= 90)).sum().alias("61-90 days"),
        one_if((lag > 90) & (lag <= 120)).sum().alias("91-120 days"),
        one_if(lag > 120).sum().alias(">120 days"),
    ]


def shipping_delays(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    channel: LazyFrame,
    sales_key: String,
    channel_key: String,
    channel_name: String,
) raises -> LazyFrame:
    """Shipping delay counts by warehouse, ship mode and sales channel for
    orders shipped in one year of months (q62, q99)."""
    var keys: List[String] = ["w_substr", "sm_type", channel_name]
    return (
        t[sales]
        .lazy()
        .join(
            dates(
                t,
                col("d_month_seq").is_between(
                    lit(Int64(1200)), lit(Int64(1211))
                ),
                "d_sk",
            ),
            left_on=[prefix + "_ship_date_sk"],
            right_on=["d_sk"],
        )
        .join(
            t["warehouse"]
            .lazy()
            .select_exprs(
                [
                    col("w_warehouse_sk"),
                    col("w_warehouse_name")
                    .str()
                    .slice(0, 20)
                    .alias("w_substr"),
                ]
            ),
            left_on=[prefix + "_warehouse_sk"],
            right_on=["w_warehouse_sk"],
        )
        .join(
            t["ship_mode"].lazy(),
            left_on=[prefix + "_ship_mode_sk"],
            right_on=["sm_ship_mode_sk"],
        )
        .join(channel, left_on=[sales_key], right_on=[channel_key])
        .group_by(keys)
        .agg(day_buckets(prefix + "_ship_date_sk", prefix + "_sold_date_sk"))
    )


def tickets(
    t: Dict[String, DataFrame],
    days: Expr,
    stores: Expr,
    households: Expr,
    keys: List[String],
    sums: List[Expr],
    bought_city: Bool = False,
) raises -> LazyFrame:
    """Store sales on chosen days, in chosen stores, to chosen households,
    summed per ticket (q34, q46, q68, q73, q79). With `bought_city`, each
    sale carries the city of its address as `bought_city`."""
    var sold = (
        t["store_sales"]
        .lazy()
        .join(
            dates(t, days, "d_sk"),
            left_on=["ss_sold_date_sk"],
            right_on=["d_sk"],
        )
        .join(
            t["store"].lazy().filter(stores),
            left_on=["ss_store_sk"],
            right_on=["s_store_sk"],
        )
        .join(
            t["household_demographics"].lazy().filter(households),
            left_on=["ss_hdemo_sk"],
            right_on=["hd_demo_sk"],
        )
    )
    if bought_city:
        sold = sold.join(
            t["customer_address"]
            .lazy()
            .select_exprs(
                [
                    col("ca_address_sk").alias("bought_sk"),
                    col("ca_city").alias("bought_city"),
                ]
            ),
            left_on=["ss_addr_sk"],
            right_on=["bought_sk"],
        )
    return sold.group_by(keys).agg(sums)


def in_three_years() -> Expr:
    return col("d_year").is_in(ints([1999, 2000, 2001]))


def excess_discount(
    t: Dict[String, DataFrame], sales: String, prefix: String, maker: Int
) raises -> LazyFrame:
    """The discounts of one maker's items, over 90 days, that exceed 1.3
    times the item's average discount in those days (q32, q92). The SQL
    asks for that average per row; here it is one group-by joined back."""
    var amount = prefix + "_ext_discount_amt"
    var in_range = (
        t[sales]
        .lazy()
        .join(
            dates(
                t,
                col("d_date").is_between(
                    date_lit("2000-01-27"), date_lit("2000-04-26")
                ),
                "d_sk",
            ),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_sk"],
        )
    )
    var typical = (
        in_range.group_by([prefix + "_item_sk"])
        .agg([col(amount).cast(DataType.FLOAT64).mean().alias("typical")])
        .select_exprs(
            [col(prefix + "_item_sk").alias("typical_item"), col("typical")]
        )
    )
    return (
        in_range.join(
            t["item"].lazy().filter(col("i_manufact_id") == lit(Int64(maker))),
            left_on=[prefix + "_item_sk"],
            right_on=["i_item_sk"],
        )
        .join(typical, left_on=[prefix + "_item_sk"], right_on=["typical_item"])
        .filter(col(amount).cast(DataType.FLOAT64) > lit(1.3) * col("typical"))
        .select_exprs([col(amount).sum().alias("excess")])
    )


def buyers(
    t: Dict[String, DataFrame],
    sales: String,
    date_key: String,
    customer_key: String,
    days: Expr,
    marker: String,
) raises -> LazyFrame:
    """The customers with a sale through one channel on the chosen days,
    once each, as `<marker>_sk` and a second copy named `marker`: the
    first is a join key, the second says after a left join whether the
    customer was found (q10, q35, q69)."""
    return (
        t[sales]
        .lazy()
        .join(dates(t, days, "d_sk"), left_on=[date_key], right_on=["d_sk"])
        .select_exprs(
            [
                col(customer_key).alias(marker + "_sk"),
                col(customer_key).alias(marker),
            ]
        )
        .unique()
    )


def active_customers(
    t: Dict[String, DataFrame], days: Expr, places: Expr, also_elsewhere: Bool
) raises -> LazyFrame:
    """Customers at the chosen addresses who bought in a store on the
    chosen days and, on the same days, either also bought on the web or by
    catalog (`also_elsewhere`) or did neither, with their demographics."""
    var found = (
        t["customer"]
        .lazy()
        .join(
            t["customer_address"].lazy().filter(places),
            left_on=["c_current_addr_sk"],
            right_on=["ca_address_sk"],
        )
        .join(
            t["customer_demographics"].lazy(),
            left_on=["c_current_cdemo_sk"],
            right_on=["cd_demo_sk"],
        )
        .join(
            buyers(
                t,
                "store_sales",
                "ss_sold_date_sk",
                "ss_customer_sk",
                days,
                "st",
            ),
            left_on=["c_customer_sk"],
            right_on=["st_sk"],
            how="semi",
        )
        .join(
            buyers(
                t,
                "web_sales",
                "ws_sold_date_sk",
                "ws_bill_customer_sk",
                days,
                "web",
            ),
            left_on=["c_customer_sk"],
            right_on=["web_sk"],
            how="left",
        )
        .join(
            buyers(
                t,
                "catalog_sales",
                "cs_sold_date_sk",
                "cs_ship_customer_sk",
                days,
                "cat",
            ),
            left_on=["c_customer_sk"],
            right_on=["cat_sk"],
            how="left",
        )
    )
    if also_elsewhere:
        return found.filter(col("web").is_not_null() | col("cat").is_not_null())
    return found.filter(col("web").is_null() & col("cat").is_null())


def split_shipments(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    returns: String,
    returned_order: String,
    first_day: String,
    last_day: String,
    state: String,
    channel: LazyFrame,
    sales_key: String,
    channel_key: String,
) raises -> LazyFrame:
    """Orders shipped in a date range to one state through chosen outlets
    that left from more than one warehouse and were never returned: how
    many, and their shipping cost and profit (q16, q94)."""
    var order = prefix + "_order_number"
    var warehouse = prefix + "_warehouse_sk"
    # Orders with two different warehouses among their rows.
    var split = (
        t[sales]
        .lazy()
        .group_by([order])
        .agg(
            [
                col(warehouse).min().alias("first_warehouse"),
                col(warehouse).max().alias("last_warehouse"),
            ]
        )
        .filter(col("first_warehouse") != col("last_warehouse"))
        .select_exprs([col(order).alias("split_order")])
    )
    return (
        t[sales]
        .lazy()
        .join(
            dates(
                t,
                col("d_date").is_between(
                    date_lit(first_day), date_lit(last_day)
                ),
                "d_sk",
            ),
            left_on=[prefix + "_ship_date_sk"],
            right_on=["d_sk"],
        )
        .join(
            t["customer_address"].lazy().filter(col("ca_state") == state),
            left_on=[prefix + "_ship_addr_sk"],
            right_on=["ca_address_sk"],
        )
        .join(channel, left_on=[sales_key], right_on=[channel_key])
        .join(split, left_on=[order], right_on=["split_order"], how="semi")
        .join(
            t[returns].lazy(),
            left_on=[order],
            right_on=[returned_order],
            how="anti",
        )
        .select_exprs(
            [
                col(order).n_unique().alias("order count"),
                col(prefix + "_ext_ship_cost")
                .sum(min_count=1)
                .alias("total shipping cost"),
                col(prefix + "_net_profit")
                .sum(min_count=1)
                .alias("total net profit"),
            ]
        )
    )


def returners(
    t: Dict[String, DataFrame],
    returns: String,
    date_key: String,
    customer_key: String,
    address_key: String,
    amount: String,
    year: Int,
) raises -> LazyFrame:
    """Customers whose returns in a year, by their address's state, total
    more than 1.2 times the average such total in that state (q30, q81).
    The SQL compares each total with a correlated average; here the
    averages by state are one group-by joined back."""
    var totals = (
        t[returns]
        .lazy()
        .join(
            dates(t, col("d_year") == lit(Int64(year)), "d_sk"),
            left_on=[date_key],
            right_on=["d_sk"],
        )
        .join(
            t["customer_address"]
            .lazy()
            .select_exprs(
                [
                    col("ca_address_sk").alias("return_address"),
                    col("ca_state").alias("ctr_state"),
                ]
            ),
            left_on=[address_key],
            right_on=["return_address"],
        )
        .group_by([customer_key, "ctr_state"])
        .agg([col(amount).sum(min_count=1).alias("ctr_total_return")])
        .select_exprs(
            [
                col(customer_key).alias("ctr_customer_sk"),
                col("ctr_state"),
                col("ctr_total_return"),
            ]
        )
    )
    var typical = (
        totals.group_by(["ctr_state"])
        .agg(
            [
                col("ctr_total_return")
                .cast(DataType.FLOAT64)
                .mean()
                .alias("typical")
            ]
        )
        .select_exprs([col("ctr_state").alias("typical_state"), col("typical")])
    )
    return (
        totals.join(typical, left_on=["ctr_state"], right_on=["typical_state"])
        .filter(
            col("ctr_total_return").cast(DataType.FLOAT64)
            > lit(1.2) * col("typical")
        )
        .join(
            t["customer"].lazy(),
            left_on=["ctr_customer_sk"],
            right_on=["c_customer_sk"],
        )
        .join(
            t["customer_address"].lazy().filter(col("ca_state") == "GA"),
            left_on=["c_current_addr_sk"],
            right_on=["ca_address_sk"],
        )
    )


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
            .agg(day_buckets("sr_returned_date_sk", "ss_sold_date_sk"))
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
    if q == "q21":
        var day = date_lit("2000-03-11")
        var none = lit(Int64(0))
        var grouped = (
            t["inventory"]
            .lazy()
            .join(
                t["warehouse"].lazy(),
                left_on=["inv_warehouse_sk"],
                right_on=["w_warehouse_sk"],
            )
            .join(
                t["item"]
                .lazy()
                .filter(between(t["item"], "i_current_price", "0.99", "1.49")),
                left_on=["inv_item_sk"],
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
                left_on=["inv_date_sk"],
                right_on=["d_date_sk"],
            )
            .group_by(["w_warehouse_name", "i_item_id"])
            .agg(
                [
                    when(col("d_date") < day)
                    .then(col("inv_quantity_on_hand"))
                    .otherwise(none)
                    .sum()
                    .alias("inv_before"),
                    when(col("d_date") >= day)
                    .then(col("inv_quantity_on_hand"))
                    .otherwise(none)
                    .sum()
                    .alias("inv_after"),
                ]
            )
            .filter(
                (col("inv_before") > 0)
                & (
                    col("inv_after").cast(DataType.FLOAT64)
                    / col("inv_before").cast(DataType.FLOAT64)
                ).is_between(lit(2.0 / 3.0), lit(1.5))
            )
        )
        return ascending(
            grouped, ["w_warehouse_name", "i_item_id"], nulls_first=True
        ).head(100)
    if q == "q32":
        return excess_discount(t, "catalog_sales", "cs", 977)
    if q == "q34" or q == "q73":
        var per_car = col("hd_dep_count").cast(DataType.FLOAT64) / col(
            "hd_vehicle_count"
        ).cast(DataType.FLOAT64)
        var potential: List[String] = [">10000", "Unknown"]
        var days: Expr
        var stores: Expr
        var low = 15
        var high = 20
        var ratio = 1.2
        if q == "q34":
            days = (
                col("d_dom").is_between(lit(Int64(1)), lit(Int64(3)))
                | col("d_dom").is_between(lit(Int64(25)), lit(Int64(28)))
            ) & in_three_years()
            stores = col("s_county") == "Williamson County"
        else:
            days = (
                col("d_dom").is_between(lit(Int64(1)), lit(Int64(2)))
                & in_three_years()
            )
            var counties: List[String] = [
                "Orange County",
                "Bronx County",
                "Franklin Parish",
                "Williamson County",
            ]
            stores = col("s_county").is_in(counties)
            low = 1
            high = 5
            ratio = 1.0
        var keys: List[String] = ["ss_ticket_number", "ss_customer_sk"]
        var found = (
            tickets(
                t,
                days,
                stores,
                col("hd_buy_potential").is_in(potential)
                & (col("hd_vehicle_count") > 0)
                & (per_car > lit(ratio)),
                keys,
                [col("ss_ticket_number").len().alias("cnt")],
            )
            .filter(
                col("cnt")
                .cast(DataType.INT64)
                .is_between(lit(Int64(low)), lit(Int64(high)))
            )
            .join(
                t["customer"].lazy(),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .select(
                [
                    "c_last_name",
                    "c_first_name",
                    "c_salutation",
                    "c_preferred_cust_flag",
                    "ss_ticket_number",
                    "cnt",
                ]
            )
        )
        if q == "q73":
            return ordered(found, ["cnt", "c_last_name"], [True, False])
        return ordered(
            found,
            [
                "c_last_name",
                "c_first_name",
                "c_salutation",
                "c_preferred_cust_flag",
                "ss_ticket_number",
            ],
            [False, False, False, True, False],
            nulls_first=True,
        )
    if q == "q41":
        # The SQL counts, per item, the items of the same maker that match
        # one of eight descriptions, and keeps items where any does: the
        # items of the makers with such an item.
        var described = Optional[Expr]()
        var shapes: List[List[String]] = [
            [
                "Women",
                "powder",
                "khaki",
                "Ounce",
                "Oz",
                "medium",
                "extra large",
            ],
            ["Women", "brown", "honeydew", "Bunch", "Ton", "N/A", "small"],
            ["Men", "floral", "deep", "N/A", "Dozen", "petite", "petite"],
            [
                "Men",
                "light",
                "cornflower",
                "Box",
                "Pound",
                "medium",
                "extra large",
            ],
            [
                "Women",
                "midnight",
                "snow",
                "Pallet",
                "Gross",
                "medium",
                "extra large",
            ],
            ["Women", "cyan", "papaya", "Cup", "Dram", "N/A", "small"],
            ["Men", "orange", "frosted", "Each", "Tbl", "petite", "petite"],
            ["Men", "forest", "ghost", "Lb", "Bundle", "medium", "extra large"],
        ]
        for shape in shapes:
            var colors: List[String] = [shape[1], shape[2]]
            var units: List[String] = [shape[3], shape[4]]
            var sizes: List[String] = [shape[5], shape[6]]
            var one = (
                (col("i_category") == shape[0])
                & col("i_color").is_in(colors)
                & col("i_units").is_in(units)
                & col("i_size").is_in(sizes)
            )
            described = Optional(
                (described.value() | one) if described else one.copy()
            )
        var makers = (
            t["item"]
            .lazy()
            .filter(described.value())
            .select_exprs([col("i_manufact").alias("maker")])
            .unique()
        )
        var names = (
            t["item"]
            .lazy()
            .filter(
                col("i_manufact_id").is_between(
                    lit(Int64(738)), lit(Int64(778))
                )
            )
            .join(
                makers, left_on=["i_manufact"], right_on=["maker"], how="semi"
            )
            .select(["i_product_name"])
            .unique()
        )
        return ascending(names, ["i_product_name"]).head(100)
    if q == "q45":
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
        # Item ids of ten chosen items; a second copy of the id marks the
        # sales of an item with one of them.
        var listed = (
            t["item"]
            .lazy()
            .filter(
                col("i_item_sk").is_in(
                    ints([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
                )
            )
            .select_exprs(
                [
                    col("i_item_id").alias("listed_id"),
                    col("i_item_id").alias("listed"),
                ]
            )
            .unique()
        )
        var grouped = (
            t["web_sales"]
            .lazy()
            .join(
                dates(t, (col("d_qoy") == 2) & (col("d_year") == 2001), "d_sk"),
                left_on=["ws_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["customer"].lazy(),
                left_on=["ws_bill_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_address"].lazy(),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["item"].lazy(), left_on=["ws_item_sk"], right_on=["i_item_sk"]
            )
            .join(
                listed,
                left_on=["i_item_id"],
                right_on=["listed_id"],
                how="left",
            )
            .filter(
                col("ca_zip").str().slice(0, 5).is_in(zips)
                | col("listed").is_not_null()
            )
            .group_by(["ca_zip", "ca_city"])
            .agg([col("ws_sales_price").sum().alias("total")])
        )
        return ascending(grouped, ["ca_zip", "ca_city"]).head(100)
    if q == "q46" or q == "q68":
        var cities: List[String] = ["Fairview", "Midway"]
        var keys: List[String] = [
            "ss_ticket_number",
            "ss_customer_sk",
            "ss_addr_sk",
            "bought_city",
        ]
        var days: Expr
        var sums: List[Expr]
        var shown: List[String]
        if q == "q46":
            days = col("d_dow").is_in(ints([6, 0])) & in_three_years()
            sums = [
                col("ss_coupon_amt").sum().alias("amt"),
                col("ss_net_profit").sum().alias("profit"),
            ]
            shown = [
                "c_last_name",
                "c_first_name",
                "ca_city",
                "bought_city",
                "ss_ticket_number",
                "amt",
                "profit",
            ]
        else:
            days = (
                col("d_dom").is_between(lit(Int64(1)), lit(Int64(2)))
                & in_three_years()
            )
            sums = [
                col("ss_ext_sales_price").sum().alias("extended_price"),
                col("ss_ext_list_price").sum().alias("list_price"),
                col("ss_ext_tax").sum().alias("extended_tax"),
            ]
            shown = [
                "c_last_name",
                "c_first_name",
                "ca_city",
                "bought_city",
                "ss_ticket_number",
                "extended_price",
                "extended_tax",
                "list_price",
            ]
        var found = (
            tickets(
                t,
                days,
                col("s_city").is_in(cities),
                (col("hd_dep_count") == 4) | (col("hd_vehicle_count") == 3),
                keys,
                sums,
                bought_city=True,
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
            .filter(col("ca_city") != col("bought_city"))
            .select(shown)
        )
        if q == "q46":
            return ascending(
                found,
                [
                    "c_last_name",
                    "c_first_name",
                    "ca_city",
                    "bought_city",
                    "ss_ticket_number",
                ],
                nulls_first=True,
            ).head(100)
        return ascending(
            found, ["c_last_name", "ss_ticket_number"], nulls_first=True
        ).head(100)
    if q == "q62":
        return ascending(
            shipping_delays(
                t,
                "web_sales",
                "ws",
                t["web_site"].lazy(),
                "ws_web_site_sk",
                "web_site_sk",
                "web_name",
            ),
            ["w_substr", "sm_type", "web_name"],
            nulls_first=True,
        ).head(100)
    if q == "q79":
        var keys: List[String] = [
            "ss_ticket_number",
            "ss_customer_sk",
            "ss_addr_sk",
            "s_city",
        ]
        var found = (
            tickets(
                t,
                (col("d_dow") == 1) & in_three_years(),
                col("s_number_employees").is_between(
                    lit(Int64(200)), lit(Int64(295))
                ),
                (col("hd_dep_count") == 6) | (col("hd_vehicle_count") > 2),
                keys,
                # SQL's sum of no values is null, and profit orders the rows.
                [
                    col("ss_coupon_amt").sum(min_count=1).alias("amt"),
                    col("ss_net_profit").sum(min_count=1).alias("profit"),
                ],
            )
            .join(
                t["customer"].lazy(),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .select_exprs(
                [
                    col("c_last_name"),
                    col("c_first_name"),
                    col("s_city").str().slice(0, 30).alias("city"),
                    col("ss_ticket_number"),
                    col("amt"),
                    col("profit"),
                ]
            )
        )
        return found.sort(
            [
                "c_last_name",
                "c_first_name",
                "city",
                "profit",
                "ss_ticket_number",
            ],
            descending=[False, False, False, False, False],
            nulls_last=[False, False, False, False, True],
        ).head(100)
    if q == "q92":
        return excess_discount(t, "web_sales", "ws", 350)
    if q == "q93":
        ref sales = t["store_sales"]
        var price_type = sales.column("ss_sales_price").dtype()
        var kept = col("ss_quantity") - col("sr_return_quantity")
        var grouped = (
            sales.lazy()
            .join(
                t["store_returns"]
                .lazy()
                .join(
                    t["reason"]
                    .lazy()
                    .filter(col("r_reason_desc") == "reason 28"),
                    left_on=["sr_reason_sk"],
                    right_on=["r_reason_sk"],
                ),
                left_on=["ss_item_sk", "ss_ticket_number"],
                right_on=["sr_item_sk", "sr_ticket_number"],
            )
            .with_columns(
                [
                    (
                        when(col("sr_return_quantity").is_not_null())
                        .then(kept)
                        .otherwise(col("ss_quantity"))
                        .cast(price_type)
                        * col("ss_sales_price")
                    ).alias("act_sales")
                ]
            )
            .group_by(["ss_customer_sk"])
            # SQL's sum of no values is null, and nulls sort first here.
            .agg([col("act_sales").sum(min_count=1).alias("sumsales")])
        )
        return ascending(
            grouped, ["sumsales", "ss_customer_sk"], nulls_first=True
        ).head(100)
    if q == "q99":
        var grouped = shipping_delays(
            t,
            "catalog_sales",
            "cs",
            t["call_center"].lazy(),
            "cs_call_center_sk",
            "cc_call_center_sk",
            "cc_name",
        ).select_exprs(
            [
                col("w_substr"),
                col("sm_type"),
                col("cc_name").str().to_lowercase().alias("cc_name_lower"),
                col("30 days"),
                col("31-60 days"),
                col("61-90 days"),
                col("91-120 days"),
                col(">120 days"),
            ]
        )
        return ascending(
            grouped, ["w_substr", "sm_type", "cc_name_lower"], nulls_first=True
        ).head(100)
    if q == "q6":
        # The month of January 2001, read first, as a user would.
        var month = (
            t["date_dim"]
            .lazy()
            .filter((col("d_year") == 2001) & (col("d_moy") == 1))
            .select(["d_month_seq"])
            .unique()
            .collect()
            .item()
            .int64()
        )
        # Items priced over 1.2 times their category's average. An item
        # without a category has no average to compare with in the SQL.
        var price = col("i_current_price").cast(DataType.FLOAT64)
        var costly = (
            t["item"]
            .lazy()
            .filter(col("i_category").is_not_null())
            .filter(price > lit(1.2) * price.mean().over("i_category"))
            .select(["i_item_sk"])
        )
        var grouped = (
            t["store_sales"]
            .lazy()
            .join(
                dates(t, col("d_month_seq") == lit(month), "d_sk"),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(costly, left_on=["ss_item_sk"], right_on=["i_item_sk"])
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
            .group_by(["ca_state"])
            .agg([col("ca_state").len().alias("cnt")])
            .filter(col("cnt").cast(DataType.INT64) >= 10)
            .select_exprs([col("ca_state").alias("state"), col("cnt")])
        )
        return ascending(grouped, ["cnt", "state"], nulls_first=True).head(100)
    if q == "q9":
        # Five quantity bands; each reports the average discount when the
        # band has more sales than a given count, else the average paid.
        var lows: List[Int] = [1, 21, 41, 61, 81]
        var counts: List[Int] = [74129, 122840, 56580, 10097, 165306]
        var buckets = List[Expr]()
        for i in range(5):
            var band = col("ss_quantity").is_between(
                lit(Int64(lows[i])), lit(Int64(lows[i] + 19))
            )
            buckets.append(
                when(one_if(band).sum() > lit(Int64(counts[i])))
                .then(
                    when(band)
                    .then(col("ss_ext_discount_amt").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
                .otherwise(
                    when(band)
                    .then(col("ss_net_paid").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
                .alias("bucket" + String(i + 1))
            )
        return t["store_sales"].lazy().select_exprs(buckets)
    if q == "q28":
        # Six quantity bands, each a filter of its own with three
        # statistics of the list price, side by side in one row.
        ref sales = t["store_sales"]
        var lows: List[Int] = [0, 6, 11, 16, 21, 26]
        var highs: List[Int] = [5, 10, 15, 20, 25, 30]
        var prices: List[Int] = [8, 90, 142, 135, 122, 154]
        var coupons: List[Int] = [459, 2323, 12214, 6071, 836, 7326]
        var costs: List[Int] = [57, 31, 79, 38, 17, 7]
        # The six one-row results sit side by side through a cross join.
        var row = Optional[LazyFrame]()
        for i in range(6):
            var tag = "B" + String(i + 1)
            var band = (
                sales.lazy()
                .filter(
                    col("ss_quantity").is_between(
                        lit(Int64(lows[i])), lit(Int64(highs[i]))
                    )
                    & (
                        between(
                            sales,
                            "ss_list_price",
                            String(prices[i]),
                            String(prices[i] + 10),
                        )
                        | between(
                            sales,
                            "ss_coupon_amt",
                            String(coupons[i]),
                            String(coupons[i] + 1000),
                        )
                        | between(
                            sales,
                            "ss_wholesale_cost",
                            String(costs[i]),
                            String(costs[i] + 20),
                        )
                    )
                    # count(DISTINCT) leaves nulls out; the average and the
                    # count of the column do too.
                    & col("ss_list_price").is_not_null()
                )
                .select_exprs(
                    [
                        col("ss_list_price").mean().alias(tag + "_LP"),
                        col("ss_list_price").count().alias(tag + "_CNT"),
                        col("ss_list_price").n_unique().alias(tag + "_CNTD"),
                    ]
                )
            )
            row = Optional(
                row.value().join(band, how="cross") if row else band^
            )
        return row.take()
    if q == "q61":
        # November 1998 jewelry sales in one time zone, and the part of
        # them sold on a mail, email or TV promotion.
        var promoted = (
            t["promotion"]
            .lazy()
            .filter(
                (col("p_channel_dmail") == "Y")
                | (col("p_channel_email") == "Y")
                | (col("p_channel_tv") == "Y")
            )
            .select_exprs(
                [col("p_promo_sk"), col("p_promo_sk").alias("promoted")]
            )
        )
        var offset = like(t["store"], "s_gmt_offset", "-5")
        var both = (
            t["store_sales"]
            .lazy()
            .join(
                dates(
                    t, (col("d_year") == 1998) & (col("d_moy") == 11), "d_sk"
                ),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["store"].lazy().filter(col("s_gmt_offset") == offset),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["item"].lazy().filter(col("i_category") == "Jewelry"),
                left_on=["ss_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                t["customer"].lazy(),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_address"]
                .lazy()
                .filter(
                    col("ca_gmt_offset")
                    == like(t["customer_address"], "ca_gmt_offset", "-5")
                ),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                promoted,
                left_on=["ss_promo_sk"],
                right_on=["p_promo_sk"],
                how="left",
            )
            .select_exprs(
                [
                    when(col("promoted").is_not_null())
                    .then(col("ss_ext_sales_price"))
                    .end()
                    .sum(min_count=1)
                    .alias("promotions"),
                    col("ss_ext_sales_price").sum(min_count=1).alias("total"),
                ]
            )
        )
        return both.with_columns(
            [
                (
                    col("promotions").cast(DataType.FLOAT64)
                    / col("total").cast(DataType.FLOAT64)
                    * lit(100.0)
                ).alias("share")
            ]
        )
    if q == "q65":
        var revenue = (
            t["store_sales"]
            .lazy()
            .join(
                dates(
                    t,
                    col("d_month_seq").is_between(
                        lit(Int64(1176)), lit(Int64(1187))
                    ),
                    "d_sk",
                ),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .group_by(["ss_store_sk", "ss_item_sk"])
            .agg([col("ss_sales_price").sum(min_count=1).alias("revenue")])
        )
        var typical = (
            revenue.group_by(["ss_store_sk"])
            .agg([col("revenue").cast(DataType.FLOAT64).mean().alias("ave")])
            .select_exprs([col("ss_store_sk").alias("ave_store"), col("ave")])
        )
        var low = (
            revenue.join(
                typical, left_on=["ss_store_sk"], right_on=["ave_store"]
            )
            .filter(
                col("revenue").cast(DataType.FLOAT64) <= lit(0.1) * col("ave")
            )
            .join(
                t["store"].lazy(),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["item"].lazy(), left_on=["ss_item_sk"], right_on=["i_item_sk"]
            )
            .select(
                [
                    "s_store_name",
                    "i_item_desc",
                    "revenue",
                    "i_current_price",
                    "i_wholesale_cost",
                    "i_brand",
                ]
            )
        )
        return ascending(
            low, ["s_store_name", "i_item_desc"], nulls_first=True
        ).head(100)
    if q == "q88":
        # Sales at one store to three kinds of household, counted by half
        # hour from 8:30 to 12:30.
        var households = (
            ((col("hd_dep_count") == 4) & (col("hd_vehicle_count") <= 6))
            | ((col("hd_dep_count") == 2) & (col("hd_vehicle_count") <= 4))
            | ((col("hd_dep_count") == 0) & (col("hd_vehicle_count") <= 2))
        )
        var names: List[String] = [
            "h8_30_to_9",
            "h9_to_9_30",
            "h9_30_to_10",
            "h10_to_10_30",
            "h10_30_to_11",
            "h11_to_11_30",
            "h11_30_to_12",
            "h12_to_12_30",
        ]
        var counts = List[Expr]()
        for i in range(8):
            var hour = 8 + (i + 1) // 2
            var half = (col("t_minute") >= 30) if i % 2 == 0 else (
                col("t_minute") < 30
            )
            counts.append(
                one_if((col("t_hour") == lit(Int64(hour))) & half)
                .sum()
                .alias(names[i])
            )
        return (
            t["store_sales"]
            .lazy()
            .join(
                t["household_demographics"].lazy().filter(households),
                left_on=["ss_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["store"].lazy().filter(col("s_store_name") == "ese"),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["time_dim"]
                .lazy()
                .filter(
                    col("t_hour").is_between(lit(Int64(8)), lit(Int64(12)))
                ),
                left_on=["ss_sold_time_sk"],
                right_on=["t_time_sk"],
            )
            .select_exprs(counts)
        )
    if q == "q90":
        var counted = (
            t["web_sales"]
            .lazy()
            .join(
                t["household_demographics"]
                .lazy()
                .filter(col("hd_dep_count") == 6),
                left_on=["ws_ship_hdemo_sk"],
                right_on=["hd_demo_sk"],
            )
            .join(
                t["web_page"]
                .lazy()
                .filter(
                    col("wp_char_count").is_between(
                        lit(Int64(5000)), lit(Int64(5200))
                    )
                ),
                left_on=["ws_web_page_sk"],
                right_on=["wp_web_page_sk"],
            )
            .join(
                t["time_dim"].lazy(),
                left_on=["ws_sold_time_sk"],
                right_on=["t_time_sk"],
            )
            .select_exprs(
                [
                    one_if(
                        col("t_hour").is_between(lit(Int64(8)), lit(Int64(9)))
                    )
                    .sum()
                    .alias("amc"),
                    one_if(
                        col("t_hour").is_between(lit(Int64(19)), lit(Int64(20)))
                    )
                    .sum()
                    .alias("pmc"),
                ]
            )
        )
        return counted.select_exprs(
            [
                when(col("pmc") != 0)
                .then(
                    col("amc").cast(DataType.FLOAT64)
                    / col("pmc").cast(DataType.FLOAT64)
                )
                .end()
                .alias("am_pm_ratio")
            ]
        )
    if q == "q10" or q == "q69":
        var keys: List[String] = [
            "cd_gender",
            "cd_marital_status",
            "cd_education_status",
            "cd_purchase_estimate",
            "cd_credit_rating",
        ]
        var shown = List[Expr]()
        var customers: LazyFrame
        if q == "q10":
            var counties: List[String] = [
                "Rush County",
                "Toole County",
                "Jefferson County",
                "Dona Ana County",
                "La Porte County",
            ]
            keys.append("cd_dep_count")
            keys.append("cd_dep_employed_count")
            keys.append("cd_dep_college_count")
            customers = active_customers(
                t,
                (col("d_year") == 2002)
                & col("d_moy").is_between(lit(Int64(1)), lit(Int64(4))),
                col("ca_county").is_in(counties),
                True,
            )
        else:
            var states: List[String] = ["KY", "GA", "NM"]
            customers = active_customers(
                t,
                (col("d_year") == 2001)
                & col("d_moy").is_between(lit(Int64(4)), lit(Int64(6))),
                col("ca_state").is_in(states),
                False,
            )
        # The SQL lists the count once after each key from the third on.
        for i in range(len(keys)):
            shown.append(col(keys[i]))
            if i >= 2:
                shown.append(col("cnt").alias("cnt" + String(i - 1)))
        var grouped = (
            customers.group_by(keys)
            .agg([col("cd_gender").len().alias("cnt")])
            .select_exprs(shown)
        )
        return ascending(grouped, keys).head(100)
    if q == "q35":
        var keys: List[String] = [
            "ca_state",
            "cd_gender",
            "cd_marital_status",
            "cd_dep_count",
            "cd_dep_employed_count",
            "cd_dep_college_count",
        ]
        var aggregates: List[Expr] = [col("ca_state").len().alias("cnt")]
        var shown = List[Expr]()
        for i in range(6):
            shown.append(col(keys[i]))
            if i >= 3:
                var n = String(i - 2)
                var value = col(keys[i])
                aggregates.append(value.min().alias("min" + n))
                aggregates.append(value.max().alias("max" + n))
                aggregates.append(value.mean().alias("avg" + n))
                shown.append(col("cnt").alias("cnt" + n))
                shown.append(col("min" + n))
                shown.append(col("max" + n))
                shown.append(col("avg" + n))
        var grouped = (
            active_customers(
                t,
                (col("d_year") == 2002) & (col("d_qoy") < 4),
                col("ca_address_sk").is_not_null(),
                True,
            )
            .group_by(keys)
            .agg(aggregates)
            .select_exprs(shown)
        )
        return ascending(grouped, keys, nulls_first=True).head(100)
    if q == "q16":
        return split_shipments(
            t,
            "catalog_sales",
            "cs",
            "catalog_returns",
            "cr_order_number",
            "2002-02-01",
            "2002-04-02",
            "GA",
            t["call_center"]
            .lazy()
            .filter(col("cc_county") == "Williamson County"),
            "cs_call_center_sk",
            "cc_call_center_sk",
        )
    if q == "q94":
        return split_shipments(
            t,
            "web_sales",
            "ws",
            "web_returns",
            "wr_order_number",
            "1999-02-01",
            "1999-04-02",
            "IL",
            t["web_site"].lazy().filter(col("web_company_name") == "pri"),
            "ws_web_site_sk",
            "web_site_sk",
        )
    if q == "q1":
        var totals = (
            t["store_returns"]
            .lazy()
            .join(
                dates(t, col("d_year") == 2000, "d_sk"),
                left_on=["sr_returned_date_sk"],
                right_on=["d_sk"],
            )
            .group_by(["sr_customer_sk", "sr_store_sk"])
            .agg([col("sr_return_amt").sum(min_count=1).alias("total")])
        )
        var typical = (
            totals.group_by(["sr_store_sk"])
            .agg([col("total").cast(DataType.FLOAT64).mean().alias("typical")])
            .select_exprs(
                [col("sr_store_sk").alias("typical_store"), col("typical")]
            )
        )
        var found = (
            totals.join(
                typical, left_on=["sr_store_sk"], right_on=["typical_store"]
            )
            .filter(
                col("total").cast(DataType.FLOAT64) > lit(1.2) * col("typical")
            )
            .join(
                t["store"].lazy().filter(col("s_state") == "TN"),
                left_on=["sr_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["customer"].lazy(),
                left_on=["sr_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .select(["c_customer_id"])
        )
        return ascending(found, ["c_customer_id"]).head(100)
    if q == "q24":
        # Returned store sales by customer, store and item attributes, then
        # peach items whose total passes 5% of the average over all. The
        # grouped table is collected once and read twice.
        var keys: List[String] = [
            "c_last_name",
            "c_first_name",
            "s_store_name",
            "ca_state",
            "s_state",
            "i_color",
            "i_current_price",
            "i_manager_id",
            "i_units",
            "i_size",
        ]
        var ssales = (
            t["store_sales"]
            .lazy()
            .join(
                t["store_returns"]
                .lazy()
                .select(["sr_ticket_number", "sr_item_sk"]),
                left_on=["ss_ticket_number", "ss_item_sk"],
                right_on=["sr_ticket_number", "sr_item_sk"],
            )
            .join(
                t["store"].lazy().filter(col("s_market_id") == 8),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["item"].lazy(), left_on=["ss_item_sk"], right_on=["i_item_sk"]
            )
            .join(
                t["customer"].lazy(),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_address"].lazy(),
                left_on=["c_current_addr_sk", "s_zip"],
                right_on=["ca_address_sk", "ca_zip"],
            )
            .filter(
                col("c_birth_country") != col("ca_country").str().to_uppercase()
            )
            .group_by(keys)
            .agg([col("ss_net_paid").sum(min_count=1).alias("netpaid")])
            .collect()
        )
        # The cutoff is 5% of the mean over every row of ssales; with no
        # rows the mean is null and the comparison keeps nothing.
        var average = ssales.select(
            col("netpaid").cast(DataType.FLOAT64).mean()
        ).item()
        var above = lit(False)
        if not average.is_null():
            above = col("paid").cast(DataType.FLOAT64) > lit(
                average.float64() * 0.05
            )
        return (
            ssales.lazy()
            .filter(col("i_color") == "peach")
            .group_by(["c_last_name", "c_first_name", "s_store_name"])
            .agg([col("netpaid").sum(min_count=1).alias("paid")])
            .filter(above)
            .sort(["c_last_name", "c_first_name", "s_store_name"])
        )
    if q == "q30":
        var found = returners(
            t,
            "web_returns",
            "wr_returned_date_sk",
            "wr_returning_customer_sk",
            "wr_returning_addr_sk",
            "wr_return_amt",
            2002,
        )
        var shown: List[String] = [
            "c_customer_id",
            "c_salutation",
            "c_first_name",
            "c_last_name",
            "c_preferred_cust_flag",
            "c_birth_day",
            "c_birth_month",
            "c_birth_year",
            "c_birth_country",
            "c_login",
            "c_email_address",
            "c_last_review_date_sk",
            "ctr_total_return",
        ]
        return ascending(found.select(shown), shown, nulls_first=True).head(100)
    if q == "q39":
        # Warehouse-item months of 2001 whose stock varies more than its
        # mean, January beside February.
        var quantity = col("inv_quantity_on_hand").cast(DataType.FLOAT64)
        var stock = (
            t["inventory"]
            .lazy()
            .join(
                t["item"].lazy().select(["i_item_sk"]),
                left_on=["inv_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                t["warehouse"]
                .lazy()
                .select(["w_warehouse_sk", "w_warehouse_name"]),
                left_on=["inv_warehouse_sk"],
                right_on=["w_warehouse_sk"],
            )
            .join(
                t["date_dim"]
                .lazy()
                .filter(col("d_year") == 2001)
                .select(["d_date_sk", "d_moy"]),
                left_on=["inv_date_sk"],
                right_on=["d_date_sk"],
            )
            .group_by(
                ["w_warehouse_name", "inv_warehouse_sk", "inv_item_sk", "d_moy"]
            )
            .agg([quantity.std().alias("stdev"), quantity.mean().alias("mean")])
            .filter(
                (col("mean") != lit(0.0))
                & (col("stdev") / col("mean") > lit(1.0))
            )
            .with_columns([(col("stdev") / col("mean")).alias("cov")])
        )
        var january = stock.filter(col("d_moy") == 1).select_exprs(
            [
                col("inv_warehouse_sk").alias("wsk1"),
                col("inv_item_sk").alias("isk1"),
                col("d_moy").alias("dmoy1"),
                col("mean").alias("mean1"),
                col("cov").alias("cov1"),
            ]
        )
        var february = stock.filter(col("d_moy") == 2).select_exprs(
            [
                col("inv_warehouse_sk").alias("wsk2"),
                col("inv_item_sk").alias("isk2"),
                col("d_moy").alias("dmoy2"),
                col("mean").alias("mean2"),
                col("cov").alias("cov2"),
            ]
        )
        var paired = january.join(
            february, left_on=["isk1", "wsk1"], right_on=["isk2", "wsk2"]
        ).select_exprs(
            [
                col("wsk1"),
                col("isk1"),
                col("dmoy1"),
                col("mean1"),
                col("cov1"),
                col("wsk1").alias("wsk2"),
                col("isk1").alias("isk2"),
                col("dmoy2"),
                col("mean2"),
                col("cov2"),
            ]
        )
        return ascending(
            paired,
            [
                "wsk1",
                "isk1",
                "dmoy1",
                "mean1",
                "cov1",
                "dmoy2",
                "mean2",
                "cov2",
            ],
            nulls_first=True,
        )
    if q == "q59":
        var days: List[String] = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        var short: List[String] = [
            "sun",
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
        ]
        var sums = List[Expr]()
        for i in range(7):
            sums.append(
                when(col("d_day_name") == days[i])
                .then(col("ss_sales_price"))
                .end()
                .sum(min_count=1)
                .alias(short[i] + "_sales")
            )
        var weekly = (
            t["store_sales"]
            .lazy()
            .join(
                t["date_dim"]
                .lazy()
                .select(["d_date_sk", "d_week_seq", "d_day_name"]),
                left_on=["ss_sold_date_sk"],
                right_on=["d_date_sk"],
            )
            .group_by(["d_week_seq", "ss_store_sk"])
            .agg(sums)
            .join(
                t["store"]
                .lazy()
                .select(["s_store_sk", "s_store_name", "s_store_id"]),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
        )
        # As the SQL has it, each week's row joins every one of its days.
        var weeks = (
            t["date_dim"]
            .lazy()
            .select_exprs([col("d_week_seq").alias("week"), col("d_month_seq")])
        )
        var first = weekly.join(
            weeks.filter(
                col("d_month_seq").is_between(
                    lit(Int64(1212)), lit(Int64(1223))
                )
            ),
            left_on=["d_week_seq"],
            right_on=["week"],
        ).select_exprs(
            [
                col("s_store_name").alias("s_store_name1"),
                col("d_week_seq").alias("d_week_seq1"),
                col("s_store_id").alias("s_store_id1"),
                (col("d_week_seq") + lit(Int64(52))).alias("next_week"),
                col("sun_sales").alias("sun_sales1"),
                col("mon_sales").alias("mon_sales1"),
                col("tue_sales").alias("tue_sales1"),
                col("wed_sales").alias("wed_sales1"),
                col("thu_sales").alias("thu_sales1"),
                col("fri_sales").alias("fri_sales1"),
                col("sat_sales").alias("sat_sales1"),
            ]
        )
        var second = weekly.join(
            weeks.filter(
                col("d_month_seq").is_between(
                    lit(Int64(1224)), lit(Int64(1235))
                )
            ),
            left_on=["d_week_seq"],
            right_on=["week"],
        ).select_exprs(
            [
                col("d_week_seq").alias("d_week_seq2"),
                col("s_store_id").alias("s_store_id2"),
                col("sun_sales").alias("sun_sales2"),
                col("mon_sales").alias("mon_sales2"),
                col("tue_sales").alias("tue_sales2"),
                col("wed_sales").alias("wed_sales2"),
                col("thu_sales").alias("thu_sales2"),
                col("fri_sales").alias("fri_sales2"),
                col("sat_sales").alias("sat_sales2"),
            ]
        )
        var ratios = List[Expr]()
        ratios.append(col("s_store_name1"))
        ratios.append(col("s_store_id1"))
        ratios.append(col("d_week_seq1"))
        for i in range(7):
            ratios.append(
                (
                    col(short[i] + "_sales1").cast(DataType.FLOAT64)
                    / col(short[i] + "_sales2").cast(DataType.FLOAT64)
                ).alias(short[i] + "_sales_ratio")
            )
        var paired = first.join(
            second,
            left_on=["s_store_id1", "next_week"],
            right_on=["s_store_id2", "d_week_seq2"],
        ).select_exprs(ratios)
        return ascending(
            paired,
            ["s_store_name1", "s_store_id1", "d_week_seq1"],
            nulls_first=True,
        ).head(100)
    if q == "q81":
        var found = returners(
            t,
            "catalog_returns",
            "cr_returned_date_sk",
            "cr_returning_customer_sk",
            "cr_returning_addr_sk",
            "cr_return_amt_inc_tax",
            2000,
        )
        var shown: List[String] = [
            "c_customer_id",
            "c_salutation",
            "c_first_name",
            "c_last_name",
            "ca_street_number",
            "ca_street_name",
            "ca_street_type",
            "ca_suite_number",
            "ca_city",
            "ca_county",
            "ca_state",
            "ca_zip",
            "ca_country",
            "ca_gmt_offset",
            "ca_location_type",
            "ctr_total_return",
        ]
        return ascending(found.select(shown), shown).head(100)
    if q == "q95":
        # Orders shipped from two warehouses that were also returned: the
        # SQL's second IN joins returns to those orders, so it is the
        # returned orders among the split ones.
        var split = (
            t["web_sales"]
            .lazy()
            .group_by(["ws_order_number"])
            .agg(
                [
                    col("ws_warehouse_sk").min().alias("first_warehouse"),
                    col("ws_warehouse_sk").max().alias("last_warehouse"),
                ]
            )
            .filter(col("first_warehouse") != col("last_warehouse"))
            .select_exprs([col("ws_order_number").alias("split_order")])
        )
        return (
            t["web_sales"]
            .lazy()
            .join(
                dates(
                    t,
                    col("d_date").is_between(
                        date_lit("1999-02-01"), date_lit("1999-04-02")
                    ),
                    "d_sk",
                ),
                left_on=["ws_ship_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["customer_address"].lazy().filter(col("ca_state") == "IL"),
                left_on=["ws_ship_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["web_site"].lazy().filter(col("web_company_name") == "pri"),
                left_on=["ws_web_site_sk"],
                right_on=["web_site_sk"],
            )
            .join(
                split,
                left_on=["ws_order_number"],
                right_on=["split_order"],
                how="semi",
            )
            .join(
                t["web_returns"].lazy().select(["wr_order_number"]),
                left_on=["ws_order_number"],
                right_on=["wr_order_number"],
                how="semi",
            )
            .select_exprs(
                [
                    col("ws_order_number").n_unique().alias("order count"),
                    col("ws_ext_ship_cost")
                    .sum(min_count=1)
                    .alias("total shipping cost"),
                    col("ws_net_profit")
                    .sum(min_count=1)
                    .alias("total net profit"),
                ]
            )
        )
    raise unsupported("not translated")


def main() raises:
    run[query]()
