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


def q4_year_total(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    customer_key: String,
    total: Expr,
    sale_type: String,
    keys: List[String],
    names: List[String],
    years: List[Int],
) raises -> LazyFrame:
    """One channel's rows of the year_total table of q4, q11 and q74: a
    customer's sales total per year, grouped by `keys` and renamed to
    `names`, tagged with `sale_type`. With `years`, only those years."""
    var days = t["date_dim"].lazy()
    if len(years) > 0:
        days = days.filter(col("d_year").is_in(ints(years)))
    var shown = List[Expr]()
    for i in range(len(keys)):
        shown.append(col(keys[i]).alias(names[i]))
    shown.append(col("year_total"))
    shown.append(lit(sale_type).alias("sale_type"))
    return (
        t[sales]
        .lazy()
        .join(
            t["customer"].lazy(),
            left_on=[customer_key],
            right_on=["c_customer_sk"],
        )
        .join(
            days.select(["d_date_sk", "d_year"]),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_date_sk"],
        )
        .group_by(keys)
        .agg([total.sum(min_count=1).alias("year_total")])
        .select_exprs(shown)
    )


def q4_channel_year(
    year_total: LazyFrame,
    sale_type: String,
    year: Int,
    year_name: String,
    tag: String,
    first: Bool,
) raises -> LazyFrame:
    """One alias of year_total in q4, q11 and q74 (t_s_firstyear, ...):
    the rows of one channel and year, keyed by `<tag>_id` with the total
    as `<tag>_total`; a first year keeps only positive totals."""
    var keep = (col("sale_type") == sale_type) & (
        col(year_name) == lit(Int64(year))
    )
    if first:
        keep = keep & (col("year_total").cast(DataType.FLOAT64) > lit(0.0))
    return year_total.filter(keep).select_exprs(
        [
            col("customer_id").alias(tag + "_id"),
            col("year_total").cast(DataType.FLOAT64).alias(tag + "_total"),
        ]
    )


def q23_channel(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    frequent: LazyFrame,
    best: LazyFrame,
    price_type: DataType,
) raises -> LazyFrame:
    """One channel's sales in February 2000 of frequent items to the best
    store customers, summed by customer name (q23)."""
    return (
        t[sales]
        .lazy()
        .join(
            dates(
                t,
                (col("d_year") == 2000) & (col("d_moy") == 2),
                "d_sk",
            ),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_sk"],
        )
        .join(
            frequent,
            left_on=[prefix + "_item_sk"],
            right_on=["item_sk"],
        )
        .join(
            best,
            left_on=[prefix + "_bill_customer_sk"],
            right_on=["best_sk"],
        )
        .join(
            t["customer"]
            .lazy()
            .select(["c_customer_sk", "c_last_name", "c_first_name"]),
            left_on=[prefix + "_bill_customer_sk"],
            right_on=["c_customer_sk"],
        )
        .with_columns(
            [
                (
                    col(prefix + "_quantity").cast(price_type)
                    * col(prefix + "_list_price")
                ).alias("value")
            ]
        )
        .group_by(["c_last_name", "c_first_name"])
        .agg([col("value").sum(min_count=1).alias("sales")])
    )


def q27_no_text() -> Expr:
    """A null string literal: the SQL's NULL AS s_state (q27)."""
    from dataframe import null

    return null(DataType.STRING)


def q33_channel_totals(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    address: String,
    key: String,
    chosen: Expr,
    days: Expr,
) raises -> LazyFrame:
    """One channel's sales per `key` (q33, q56, q60): items whose `key` is
    among the `chosen` items' keys, sold in `days` to an address at GMT-5."""
    var keys = (
        t["item"].lazy().filter(chosen).select_exprs([col(key).alias("k_in")])
    )
    var items = (
        t["item"]
        .lazy()
        .select(["i_item_sk", key])
        .join(keys, left_on=[key], right_on=["k_in"], how="semi")
    )
    ref places = t["customer_address"]
    return (
        t[sales]
        .lazy()
        .join(
            dates(t, days, "d_sk"),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_sk"],
        )
        .join(
            places.lazy().filter(
                col("ca_gmt_offset") == like(places, "ca_gmt_offset", "-5")
            ),
            left_on=[address],
            right_on=["ca_address_sk"],
        )
        .join(items, left_on=[prefix + "_item_sk"], right_on=["i_item_sk"])
        .group_by([key])
        .agg(
            [
                col(prefix + "_ext_sales_price")
                .sum(min_count=1)
                .alias("total_sales")
            ]
        )
    )


def q33_union_totals(
    t: Dict[String, DataFrame], key: String, chosen: Expr, days: Expr
) raises -> LazyFrame:
    """The store, catalog and web totals per `key`, UNION ALL, summed."""
    return (
        q33_channel_totals(
            t, "store_sales", "ss", "ss_addr_sk", key, chosen, days
        )
        .concat(
            q33_channel_totals(
                t, "catalog_sales", "cs", "cs_bill_addr_sk", key, chosen, days
            )
        )
        .concat(
            q33_channel_totals(
                t, "web_sales", "ws", "ws_bill_addr_sk", key, chosen, days
            )
        )
        .group_by([key])
        .agg([col("total_sales").sum(min_count=1).alias("total_sales")])
    )


def q38_channel_tags(t: Dict[String, DataFrame]) raises -> LazyFrame:
    """The distinct (last name, first name, date) of each channel's buyers
    in months 1200 to 1211, tagged 1 (store), 2 (catalog) or 4 (web), then
    the sum of tags per triple (q38, q87). SQL INTERSECT and EXCEPT match
    null names as equal, as grouping does and a join does not."""
    var keys: List[String] = ["c_last_name", "c_first_name", "d_date"]
    var tables: List[String] = ["store_sales", "catalog_sales", "web_sales"]
    var date_keys: List[String] = [
        "ss_sold_date_sk",
        "cs_sold_date_sk",
        "ws_sold_date_sk",
    ]
    var buyer_keys: List[String] = [
        "ss_customer_sk",
        "cs_bill_customer_sk",
        "ws_bill_customer_sk",
    ]
    var days = (
        t["date_dim"]
        .lazy()
        .filter(
            col("d_month_seq").is_between(lit(Int64(1200)), lit(Int64(1211)))
        )
        .select(["d_date_sk", "d_date"])
    )
    var people = (
        t["customer"]
        .lazy()
        .select(["c_customer_sk", "c_last_name", "c_first_name"])
    )
    var tagged = List[LazyFrame]()
    for i in range(3):
        tagged.append(
            t[tables[i]]
            .lazy()
            .join(days, left_on=[date_keys[i]], right_on=["d_date_sk"])
            .join(people, left_on=[buyer_keys[i]], right_on=["c_customer_sk"])
            .select(keys)
            .unique()
            .with_columns([lit(Int64(1 << i)).alias("tag")])
        )
    return (
        tagged[0]
        .concat(tagged[1])
        .concat(tagged[2])
        .group_by(keys)
        .agg([col("tag").sum().alias("tags")])
    )


def q66_warehouse_months(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    price: String,
    paid: String,
) raises -> LazyFrame:
    """One channel's 2001 sales and net paid by warehouse and month, for
    orders in the time window shipped by DHL or BARIAN (q66)."""
    ref frame = t[sales]
    # Money times quantity: decimals combine only with decimals, so the
    # quantity is read as DuckDB reads a BIGINT there, as decimal(18,0).
    var counted = DataType.FLOAT64
    if frame.column(price).dtype().is_decimal():
        counted = DataType.decimal(18, 0, 64)
    var quantity = col(prefix + "_quantity").cast(counted)
    var zero = like(frame, price, "0") * lit(Int64(0)).cast(counted)
    var months: List[String] = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    var keys: List[String] = [
        "w_warehouse_name",
        "w_warehouse_sq_ft",
        "w_city",
        "w_county",
        "w_state",
        "w_country",
        "d_year",
    ]
    var sums = List[Expr]()
    for i in range(12):
        sums.append(
            when(col("d_moy") == i + 1)
            .then(col(price) * quantity)
            .otherwise(zero)
            .sum(min_count=1)
            .alias(months[i] + "_sales")
        )
    for i in range(12):
        sums.append(
            when(col("d_moy") == i + 1)
            .then(col(paid) * quantity)
            .otherwise(zero)
            .sum(min_count=1)
            .alias(months[i] + "_net")
        )
    var shown = List[Expr]()
    for i in range(6):
        shown.append(col(keys[i]))
    shown.append(lit("DHL,BARIAN").alias("ship_carriers"))
    shown.append(col("d_year").alias("year_"))
    for i in range(12):
        shown.append(col(months[i] + "_sales"))
    for i in range(12):
        shown.append(col(months[i] + "_net"))
    return (
        frame.lazy()
        .join(
            t["warehouse"].lazy(),
            left_on=[prefix + "_warehouse_sk"],
            right_on=["w_warehouse_sk"],
        )
        .join(
            t["date_dim"]
            .lazy()
            .filter(col("d_year") == 2001)
            .select(["d_date_sk", "d_year", "d_moy"]),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_date_sk"],
        )
        .join(
            t["time_dim"]
            .lazy()
            .filter(
                col("t_time").is_between(
                    lit(Int64(30838)), lit(Int64(30838 + 28800))
                )
            )
            .select(["t_time_sk"]),
            left_on=[prefix + "_sold_time_sk"],
            right_on=["t_time_sk"],
        )
        .join(
            t["ship_mode"]
            .lazy()
            .filter(col("sm_carrier").is_in(["DHL", "BARIAN"]))
            .select(["sm_ship_mode_sk"]),
            left_on=[prefix + "_ship_mode_sk"],
            right_on=["sm_ship_mode_sk"],
        )
        .group_by(keys)
        .agg(sums)
        .select_exprs(shown)
    )


def q47_neighbours(
    t: Dict[String, DataFrame],
    sales: String,
    prefix: String,
    place: String,
    sales_key: String,
    place_key: String,
    names: List[String],
) raises -> LazyFrame:
    """Monthly sales of a category, brand and place around 1999 (q47, q57),
    with the year's monthly average and the months ranked within the
    place; each 1999 month beside the one ranked before and after it, kept
    when it is more than 10% off the average. Adds `diff`, the first sort
    key."""
    var keys: List[String] = ["i_category", "i_brand"]
    for name in names:
        keys.append(name)
    var year_keys = keys.copy()
    year_keys.append("d_year")
    var group_keys = year_keys.copy()
    group_keys.append("d_moy")
    var months = (
        t["item"]
        .lazy()
        .join(
            t[sales].lazy(),
            left_on=["i_item_sk"],
            right_on=[prefix + "_item_sk"],
        )
        .join(
            t["date_dim"]
            .lazy()
            .filter(
                (col("d_year") == lit(Int64(1999)))
                | (
                    (col("d_year") == lit(Int64(1998)))
                    & (col("d_moy") == lit(Int64(12)))
                )
                | (
                    (col("d_year") == lit(Int64(2000)))
                    & (col("d_moy") == lit(Int64(1)))
                )
            )
            .select(["d_date_sk", "d_year", "d_moy"]),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_date_sk"],
        )
        .join(
            t[place].lazy(),
            left_on=[sales_key],
            right_on=[place_key],
        )
        .group_by(group_keys)
        .agg([col(prefix + "_sales_price").sum(min_count=1).alias("sum_sales")])
        .with_columns(
            [
                col("sum_sales")
                .cast(DataType.FLOAT64)
                .mean()
                .over(year_keys)
                .alias("avg_monthly_sales"),
                # ORDER BY d_year, d_moy as one key: months are 1 to 12.
                (col("d_year") * lit(Int64(100)) + col("d_moy"))
                .rank(method="min")
                .over(keys)
                .alias("rn"),
            ]
        )
    )
    var lag_keys = List[String]()
    var lead_keys = List[String]()
    var lag = List[Expr]()
    var lead = List[Expr]()
    for key in keys:
        lag_keys.append("lag_" + key)
        lead_keys.append("lead_" + key)
        lag.append(col(key).alias("lag_" + key))
        lead.append(col(key).alias("lead_" + key))
    lag_keys.append("lag_rn")
    lead_keys.append("lead_rn")
    lag.append((col("rn") + lit(Int64(1))).alias("lag_rn"))
    lead.append((col("rn") - lit(Int64(1))).alias("lead_rn"))
    lag.append(col("sum_sales").alias("psum"))
    lead.append(col("sum_sales").alias("nsum"))
    var rank_keys = keys.copy()
    rank_keys.append("rn")
    var sales_value = col("sum_sales").cast(DataType.FLOAT64)
    var shown = List[Expr]()
    for key in group_keys:
        shown.append(col(key))
    shown.append(col("avg_monthly_sales"))
    shown.append(col("sum_sales"))
    shown.append(col("psum"))
    shown.append(col("nsum"))
    shown.append((sales_value - col("avg_monthly_sales")).alias("diff"))
    return (
        months.join(
            months.select_exprs(lag), left_on=rank_keys, right_on=lag_keys
        )
        .join(months.select_exprs(lead), left_on=rank_keys, right_on=lead_keys)
        .filter(
            (col("d_year") == lit(Int64(1999)))
            & (col("avg_monthly_sales") > lit(0.0))
            & (
                (sales_value - col("avg_monthly_sales")).abs()
                / col("avg_monthly_sales")
                > lit(0.1)
            )
        )
        .select_exprs(shown)
    )


def q47_sorted(
    frame: LazyFrame, names: List[String], diff_nulls_first: Bool
) raises -> LazyFrame:
    """Sort by `diff`, then the shown columns (nulls last), and show those."""
    var by: List[String] = ["diff"]
    for name in names:
        by.append(name)
    var nulls_last = List[Bool](length=len(by), fill=True)
    nulls_last[0] = not diff_nulls_first
    return (
        frame.sort(
            by,
            descending=List[Bool](length=len(by), fill=False),
            nulls_last=nulls_last,
        )
        .head(100)
        .select(names)
    )


def q49_channel(
    t: Dict[String, DataFrame],
    channel: String,
    sales: String,
    returns: String,
    sale_keys: List[String],
    return_keys: List[String],
    s: String,
    r: String,
    returned_amount: String,
) raises -> LazyFrame:
    """One channel's items (q49) of December 2001 among the ten best by
    return ratio or by currency ratio, over sales with a large return."""
    ref sold = t[sales]
    ref back = t[returns]
    var quantity = s + "_quantity"
    var returned = r + "_return_quantity"
    var paid = s + "_net_paid"
    return (
        sold.lazy()
        .join(back.lazy(), left_on=sale_keys, right_on=return_keys, how="left")
        .filter(
            (col(returned_amount) > like(back, returned_amount, "10000"))
            & (col(s + "_net_profit") > like(sold, s + "_net_profit", "1"))
            & (col(paid) > like(sold, paid, "0"))
            & (col(quantity) > lit(Int64(0)))
        )
        .join(
            dates(
                t,
                (col("d_year") == lit(Int64(2001)))
                & (col("d_moy") == lit(Int64(12))),
                "d_sk",
            ),
            left_on=[s + "_sold_date_sk"],
            right_on=["d_sk"],
        )
        .group_by([s + "_item_sk"])
        .agg(
            [
                coalesce([col(returned), lit(Int64(0))])
                .sum(min_count=1)
                .alias("returned"),
                coalesce([col(quantity), lit(Int64(0))])
                .sum(min_count=1)
                .alias("sold"),
                coalesce(
                    [col(returned_amount), like(back, returned_amount, "0")]
                )
                .sum(min_count=1)
                .alias("returned_amount"),
                coalesce([col(paid), like(sold, paid, "0")])
                .sum(min_count=1)
                .alias("paid"),
            ]
        )
        .with_columns(
            [
                (
                    col("returned").cast(DataType.FLOAT64)
                    / col("sold").cast(DataType.FLOAT64)
                ).alias("return_ratio"),
                (
                    col("returned_amount").cast(DataType.FLOAT64)
                    / col("paid").cast(DataType.FLOAT64)
                ).alias("currency_ratio"),
            ]
        )
        .with_columns(
            [
                col("return_ratio").rank(method="min").alias("return_rank"),
                col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter(
            (col("return_rank") <= lit(Int64(10)))
            | (col("currency_rank") <= lit(Int64(10)))
        )
        .select_exprs(
            [
                lit(channel).alias("channel"),
                col(s + "_item_sk").alias("item"),
                col("return_ratio"),
                col("return_rank"),
                col("currency_rank"),
            ]
        )
    )


def q51_cumulative(
    t: Dict[String, DataFrame], sales: String, prefix: String, name: String
) raises -> LazyFrame:
    """One channel's sales (q51) per item and day of 1200-1211 with the
    item's running total up to that day. SQL's running SUM skips nulls and
    is null until a value is seen, so a day of only null prices keeps the
    total so far: running sum of the day totals (zero when none), null
    while the running count of prices is zero."""
    var item = prefix + "_item_sk"
    var price = prefix + "_sales_price"
    var daily = (
        t[sales]
        .lazy()
        .filter(col(item).is_not_null())
        .join(
            t["date_dim"]
            .lazy()
            .filter(
                col("d_month_seq").is_between(
                    lit(Int64(1200)), lit(Int64(1211))
                )
            )
            .select(["d_date_sk", "d_date"]),
            left_on=[prefix + "_sold_date_sk"],
            right_on=["d_date_sk"],
        )
        .group_by([item, "d_date"])
        .agg(
            [
                col(price).sum().alias("day_total"),
                col(price).count().alias("day_count"),
            ]
        )
    )
    return ascending(daily, [item, "d_date"]).select_exprs(
        [
            col(item).alias("item_sk"),
            col("d_date"),
            when(col("day_count").cum_sum().over(item) > lit(Int64(0)))
            .then(col("day_total").cum_sum().over(item))
            .end()
            .alias(name),
        ]
    )


def q51_running_max(value: String, name: String) -> Expr:
    """SQL's running MAX by item: skips nulls, null until a value is seen.
    A null row takes the item's smallest value, which cannot raise the
    running max."""
    return (
        when(col(value).cum_count().over("item_sk") > lit(Int64(0)))
        .then(
            col(value)
            .fill_null(col(value).min().over("item_sk"))
            .cum_max()
            .over("item_sk")
        )
        .end()
        .alias(name)
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
    if q == "q12" or q == "q20" or q == "q98":
        # Each item's revenue in a month of three categories, and its share
        # of its class's revenue: sum(sum(x)) OVER (PARTITION BY i_class).
        var sales = String("web_sales")
        var prefix = String("ws")
        if q == "q20":
            sales = "catalog_sales"
            prefix = "cs"
        elif q == "q98":
            sales = "store_sales"
            prefix = "ss"
        var categories: List[String] = ["Sports", "Books", "Home"]
        var revenue = col("itemrevenue").cast(DataType.FLOAT64)
        var grouped = (
            t[sales]
            .lazy()
            .join(
                t["item"].lazy().filter(col("i_category").is_in(categories)),
                left_on=[prefix + "_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                dates(
                    t,
                    col("d_date").is_between(
                        date_lit("1999-02-22"), date_lit("1999-03-24")
                    ),
                    "d_sk",
                ),
                left_on=[prefix + "_sold_date_sk"],
                right_on=["d_sk"],
            )
            .group_by(
                [
                    "i_item_id",
                    "i_item_desc",
                    "i_category",
                    "i_class",
                    "i_current_price",
                ]
            )
            .agg(
                [
                    col(prefix + "_ext_sales_price")
                    .sum(min_count=1)
                    .alias("itemrevenue")
                ]
            )
            .with_columns(
                (
                    revenue
                    * lit(100.0)
                    / revenue.sum(min_count=1).over("i_class")
                ).alias("revenueratio")
            )
        )
        var sorted = ascending(
            grouped,
            [
                "i_category",
                "i_class",
                "i_item_id",
                "i_item_desc",
                "revenueratio",
            ],
            nulls_first=q != "q12",
        )
        if q == "q98":
            return sorted^
        return sorted.head(100)
    if q == "q53" or q == "q63" or q == "q89":
        # Store sales of chosen items summed per group, kept where the sum
        # is more than 10% away from avg(sum(x)) over a coarser partition.
        def q53_names(values: List[String]) -> List[String]:
            return values.copy()

        var items: Expr
        var days: Expr
        var keys: List[String]
        var partition: List[String]
        var average = String("avg_monthly_sales")
        if q == "q89":
            days = col("d_year") == 1999
            items = (
                col("i_category").is_in(
                    q53_names(["Books", "Electronics", "Sports"])
                )
                & col("i_class").is_in(
                    q53_names(["computers", "stereo", "football"])
                )
            ) | (
                col("i_category").is_in(q53_names(["Men", "Jewelry", "Women"]))
                & col("i_class").is_in(
                    q53_names(["shirts", "birdal", "dresses"])
                )
            )
            keys = [
                "i_category",
                "i_class",
                "i_brand",
                "s_store_name",
                "s_company_name",
                "d_moy",
            ]
            partition = [
                "i_category",
                "i_brand",
                "s_store_name",
                "s_company_name",
            ]
        else:
            days = col("d_month_seq").is_between(
                lit(Int64(1200)), lit(Int64(1211))
            )
            items = (
                col("i_category").is_in(
                    q53_names(["Books", "Children", "Electronics"])
                )
                & col("i_class").is_in(
                    q53_names(
                        ["personal", "portable", "reference", "self-help"]
                    )
                )
                & col("i_brand").is_in(
                    q53_names(
                        [
                            "scholaramalgamalg #14",
                            "scholaramalgamalg #7",
                            "exportiunivamalg #9",
                            "scholaramalgamalg #9",
                        ]
                    )
                )
            ) | (
                col("i_category").is_in(q53_names(["Women", "Music", "Men"]))
                & col("i_class").is_in(
                    q53_names(
                        ["accessories", "classical", "fragrances", "pants"]
                    )
                )
                & col("i_brand").is_in(
                    q53_names(
                        [
                            "amalgimporto #1",
                            "edu packscholar #1",
                            "exportiimporto #1",
                            "importoamalg #1",
                        ]
                    )
                )
            )
            if q == "q53":
                keys = ["i_manufact_id", "d_qoy"]
                partition = ["i_manufact_id"]
                average = "avg_quarterly_sales"
            else:
                keys = ["i_manager_id", "d_moy"]
                partition = ["i_manager_id"]
        var store = t["store"].lazy()
        if q != "q89":
            store = store.select(["s_store_sk"])
        var total = col("sum_sales").cast(DataType.FLOAT64)
        var mean = col(average)
        var grouped = (
            t["store_sales"]
            .lazy()
            .join(
                t["item"].lazy().filter(items),
                left_on=["ss_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                t["date_dim"]
                .lazy()
                .filter(days)
                .select_exprs(
                    [
                        col("d_date_sk").alias("d_sk"),
                        col(keys[len(keys) - 1]),
                    ]
                ),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(store, left_on=["ss_store_sk"], right_on=["s_store_sk"])
            .group_by(keys)
            .agg([col("ss_sales_price").sum(min_count=1).alias("sum_sales")])
            .with_columns(total.mean().over(partition).alias(average))
        )
        if q == "q89":
            var kept = grouped.filter(
                (mean != lit(0.0)) & ((total - mean).abs() / mean > lit(0.1))
            ).with_columns((total - mean).alias("q89_gap"))
            return (
                ascending(
                    kept,
                    [
                        "q89_gap",
                        "s_store_name",
                        "i_category",
                        "i_class",
                        "i_brand",
                        "s_company_name",
                        "d_moy",
                        "sum_sales",
                        average,
                    ],
                )
                .head(100)
                .drop(["q89_gap"])
            )
        var kept = grouped.filter(
            (mean > lit(0.0)) & ((total - mean).abs() / mean > lit(0.1))
        ).select([partition[0], "sum_sales", average])
        var order: List[String] = [average, "sum_sales", partition[0]]
        if q == "q63":
            order = [partition[0], average, "sum_sales"]
        return ascending(kept, order).head(100)
    if q == "q31":
        # Sales by county and quarter for each channel; each WITH table is
        # one plan, filtered to a quarter of 2000 in each of its three uses.
        def q31_channel(
            t: Dict[String, DataFrame],
            sales: String,
            date_key: String,
            address_key: String,
            amount: String,
        ) raises -> LazyFrame:
            return (
                t[sales]
                .lazy()
                .join(
                    t["date_dim"]
                    .lazy()
                    .select(["d_date_sk", "d_qoy", "d_year"]),
                    left_on=[date_key],
                    right_on=["d_date_sk"],
                )
                .join(
                    t["customer_address"]
                    .lazy()
                    .select(["ca_address_sk", "ca_county"]),
                    left_on=[address_key],
                    right_on=["ca_address_sk"],
                )
                .group_by(["ca_county", "d_qoy", "d_year"])
                .agg(
                    [
                        col(amount)
                        .sum(min_count=1)
                        .cast(DataType.FLOAT64)
                        .alias("total")
                    ]
                )
            )

        var ss = q31_channel(
            t,
            "store_sales",
            "ss_sold_date_sk",
            "ss_addr_sk",
            "ss_ext_sales_price",
        )
        var ws = q31_channel(
            t,
            "web_sales",
            "ws_sold_date_sk",
            "ws_bill_addr_sk",
            "ws_ext_sales_price",
        )

        def q31_quarter(
            frame: LazyFrame, quarter: Int, name: String
        ) raises -> LazyFrame:
            return frame.filter(
                (col("d_qoy") == quarter) & (col("d_year") == 2000)
            ).select_exprs(
                [
                    col("ca_county").alias(name + "_county"),
                    col("d_year").alias(name + "_year"),
                    col("total").alias(name),
                ]
            )

        var paired = (
            q31_quarter(ss, 1, "ss1")
            .join(
                q31_quarter(ss, 2, "ss2").select(["ss2_county", "ss2"]),
                left_on=["ss1_county"],
                right_on=["ss2_county"],
            )
            .join(
                q31_quarter(ss, 3, "ss3").select(["ss3_county", "ss3"]),
                left_on=["ss1_county"],
                right_on=["ss3_county"],
            )
            .join(
                q31_quarter(ws, 1, "ws1").select(["ws1_county", "ws1"]),
                left_on=["ss1_county"],
                right_on=["ws1_county"],
            )
            .join(
                q31_quarter(ws, 2, "ws2").select(["ws2_county", "ws2"]),
                left_on=["ss1_county"],
                right_on=["ws2_county"],
            )
            .join(
                q31_quarter(ws, 3, "ws3").select(["ws3_county", "ws3"]),
                left_on=["ss1_county"],
                right_on=["ws3_county"],
            )
            # The SQL's CASE gives null for a total that is not positive,
            # and a comparison with null keeps nothing.
            .filter(
                (col("ws1") > 0.0)
                & (col("ss1") > 0.0)
                & (col("ws2") / col("ws1") > col("ss2") / col("ss1"))
                & (col("ws2") > 0.0)
                & (col("ss2") > 0.0)
                & (col("ws3") / col("ws2") > col("ss3") / col("ss2"))
            )
            .select_exprs(
                [
                    col("ss1_county").alias("ca_county"),
                    col("ss1_year").alias("d_year"),
                    (col("ws2") / col("ws1")).alias("web_q1_q2_increase"),
                    (col("ss2") / col("ss1")).alias("store_q1_q2_increase"),
                    (col("ws3") / col("ws2")).alias("web_q2_q3_increase"),
                    (col("ss3") / col("ss2")).alias("store_q2_q3_increase"),
                ]
            )
        )
        return ascending(paired, ["ca_county"])
    if q == "q58" or q == "q83":
        # Revenue (q58) or returned quantity (q83) by item id in each of
        # three channels on the days of chosen weeks, for items found in
        # all three. The days are one plan used by each channel.
        var weeks: LazyFrame
        if q == "q58":
            weeks = (
                t["date_dim"]
                .lazy()
                .filter(col("d_date") == date_lit("2000-01-03"))
                .select_exprs([col("d_week_seq").alias("week")])
            )
        else:
            weeks = (
                t["date_dim"]
                .lazy()
                .filter(
                    (col("d_date") == date_lit("2000-06-30"))
                    | (col("d_date") == date_lit("2000-09-27"))
                    | (col("d_date") == date_lit("2000-11-17"))
                )
                .select_exprs([col("d_week_seq").alias("week")])
            )
        var week_days = (
            t["date_dim"]
            .lazy()
            .join(weeks, left_on=["d_week_seq"], right_on=["week"], how="semi")
            .select_exprs([col("d_date").alias("day")])
        )
        var days = (
            t["date_dim"]
            .lazy()
            .join(week_days, left_on=["d_date"], right_on=["day"], how="semi")
            .select_exprs([col("d_date_sk").alias("d_sk")])
        )

        def q58_channel(
            t: Dict[String, DataFrame],
            days: LazyFrame,
            sales: String,
            item_key: String,
            date_key: String,
            amount: String,
            name: String,
        ) raises -> LazyFrame:
            return (
                t[sales]
                .lazy()
                .join(
                    t["item"].lazy().select(["i_item_sk", "i_item_id"]),
                    left_on=[item_key],
                    right_on=["i_item_sk"],
                )
                .join(days, left_on=[date_key], right_on=["d_sk"])
                .group_by(["i_item_id"])
                .agg([col(amount).sum(min_count=1).alias(name)])
                .select_exprs([col("i_item_id").alias(name + "_id"), col(name)])
            )

        var first: LazyFrame
        var second: LazyFrame
        var third: LazyFrame
        var names: List[String]
        if q == "q58":
            names = ["ss_item_rev", "cs_item_rev", "ws_item_rev"]
            first = q58_channel(
                t,
                days,
                "store_sales",
                "ss_item_sk",
                "ss_sold_date_sk",
                "ss_ext_sales_price",
                names[0],
            )
            second = q58_channel(
                t,
                days,
                "catalog_sales",
                "cs_item_sk",
                "cs_sold_date_sk",
                "cs_ext_sales_price",
                names[1],
            )
            third = q58_channel(
                t,
                days,
                "web_sales",
                "ws_item_sk",
                "ws_sold_date_sk",
                "ws_ext_sales_price",
                names[2],
            )
        else:
            names = ["sr_item_qty", "cr_item_qty", "wr_item_qty"]
            first = q58_channel(
                t,
                days,
                "store_returns",
                "sr_item_sk",
                "sr_returned_date_sk",
                "sr_return_quantity",
                names[0],
            )
            second = q58_channel(
                t,
                days,
                "catalog_returns",
                "cr_item_sk",
                "cr_returned_date_sk",
                "cr_return_quantity",
                names[1],
            )
            third = q58_channel(
                t,
                days,
                "web_returns",
                "wr_item_sk",
                "wr_returned_date_sk",
                "wr_return_quantity",
                names[2],
            )
        var joined = first.join(
            second, left_on=[names[0] + "_id"], right_on=[names[1] + "_id"]
        ).join(third, left_on=[names[0] + "_id"], right_on=[names[2] + "_id"])
        var a = col(names[0]).cast(DataType.FLOAT64)
        var b = col(names[1]).cast(DataType.FLOAT64)
        var c = col(names[2]).cast(DataType.FLOAT64)
        var shown = List[Expr]()
        shown.append(col(names[0] + "_id").alias("item_id"))
        var values: List[Expr] = [a.copy(), b.copy(), c.copy()]
        var labels: List[String] = ["ss_dev", "cs_dev", "ws_dev"]
        if q == "q83":
            labels = ["sr_dev", "cr_dev", "wr_dev"]
        if q == "q58":
            # The exact sum is cast once; the SQL divides decimals as
            # doubles.
            var average = (col(names[0]) + col(names[1]) + col(names[2])).cast(
                DataType.FLOAT64
            ) / lit(3.0)
            joined = joined.filter(
                a.is_between(lit(0.9) * b, lit(1.1) * b)
                & a.is_between(lit(0.9) * c, lit(1.1) * c)
                & b.is_between(lit(0.9) * a, lit(1.1) * a)
                & b.is_between(lit(0.9) * c, lit(1.1) * c)
                & c.is_between(lit(0.9) * a, lit(1.1) * a)
                & c.is_between(lit(0.9) * b, lit(1.1) * b)
            )
            for i in range(3):
                shown.append(col(names[i]))
                shown.append(
                    (values[i] / average * lit(100.0)).alias(labels[i])
                )
            shown.append(average.alias("average"))
        else:
            var total = (col(names[0]) + col(names[1]) + col(names[2])).cast(
                DataType.FLOAT64
            )
            for i in range(3):
                shown.append(col(names[i]))
                shown.append(
                    (values[i] / total / lit(3.0) * lit(100.0)).alias(labels[i])
                )
            shown.append((total / lit(3.0)).alias("average"))
        return ascending(
            joined.select_exprs(shown), ["item_id", names[0]], nulls_first=True
        ).head(100)
    if q == "q64":
        # Items whose catalog list price total is more than twice what was
        # refunded on them (cs_ui), then returned store sales of those items
        # by store, addresses and years (cross_sales), paired 1999 to 2000.
        ref catalog = t["catalog_sales"]
        var refunded = (
            col("cr_refunded_cash")
            + col("cr_reversed_charge")
            + col("cr_store_credit")
        )
        var cs_ui = (
            catalog.lazy()
            .join(
                t["catalog_returns"].lazy(),
                left_on=["cs_item_sk", "cs_order_number"],
                right_on=["cr_item_sk", "cr_order_number"],
            )
            .group_by(["cs_item_sk"])
            .agg(
                [
                    col("cs_ext_list_price").sum(min_count=1).alias("sale"),
                    refunded.sum(min_count=1).alias("refund"),
                ]
            )
            .filter(
                col("sale")
                > like(catalog, "cs_ext_list_price", "2") * col("refund")
            )
            .select(["cs_item_sk"])
        )

        def q64_years(name: String) raises -> List[Expr]:
            return [
                col("d_date_sk").alias(name + "_sk"),
                col("d_year").alias(name),
            ]

        def q64_address(prefix: String) raises -> List[Expr]:
            return [
                col("ca_address_sk").alias(prefix + "_address_sk"),
                col("ca_street_number").alias(prefix + "_street_number"),
                col("ca_street_name").alias(prefix + "_street_name"),
                col("ca_city").alias(prefix + "_city"),
                col("ca_zip").alias(prefix + "_zip"),
            ]

        var band = (
            t["household_demographics"]
            .lazy()
            .select(["hd_demo_sk", "hd_income_band_sk"])
            .join(
                t["income_band"].lazy().select(["ib_income_band_sk"]),
                left_on=["hd_income_band_sk"],
                right_on=["ib_income_band_sk"],
            )
        )
        var keys: List[String] = [
            "product_name",
            "ss_item_sk",
            "store_name",
            "store_zip",
            "b_street_number",
            "b_street_name",
            "b_city",
            "b_zip",
            "c_street_number",
            "c_street_name",
            "c_city",
            "c_zip",
            "syear",
            "fsyear",
            "s2year",
        ]
        var cross_sales = (
            t["store_sales"]
            .lazy()
            .join(
                t["store_returns"]
                .lazy()
                .select(["sr_item_sk", "sr_ticket_number"]),
                left_on=["ss_item_sk", "ss_ticket_number"],
                right_on=["sr_item_sk", "sr_ticket_number"],
            )
            .join(cs_ui, left_on=["ss_item_sk"], right_on=["cs_item_sk"])
            .join(
                t["item"]
                .lazy()
                .filter(
                    col("i_color").is_in(
                        [
                            "purple",
                            "burlywood",
                            "indian",
                            "spring",
                            "floral",
                            "medium",
                        ]
                    )
                    & between(t["item"], "i_current_price", "64", "74")
                    & between(t["item"], "i_current_price", "65", "79")
                )
                .select_exprs(
                    [
                        col("i_item_sk").alias("item_sk"),
                        col("i_product_name").alias("product_name"),
                    ]
                ),
                left_on=["ss_item_sk"],
                right_on=["item_sk"],
            )
            .join(
                t["date_dim"].lazy().select_exprs(q64_years("syear")),
                left_on=["ss_sold_date_sk"],
                right_on=["syear_sk"],
            )
            .join(
                t["store"]
                .lazy()
                .select_exprs(
                    [
                        col("s_store_sk"),
                        col("s_store_name").alias("store_name"),
                        col("s_zip").alias("store_zip"),
                    ]
                ),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["customer"]
                .lazy()
                .select(
                    [
                        "c_customer_sk",
                        "c_current_cdemo_sk",
                        "c_current_hdemo_sk",
                        "c_current_addr_sk",
                        "c_first_sales_date_sk",
                        "c_first_shipto_date_sk",
                    ]
                ),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                t["customer_demographics"]
                .lazy()
                .select_exprs(
                    [
                        col("cd_demo_sk").alias("cd1_sk"),
                        col("cd_marital_status").alias("cd1_marital"),
                    ]
                ),
                left_on=["ss_cdemo_sk"],
                right_on=["cd1_sk"],
            )
            .join(
                t["customer_demographics"]
                .lazy()
                .select_exprs(
                    [
                        col("cd_demo_sk").alias("cd2_sk"),
                        col("cd_marital_status").alias("cd2_marital"),
                    ]
                ),
                left_on=["c_current_cdemo_sk"],
                right_on=["cd2_sk"],
            )
            .filter(col("cd1_marital") != col("cd2_marital"))
            .join(
                t["promotion"].lazy().select(["p_promo_sk"]),
                left_on=["ss_promo_sk"],
                right_on=["p_promo_sk"],
            )
            .join(
                band.select_exprs([col("hd_demo_sk").alias("hd1_sk")]),
                left_on=["ss_hdemo_sk"],
                right_on=["hd1_sk"],
            )
            .join(
                band.select_exprs([col("hd_demo_sk").alias("hd2_sk")]),
                left_on=["c_current_hdemo_sk"],
                right_on=["hd2_sk"],
            )
            .join(
                t["customer_address"].lazy().select_exprs(q64_address("b")),
                left_on=["ss_addr_sk"],
                right_on=["b_address_sk"],
            )
            .join(
                t["customer_address"].lazy().select_exprs(q64_address("c")),
                left_on=["c_current_addr_sk"],
                right_on=["c_address_sk"],
            )
            .join(
                t["date_dim"].lazy().select_exprs(q64_years("fsyear")),
                left_on=["c_first_sales_date_sk"],
                right_on=["fsyear_sk"],
            )
            .join(
                t["date_dim"].lazy().select_exprs(q64_years("s2year")),
                left_on=["c_first_shipto_date_sk"],
                right_on=["s2year_sk"],
            )
            .group_by(keys)
            .agg(
                [
                    col("ss_item_sk").len().alias("cnt"),
                    col("ss_wholesale_cost").sum(min_count=1).alias("s1"),
                    col("ss_list_price").sum(min_count=1).alias("s2"),
                    col("ss_coupon_amt").sum(min_count=1).alias("s3"),
                ]
            )
        )
        var cs2 = cross_sales.filter(col("syear") == 2000).select_exprs(
            [
                col("ss_item_sk").alias("item_sk2"),
                col("store_name").alias("store_name2"),
                col("store_zip").alias("store_zip2"),
                col("s1").alias("s12"),
                col("s2").alias("s22"),
                col("s3").alias("s32"),
                col("syear").alias("syear2"),
                col("cnt").alias("cnt2"),
            ]
        )
        var paired = (
            cross_sales.filter(col("syear") == 1999)
            .join(
                cs2,
                left_on=["ss_item_sk", "store_name", "store_zip"],
                right_on=["item_sk2", "store_name2", "store_zip2"],
            )
            .filter(col("cnt2") <= col("cnt"))
            .select_exprs(
                [
                    col("product_name"),
                    col("store_name"),
                    col("store_zip"),
                    col("b_street_number"),
                    col("b_street_name"),
                    col("b_city"),
                    col("b_zip"),
                    col("c_street_number"),
                    col("c_street_name"),
                    col("c_city"),
                    col("c_zip"),
                    col("syear").alias("cs1syear"),
                    col("cnt").alias("cs1cnt"),
                    col("s1").alias("s11"),
                    col("s2").alias("s21"),
                    col("s3").alias("s31"),
                    col("s12"),
                    col("s22"),
                    col("s32"),
                    col("syear2").alias("syear"),
                    col("cnt2").alias("cnt"),
                ]
            )
        )
        return ascending(
            paired, ["product_name", "store_name", "cnt", "s11", "s12"]
        )
    if q == "q78":
        # Store sales of 2000 that were not returned, by year, item and
        # customer, beside the same customer's unreturned web and catalog
        # purchases of that item that year. LEFT JOIN ... IS NULL on the
        # returns is an anti join.
        def q78_channel(
            t: Dict[String, DataFrame],
            sales: String,
            returns: String,
            sale_keys: List[String],
            return_keys: List[String],
            date_key: String,
            item_key: String,
            customer_key: String,
            prefix: String,
            side: String,
        ) raises -> LazyFrame:
            return (
                t[sales]
                .lazy()
                .join(
                    t[returns].lazy().select(return_keys),
                    left_on=sale_keys,
                    right_on=return_keys,
                    how="anti",
                )
                .join(
                    t["date_dim"].lazy().select(["d_date_sk", "d_year"]),
                    left_on=[date_key],
                    right_on=["d_date_sk"],
                )
                .group_by(["d_year", item_key, customer_key])
                .agg(
                    [
                        col(prefix + "_quantity")
                        .sum(min_count=1)
                        .alias(side + "_qty"),
                        col(prefix + "_wholesale_cost")
                        .sum(min_count=1)
                        .alias(side + "_wc"),
                        col(prefix + "_sales_price")
                        .sum(min_count=1)
                        .alias(side + "_sp"),
                    ]
                )
                .select_exprs(
                    [
                        col("d_year").alias(side + "_sold_year"),
                        col(item_key).alias(side + "_item_sk"),
                        col(customer_key).alias(side + "_customer_sk"),
                        col(side + "_qty"),
                        col(side + "_wc"),
                        col(side + "_sp"),
                    ]
                )
            )

        var ws = q78_channel(
            t,
            "web_sales",
            "web_returns",
            ["ws_order_number", "ws_item_sk"],
            ["wr_order_number", "wr_item_sk"],
            "ws_sold_date_sk",
            "ws_item_sk",
            "ws_bill_customer_sk",
            "ws",
            "ws",
        )
        var cs = q78_channel(
            t,
            "catalog_sales",
            "catalog_returns",
            ["cs_order_number", "cs_item_sk"],
            ["cr_order_number", "cr_item_sk"],
            "cs_sold_date_sk",
            "cs_item_sk",
            "cs_bill_customer_sk",
            "cs",
            "cs",
        )
        var ss = q78_channel(
            t,
            "store_sales",
            "store_returns",
            ["ss_ticket_number", "ss_item_sk"],
            ["sr_ticket_number", "sr_item_sk"],
            "ss_sold_date_sk",
            "ss_item_sk",
            "ss_customer_sk",
            "ss",
            "ss",
        )
        ref web = t["web_sales"]
        var zero_wc = like(web, "ws_wholesale_cost", "0")
        var zero_sp = like(web, "ws_sales_price", "0")
        var other_qty = coalesce([col("ws_qty"), lit(Int64(0))]) + coalesce(
            [col("cs_qty"), lit(Int64(0))]
        )
        var combined = (
            ss.join(
                ws,
                left_on=["ss_sold_year", "ss_item_sk", "ss_customer_sk"],
                right_on=["ws_sold_year", "ws_item_sk", "ws_customer_sk"],
                how="left",
            )
            .join(
                cs,
                left_on=["ss_sold_year", "ss_item_sk", "ss_customer_sk"],
                right_on=["cs_sold_year", "cs_item_sk", "cs_customer_sk"],
                how="left",
            )
            .filter(
                (
                    (coalesce([col("ws_qty"), lit(Int64(0))]) > 0)
                    | (coalesce([col("cs_qty"), lit(Int64(0))]) > 0)
                )
                & (col("ss_sold_year") == 2000)
            )
            .select_exprs(
                [
                    col("ss_sold_year"),
                    col("ss_item_sk"),
                    col("ss_customer_sk"),
                    (
                        col("ss_qty").cast(DataType.FLOAT64)
                        / other_qty.cast(DataType.FLOAT64)
                    )
                    .round(2)
                    .alias("ratio"),
                    col("ss_qty").alias("store_qty"),
                    col("ss_wc").alias("store_wholesale_cost"),
                    col("ss_sp").alias("store_sales_price"),
                    other_qty.alias("other_chan_qty"),
                    (
                        coalesce([col("ws_wc"), zero_wc.copy()])
                        + coalesce([col("cs_wc"), zero_wc.copy()])
                    ).alias("other_chan_wholesale_cost"),
                    (
                        coalesce([col("ws_sp"), zero_sp.copy()])
                        + coalesce([col("cs_sp"), zero_sp.copy()])
                    ).alias("other_chan_sales_price"),
                ]
            )
        )
        return ordered(
            combined,
            [
                "ss_sold_year",
                "ss_item_sk",
                "ss_customer_sk",
                "store_qty",
                "store_wholesale_cost",
                "store_sales_price",
                "other_chan_qty",
                "other_chan_wholesale_cost",
                "other_chan_sales_price",
                "ratio",
            ],
            [False, False, False, True, True, True, False, False, False, False],
        ).head(100)
    if q == "q97":
        # Distinct (customer, item) pairs bought in the store and from the
        # catalog in one year, counted by which side of a full join holds
        # them. The right keys stay separate (coalesce=False) so each
        # side's IS NOT NULL reads its own key, as in the SQL.
        var year = dates(
            t,
            col("d_month_seq").is_between(lit(Int64(1200)), lit(Int64(1211))),
            "d_sk",
        )
        var ssci = (
            t["store_sales"]
            .lazy()
            .join(year, left_on=["ss_sold_date_sk"], right_on=["d_sk"])
            .select_exprs(
                [
                    col("ss_customer_sk").alias("customer_sk"),
                    col("ss_item_sk").alias("item_sk"),
                ]
            )
            .unique()
        )
        var csci = (
            t["catalog_sales"]
            .lazy()
            .join(year, left_on=["cs_sold_date_sk"], right_on=["d_sk"])
            .select_exprs(
                [
                    col("cs_bill_customer_sk").alias("c_customer_sk"),
                    col("cs_item_sk").alias("c_item_sk"),
                ]
            )
            .unique()
        )
        var store = col("customer_sk").is_not_null()
        var catalog = col("c_customer_sk").is_not_null()
        return (
            ssci.join(
                csci,
                left_on=["customer_sk", "item_sk"],
                right_on=["c_customer_sk", "c_item_sk"],
                how="full",
                coalesce=False,
            )
            .select_exprs(
                [
                    one_if(store & ~catalog)
                    .sum(min_count=1)
                    .alias("store_only"),
                    one_if(~store & catalog)
                    .sum(min_count=1)
                    .alias("catalog_only"),
                    one_if(store & catalog)
                    .sum(min_count=1)
                    .alias("store_and_catalog"),
                ]
            )
            .head(100)
        )
    if q == "q2":
        # Web and catalog sales (UNION ALL) summed by week and weekday,
        # each week of 2001 beside the week 53 later. As the SQL has it,
        # each week's row joins every one of its days in the year.
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
        var wscs = (
            t["web_sales"]
            .lazy()
            .select_exprs(
                [
                    col("ws_sold_date_sk").alias("sold_date_sk"),
                    col("ws_ext_sales_price").alias("sales_price"),
                ]
            )
            .concat(
                t["catalog_sales"]
                .lazy()
                .select_exprs(
                    [
                        col("cs_sold_date_sk").alias("sold_date_sk"),
                        col("cs_ext_sales_price").alias("sales_price"),
                    ]
                )
            )
        )
        var sums = List[Expr]()
        for i in range(7):
            sums.append(
                when(col("d_day_name") == days[i])
                .then(col("sales_price"))
                .end()
                .sum(min_count=1)
                .alias(short[i] + "_sales")
            )
        var wswscs = (
            wscs.join(
                t["date_dim"]
                .lazy()
                .select(["d_date_sk", "d_week_seq", "d_day_name"]),
                left_on=["sold_date_sk"],
                right_on=["d_date_sk"],
            )
            .group_by(["d_week_seq"])
            .agg(sums)
        )
        var y_cols: List[Expr] = [col("d_week_seq").alias("d_week_seq1")]
        var z_cols: List[Expr] = [
            (col("d_week_seq") - lit(Int64(53))).alias("week_before")
        ]
        for i in range(7):
            y_cols.append(col(short[i] + "_sales").alias(short[i] + "_sales1"))
            z_cols.append(col(short[i] + "_sales").alias(short[i] + "_sales2"))
        var y = wswscs.join(
            t["date_dim"]
            .lazy()
            .filter(col("d_year") == 2001)
            .select_exprs([col("d_week_seq").alias("week")]),
            left_on=["d_week_seq"],
            right_on=["week"],
        ).select_exprs(y_cols)
        var z = wswscs.join(
            t["date_dim"]
            .lazy()
            .filter(col("d_year") == 2002)
            .select_exprs([col("d_week_seq").alias("week")]),
            left_on=["d_week_seq"],
            right_on=["week"],
        ).select_exprs(z_cols)
        var shown: List[Expr] = [col("d_week_seq1")]
        for i in range(7):
            # The SQL leaves the seventh ratio unnamed.
            var name = "r" + String(i + 1)
            shown.append(
                (
                    col(short[i] + "_sales1").cast(DataType.FLOAT64)
                    / col(short[i] + "_sales2").cast(DataType.FLOAT64)
                )
                .round(2)
                .alias(name)
            )
        var paired = y.join(
            z, left_on=["d_week_seq1"], right_on=["week_before"]
        ).select_exprs(shown)
        return ascending(paired, ["d_week_seq1"], nulls_first=True)
    if q == "q4" or q == "q11" or q == "q74":
        # year_total is a UNION ALL of per-channel yearly totals, read once
        # per alias of the SQL's self join.
        var keys: List[String] = [
            "c_customer_id",
            "c_first_name",
            "c_last_name",
            "c_preferred_cust_flag",
            "c_birth_country",
            "c_login",
            "c_email_address",
            "d_year",
        ]
        var names: List[String] = [
            "customer_id",
            "customer_first_name",
            "customer_last_name",
            "customer_preferred_cust_flag",
            "customer_birth_country",
            "customer_login",
            "customer_email_address",
            "dyear",
        ]
        var years = List[Int]()
        var year_name = "dyear"
        if q == "q74":
            keys = ["c_customer_id", "c_first_name", "c_last_name", "d_year"]
            names = [
                "customer_id",
                "customer_first_name",
                "customer_last_name",
                "year_",
            ]
            years = [2001, 2002]
            year_name = "year_"
        var store_total: Expr
        var web_total: Expr
        if q == "q4":
            # DuckDB divides decimals as doubles.
            store_total = (
                (
                    col("ss_ext_list_price")
                    - col("ss_ext_wholesale_cost")
                    - col("ss_ext_discount_amt")
                    + col("ss_ext_sales_price")
                ).cast(DataType.FLOAT64)
            ) / lit(2.0)
            web_total = (
                (
                    col("ws_ext_list_price")
                    - col("ws_ext_wholesale_cost")
                    - col("ws_ext_discount_amt")
                    + col("ws_ext_sales_price")
                ).cast(DataType.FLOAT64)
            ) / lit(2.0)
        elif q == "q11":
            store_total = col("ss_ext_list_price") - col("ss_ext_discount_amt")
            web_total = col("ws_ext_list_price") - col("ws_ext_discount_amt")
        else:
            store_total = col("ss_net_paid")
            web_total = col("ws_net_paid")
        var year_total = q4_year_total(
            t,
            "store_sales",
            "ss",
            "ss_customer_sk",
            store_total,
            "s",
            keys,
            names,
            years,
        )
        if q == "q4":
            year_total = year_total.concat(
                q4_year_total(
                    t,
                    "catalog_sales",
                    "cs",
                    "cs_bill_customer_sk",
                    (
                        (
                            col("cs_ext_list_price")
                            - col("cs_ext_wholesale_cost")
                            - col("cs_ext_discount_amt")
                            + col("cs_ext_sales_price")
                        ).cast(DataType.FLOAT64)
                    )
                    / lit(2.0),
                    "c",
                    keys,
                    names,
                    years,
                )
            )
        year_total = year_total.concat(
            q4_year_total(
                t,
                "web_sales",
                "ws",
                "ws_bill_customer_sk",
                web_total,
                "w",
                keys,
                names,
                years,
            )
        )
        var shown: List[String] = [
            "customer_id",
            "customer_first_name",
            "customer_last_name",
        ]
        if q != "q74":
            shown.append("customer_preferred_cust_flag")
        var second_cols: List[Expr] = [
            col("customer_id").alias("s2_id"),
            col("year_total").cast(DataType.FLOAT64).alias("s2_total"),
        ]
        for name in shown:
            second_cols.append(col(name))
        var found = (
            q4_channel_year(year_total, "s", 2001, year_name, "s1", True)
            .join(
                year_total.filter(
                    (col("sale_type") == "s")
                    & (col(year_name) == lit(Int64(2002)))
                ).select_exprs(second_cols),
                left_on=["s1_id"],
                right_on=["s2_id"],
            )
            .join(
                q4_channel_year(year_total, "w", 2001, year_name, "w1", True),
                left_on=["s1_id"],
                right_on=["w1_id"],
            )
            .join(
                q4_channel_year(year_total, "w", 2002, year_name, "w2", False),
                left_on=["s1_id"],
                right_on=["w2_id"],
            )
        )
        var web_growth = col("w2_total") / col("w1_total")
        var store_growth = col("s2_total") / col("s1_total")
        if q == "q4":
            found = (
                found.join(
                    q4_channel_year(
                        year_total, "c", 2001, year_name, "c1", True
                    ),
                    left_on=["s1_id"],
                    right_on=["c1_id"],
                )
                .join(
                    q4_channel_year(
                        year_total, "c", 2002, year_name, "c2", False
                    ),
                    left_on=["s1_id"],
                    right_on=["c2_id"],
                )
                .filter(
                    (col("c2_total") / col("c1_total") > store_growth)
                    & (col("c2_total") / col("c1_total") > web_growth)
                )
            )
        else:
            found = found.filter(web_growth > store_growth)
        var by = shown.copy()
        if q == "q74":
            by = ["customer_id"]
        return ascending(found.select(shown), by, nulls_first=True).head(100)
    if q == "q8":
        var listed = List[Expr]()
        for code in """
            24128 76232 65084 87816 83926 77556 20548 26231 43848 15126 91137
            61265 98294 25782 17920 18426 98235 40081 84093 28577 55565 17183
            54601 67897 22752 86284 18376 38607 45200 21756 29741 96765 23932
            89360 29839 25989 28898 91068 72550 10390 18845 47770 82636 41367
            76638 86198 81312 37126 39192 88424 72175 81426 53672 10445 42666
            66864 66708 41248 48583 82276 18842 78890 49448 14089 38122 34425
            79077 19849 43285 39861 66162 77610 13695 99543 83444 83041 12305
            57665 68341 25003 57834 62878 49130 81096 18840 27700 23470 50412
            21195 16021 76107 71954 68309 18119 98359 64544 10336 86379 27068
            39736 98569 28915 24206 56529 57647 54917 42961 91110 63981 14922
            36420 23006 67467 32754 30903 20260 31671 51798 72325 85816 68621
            13955 36446 41766 68806 16725 15146 22744 35850 88086 51649 18270
            52867 39972 96976 63792 11376 94898 13595 10516 90225 58943 39371
            94945 28587 96576 57855 28488 26105 83933 25858 34322 44438 73171
            30122 34102 22685 71256 78451 54364 13354 45375 40558 56458 28286
            45266 47305 69399 83921 26233 11101 15371 69913 35942 15882 25631
            24610 44165 99076 33786 70738 26653 14328 72305 62496 22152 10144
            64147 48425 14663 21076 18799 30450 63089 81019 68893 24996 51200
            51211 45692 92712 70466 79994 22437 25280 38935 71791 73134 56571
            14060 19505 72425 56575 74351 68786 51650 20004 18383 76614 11634
            18906 15765 41368 73241 76698 78567 97189 28545 76231 75691 22246
            51061 90578 56691 68014 51103 94167 57047 14867 73520 15734 63435
            25733 35474 24676 94627 53535 17879 15559 53268 59166 11928 59402
            33282 45721 43933 68101 33515 36634 71286 19736 58058 55253 67473
            41918 19515 36495 19430 22351 77191 91393 49156 50298 87501 18652
            53179 18767 63193 23968 65164 68880 21286 72823 58470 67301 13394
            31016 70372 67030 40604 24317 45748 39127 26065 77721 31029 31880
            60576 24671 45549 13376 50016 33123 19769 22927 97789 46081 72151
            15723 46136 51949 68100 96888 64528 14171 79777 28709 11489 25103
            32213 78668 22245 15798 27156 37930 62971 21337 51622 67853 10567
            38415 15455 58263 42029 60279 37125 56240 88190 50308 26859 64457
            89091 82136 62377 36233 63837 58078 17043 30010 60099 28810 98025
            29178 87343 73273 30469 64034 39516 86057 21309 90257 67875 40162
            11356 73650 61810 72013 30431 22461 19512 13375 55307 30625 83849
            68908 26689 96451 38193 46820 88885 84935 69035 83144 47537 56616
            94983 48033 69952 25486 61547 27385 61860 58048 56910 16807 17871
            35258 31387 35458 35576
        """.split():
            listed.append(lit(String(code)))
        var five = col("ca_zip").str().slice(0, 5)
        # INTERSECT: the listed zips that more than 10 preferred customers
        # live at, once each.
        var crowded = (
            t["customer_address"]
            .lazy()
            .join(
                t["customer"]
                .lazy()
                .filter(col("c_preferred_cust_flag") == "Y"),
                left_on=["ca_address_sk"],
                right_on=["c_current_addr_sk"],
            )
            .group_by(["ca_zip"])
            .agg([col("ca_zip").len().alias("cnt")])
            .filter(col("cnt") > 10)
            .select_exprs([five.alias("crowded_zip")])
        )
        var v1 = (
            t["customer_address"]
            .lazy()
            .filter(five.is_in(listed))
            .select_exprs([five.alias("ca_zip")])
            .join(
                crowded,
                left_on=["ca_zip"],
                right_on=["crowded_zip"],
                how="semi",
            )
            .unique()
            .select_exprs([col("ca_zip").str().slice(0, 2).alias("v1_zip2")])
        )
        var grouped = (
            t["store_sales"]
            .lazy()
            .join(
                dates(
                    t,
                    (col("d_qoy") == 2) & (col("d_year") == 1998),
                    "d_sk",
                ),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["store"]
                .lazy()
                .select_exprs(
                    [
                        col("s_store_sk"),
                        col("s_store_name"),
                        col("s_zip").str().slice(0, 2).alias("s_zip2"),
                    ]
                ),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(v1, left_on=["s_zip2"], right_on=["v1_zip2"])
            .group_by(["s_store_name"])
            .agg(
                [
                    col("ss_net_profit")
                    .sum(min_count=1)
                    .alias("sum(ss_net_profit)")
                ]
            )
        )
        return ascending(grouped, ["s_store_name"]).head(100)
    if q == "q23":
        var price_type = t["store_sales"].column("ss_sales_price").dtype()
        var four_years = col("d_year").is_in(ints([2000, 2001, 2002, 2003]))
        var store_value = col("ss_quantity").cast(price_type) * col(
            "ss_sales_price"
        )
        # Items sold more than 4 times on one day; an item appears once per
        # such day, and the joins below match each of those rows.
        var frequent = (
            t["store_sales"]
            .lazy()
            .join(
                t["date_dim"]
                .lazy()
                .filter(four_years)
                .select(["d_date_sk", "d_date"]),
                left_on=["ss_sold_date_sk"],
                right_on=["d_date_sk"],
            )
            .join(
                t["item"]
                .lazy()
                .select_exprs(
                    [
                        col("i_item_desc").str().slice(0, 30).alias("itemdesc"),
                        col("i_item_sk"),
                    ]
                ),
                left_on=["ss_item_sk"],
                right_on=["i_item_sk"],
            )
            .group_by(["itemdesc", "ss_item_sk", "d_date"])
            .agg([col("ss_item_sk").len().alias("cnt")])
            .filter(col("cnt") > 4)
            .select_exprs([col("ss_item_sk").alias("item_sk")])
        )
        var max_store_sales = (
            t["store_sales"]
            .lazy()
            .join(
                t["customer"].lazy().select(["c_customer_sk"]),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .join(
                dates(t, four_years, "d_sk"),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .with_columns([store_value.alias("value")])
            .group_by(["ss_customer_sk"])
            .agg([col("value").sum(min_count=1).alias("csales")])
            .select_exprs([col("csales").max().alias("tpcds_cmax")])
        )
        var best = (
            t["store_sales"]
            .lazy()
            .join(
                t["customer"].lazy().select(["c_customer_sk"]),
                left_on=["ss_customer_sk"],
                right_on=["c_customer_sk"],
            )
            .with_columns([store_value.alias("value")])
            .group_by(["ss_customer_sk"])
            .agg([col("value").sum(min_count=1).alias("ssales")])
            .join(max_store_sales, how="cross")
            .filter(
                col("ssales").cast(DataType.FLOAT64)
                > lit(0.5) * col("tpcds_cmax").cast(DataType.FLOAT64)
            )
            .select_exprs([col("ss_customer_sk").alias("best_sk")])
        )
        var list_type = t["catalog_sales"].column("cs_list_price").dtype()

        var both = q23_channel(
            t, "catalog_sales", "cs", frequent, best, list_type
        ).concat(q23_channel(t, "web_sales", "ws", frequent, best, list_type))
        return ascending(
            both, ["c_last_name", "c_first_name", "sales"], nulls_first=True
        ).head(100)
    if q == "q27":
        # The SQL's UNION ALL form of a rollup: by item and state, by item,
        # and over all rows.
        var results = (
            t["store_sales"]
            .lazy()
            .join(
                t["customer_demographics"]
                .lazy()
                .filter(
                    (col("cd_gender") == "M")
                    & (col("cd_marital_status") == "S")
                    & (col("cd_education_status") == "College")
                )
                .select(["cd_demo_sk"]),
                left_on=["ss_cdemo_sk"],
                right_on=["cd_demo_sk"],
            )
            .join(
                dates(t, col("d_year") == 2002, "d_sk"),
                left_on=["ss_sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["store"]
                .lazy()
                .filter(col("s_state") == "TN")
                .select(["s_store_sk", "s_state"]),
                left_on=["ss_store_sk"],
                right_on=["s_store_sk"],
            )
            .join(
                t["item"].lazy().select(["i_item_sk", "i_item_id"]),
                left_on=["ss_item_sk"],
                right_on=["i_item_sk"],
            )
        )
        var averages: List[Expr] = [
            col("ss_quantity").mean().alias("agg1"),
            col("ss_list_price").mean().alias("agg2"),
            col("ss_coupon_amt").mean().alias("agg3"),
            col("ss_sales_price").mean().alias("agg4"),
        ]
        var shown: List[Expr] = [
            col("i_item_id"),
            col("s_state"),
            col("g_state"),
            col("agg1"),
            col("agg2"),
            col("agg3"),
            col("agg4"),
        ]
        var by_state = (
            results.group_by(["i_item_id", "s_state"])
            .agg(averages)
            .with_columns([lit(Int64(0)).alias("g_state")])
            .select_exprs(shown)
        )
        var by_item = (
            results.group_by(["i_item_id"])
            .agg(averages)
            .with_columns(
                [
                    q27_no_text().alias("s_state"),
                    lit(Int64(1)).alias("g_state"),
                ]
            )
            .select_exprs(shown)
        )
        var overall = (
            results.select_exprs(averages)
            .with_columns(
                [
                    q27_no_text().alias("i_item_id"),
                    q27_no_text().alias("s_state"),
                    lit(Int64(1)).alias("g_state"),
                ]
            )
            .select_exprs(shown)
        )
        return ascending(
            by_state.concat(by_item).concat(overall),
            ["i_item_id", "s_state"],
            nulls_first=True,
        ).head(100)
    if q == "q33":
        var found = q33_union_totals(
            t,
            "i_manufact_id",
            col("i_category") == "Electronics",
            (col("d_year") == 1998) & (col("d_moy") == 5),
        )
        return ascending(found, ["total_sales"]).head(100)
    if q == "q56":
        var found = q33_union_totals(
            t,
            "i_item_id",
            col("i_color").is_in(["slate", "blanched", "burnished"]),
            (col("d_year") == 2001) & (col("d_moy") == 2),
        )
        return ascending(
            found, ["total_sales", "i_item_id"], nulls_first=True
        ).head(100)
    if q == "q60":
        var found = q33_union_totals(
            t,
            "i_item_id",
            col("i_category") == "Music",
            (col("d_year") == 1998) & (col("d_moy") == 9),
        )
        return ascending(found, ["i_item_id", "total_sales"]).head(100)
    if q == "q38" or q == "q87":
        # INTERSECT keeps the triples every channel has (tags 1+2+4);
        # store EXCEPT catalog EXCEPT web the ones only the store has.
        var wanted = 7 if q == "q38" else 1
        return (
            q38_channel_tags(t)
            .filter(col("tags") == wanted)
            .select_exprs([col("tags").len().alias("count_star()")])
        )
    if q == "q54":
        var channel_sales = (
            t["catalog_sales"]
            .lazy()
            .select_exprs(
                [
                    col("cs_sold_date_sk").alias("sold_date_sk"),
                    col("cs_bill_customer_sk").alias("customer_sk"),
                    col("cs_item_sk").alias("item_sk"),
                ]
            )
            .concat(
                t["web_sales"]
                .lazy()
                .select_exprs(
                    [
                        col("ws_sold_date_sk").alias("sold_date_sk"),
                        col("ws_bill_customer_sk").alias("customer_sk"),
                        col("ws_item_sk").alias("item_sk"),
                    ]
                )
            )
        )
        var my_customers = (
            channel_sales.join(
                t["item"]
                .lazy()
                .filter(
                    (col("i_category") == "Women")
                    & (col("i_class") == "maternity")
                )
                .select(["i_item_sk"]),
                left_on=["item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                dates(
                    t, (col("d_moy") == 12) & (col("d_year") == 1998), "d_sk"
                ),
                left_on=["sold_date_sk"],
                right_on=["d_sk"],
            )
            .join(
                t["customer"]
                .lazy()
                .select(["c_customer_sk", "c_current_addr_sk"]),
                left_on=["customer_sk"],
                right_on=["c_customer_sk"],
            )
            .select_exprs(
                [
                    col("customer_sk").alias("c_customer_sk"),
                    col("c_current_addr_sk"),
                ]
            )
            .unique()
        )
        # The two scalar subqueries: the month after December 1998 and
        # the third month after it, as one row crossed with date_dim.
        var bounds = (
            t["date_dim"]
            .lazy()
            .filter((col("d_year") == 1998) & (col("d_moy") == 12))
            .select_exprs(
                [
                    (col("d_month_seq") + lit(Int64(1))).alias("low_seq"),
                    (col("d_month_seq") + lit(Int64(3))).alias("high_seq"),
                ]
            )
            .unique()
        )
        var months = (
            t["date_dim"]
            .lazy()
            .select(["d_date_sk", "d_month_seq"])
            .join(bounds, how="cross")
            .filter(
                (col("d_month_seq") >= col("low_seq"))
                & (col("d_month_seq") <= col("high_seq"))
            )
            .select_exprs([col("d_date_sk").alias("m_sk")])
        )
        var segments = (
            my_customers.join(
                t["store_sales"]
                .lazy()
                .select(
                    ["ss_customer_sk", "ss_sold_date_sk", "ss_ext_sales_price"]
                ),
                left_on=["c_customer_sk"],
                right_on=["ss_customer_sk"],
            )
            .join(months, left_on=["ss_sold_date_sk"], right_on=["m_sk"])
            .join(
                t["customer_address"]
                .lazy()
                .select(["ca_address_sk", "ca_county", "ca_state"]),
                left_on=["c_current_addr_sk"],
                right_on=["ca_address_sk"],
            )
            .join(
                t["store"].lazy().select(["s_county", "s_state"]),
                left_on=["ca_county", "ca_state"],
                right_on=["s_county", "s_state"],
            )
            .group_by(["c_customer_sk"])
            .agg([col("ss_ext_sales_price").sum(min_count=1).alias("revenue")])
            .select_exprs(
                [
                    (col("revenue").cast(DataType.FLOAT64) / lit(50.0))
                    .round()
                    .cast(DataType.INT32)
                    .alias("segment")
                ]
            )
            .group_by(["segment"])
            .agg([col("segment").len().alias("num_customers")])
            .select_exprs(
                [
                    col("segment"),
                    col("num_customers"),
                    (col("segment") * lit(Int32(50))).alias("segment_base"),
                ]
            )
        )
        return ascending(
            segments,
            ["segment", "num_customers", "segment_base"],
            nulls_first=True,
        ).head(100)
    if q == "q66":
        var months: List[String] = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
        var keys: List[String] = [
            "w_warehouse_name",
            "w_warehouse_sq_ft",
            "w_city",
            "w_county",
            "w_state",
            "w_country",
            "ship_carriers",
            "year_",
        ]
        var sums = List[Expr]()
        for i in range(12):
            var name = months[i] + "_sales"
            sums.append(col(name).sum(min_count=1).alias(name))
        var area = col("w_warehouse_sq_ft").cast(DataType.FLOAT64)
        for i in range(12):
            sums.append(
                (col(months[i] + "_sales").cast(DataType.FLOAT64) / area)
                .sum(min_count=1)
                .alias(months[i] + "_sales_per_sq_foot")
            )
        for i in range(12):
            var name = months[i] + "_net"
            sums.append(col(name).sum(min_count=1).alias(name))
        var grouped = (
            q66_warehouse_months(
                t, "web_sales", "ws", "ws_ext_sales_price", "ws_net_paid"
            )
            .concat(
                q66_warehouse_months(
                    t,
                    "catalog_sales",
                    "cs",
                    "cs_sales_price",
                    "cs_net_paid_inc_tax",
                )
            )
            .group_by(keys)
            .agg(sums)
        )
        return ascending(grouped, ["w_warehouse_name"], nulls_first=True).head(
            100
        )
    if q == "q71":
        var prefixes: List[String] = ["ws", "cs", "ss"]
        var tables: List[String] = ["web_sales", "catalog_sales", "store_sales"]
        var days = dates(
            t, (col("d_moy") == 11) & (col("d_year") == 1999), "d_sk"
        )
        var parts = List[LazyFrame]()
        for i in range(3):
            var p = prefixes[i]
            parts.append(
                t[tables[i]]
                .lazy()
                .join(days, left_on=[p + "_sold_date_sk"], right_on=["d_sk"])
                .select_exprs(
                    [
                        col(p + "_ext_sales_price").alias("ext_price"),
                        col(p + "_sold_date_sk").alias("sold_date_sk"),
                        col(p + "_item_sk").alias("sold_item_sk"),
                        col(p + "_sold_time_sk").alias("time_sk"),
                    ]
                )
            )
        var grouped = (
            parts[0]
            .concat(parts[1])
            .concat(parts[2])
            .join(
                t["item"].lazy().filter(col("i_manager_id") == 1),
                left_on=["sold_item_sk"],
                right_on=["i_item_sk"],
            )
            .join(
                t["time_dim"]
                .lazy()
                .filter(col("t_meal_time").is_in(["breakfast", "dinner"])),
                left_on=["time_sk"],
                right_on=["t_time_sk"],
            )
            .group_by(["i_brand", "i_brand_id", "t_hour", "t_minute"])
            .agg([col("ext_price").sum(min_count=1).alias("ext_price")])
            .select_exprs(
                [
                    col("i_brand_id").alias("brand_id"),
                    col("i_brand").alias("brand"),
                    col("t_hour"),
                    col("t_minute"),
                    col("ext_price"),
                ]
            )
        )
        return ordered(
            grouped,
            ["ext_price", "brand_id", "t_hour"],
            [True, False, False],
            nulls_first=True,
        )
    if q == "q75":
        var prefixes: List[String] = ["cs", "ss", "ws"]
        var tables: List[String] = ["catalog_sales", "store_sales", "web_sales"]
        var returns: List[String] = [
            "catalog_returns",
            "store_returns",
            "web_returns",
        ]
        var sale_keys: List[String] = [
            "cs_order_number",
            "ss_ticket_number",
            "ws_order_number",
        ]
        var return_keys: List[String] = [
            "cr_order_number",
            "sr_ticket_number",
            "wr_order_number",
        ]
        var return_items: List[String] = [
            "cr_item_sk",
            "sr_item_sk",
            "wr_item_sk",
        ]
        var return_counts: List[String] = [
            "cr_return_quantity",
            "sr_return_quantity",
            "wr_return_quantity",
        ]
        var return_amounts: List[String] = [
            "cr_return_amount",
            "sr_return_amt",
            "wr_return_amt",
        ]
        var keys: List[String] = [
            "d_year",
            "i_brand_id",
            "i_class_id",
            "i_category_id",
            "i_manufact_id",
        ]
        var books = (
            t["item"]
            .lazy()
            .filter(col("i_category") == "Books")
            .select(
                [
                    "i_item_sk",
                    "i_brand_id",
                    "i_class_id",
                    "i_category_id",
                    "i_manufact_id",
                ]
            )
        )
        var details = List[LazyFrame]()
        for i in range(3):
            var p = prefixes[i]
            ref back = t[returns[i]]
            var no_count = lit(Int64(0)).cast(
                back.column(return_counts[i]).dtype()
            )
            details.append(
                t[tables[i]]
                .lazy()
                .join(books, left_on=[p + "_item_sk"], right_on=["i_item_sk"])
                .join(
                    t["date_dim"].lazy().select(["d_date_sk", "d_year"]),
                    left_on=[p + "_sold_date_sk"],
                    right_on=["d_date_sk"],
                )
                .join(
                    back.lazy().select(
                        [
                            return_keys[i],
                            return_items[i],
                            return_counts[i],
                            return_amounts[i],
                        ]
                    ),
                    left_on=[sale_keys[i], p + "_item_sk"],
                    right_on=[return_keys[i], return_items[i]],
                    how="left",
                )
                .select_exprs(
                    [
                        col("d_year"),
                        col("i_brand_id"),
                        col("i_class_id"),
                        col("i_category_id"),
                        col("i_manufact_id"),
                        (
                            col(p + "_quantity")
                            - coalesce([col(return_counts[i]), no_count.copy()])
                        ).alias("sales_cnt"),
                        (
                            col(p + "_ext_sales_price")
                            - coalesce(
                                [
                                    col(return_amounts[i]),
                                    like(back, return_amounts[i], "0"),
                                ]
                            )
                        ).alias("sales_amt"),
                    ]
                )
            )
        var all_sales = (
            details[0]
            .concat(details[1])
            .concat(details[2])
            .unique()
            .group_by(keys)
            .agg(
                [
                    col("sales_cnt").sum(min_count=1).alias("sales_cnt"),
                    col("sales_amt").sum(min_count=1).alias("sales_amt"),
                ]
            )
        )
        var prev_yr = all_sales.filter(col("d_year") == 2001).select_exprs(
            [
                col("d_year").alias("prev_year"),
                col("i_brand_id").alias("p_brand_id"),
                col("i_class_id").alias("p_class_id"),
                col("i_category_id").alias("p_category_id"),
                col("i_manufact_id").alias("p_manufact_id"),
                col("sales_cnt").alias("prev_yr_cnt"),
                col("sales_amt").alias("prev_yr_amt"),
            ]
        )
        var compared = (
            all_sales.filter(col("d_year") == 2002)
            .join(
                prev_yr,
                left_on=[
                    "i_brand_id",
                    "i_class_id",
                    "i_category_id",
                    "i_manufact_id",
                ],
                right_on=[
                    "p_brand_id",
                    "p_class_id",
                    "p_category_id",
                    "p_manufact_id",
                ],
            )
            .filter(
                col("sales_cnt").cast(DataType.FLOAT64)
                / col("prev_yr_cnt").cast(DataType.FLOAT64)
                < lit(0.9)
            )
            .select_exprs(
                [
                    col("prev_year"),
                    col("d_year").alias("year_"),
                    col("i_brand_id"),
                    col("i_class_id"),
                    col("i_category_id"),
                    col("i_manufact_id"),
                    col("prev_yr_cnt"),
                    col("sales_cnt").alias("curr_yr_cnt"),
                    (col("sales_cnt") - col("prev_yr_cnt")).alias(
                        "sales_cnt_diff"
                    ),
                    (col("sales_amt") - col("prev_yr_amt")).alias(
                        "sales_amt_diff"
                    ),
                ]
            )
        )
        return ascending(compared, ["sales_cnt_diff", "sales_amt_diff"]).head(
            100
        )
    if q == "q76":
        var channels: List[String] = ["store", "web", "catalog"]
        var missing: List[String] = [
            "ss_store_sk",
            "ws_ship_customer_sk",
            "cs_ship_addr_sk",
        ]
        var tables: List[String] = ["store_sales", "web_sales", "catalog_sales"]
        var prefixes: List[String] = ["ss", "ws", "cs"]
        var keys: List[String] = [
            "channel",
            "col_name",
            "d_year",
            "d_qoy",
            "i_category",
        ]
        var parts = List[LazyFrame]()
        for i in range(3):
            var p = prefixes[i]
            parts.append(
                t[tables[i]]
                .lazy()
                .filter(col(missing[i]).is_null())
                .join(
                    t["date_dim"]
                    .lazy()
                    .select(["d_date_sk", "d_year", "d_qoy"]),
                    left_on=[p + "_sold_date_sk"],
                    right_on=["d_date_sk"],
                )
                .join(
                    t["item"].lazy().select(["i_item_sk", "i_category"]),
                    left_on=[p + "_item_sk"],
                    right_on=["i_item_sk"],
                )
                .select_exprs(
                    [
                        lit(channels[i]).alias("channel"),
                        lit(missing[i]).alias("col_name"),
                        col("d_year"),
                        col("d_qoy"),
                        col("i_category"),
                        col(p + "_ext_sales_price").alias("ext_sales_price"),
                    ]
                )
            )
        var grouped = (
            parts[0]
            .concat(parts[1])
            .concat(parts[2])
            .group_by(keys)
            .agg(
                [
                    col("channel").len().alias("sales_cnt"),
                    col("ext_sales_price").sum(min_count=1).alias("sales_amt"),
                ]
            )
        )
        return ascending(grouped, keys, nulls_first=True).head(100)
    if q == "q44":
        # Store 4's items with an average profit above 90% of its average
        # on sales without an address, ten best beside ten worst by rank.
        ref sales = t["store_sales"]
        var profit = col("ss_net_profit").cast(DataType.FLOAT64)
        var baseline = (
            sales.lazy()
            .filter(
                (col("ss_store_sk") == lit(Int64(4)))
                & col("ss_addr_sk").is_null()
            )
            .group_by(["ss_store_sk"])
            .agg([profit.mean().alias("baseline")])
            .select(["baseline"])
        )
        var items = (
            sales.lazy()
            .filter(col("ss_store_sk") == lit(Int64(4)))
            .group_by(["ss_item_sk"])
            .agg([profit.mean().alias("rank_col")])
            .join(baseline, how="cross")
            .filter(col("rank_col") > lit(0.9) * col("baseline"))
        )
        var best = (
            items.with_columns(
                [col("rank_col").rank(method="min").alias("rnk")]
            )
            .filter(col("rnk") < lit(Int64(11)))
            .select_exprs([col("ss_item_sk").alias("best_sk"), col("rnk")])
        )
        var worst = (
            items.with_columns(
                [
                    col("rank_col")
                    .rank(method="min", descending=True)
                    .alias("worst_rnk")
                ]
            )
            .filter(col("worst_rnk") < lit(Int64(11)))
            .select_exprs(
                [col("ss_item_sk").alias("worst_sk"), col("worst_rnk")]
            )
        )
        var paired = (
            best.join(worst, left_on=["rnk"], right_on=["worst_rnk"])
            .join(
                t["item"]
                .lazy()
                .select_exprs(
                    [
                        col("i_item_sk").alias("best_item"),
                        col("i_product_name").alias("best_performing"),
                    ]
                ),
                left_on=["best_sk"],
                right_on=["best_item"],
            )
            .join(
                t["item"]
                .lazy()
                .select_exprs(
                    [
                        col("i_item_sk").alias("worst_item"),
                        col("i_product_name").alias("worst_performing"),
                    ]
                ),
                left_on=["worst_sk"],
                right_on=["worst_item"],
            )
            .select(["rnk", "best_performing", "worst_performing"])
        )
        return ascending(paired, ["rnk"]).head(100)
    if q == "q47":
        var shown: List[String] = [
            "i_category",
            "i_brand",
            "s_store_name",
            "s_company_name",
            "d_year",
            "d_moy",
            "avg_monthly_sales",
            "sum_sales",
            "psum",
            "nsum",
        ]
        var found = q47_neighbours(
            t,
            "store_sales",
            "ss",
            "store",
            "ss_store_sk",
            "s_store_sk",
            ["s_store_name", "s_company_name"],
        )
        return q47_sorted(found, shown, diff_nulls_first=False)
    if q == "q57":
        var shown: List[String] = [
            "i_category",
            "i_brand",
            "cc_name",
            "d_year",
            "d_moy",
            "avg_monthly_sales",
            "sum_sales",
            "psum",
            "nsum",
        ]
        var found = q47_neighbours(
            t,
            "catalog_sales",
            "cs",
            "call_center",
            "cs_call_center_sk",
            "cc_call_center_sk",
            ["cc_name"],
        )
        return q47_sorted(found, shown, diff_nulls_first=True)
    if q == "q51":
        var both = q51_cumulative(t, "web_sales", "ws", "web_sales").join(
            q51_cumulative(t, "store_sales", "ss", "store_sales"),
            on=["item_sk", "d_date"],
            how="full",
        )
        # The running maxima follow d_date within each item: sort, then
        # window over the item, which keeps that order.
        var found = (
            ascending(both, ["item_sk", "d_date"])
            .with_columns(
                [
                    q51_running_max("web_sales", "web_cumulative"),
                    q51_running_max("store_sales", "store_cumulative"),
                ]
            )
            .filter(col("web_cumulative") > col("store_cumulative"))
            .select(
                [
                    "item_sk",
                    "d_date",
                    "web_sales",
                    "store_sales",
                    "web_cumulative",
                    "store_cumulative",
                ]
            )
        )
        return ascending(found, ["item_sk", "d_date"], nulls_first=True).head(
            100
        )
    if q == "q49":
        var web = q49_channel(
            t,
            "web",
            "web_sales",
            "web_returns",
            ["ws_order_number", "ws_item_sk"],
            ["wr_order_number", "wr_item_sk"],
            "ws",
            "wr",
            "wr_return_amt",
        )
        var catalog = q49_channel(
            t,
            "catalog",
            "catalog_sales",
            "catalog_returns",
            ["cs_order_number", "cs_item_sk"],
            ["cr_order_number", "cr_item_sk"],
            "cs",
            "cr",
            "cr_return_amount",
        )
        var store = q49_channel(
            t,
            "store",
            "store_sales",
            "store_returns",
            ["ss_ticket_number", "ss_item_sk"],
            ["sr_ticket_number", "sr_item_sk"],
            "ss",
            "sr",
            "sr_return_amt",
        )
        # UNION: UNION ALL, then distinct rows.
        var union = web.concat(catalog).concat(store).unique()
        return ascending(
            union,
            ["channel", "return_rank", "currency_rank", "item"],
            nulls_first=True,
        ).head(100)
    raise unsupported("not translated")


def main() raises:
    run[query]()
