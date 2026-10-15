"""An inner join chain nested as a join's right input joins the outer chain
from the table that supplies the outer join's keys (DuckDB's flattened join
graph). Every result equals the same joins over separately collected
inputs, with the nested chain's output columns: its left keys, which the
walk drops where it meets a join from its right side, come back from their
equal partners, and its filter still applies. A nested chain whose base
supplies the keys, and one whose names clash, are left as written."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def table(
    name: String, keys: List[String], rows: Int, spans: List[Int]
) raises -> LazyFrame:
    var columns = List[Series]()
    for k in range(len(keys)):
        var values = List[Int64]()
        for row in range(rows):
            values.append(Int64((row * (7 + 2 * k)) % spans[k]))
        columns.append(Series(keys[k], Column[Int64](values^)))
    var payload = List[Int64]()
    for row in range(rows):
        payload.append(Int64(row))
    columns.append(Series(name + "_v", Column[Int64](payload^)))
    return DataFrame(columns^).lazy()


def regions() raises -> LazyFrame:
    return table("r", ["r_key"], 5, [5])


def nations() raises -> LazyFrame:
    return table("n", ["n_key", "n_region"], 25, [25, 5])


def suppliers() raises -> LazyFrame:
    return table("s", ["s_key", "s_nation"], 400, [400, 25])


def stock() raises -> LazyFrame:
    return table("ps", ["ps_part", "ps_supp"], 8000, [2000, 400])


def parts() raises -> LazyFrame:
    return table("p", ["p_key"], 2000, [2000]).filter(
        col("p_v") % lit(Int64(29)) == lit(Int64(0))
    )


def test_a_nested_chain_joins_from_the_key_supplier() raises:
    # Parts joined to one region's stock, written region first (PDS-H q2's
    # shape): the stock table supplies the outer key.
    var region_stock = (
        regions()
        .filter(col("r_v") == lit(Int64(2)))
        .join(nations(), left_on=["r_key"], right_on=["n_region"])
        .join(suppliers(), left_on=["n_key"], right_on=["s_nation"])
        .join(stock(), left_on=["s_key"], right_on=["ps_supp"])
    )
    var query = parts().join(
        region_stock, left_on=["p_key"], right_on=["ps_part"]
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var flattened = query.collect()
    var took = "dataframe-path: lazy.flatten_join_tree\n" in observed.text()
    _ = observed^
    _ = trace^
    assert_true(took, "the nested join chain was not flattened")
    var eager = (
        parts()
        .collect()
        .join(region_stock.collect(), left_on=["p_key"], right_on=["ps_part"])
    )
    assert_true(eager.height() > 0)
    assert_equal(flattened.columns(), eager.columns())
    assert_frame(
        flattened.sort(["p_key", "s_key"]), eager.sort(["p_key", "s_key"])
    )


def test_chains_keyed_at_their_base_or_with_clashing_names_stay() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    # The nested chain's base (stock) supplies the key: already rooted.
    var rooted = stock().join(
        suppliers(), left_on=["ps_supp"], right_on=["s_key"]
    )
    var observed = StrategyTrace()
    var first = parts().join(rooted, left_on=["p_key"], right_on=["ps_part"])
    var result = first.collect()
    assert_true(
        "dataframe-path: lazy.flatten_join_tree\n" not in observed.text(),
        "a chain keyed at its base was flattened",
    )
    _ = observed^
    assert_frame(
        result.sort(["p_key", "ps_supp"]),
        parts()
        .collect()
        .join(rooted.collect(), left_on=["p_key"], right_on=["ps_part"])
        .sort(["p_key", "ps_supp"]),
    )
    # The outer side holds a name the nested chain also has.
    var clashing = parts().with_columns([col("p_v").alias("s_v")])
    var nested = suppliers().join(
        stock(), left_on=["s_key"], right_on=["ps_supp"]
    )
    var observed_clash = StrategyTrace()
    var second = clashing.join(nested, left_on=["p_key"], right_on=["ps_part"])
    var clash_result = second.collect()
    assert_true(
        "dataframe-path: lazy.flatten_join_tree\n" not in observed_clash.text(),
        "a chain sharing a name with the outer side was flattened",
    )
    _ = observed_clash^
    _ = trace^
    assert_frame(
        clash_result.sort(["p_key", "s_key"]),
        clashing.collect()
        .join(nested.collect(), left_on=["p_key"], right_on=["ps_part"])
        .sort(["p_key", "s_key"]),
    )
