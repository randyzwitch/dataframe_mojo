"""Sideways join-key filters preserve scalar pairs through chains and renames."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col, lit
from dataframe.lazy import JOIN, SCAN_FRAME, _key_set_filter
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.join_pushdown_reference import (
    PushdownRow,
    row,
    fact_item,
    fact_day,
    fact_item_valid,
    fact_day_valid,
    quantity,
    payload,
    payload_valid,
    calendar_day,
    bucket,
    selected_keys,
    key_filter,
    single_join_rows,
    stock_rows,
    order_item,
    order_day,
    order_kind,
    order_valid,
    sold_rows,
    compound_keys,
    compound_join_rows,
)


def _column(
    name: String, var values: List[Int64], var valid: List[Bool] = List[Bool]()
) raises -> Series:
    if not valid:
        valid = List[Bool](length=len(values), fill=True)
    return Series(name, Column[Int64](values^, valid^))


def _facts(n: Int, layout: Int) raises -> DataFrame:
    var ids = List[Int64]()
    var items = List[Int64]()
    var days = List[Int64]()
    var quantities = List[Int64]()
    var values = List[Int64]()
    var item_valid = List[Bool]()
    var day_valid = List[Bool]()
    var value_valid = List[Bool]()
    for index in range(n):
        ids.append(Int64(index))
        items.append(fact_item(index))
        days.append(fact_day(index))
        quantities.append(quantity(index))
        values.append(payload(index))
        item_valid.append(fact_item_valid(index))
        day_valid.append(fact_day_valid(index))
        value_valid.append(payload_valid(index))
    var frame = DataFrame(
        [
            _column("fact_id", ids^),
            _column("fact_item", items^, item_valid^),
            _column("fact_day", days^, day_valid^),
            _column("quantity", quantities^),
            _column("payload", values^, value_valid^),
        ]
    )
    if layout == 1:
        return offset_layout(frame)
    if layout == 2:
        return chunk_layout(frame)
    return frame^


def _calendar() raises -> DataFrame:
    var days = List[Int64]()
    var buckets = List[Int64]()
    for index in range(129):
        days.append(calendar_day(index))
        buckets.append(bucket(index))
    return DataFrame(
        [
            _column("calendar_value", days.copy()),
            _column("calendar_day", days^),
            _column("bucket", buckets^),
        ]
    )


def _stock(n: Int, layout: Int, filter_quantity: Bool) raises -> LazyFrame:
    var facts = _facts(n, layout).lazy()
    if filter_quantity:
        facts = facts.filter(col("quantity") > lit(Int64(-11)))
    return facts.join(
        _calendar().lazy(),
        left_on=["fact_day"],
        right_on=["calendar_day"],
    ).select_exprs(
        [
            col("fact_id"),
            col("fact_item").alias("join_item"),
            col("calendar_value").alias("join_day"),
            col("bucket"),
            col("payload"),
        ]
    )


def _small(count: Int, missing: Bool) raises -> LazyFrame:
    var keys = selected_keys(count, missing)
    var ids = List[Int64]()
    for index in range(len(keys[0])):
        ids.append(Int64(index))
    return (
        DataFrame(
            [
                _column("left_id", ids^),
                _column("item", keys[0].copy(), keys[1].copy()),
            ]
        )
        .lazy()
        .filter(col("left_id") >= lit(Int64(0)))
    )


def _sold() raises -> LazyFrame:
    var ids = List[Int64]()
    var items = List[Int64]()
    var days = List[Int64]()
    var kinds = List[Int64]()
    var valid = List[Bool]()
    for index in range(257):
        ids.append(Int64(index))
        items.append(order_item(index))
        days.append(order_day(index))
        kinds.append(order_kind(index))
        valid.append(order_valid(index))
    var orders = DataFrame(
        [
            _column("left_id", ids^),
            _column("item", items^, valid^),
            _column("order_day", days^),
            _column("order_kind", kinds^),
        ]
    )
    var kind_keys = List[Int64]()
    var flags = List[Int64]()
    for kind in range(5):
        kind_keys.append(Int64(kind))
        flags.append(Int64(kind % 2))
    var categories = DataFrame(
        [
            _column("kind_key", kind_keys^),
            _column("flag", flags^),
        ]
    )
    return (
        orders.lazy()
        .join(
            categories.lazy().filter(col("flag") == lit(Int64(1))),
            left_on=["order_kind"],
            right_on=["kind_key"],
        )
        .join(
            _calendar()
            .head(128)
            .lazy()
            .filter(col("calendar_day") < lit(Int64(9)))
            .select_exprs(
                [
                    col("calendar_day").alias("day_key"),
                    (col("calendar_day") + lit(Int64(0))).alias("wanted_day"),
                ]
            ),
            left_on=["order_day"],
            right_on=["day_key"],
        )
        .select(["left_id", "item", "wanted_day"])
    )


def _query(
    left: LazyFrame, stock: LazyFrame, semi: Bool, compound: Bool
) raises -> LazyFrame:
    var left_keys: List[String] = ["item"]
    var right_keys: List[String] = ["join_item"]
    if compound:
        left_keys.append("wanted_day")
        right_keys.append("join_day")
    var joined = left.join(
        stock,
        left_on=left_keys^,
        right_on=right_keys^,
        how="semi" if semi else "inner",
    )
    if semi:
        return joined.select(["left_id", "item"])
    return joined.select_exprs(
        [
            col("left_id"),
            col("item"),
            col("fact_id"),
            col("wanted_day" if compound else "join_day").alias("day"),
            col("bucket"),
            col("payload"),
        ]
    )


def _output_names(semi: Bool) -> List[String]:
    if semi:
        return ["left_id", "item"]
    return ["left_id", "item", "fact_id", "day", "bucket", "payload"]


def _assert_rows(
    actual: DataFrame, expected: List[PushdownRow], names: List[String]
) raises:
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(expected))
    for name in names:
        assert_equal(actual.column(name).dtype(), DataType.INT64)
    var used = List[Bool](length=len(expected), fill=False)
    for index in range(actual.height()):
        var matched = -1
        for candidate in range(len(expected)):
            if used[candidate]:
                continue
            var equal = True
            for column in range(len(names)):
                var cell = actual.item(index, names[column])
                if cell.is_null() == expected[candidate].valid[column]:
                    equal = False
                elif (
                    not cell.is_null()
                    and cell.int64() != expected[candidate].values[column]
                ):
                    equal = False
            if equal:
                matched = candidate
                break
        assert_true(
            matched >= 0, "Unexpected complete join row or multiplicity"
        )
        used[matched] = True
    for found in used:
        assert_true(found)


def _assert_materialized_stock(
    plan: LazyFrame, expected: List[PushdownRow], compound: Bool
) raises:
    var rewritten = plan._optimized()
    rewritten._push_join_key_sets(False, 29)
    var seen = False
    for node in rewritten._nodes:
        if node.kind != JOIN or not node.names or node.names[0] != "item":
            continue
        seen = True
        assert_equal(len(node.names), 2 if compound else 1)
        assert_equal(rewritten._nodes[node.left].kind, SCAN_FRAME)
        assert_equal(rewritten._nodes[node.right].kind, SCAN_FRAME)
        ref narrowed = rewritten._frames[rewritten._nodes[node.right].offset]
        # Assert the actual cached right-side execution, before the final
        # join. Range false positives must still be present here.
        var canonical_names: List[String] = [
            "fact_id",
            "join_item",
            "join_day",
            "bucket",
            "payload",
        ]
        var projected = List[PushdownRow]()
        for source in expected:
            var values = List[Int64]()
            var valid = List[Bool]()
            for name in narrowed.columns():
                var position = -1
                for column in range(len(canonical_names)):
                    if canonical_names[column] == name:
                        position = column
                assert_true(position >= 0)
                values.append(source.values[position])
                valid.append(source.valid[position])
            projected.append(PushdownRow(values^, valid^))
        assert_true("join_item" in narrowed.columns())
        if compound:
            assert_true("join_day" in narrowed.columns())
        _assert_rows(narrowed, projected, narrowed.columns())
    assert_true(seen, "Final key-filtered join must remain observable")


def test_join_key_filter_cutoff_and_scalar_models_have_literal_membership_anchors() raises:
    assert_equal(fact_item(1024), Int64(0))
    assert_equal(fact_day(1), Int64(13))
    assert_equal(fact_day(1024), Int64(17))
    assert_equal(quantity(1), Int64(-9))
    assert_equal(payload(1), Int64(-99))
    assert_true(not fact_item_valid(31))
    assert_equal(calendar_day(128), Int64(6))
    assert_equal(bucket(128), Int64(6))
    var literal_item: List[Int64] = [30]
    var literal_valid: List[Bool] = [True]
    var literal_stock = stock_rows(
        31, literal_item, literal_valid, False, quantity_filter=False
    )
    assert_equal(len(literal_stock), 2)
    assert_equal(literal_stock[0].values, [30, 30, 6, 4, -70])
    assert_equal(literal_stock[1].values, [30, 30, 6, 6, -70])
    assert_equal(literal_stock[0].valid, [True, True, True, True, True])
    var literal_sold = sold_rows()
    assert_equal(literal_sold[0].values, [1, 9, 5])
    assert_equal(literal_sold[1].values, [3, 13, 4])
    var keys = selected_keys(50)
    assert_true(key_filter(keys[0], keys[1], 7, False))
    assert_true(not key_filter(keys[0], keys[1], 8, False))
    keys = selected_keys(51)
    assert_true(key_filter(keys[0], keys[1], 8, True))
    for count in [0, 1, 49, 50, 51]:
        var values = List[Int64]()
        var valid = List[Bool]()
        if count:
            keys = selected_keys(count)
            values = keys[0].copy()
            valid = keys[1].copy()
        else:
            values.append(7)
            valid.append(False)
        var expression = _key_set_filter(
            _column("small", values.copy(), valid.copy()), "key", DataType.INT64
        )
        assert_true(Bool(expression))
        var ids = List[Int64]()
        var probe = List[Int64]()
        var probe_valid = List[Bool]()
        var wanted = List[PushdownRow]()
        for index in range(143):
            var value = Int64(index) if index < 142 else Int64(7)
            ids.append(Int64(index))
            probe.append(value)
            probe_valid.append(index != 142)
            if index != 142 and key_filter(values, valid, value, count > 50):
                wanted.append(row([Int64(index), value]))
        var source = DataFrame(
            [_column("id", ids^), _column("key", probe^, probe_valid^)]
        )
        _assert_rows(source.filter(expression.take()), wanted, ["id", "key"])


def test_sideways_key_lists_and_ranges_reach_right_join_chains_and_keep_every_pair() raises:
    comptime n = 8192
    for layout in range(3):
        for count in [5, 49, 50, 51]:
            var missing = count == 5
            var keys = selected_keys(count, missing)
            var narrowed = stock_rows(n, keys[0], keys[1], count > 50)
            for semi in [False, True]:
                var query = _query(
                    _small(count, missing), _stock(n, layout, True), semi, False
                )
                _assert_materialized_stock(query, narrowed, False)
                var expected = single_join_rows(n, count, missing, semi)
                assert_true(len(expected) > 0)
                for optimize in [False, True]:
                    _assert_rows(
                        query.collect(
                            optimize=optimize,
                            streaming=False,
                            batch_size=29,
                            engine="cpu",
                        ),
                        expected,
                        _output_names(semi),
                    )


def test_two_sideways_key_sets_follow_separate_providers_through_renamed_join_columns() raises:
    comptime n = 8192
    var keys = compound_keys()
    var valid = List[Bool](length=len(keys[0]), fill=True)
    var narrowed = stock_rows(n, keys[0], valid, False, keys[1], False)
    assert_true(len(narrowed) > 0)
    for layout in range(3):
        for semi in [False, True]:
            var query = _query(_sold(), _stock(n, layout, False), semi, True)
            _assert_materialized_stock(query, narrowed, True)
            var expected = compound_join_rows(n, semi)
            assert_true(len(expected) > 0)
            for optimize in [False, True]:
                _assert_rows(
                    query.collect(
                        optimize=optimize,
                        streaming=False,
                        batch_size=29,
                        engine="cpu",
                    ),
                    expected,
                    _output_names(semi),
                )
