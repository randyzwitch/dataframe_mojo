"""Join/aggregate rewrites: fixed CPU executor plus independent row relations."""
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
)
from tests.support.reference_model import (
    generated_model,
    reference_groups,
    reference_join,
    ReferenceTable,
    RowPair,
)
from tests.support.planner_reference import filtered_join_pairs
from tests.support.coverage_generators import (
    frame_from_model,
    offset_layout,
    chunk_layout,
)
from tests.support.coverage_compare import (
    assert_frame,
    assert_join_pairs,
    assert_join_model,
    assert_reference_cell,
    assert_float,
    assert_model,
)
from dataframe.lazy import JOIN, FILTER, AGG
from dataframe.expr import CAST, LIT_INT


def _layout(model: ReferenceTable, layout: Int) raises -> DataFrame:
    var source = frame_from_model(model, 7 if layout == 1 else 0)
    return chunk_layout(source) if layout == 2 else source.copy()


def _fixed(plan: LazyFrame, optimize: Bool) raises -> DataFrame:
    return plan.collect(optimize=optimize, streaming=False, engine="cpu")


def test_join_predicates_preserve_outer_null_rows_and_side_dependencies() raises:
    var shapes: List[Tuple[Int, Int]] = [
        (0, 0),
        (0, 9),
        (9, 0),
        (1, 1),
        (35, 33),
    ]
    for shape in shapes:
        for validity in range(3):
            var left_model = generated_model(shape[0], 3, validity)
            var right_model = generated_model(shape[1], 5, validity)
            for layout in range(3):
                var left = _layout(left_model, layout)
                var right = _layout(right_model, layout)
                for how in ["inner", "left", "right", "full", "semi", "anti"]:
                    var join = left.lazy().join(right.lazy(), "s", how=how)
                    for predicate in range(4):
                        if (how == "semi" or how == "anti") and predicate != 0:
                            continue
                        var expr = (
                            col("i")
                            > 0 if predicate
                            == 0 else col("f_right")
                            > 0 if predicate
                            == 1 else col("f_right").is_null() if predicate
                            == 2 else (col("i") > 0) & (col("f_right") > 0)
                        )
                        var plan = join.filter(expr)
                        var pairs = filtered_join_pairs(
                            left_model, right_model, ["s"], how, predicate
                        )
                        var raw = _fixed(plan, False)
                        var optimized = _fixed(plan, True)
                        assert_join_pairs(
                            raw, left_model, right_model, ["s"], how, pairs
                        )
                        assert_join_pairs(
                            optimized,
                            left_model,
                            right_model,
                            ["s"],
                            how,
                            pairs,
                        )
                        assert_frame(optimized, raw)
                    # A differently named right key must not attract a predicate
                    # referring to the output's coalesced left key.
                    var renamed = right.rename({"i": "right_key"})
                    var plan = (
                        left.lazy()
                        .join(
                            renamed.lazy(),
                            left_on=["i"],
                            right_on=["right_key"],
                            how=how,
                        )
                        .filter(col("i") > 0)
                    )
                    var pairs = filtered_join_pairs(
                        left_model, right_model, ["i"], how, 4
                    )
                    for optimize in [False, True]:
                        assert_join_pairs(
                            _fixed(plan, optimize),
                            left_model,
                            right_model,
                            ["i"],
                            how,
                            pairs,
                        )


def test_join_projection_retains_both_keys_suffix_payloads_and_empty_schema() raises:
    for n in [0, 1, 35]:
        for validity in range(3):
            var left_model = generated_model(n, 3, validity)
            var right_model = generated_model(n, 5, validity)
            for layout in range(3):
                var left = _layout(left_model, layout)
                var right = _layout(right_model, layout).rename(
                    {"s": "right_key"}
                )
                for how in ["inner", "left", "right", "full"]:
                    var joined = left.lazy().join(
                        right.lazy(),
                        left_on=["s"],
                        right_on=["right_key"],
                        how=how,
                    )
                    var pairs = reference_join(
                        left_model, right_model, ["s"], how
                    )
                    for optimize in [False, True]:
                        var actual = _fixed(
                            joined.select(["s", "b", "f_right"]), optimize
                        )
                        assert_equal(actual.columns(), ["s", "b", "f_right"])
                        assert_equal(
                            actual.dtypes(),
                            [DataType.STRING, DataType.BOOL, DataType.FLOAT64],
                        )
                        assert_equal(actual.height(), len(pairs))
                        for row in range(len(pairs)):
                            ref pair = pairs[row]
                            if how == "right" or pair.left < 0:
                                assert_reference_cell(
                                    actual.item(row, "s"),
                                    right_model.rows[pair.right],
                                    "s",
                                )
                            else:
                                assert_reference_cell(
                                    actual.item(row, "s"),
                                    left_model.rows[pair.left],
                                    "s",
                                )
                            if pair.left < 0:
                                assert_true(actual.item(row, "b").is_null())
                            else:
                                assert_reference_cell(
                                    actual.item(row, "b"),
                                    left_model.rows[pair.left],
                                    "b",
                                )
                            if pair.right < 0:
                                assert_true(
                                    actual.item(row, "f_right").is_null()
                                )
                            else:
                                assert_reference_cell(
                                    actual.item(row, "f_right"),
                                    right_model.rows[pair.right],
                                    "f",
                                )
                        var explicit = _fixed(
                            left.lazy().join(
                                right.lazy(),
                                left_on=["s"],
                                right_on=["right_key"],
                                how=how,
                                coalesce=False,
                            ),
                            optimize,
                        )
                        assert_equal(
                            explicit.columns(),
                            [
                                "i",
                                "f",
                                "s",
                                "b",
                                "i_right",
                                "f_right",
                                "right_key",
                                "b_right",
                            ],
                        )
                        assert_join_pairs(
                            explicit.rename({"right_key": "s_right"}),
                            left_model,
                            right_model,
                            ["s"],
                            how,
                            pairs,
                            coalesce=False,
                        )
                        var zero = _fixed(
                            joined.select(List[String]()), optimize
                        )
                        assert_equal(zero.shape(), (len(pairs), 0))
    for optimize in [False, True]:
        var left = DataFrame(List[Series](), height=3)
        var right = DataFrame(List[Series](), height=2)
        assert_equal(
            _fixed(
                left.lazy().join(right.lazy(), how="cross"), optimize
            ).shape(),
            (6, 0),
        )


def test_group_filters_preserve_aggregate_barriers_and_group_order() raises:
    for n in [0, 1, 35, 65]:
        for validity in range(3):
            var model = generated_model(n, 3, validity)
            var groups = reference_groups(model, ["b"])
            for layout in range(3):
                var source = _layout(model, layout)
                var grouped = (
                    source.lazy()
                    .group_by("b", maintain_order=True)
                    .agg(
                        [
                            col("i").sum().alias("total"),
                            col("i").count().alias("count"),
                        ]
                    )
                )
                for predicate in range(3):
                    var expr = (
                        col("b") if predicate
                        == 0 else col("total")
                        > 0 if predicate
                        == 1 else col("b") & (col("total") > 0)
                    )
                    var plan = grouped.filter(expr)
                    for optimize in [False, True]:
                        var actual = _fixed(plan, optimize)
                        assert_equal(actual.columns(), ["b", "total", "count"])
                        assert_equal(
                            actual.dtypes(),
                            [DataType.BOOL, DataType.INT64, DataType.INT64],
                        )
                        var row = 0
                        for group in groups:
                            ref key = model.rows[group[0]]
                            var cells = model.take(group)
                            var total = cells.int_sum()
                            var keep_key = key.bv and key.b
                            var keep = (
                                keep_key if predicate
                                == 0 else total
                                > 0 if predicate
                                == 1 else keep_key
                                and total > 0
                            )
                            if not keep:
                                continue
                            assert_true(row < actual.height())
                            assert_reference_cell(
                                actual.item(row, "b"), key, "b"
                            )
                            assert_equal(
                                actual.item(row, "total").int64(), Int64(total)
                            )
                            assert_equal(
                                actual.item(row, "count").int64(),
                                cells.int_count(),
                            )
                            row += 1
                        assert_equal(actual.height(), row)
                # A whole-column filter sees all groups, not source rows.
                var whole = grouped.filter(
                    col("total").cast(DataType.FLOAT64) > col("total").mean()
                )
                var grand_total = Float64(0)
                for group in groups:
                    grand_total += Float64(model.take(group).int_sum())
                var mean = grand_total / Float64(
                    len(groups)
                ) if groups else Float64(0)
                for optimize in [False, True]:
                    var actual = _fixed(whole, optimize)
                    var row = 0
                    for group in groups:
                        var cells = model.take(group)
                        if Float64(cells.int_sum()) > mean:
                            assert_reference_cell(
                                actual.item(row, "b"), model.rows[group[0]], "b"
                            )
                            assert_equal(
                                actual.item(row, "total").int64(),
                                Int64(cells.int_sum()),
                            )
                            assert_equal(
                                actual.item(row, "count").int64(),
                                cells.int_count(),
                            )
                            row += 1
                    assert_equal(actual.height(), row)


def test_reusing_lazy_subplans_preserves_branch_dependencies_and_source() raises:
    for n in [0, 1, 35]:
        var model = generated_model(n, 3, 1)
        var source = _layout(model, 2)
        var shared = (
            source.lazy()
            .filter(col("i") > 0)
            .with_columns((col("i") * 2).alias("i"))
        )
        var shared_model = model.filter_i_gt(0).double_i()
        var left_model = shared_model.project(["i", "s"])
        var right_model = shared_model.filter_i_gt(10).project(["i", "s"])
        var left = shared.select(["i", "s"])
        var right = shared.filter(col("i") > 10).select(["i", "s"])
        var join = left.join(right, "i", how="full")
        for optimize in [False, True]:
            assert_join_model(
                _fixed(join, optimize), left_model, right_model, ["i"], "full"
            )
            assert_model(_fixed(shared, optimize), shared_model)
            assert_model(_fixed(source.lazy(), optimize), model)


def test_predicate_routes_prove_safe_sides_and_aggregate_barriers() raises:
    var source = _layout(generated_model(9), 0)
    var right = source.rename(
        {
            "i": "right_key",
            "f": "right_value",
            "s": "right_text",
            "b": "right_flag",
        }
    )
    for how in ["inner", "left", "semi", "anti", "right", "full"]:
        var plan = (
            source.lazy()
            .join(right.lazy(), left_on=["i"], right_on=["right_key"], how=how)
            .filter(col("i") > 0)
            ._optimized()
        )
        ref root = plan._nodes[len(plan._nodes) - 1]
        if how in ["inner", "left", "semi", "anti"]:
            assert_equal(root.kind, JOIN)
            assert_equal(plan._nodes[root.left].kind, FILTER)
        else:
            assert_equal(root.kind, FILTER)
            assert_equal(plan._nodes[root.left].kind, JOIN)
    for how in ["inner", "left"]:
        var plan = (
            source.lazy()
            .join(right.lazy(), left_on=["i"], right_on=["right_key"], how=how)
            .filter(col("right_value") > 0)
            ._optimized()
        )
        ref root = plan._nodes[len(plan._nodes) - 1]
        assert_equal(root.kind, JOIN if how == "inner" else FILTER)
        if how == "inner":
            assert_equal(plan._nodes[root.right].kind, FILTER)
    var grouped = (
        source.lazy()
        .group_by("b", maintain_order=True)
        .agg(col("i").sum().alias("total"))
    )
    var keys = grouped.filter(col("b"))._optimized()
    assert_equal(keys._nodes[len(keys._nodes) - 1].kind, AGG)
    var totals = grouped.filter(col("total") > 0)._optimized()
    assert_equal(totals._nodes[len(totals._nodes) - 1].kind, FILTER)


def _casts(plan: LazyFrame) -> Int:
    var found = 0
    for node in plan._nodes:
        for expr in node.exprs:
            for n in expr._nodes:
                if n.op == CAST:
                    found += 1
    return found


def test_casts_of_constants_fold_to_typed_literals_once() raises:
    # DuckDB's constant folding: a cast of a column-free expression becomes
    # the literal it evaluates to while the plan is optimized, so no batch
    # parses it again. The rows are the unfolded plan's.
    var values = List[Float64]()
    for i in range(40):
        values.append(Float64(i) * 0.25)
    var source = DataFrame([Series("f", Column[Float64](values^))])
    var money = DataType.decimal(7, 2)
    var priced = source.lazy().with_columns([col("f").cast(money).alias("d")])
    var cheap = priced.filter(col("d") > lit("2.50").cast(money))
    var folded = cheap._optimized()
    assert_equal(_casts(folded) <= 1, True)
    var literal = False
    for node in folded._nodes:
        for expr in node.exprs:
            for n in expr._nodes:
                literal = literal or (
                    n.op == LIT_INT
                    and len(n.dtypes) == 1
                    and n.dtypes[0].value() == money
                )
    assert_true(literal, "the decimal constant was not folded")
    var rows = cheap.collect()
    assert_equal(rows.height(), 29)
    var ints = source.lazy().filter(col("f") > lit("7").cast(DataType.FLOAT64))
    assert_equal(_casts(ints._optimized()), 0)
    assert_equal(ints.collect().height(), 11)
    # A cast that cannot succeed stays in the plan and raises when run.
    var bad = source.lazy().filter(
        col("f") > lit("not a number").cast(DataType.FLOAT64)
    )
    assert_equal(_casts(bad._optimized()), 1)
    var raised = False
    try:
        _ = bad.collect()
    except:
        raised = True
    assert_true(raised, "an invalid constant cast must still raise")
