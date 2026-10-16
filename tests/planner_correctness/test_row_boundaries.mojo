"""Rewrite barriers preserve row identity, stable order and whole-input state."""
# contract-group: planner.row_rewrites
from std.testing import assert_equal
from dataframe import LazyFrame, col, lit
from tests.support.reference_model import ReferenceTable, generated_model
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model


def _check(plan: LazyFrame, expected: ReferenceTable) raises:
    for optimize in [False, True]:
        assert_model(
            plan.collect(optimize=optimize, streaming=False, engine="cpu"),
            expected,
        )


def test_dependent_replacements_survive_projection_and_drop_pushdown() raises:
    for n in [0, 1, 9, 65]:
        for validity in range(3):
            var model = generated_model(n, 3, validity)
            for layout in range(3):
                var source = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    source = chunk_layout(source)
                var plan = (
                    source.lazy()
                    .with_columns((col("i") + lit(Int64(3))).alias("temporary"))
                    .with_columns((col("temporary") * lit(Int64(2))).alias("i"))
                    .drop(["temporary"])
                    .filter(col("i") > 0)
                    .select(["s", "i"])
                )
                var expected = model.copy()
                for row in range(len(expected.rows)):
                    if expected.rows[row].iv:
                        expected.rows[row].i = (expected.rows[row].i + 3) * 2
                _check(plan, expected.filter_i_gt(0).project(["s", "i"]))
                # Collection never replaces the reusable source plan's cells.
                assert_model(source, model)


def test_stable_sort_and_clipped_slices_keep_filter_barriers() raises:
    var slices: List[Tuple[Int, Int]] = [
        (0, 0),
        (0, 1),
        (1, 7),
        (-7, 3),
        (-100, 8),
        (100, 8),
        (2, -1),
    ]
    for n in [0, 1, 9, 65]:
        for validity in range(3):
            var model = generated_model(n, 3, validity)
            for layout in range(3):
                var source = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    source = chunk_layout(source)
                _check(
                    source.lazy().sort("i").filter(col("i") > 0),
                    model.sort_i().filter_i_gt(0),
                )
                _check(
                    source.lazy().filter(col("i") > 0).sort("i"),
                    model.filter_i_gt(0).sort_i(),
                )
                for bounds in slices:
                    _check(
                        source.lazy()
                        .slice(bounds[0], bounds[1])
                        .filter(col("i") > 0),
                        model.slice(bounds[0], bounds[1]).filter_i_gt(0),
                    )
                    _check(
                        source.lazy().slice(bounds[0], bounds[1]).sort("i"),
                        model.slice(bounds[0], bounds[1]).sort_i(),
                    )
                    _check(
                        source.lazy()
                        .sort("i")
                        .slice(bounds[0], bounds[1])
                        .filter(col("i") > 0),
                        model.sort_i()
                        .slice(bounds[0], bounds[1])
                        .filter_i_gt(0),
                    )


def test_partition_and_prefix_dependencies_are_evaluated_before_later_filters() raises:
    for n in [0, 1, 9, 65]:
        for validity in range(3):
            var model = generated_model(n, 3, validity)
            for layout in range(3):
                var source = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    source = chunk_layout(source)
                var grouped = model.copy()
                for row in range(len(model.rows)):
                    var total = Int128(0)
                    for other in range(len(model.rows)):
                        var same = model.rows[row].sv == model.rows[other].sv
                        if same and model.rows[row].sv:
                            same = model.rows[row].s == model.rows[other].s
                        if same and model.rows[other].iv:
                            total += Int128(model.rows[other].i)
                    grouped.rows[row].i = Int64(total)
                    grouped.rows[row].iv = True
                _check(
                    source.lazy()
                    .select_exprs(
                        [col("i").sum().over("s").alias("i"), col("s")]
                    )
                    .filter(col("i") > 0)
                    .slice(1, 7),
                    grouped.project(["i", "s"]).filter_i_gt(0).slice(1, 7),
                )
                var filtered = model.filter_i_gt(-5)
                var prefix = filtered.copy()
                var total = Int128(0)
                for row in range(len(prefix.rows)):
                    if prefix.rows[row].iv:
                        total += Int128(prefix.rows[row].i)
                        prefix.rows[row].i = Int64(total)
                _check(
                    source.lazy()
                    .filter(col("i") > -5)
                    .with_columns(col("i").cum_sum().alias("i"))
                    .filter(col("i") > 0)
                    .select(["s", "i"]),
                    prefix.filter_i_gt(0).project(["s", "i"]),
                )
