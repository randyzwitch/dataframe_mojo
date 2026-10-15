"""Private ranked operand resource geometry is metadata-only and wide-safe."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Expr, Series, col
from dataframe.binding import bind
from dataframe.accelerator.rows import RowPlan
from dataframe.accelerator.ordered_rows import occurrences
from dataframe.accelerator.gpu.runtime import _preflight
from dataframe.accelerator.gpu.ranked_rows import estimate, evaluate
from dataframe.accelerator.gpu.residency import Scope, Table


def ranked_operand(expression: Expr, source: DataFrame) raises -> RowPlan:
    var bound = bind(expression, source._columns)
    var cloned = occurrences(bound, 128)
    var plan = _preflight(source.clear().lazy().select(cloned[0].expr))
    var ranks = cloned[1].copy()
    while len(ranks) < len(plan.literals):
        ranks.append(UInt32(0))
    plan.ordered_ranks = ArcPointer(ranks^)
    return plan^


def test_private_ranked_rows_resources_zero_rows_descriptors_and_wide_cells() raises:
    var source = DataFrame([Series.full_null("x", DataType.INT64, 0)])
    var plan = ranked_operand(col("x").cast(DataType.INT8), source)
    var empty = estimate(plan, 0)
    assert_equal(empty.peak_bytes, 0)
    assert_equal(empty.upload_bytes, 0)
    assert_equal(empty.download_bytes, 0)
    assert_equal(empty.kernel_launches, 0)
    assert_equal(empty.synchronizations, 0)
    var unranked = ranked_operand(col("x").cast(DataType.INT8), source)
    var wrong = unranked.ordered_ranks[].copy()
    wrong[unranked.steps[0].nodes - 1] = 0
    unranked.ordered_ranks = ArcPointer(wrong^)
    with assert_raises(contains="unranked executable helper"):
        _ = estimate(unranked, 7)
    var facts = estimate(plan, 7)
    assert_equal(facts.matrix_bytes, 21 * 7 * plan.slots)
    assert_equal(facts.descriptor_bytes, 48 * plan.source.width())
    assert_equal(facts.control_bytes, 64 + 4 * facts.partial_count)
    assert_equal(
        facts.peak_bytes,
        facts.matrix_bytes
        + facts.program_bytes
        + facts.control_bytes
        + facts.descriptor_bytes,
    )
    assert_equal(
        facts.upload_bytes, facts.program_bytes // 2 + 24 * plan.source.width()
    )
    assert_equal(facts.download_bytes, 32)
    assert_equal(facts.kernel_launches, 4)
    assert_equal(facts.synchronizations, 3)
    # Physical cell/plane offsets must not truncate at UInt32.MAX. This uses
    # no row storage, scope allocation or device context.
    plan.slots = 256
    while len(plan.slot_dtypes) < plan.slots:
        plan.slot_dtypes.append(DataType.INT64)
    var large = estimate(plan, Int(Int32.MAX))
    assert_equal(large.matrix_bytes, 21 * Int(Int32.MAX) * 256)
    assert_true(Int(Int32.MAX) * 256 > Int(UInt32.MAX))
    var scope = Scope(999999, 0)
    with assert_raises(contains="nonempty same-query device source"):
        _ = evaluate(plan, Table.empty(source.clear()), scope)
    assert_true(not scope.initialized())
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 0)
    _ = scope^
