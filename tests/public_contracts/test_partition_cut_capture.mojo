"""Original public CUT/QCUT certificates, without input evaluation."""
from std.testing import assert_equal, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit, when
from dataframe.binding import bind
from dataframe.accelerator.partition_cuts import capture, validate


def test_partition_cut_capture_keeps_original_controls_and_eager_state() raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([0, 2])),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    var probabilities: List[Float64] = [0.5]
    var labels: List[String] = ["lower", "upper"]
    var state = (
        (col("x") + lit(Int64(1)))
        .qcut(
            probabilities,
            labels=labels,
            left_closed=True,
            include_breaks=True,
            allow_duplicates=True,
        )
        .over("k")
    )
    var bound = bind(state, source._columns)
    var at = len(state._nodes) - 1
    var saved = capture(bound, source, at)
    assert_equal(saved.over_node, at)
    assert_equal(saved.key_set, "k")
    assert_equal(len(saved.specifications), 1)
    assert_equal(saved.specifications[0], UInt64(0x3FE0000000000000))
    assert_equal(saved.labels, labels)
    assert_equal(saved.qcut, True)
    assert_equal(saved.left_closed, True)
    assert_equal(saved.include_breaks, True)
    assert_equal(saved.allow_duplicates, True)
    assert_equal(saved.keys[0].dtype, DataType.INT32)
    assert_false(saved.keys[0].string_hash)
    validate(saved, state, source)
    # Changing caller lists cannot mutate captured controls or the original AST.
    probabilities[0] = 0.25
    labels[0] = "changed"
    assert_equal(saved.specifications[0], UInt64(0x3FE0000000000000))
    assert_equal(saved.labels[0], "lower")
    validate(saved, state, source)
    with assert_raises(contains="original expression metadata changed"):
        validate(saved, col("x").qcut([0.25]).over("k"), source)
    with assert_raises(contains="OVER occurrence"):
        _ = capture(bound, source, 0)
    # Capture retains the OVER under a false WHEN. Admission must schedule this
    # state eagerly; its operand faults cannot be suppressed by the consumer.
    var eager_state = col("x").qcut([0.5]).over("k")
    var eager = when(lit(False)).then(eager_state.copy()).otherwise(lit("safe"))
    var eager_bound = bind(eager, source._columns)
    for node in range(len(eager._nodes)):
        if eager._nodes[node].op == state._nodes[at].op:
            var captured = capture(eager_bound, source, node)
            assert_equal(captured.specifications[0], UInt64(0x3FE0000000000000))
            validate(captured, eager, source)
