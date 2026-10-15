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


def test_partition_nested_scope_cache_entry_snapshots() raises:
    from dataframe.accelerator.gpu.partition_state import Context
    from dataframe.accelerator.cuts import lower_cuts
    from dataframe.accelerator.partition_cuts import is_marker, marker_capture
    from dataframe.lazy import LazyFrame

    var context = Context()
    context.snapshot("projection0", "outer", "outer_state")
    assert_equal(context.dispatch("projection0", "outer", "inner_state", True, 4), True)
    # The outer scope selected fallback ids before its inner same-key scope
    # created a contiguous layout. It cannot observe that later cache change.
    assert_equal(context.dispatch("projection0", "outer", "outer_state", False, 4), False)
    assert_equal(context.dispatch("projection0", "outer", "later_state", False, 4), True)
    assert_equal(context.dispatch("projection1", "outer", "new_scope", False, 4), False)
    assert_equal(context.dispatch("projection0", "small", "single_lane", True, 1), False)
    var source = DataFrame([
        Series("x", Column[Int64]([0, 2])),
        Series("k", Column[Int32]([0, 0])),
    ])
    var state = (col("x") + lit(Int64(1))).over("k").qcut([0.5]).over("k")
    var original = bind(state, source._columns)
    var saved = capture(original, source, len(state._nodes) - 1)
    assert_equal(saved.nested, True)
    assert_equal(saved.numeric_state, False)
    var lowered = lower_cuts(source.lazy().select(state))
    var markers = 0
    var entering = 0
    var numeric = 0
    for node in lowered._nodes:
        if len(node.names2) and node.names2[0] == "metal_partition_cut_operand":
            markers += 1
            entering += Int(node.flags[0])
            numeric += Int(node.flags[1])
    assert_equal(markers, 3)
    assert_equal(entering, 1)
    assert_equal(numeric, 1)


def test_partition_lookup_reservation_retained_live_witness() raises:
    from dataframe.accelerator.gpu.residency import Scope
    from dataframe.accelerator.gpu.ranked_rows import _Reservation
    from dataframe.accelerator.gpu.partition_runtime import _finish_lookup

    # No context can be initialized here, even on a machine without Metal.
    var scope = Scope(999999, budget=31)
    var other = _Reservation(scope, 7)
    var lookup = _Reservation(scope, 24)
    assert_equal(scope.info()[0], 31)
    with assert_raises(contains="exceeds memory budget before submission"):
        _ = scope.check(1)
    # Reading the exact owner after the simulated work is the lifetime witness;
    # a lexical local alone would allow Mojo's ASAP destruction before work.
    _finish_lookup(lookup, 24)
    assert_equal(lookup.bytes, 0)
    assert_equal(scope.info()[0], 7)
    assert_equal(scope.info()[1], 31)
    _ = lookup^
    _finish_lookup(other, 7)
    _ = other^
    assert_equal(scope.info()[0], 0)
    assert_false(scope.initialized())
    var empty = _Reservation(scope, 0)
    _finish_lookup(empty, 0)
    _ = empty^
    assert_equal(scope.info()[0], 0)
