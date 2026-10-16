"""Original strict-cast producer proofs retain AST, prefix and source owners."""
# contract-group: execution.apple_parity
# contract-group: execution.strict_cast_producers
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.accelerator.producer_phases import (
    prepare_original,
    validate_prefix,
    bind_region,
    OBSERVED_STRICT_CAST,
)


def test_strict_cast_original_prefix_owner_and_bridge() raises:
    var source = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var original = (
        source.lazy()
        .with_columns(col("x").cast(DataType.INT8).alias("y"))
        .filter(col("y") > 0)
        .head(1)
    )
    var certified = prepare_original(original, False, True)
    var at = -1
    for index in range(len(certified._nodes)):
        if certified._nodes[index].producer_origin:
            at = index
    assert_true(at > 0)
    assert_equal(
        certified._nodes[at].producer_origin.value()[].phase,
        OBSERVED_STRICT_CAST,
    )
    validate_prefix(certified, at)
    var proof = bind_region(certified, at, source, -1)
    assert_equal(proof[].source_task, -1)
    var wrong = certified.copy()
    wrong._frames[0] = DataFrame([Series("x", Column[Int64]([1, 2]))])
    with assert_raises(contains="lineage changed"):
        validate_prefix(wrong, at)
    wrong = certified.copy()
    wrong._nodes[at].exprs[0] = col("x").cast(DataType.INT16).alias("y")
    with assert_raises(contains="AST or logical metadata changed"):
        validate_prefix(wrong, at)
    with assert_raises(contains="Only original entry"):
        _ = prepare_original(original, False, False)
    print("PASS strict cast original prefix/source/AST and region proof")
