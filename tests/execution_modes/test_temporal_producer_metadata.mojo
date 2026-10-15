"""CPU certificates for original resident temporal producer phases."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.lazy import SELECT
from dataframe.accelerator.producer_phases import (
    prepare_original,
    validate_prefix,
    OBSERVED_IANA,
    bind_region,
)


def test_temporal_producer_original_owner_prefix_phase_and_ast() raises:
    var dtype = DataType.datetime("us")
    var source = DataFrame(
        [Series("t", Column[Int64]([0, 1])).with_dtype(dtype)]
    )
    var original = (
        source.lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
        .head(1)
    )
    var certified = prepare_original(original, False, True)
    var producer = -1
    for at in range(len(certified._nodes)):
        if certified._nodes[at].producer_origin:
            producer = at
    assert_true(producer > 0)
    assert_equal(
        certified._nodes[producer].producer_origin.value()[].phase,
        OBSERVED_IANA,
    )
    validate_prefix(certified, producer)
    # Physical owners are part of the proof even when schema/length match.
    var wrong_source = certified.copy()
    wrong_source._frames[0] = DataFrame(
        [Series("t", Column[Int64]([0, 1])).with_dtype(dtype)]
    )
    with assert_raises(contains="lineage changed"):
        validate_prefix(wrong_source, producer)
    var wrong_kind = certified.copy()
    wrong_kind._nodes[producer].kind = SELECT
    with assert_raises(contains="geometry changed"):
        validate_prefix(wrong_kind, producer)
    var wrong_expression = certified.copy()
    wrong_expression._nodes[producer].exprs[0] = (
        col("t").dt().replace_time_zone("UTC").alias("z")
    )
    with assert_raises(contains="AST or logical metadata changed"):
        validate_prefix(wrong_expression, producer)
    with assert_raises(contains="Only original entry"):
        _ = prepare_original(original, False, False)
    print(
        "PASS original temporal producer owner, prefix, phase and AST provenance"
    )


def test_temporal_producer_bridge_immutable_owner_and_logical_mapping() raises:
    var dtype = DataType.datetime("us")
    var source = DataFrame(
        [Series("t", Column[Int64]([0, 1])).with_dtype(dtype)]
    )
    var query = (
        source.lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
    )
    var certified = prepare_original(query, False, True)
    var producer = -1
    for at in range(len(certified._nodes)):
        if certified._nodes[at].producer_origin:
            producer = at
    assert_true(producer > 0)
    var proof = bind_region(certified, producer, source, -1)
    assert_equal(proof[].source_task, -1)
    var unrelated = DataFrame(
        [Series("t", Column[Int64]([0, 1])).with_dtype(dtype)]
    )
    with assert_raises(contains="immutable source changed"):
        _ = bind_region(certified, producer, unrelated, -1)
    with assert_raises(contains="logical input changed"):
        _ = bind_region(certified, producer, source.rename({"t": "other"}), -1)
    print("PASS typed temporal raw bridge owner and logical mapping checks")
