"""Typed mode6 region bridges require the actual original source owners."""
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.accelerator.producer_phases import prepare_original, bind_region


def main() raises:
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
