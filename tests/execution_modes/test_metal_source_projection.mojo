"""Dictionary staging must see only source fields the logical query reads."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col, lit
from dataframe.accelerator.source_projection import project_sources
from dataframe.accelerator.dictionaries import (
    DictionaryCodes,
    lower_dictionary_operations,
)
from tests.support.coverage_compare import assert_frame
from tests.execution_modes.test_backend_metal import require_metal


def source() raises -> DataFrame:
    return DataFrame(
        [
            Series("key", StringColumn(["b", "a", "b", ""])),
            Series("value", Column[Int64]([3, 7, -1, 5])),
            Series(
                "unneeded",
                StringColumn(["unused1", "unused2", "unused3", "unused4"]),
            ),
        ]
    )


def test_metal_source_projection_dictionary_excludes_unused_payload() raises:
    var frame = source()
    var query = (
        frame.lazy()
        .group_by("key")
        .agg([col("value").sum().alias("sum"), col("value").max().alias("max")])
    )
    var projected = project_sources(query)
    assert_equal(projected._frames[0].width(), 2)
    assert_equal(frame.width(), 3)
    var dictionary = DictionaryCodes(projected)
    assert_equal(len(dictionary.dtype.dictionary()[]), 3)
    assert_true("unused1" not in dictionary.lookup)
    assert_frame(projected.collect(engine="cpu"), query.collect(engine="cpu"))


def test_metal_source_projection_shared_frame_scans_and_height() raises:
    var frame = source()
    var left = frame.lazy().select(List[String](["key", "value"]))
    var right = frame.lazy().select(List[String](["key", "unneeded"]))
    var query = left.join(right, on="key")
    var joined = project_sources(query)
    assert_equal(joined._frames[0].width(), 2)
    assert_equal(joined._frames[1].width(), 2)
    assert_frame(joined.collect(engine="cpu"), query.collect(engine="cpu"))
    var numeric = frame.lazy().select(col("value"))
    var projected = project_sources(numeric)
    assert_equal(projected._frames[0].height(), frame.height())
    assert_equal(projected._frames[0].width(), 1)
    assert_frame(projected.collect(engine="cpu"), numeric.collect(engine="cpu"))


def test_metal_source_projection_through_dictionary_restore() raises:
    var frame = source()
    for index in range(100):
        frame = frame.with_column(
            Series("payload" + String(index), Column[Int64]([1, 2, 3, 4]))
        )
    var query = (
        frame.lazy()
        .filter(col("key").str().contains("b"))
        .select(col("value").sum().alias("sum"))
    )
    var lowered = lower_dictionary_operations(project_sources(query))
    var projected = project_sources(lowered[0])
    assert_equal(projected._frames[0].width(), 2)
    assert_equal(frame.width(), 103)
    assert_frame(projected.collect(engine="cpu"), query.collect(engine="cpu"))


def test_metal_source_projection_keeps_computed_observations() raises:
    var frame = source()
    var query = (
        frame.lazy()
        .select_exprs(
            [col("value"), (col("value") * lit(Int64.MAX)).alias("checked")]
        )
        .select(col("value"))
    )
    var projected = project_sources(query)
    # The unused checked expression still raises on CPU/GPU; pruning an
    # identity restoration must never erase this earlier observation.
    assert_equal(len(projected._nodes[1].exprs), 2)
    var raised = False
    try:
        _ = projected.collect(engine="cpu", optimize=False)
    except:
        raised = True
    assert_true(raised)


# requires: metal
def test_metal_source_projection_public_results_and_disabled_optimizer() raises:
    var frame = source()
    var query = (
        frame.lazy()
        .group_by("key")
        .agg([col("value").sum().alias("sum"), col("value").max().alias("max")])
    )
    var expected = query.collect(engine="cpu")
    assert_frame(
        query.collect(accelerator=require_metal()).sort("key"),
        expected.sort("key"),
    )
    assert_frame(
        query.collect(accelerator=require_metal(), optimize=False).sort("key"),
        expected.sort("key"),
    )
