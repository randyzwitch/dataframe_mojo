"""Dictionary staging must see only source fields the logical query reads."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe.accelerator.source_projection import project_sources
from dataframe.accelerator.dictionaries import DictionaryCodes
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
