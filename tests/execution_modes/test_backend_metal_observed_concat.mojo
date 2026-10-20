"""Full-query sparse text concat, including metadata and downstream consumers."""
# contract-group: execution.backends
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    col,
    concat_str,
    lit,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.dictionaries import lower_dictionary_operations
from dataframe.accelerator.observed_concat import has_markers


# requires: metal
def test_native_metal_observed_concat_sparse_pairs_and_consumers() raises:
    _ = require_metal()
    var left = List[String]()
    var right = List[String]()
    var present = List[Bool]()
    var ids = List[Int64]()
    for index in range(1100):
        left.append("left" + String(index))
        right.append("right" + String(index))
        present.append(True)
        ids.append(Int64(index))
    # Distinct input pairs can have identical concatenated output labels.
    left[0] = "a"
    right[0] = "bc"
    left[1] = "ab"
    right[1] = "c"
    left[2] = "é"
    right[2] = "🙂"
    present[3] = False
    var frame = DataFrame(
        [
            Series("left", StringColumn(left^, present^)),
            Series("right", StringColumn(right^)),
            Series("id", Column[Int64](ids^)),
        ]
    )
    var query = frame.lazy().with_columns(
        [concat_str([col("left"), col("right")]).alias("joined")]
    )
    assert_true(has_markers(lower_dictionary_operations(query)[0]))
    var runtime = MetalRuntime()
    var answer = query.collect(accelerator=runtime)
    assert_true(answer.equals(query.collect(engine="cpu")))
    assert_equal(answer.item(0, "joined").string(), "abc")
    assert_equal(answer.item(1, "joined").string(), "abc")
    assert_equal(answer.item(2, "joined").string(), "é🙂")
    assert_true(answer.item(3, "joined").is_null())
    print("PASS observed concat full rows, duplicate labels, Unicode and nulls")
    var selected = (
        frame.lazy()
        .filter(col("id") < lit(Int64(6)))
        .with_columns(
            [
                concat_str(
                    [col("left").fill_null(""), col("right")], separator=", "
                ).alias("joined")
            ]
        )
    )
    var selected_result = selected.collect(accelerator=runtime)
    assert_true(selected_result.equals(selected.collect(engine="cpu")))
    assert_equal(selected_result.height(), 6)
    assert_equal(selected_result.item(3, "joined").string(), ", right3")
    print("PASS observed concat after selective filter and fill_null")
    var chained = query.with_columns(
        [concat_str([lit("["), col("joined"), lit("]")]).alias("wrapped")]
    ).sort("wrapped")
    assert_true(
        chained.collect(accelerator=runtime).equals(
            chained.collect(engine="cpu")
        )
    )
    var grouped = query.group_by("joined", maintain_order=True).agg(
        [col("id").count().alias("n")]
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var filtered = query.filter(col("joined") == lit("abc"))
    assert_equal(filtered.collect(accelerator=runtime).height(), 2)
    print("PASS dynamic concat chains, sorting, grouping and literal filtering")
    var empty = (
        frame.lazy()
        .filter(col("id") < lit(Int64(0)))
        .select(concat_str([col("left"), col("right")]).alias("joined"))
    )
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var null_only = (
        frame.lazy()
        .filter(col("id") == lit(Int64(3)))
        .select(concat_str([col("left"), col("right")]).alias("joined"))
    )
    var null_answer = null_only.collect(accelerator=runtime)
    assert_equal(null_answer.height(), 1)
    assert_true(null_answer.item(0, "joined").is_null())
    var sliced = (
        frame.lazy()
        .slice(2, 7)
        .select(concat_str([col("left"), col("right")]).alias("joined"))
    )
    assert_true(
        sliced.collect(accelerator=runtime).equals(sliced.collect(engine="cpu"))
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    assert_true(query.collect(accelerator=runtime).equals(answer))
    print("PASS observed concat empty producer, budget rejection and recovery")
