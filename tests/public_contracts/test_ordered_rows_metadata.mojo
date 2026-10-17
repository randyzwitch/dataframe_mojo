"""Ordered terminal admission and rank/control traffic are context-free."""
# contract-group: public.apple_parity
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    col,
    scan_csv,
)
from dataframe.accelerator.gpu.runtime import _preflight, estimate_resources
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.calibration import estimate
from dataframe.accelerator.sources import prepare_file_sources
from dataframe.lazy import SCAN_FRAME
from tests.tooling.contract_runner import temporary_directory, write_text


def test_ordered_rows_rank_resources_calibration_and_file_provenance() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([1, 2])),
            Series("base", Column[Int64]([2, 3])),
            Series("exp", Column[Int64]([0, 1])),
        ]
    )
    var query = frame.lazy().select(
        (
            col("x").cast(DataType.INT8).cast(DataType.INT64)
            + col("base").pow(col("exp"))
        ).alias("v")
    )
    var plan = _preflight(query, ordered_source=True)
    var legacy = _preflight(query)
    assert_equal(plan.ordered_mode, 2)
    assert_equal(legacy.ordered_mode, 0)
    var ranks = len(plan.ordered_ranks[])
    assert_true(ranks > 0)
    var program = Program(plan)
    assert_equal(
        Int(program.ordered_ranks[].unsafe_ptr()),
        Int(plan.ordered_ranks[].unsafe_ptr()),
    )
    var ranked = estimate_resources(plan)
    var ordinary = estimate_resources(legacy)
    assert_equal(ranked.peak_bytes - ordinary.peak_bytes, 8 * ranks + 24)
    # The plain plan observes its checked step itself (accelerator/faults.mojo):
    # one launch, a 4-byte zero upload, one wait and a 4-byte read.
    assert_equal(ranked.upload_bytes - ordinary.upload_bytes, 4 * ranks - 4)
    assert_equal(ranked.download_bytes - ordinary.download_bytes, 8 - 4)
    assert_equal(ranked.kernel_launches - ordinary.kernel_launches, 2 - 1)
    assert_equal(ranked.synchronizations - ordinary.synchronizations, 1 - 1)
    var facts = estimate(query)
    assert_true(facts.family.find("ordered-errors-v1=2") >= 0)
    assert_equal(facts.peak_bytes, ranked.peak_bytes)
    assert_equal(facts.accelerator[8], Float64(ranked.kernel_launches))
    assert_equal(facts.accelerator[9], Float64(ranked.synchronizations))
    assert_equal(facts.accelerator[10], Float64(ranked.upload_bytes))
    assert_equal(facts.accelerator[11], Float64(ranked.download_bytes))
    # Schema mode reads the real header but must not decode invalid data rows.
    # Replacement scans retain negative provenance even with a raw flag.
    var work = temporary_directory("/tmp/mojo-ordered-rows-metadata")
    var path = work + "/values.csv"
    write_text(path, "x,base,exp\nnot-an-integer,also-invalid,invalid\n")
    var file = scan_csv(
        path,
        CsvSchema(
            [
                CsvField("x", DataType.INT64),
                CsvField("base", DataType.INT64),
                CsvField("exp", DataType.INT64),
            ]
        ),
    )
    var staged = prepare_file_sources(
        file.select(
            (
                col("x").cast(DataType.INT8).cast(DataType.INT64)
                + col("base").pow(col("exp"))
            ).alias("v")
        ),
        empty=True,
    )
    assert_equal(staged.failure, "")
    var found = 0
    for operation in staged.query._nodes:
        if operation.kind == SCAN_FRAME:
            assert_equal(operation.names2, ["metal_staged_file_source"])
            found += 1
    assert_equal(found, 1)
    assert_equal(_preflight(staged.query, ordered_source=True).ordered_mode, 0)
