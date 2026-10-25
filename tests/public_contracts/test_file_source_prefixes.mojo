"""CPU planning observes missing file schemas before value errors."""
# contract-group: public.accelerator_staging
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
from dataframe.decimal import pow10
from dataframe.lazy import LazyFrame


def _missing() raises -> LazyFrame:
    return scan_csv(
        "/tmp/mojo-source-prefixes-do-not-open/missing.csv",
        CsvSchema([CsvField("right", DataType.INT64)]),
    )


def _overflow() raises -> LazyFrame:
    var amount = Int128(6) * pow10(37)
    return (
        DataFrame(
            [
                Series("v", Column[Int128]([amount, amount])).with_dtype(
                    DataType.decimal(38, 0)
                ),
                Series("k", Column[Int64]([0, 0])),
            ]
        )
        .lazy()
        .select_exprs([col("v").sum().over("k").alias("total"), col("k")])
    )


def _error(query: LazyFrame, optimize: Bool = True) raises -> String:
    try:
        _ = query.collect(engine="cpu", optimize=optimize, streaming=False)
    except error:
        return String(error)
    raise Error("CPU fixture unexpectedly succeeded")


def _planning_error(query: LazyFrame) raises -> String:
    try:
        _ = query._optimized()
    except error:
        return String(error)
    raise Error("CPU planning fixture unexpectedly succeeded")


def test_file_source_prefixes_cpu_projection_metadata_precedes_data_errors() raises:
    var raw = _overflow().join(_missing(), how="cross")
    # A plain cross join needs no schema IO in these CPU optimizer passes:
    # its left producer reaches value evaluation before the missing right file.
    _ = raw._optimized()
    assert_true("decimal precision" in _error(raw))
    # A projection above that same join asks the CPU optimizer which input
    # supplies the selected names. The missing right schema is now observed
    # during planning, before any decimal arithmetic.
    var projected = raw.select(col("total"))
    var planning = _planning_error(projected)
    assert_true("decimal precision" not in planning)
    assert_equal(_error(projected), planning)
    # Without the optimizer, value evaluation reaches the overflow first.
    assert_true("decimal precision" in _error(projected, optimize=False))


def test_file_source_prefixes_cpu_predicate_metadata_precedes_data_errors() raises:
    var missing = scan_csv(
        "/tmp/mojo-source-prefixes-do-not-open/missing.csv",
        CsvSchema([CsvField("k", DataType.INT64)]),
    )
    var query = _overflow().join(missing, on="k").filter(col("total") > 0)
    # CPU _push_predicates calls _passes -> _columns_of(left/right) for this
    # join filter, before _push_projections.
    var planning = _planning_error(query)
    assert_true("decimal precision" not in planning)
    assert_equal(_error(query), planning)
    assert_true("decimal precision" in _error(query, optimize=False))
