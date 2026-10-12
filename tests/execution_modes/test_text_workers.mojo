"""Dispatch-sized typed text work has independent rows and deterministic errors."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.parallel import worker_count, _helpers
from dataframe.string_view import StringViewBuilder
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES

comptime ROWS = 262145


def _source(view: Bool) raises -> DataFrame:
    var pool: List[String] = [
        "7",
        "3",
        "1.5",
        "bad",
        "255",
        "0000000000000000000000000007",
        "",
    ]
    var words: List[String] = [
        "",
        "é",
        "abcde",
        "externalabcdefghijklmnop",
        "😀",
    ]
    var texts = List[String]()
    var labels = List[String]()
    var valid = List[Bool]()
    var keep = List[Bool]()
    var builder = StringViewBuilder()
    var label_builder = StringViewBuilder()
    for row in range(ROWS + 7):
        var logical = row - 7
        var present = row >= 7 and logical % 11 != 3
        var text = pool[max(0, logical) % len(pool)]
        var label = words[max(0, logical) % len(words)]
        texts.append(text)
        labels.append(label)
        valid.append(present)
        keep.append(row >= 7 and logical % 7 < 2)
        if present:
            builder.append(StringSlice(text))
            label_builder.append(StringSlice(label))
        else:
            builder.append_null()
            label_builder.append_null()
    var s = Series("s", StringColumn(builder^.finish())) if view else Series(
        "s", Column[String](texts^, valid.copy())
    )
    var label = Series(
        "label", StringColumn(label_builder^.finish())
    ) if view else Series("label", Column[String](labels^, valid.copy()))
    return DataFrame([s^, label^, Series("keep", Column[Bool](keep^))]).slice(
        7, ROWS
    )


def _expressions() -> List[Expr]:
    var result = List[Expr]()
    comptime for i in range(len(INTEGER_DTYPES)):
        result.append(
            col("s")
            .cast(DataType.of(INTEGER_DTYPES[i]), strict=False)
            .alias(String(INTEGER_DTYPES[i]))
        )
    comptime for i in range(len(FLOAT_DTYPES)):
        result.append(
            col("s")
            .cast(DataType.of(FLOAT_DTYPES[i]), strict=False)
            .alias(String(FLOAT_DTYPES[i]))
        )
    result.append(
        when(col("keep"))
        .then(col("s").cast(DataType.FLOAT64))
        .otherwise(lit(Float64(-1)))
        .alias("masked")
    )
    result.append(col("label").eq(lit("é")).alias("equal"))
    result.append(col("label").str().contains("é").alias("contains"))
    result.append(col("label").str().len_chars().alias("characters"))
    result.append(col("label").is_in([lit("é"), lit("abcde")]).alias("member"))
    result.append(col("label").str().like("%abc%").alias("like"))
    result.append(col("label").str().slice(1, 2).alias("slice"))
    result.append((lit("é") > col("label")).alias("reverse_less"))
    return result^


def _assert_result(actual: DataFrame) raises:
    assert_equal(actual.height(), ROWS)
    assert_equal(
        actual.columns(),
        [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
            "masked",
            "equal",
            "contains",
            "characters",
            "member",
            "like",
            "slice",
            "reverse_less",
        ],
    )
    var numbers: List[Float64] = [7, 3, 1.5, 0, 255, 7, 0]
    var lengths: List[Int64] = [0, 1, 5, 24, 1]
    var slices: List[String] = ["", "", "bc", "xt", ""]
    var less: List[Bool] = [True, False, True, True, False]
    for row in range(ROWS):
        var present = row % 11 != 3
        var token = row % 7
        comptime for i in range(len(INTEGER_DTYPES)):
            comptime D = INTEGER_DTYPES[i]
            var cell = actual.item(row, String(D))
            var parsed = (
                present
                and token != 2
                and token != 3
                and token != 6
                and not (D == DType.int8 and token == 4)
            )
            assert_equal(cell.is_null(), not parsed)
            assert_true(actual.column(String(D)).dtype() == DataType.of(D))
            if parsed:
                assert_equal(
                    cell.numeric[D]().cast[DType.int128](),
                    Int128(numbers[token]),
                )
        comptime for i in range(len(FLOAT_DTYPES)):
            comptime D = FLOAT_DTYPES[i]
            var cell = actual.item(row, String(D))
            var parsed = present and token != 3 and token != 6
            assert_equal(cell.is_null(), not parsed)
            assert_true(actual.column(String(D)).dtype() == DataType.of(D))
            if parsed:
                assert_equal(cell.numeric[D](), numbers[token].cast[D]())
        var masked = actual.item(row, "masked")
        assert_equal(masked.is_null(), token < 2 and not present)
        if not masked.is_null():
            assert_equal(
                masked.float64(), numbers[token] if token < 2 else Float64(-1)
            )
        for name in ["equal", "contains"]:
            assert_equal(actual.item(row, name).is_null(), not present)
            if present:
                assert_equal(actual.item(row, name).bool(), row % 5 == 1)
        assert_true(actual.column("characters").dtype() == DataType.INT64)
        assert_equal(actual.item(row, "characters").is_null(), not present)
        if present:
            assert_equal(
                actual.item(row, "characters").int64(), lengths[row % 5]
            )
        for name in ["member", "like", "slice", "reverse_less"]:
            assert_equal(actual.item(row, name).is_null(), not present)
        if present:
            assert_equal(
                actual.item(row, "member").bool(), row % 5 == 1 or row % 5 == 2
            )
            assert_equal(
                actual.item(row, "like").bool(), row % 5 == 2 or row % 5 == 3
            )
            assert_equal(actual.item(row, "slice").string(), slices[row % 5])
            assert_equal(actual.item(row, "reverse_less").bool(), less[row % 5])


def test_dispatch_sized_text_casts_predicates_and_masked_errors_match_literal_rows_on_every_worker() raises:
    for view in [False, True]:
        var source = _source(view)
        for workers in [1, 2, 4]:
            var setting = WorkerSetting(workers)
            assert_equal(worker_count(ROWS), workers)
            _assert_result(source.select_exprs(_expressions()))
            assert_equal(_helpers().load(), Int64(0))
            _ = setting^


def test_parallel_strict_text_casts_report_earliest_logical_row_and_allow_retry() raises:
    var values = List[String](length=ROWS + 7, fill="7")
    for row in [65535, 65536, 131072, 196608]:
        values[row + 7] = "bad" + String(row)
    var source = DataFrame([Series("s", Column[String](values^))]).slice(
        7, ROWS
    )
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        with assert_raises(contains="failed at row 65535 for value 'bad65535'"):
            _ = source.select(col("s").cast(DataType.INT64))
        assert_equal(_helpers().load(), Int64(0))
        var retry = source.select(col("s").cast(DataType.INT64, strict=False))
        assert_equal(retry.height(), ROWS)
        for row in range(ROWS):
            var bad = (
                row == 65535 or row == 65536 or row == 131072 or row == 196608
            )
            assert_equal(retry.item(row, "s").is_null(), bad)
            if not bad:
                assert_equal(retry.item(row, "s").int64(), Int64(7))
        _ = setting^
