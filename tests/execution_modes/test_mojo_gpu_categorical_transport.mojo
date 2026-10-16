"""Literal original categorical codes, dictionaries and decoded ROW semantics."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.accelerator.code_domains import same_dictionary
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _dtype(labels: List[String]) -> DataType:
    var dictionary = CategoricalDictionary()
    for label in labels:
        dictionary.append(label)
    return DataType.categorical(dictionary^)


def _cat(
    name: String, dtype: DataType, codes: List[UInt32], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[UInt32](codes.copy(), valid)).with_dtype(dtype)


def _codes(series: Series, expected: List[Int], labels: List[String]) raises:
    assert_true(series.dtype().is_categorical())
    assert_equal(len(series), len(expected))
    var dictionary = series.dtype().dictionary()
    assert_equal(len(dictionary[]), len(labels))
    for code in range(len(labels)):
        assert_equal(String(dictionary[].get(code)), labels[code])
    var flat = series.rechunk()
    ref codes = flat._data[Column[UInt32]]
    for row in range(len(expected)):
        assert_equal(codes._valid(row), expected[row] >= 0)
        if expected[row] >= 0:
            assert_equal(codes._get(row), UInt32(expected[row]))


def _integers(frame: DataFrame, name: String, expected: List[Int]) raises:
    for row in range(len(expected)):
        assert_equal(frame.item(row, name).is_null(), expected[row] < 0)
        if expected[row] >= 0:
            assert_equal(frame.item(row, name).int64(), Int64(expected[row]))


def _collect(query: LazyFrame, gpu: Bool) raises -> DataFrame:
    return query.collect(
        accelerator=MetalRuntime(), optimize=False
    ) if gpu else query.collect(engine="cpu", streaming=False, optimize=False)


# requires: metal
def test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over() raises:
    var labels: List[String] = ["dup", "dup", "z", "unused"]
    var dtype = _dtype(labels)
    var raw = DataFrame(
        [
            _cat(
                "k",
                dtype,
                [0, 1, 2, 0, 1, 0],
                [True, True, True, False, True, True],
            ),
            Series("v", Column[Int64]([1, 2, 4, 8, 16, 32])),
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    for layout in range(3):
        var source = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for gpu in [False, True]:
            var projected = _collect(
                source.lazy().select_exprs(
                    [col("k").alias("renamed"), col("v")]
                ),
                gpu,
            )
            _codes(projected.column("renamed"), [0, 1, 2, -1, 1, 0], labels)
            assert_true(
                same_dictionary(projected.column("renamed").dtype(), dtype)
            )
            var filtered = _collect(
                source.lazy()
                .filter(col("id") >= 1)
                .sort("v", descending=True)
                .select(col("k")),
                gpu,
            )
            _codes(filtered.column("k"), [0, 1, -1, 2, 1], labels)
            var sorted = _collect(source.lazy().sort("k").select(col("k")), gpu)
            _codes(sorted.column("k"), [0, 0, 1, 1, 2, -1], labels)
            var grouped = _collect(
                source.lazy()
                .group_by("k", maintain_order=True)
                .agg(col("v").sum()),
                gpu,
            )
            _codes(grouped.column("k"), [0, 1, 2, -1], labels)
            _integers(grouped, "v", [33, 18, 4, 8])
            var window = _collect(
                source.lazy().select_exprs(
                    [col("k"), col("v").sum().over("k").alias("sum")]
                ),
                gpu,
            )
            _codes(window.column("k"), [0, 1, 2, -1, 1, 0], labels)
            _integers(window, "sum", [33, 18, 4, 8, 18, 33])
            # Explicit ROW reads CAT as String. Equal duplicate labels become
            # one partition even though a bare categorical key keeps codes.
            var decoded = _collect(
                source.lazy()
                .with_columns(
                    when(lit(True))
                    .then(col("k"))
                    .otherwise(col("k"))
                    .alias("text")
                )
                .select(col("v").sum().over("text").alias("sum")),
                gpu,
            )
            _integers(decoded, "sum", [51, 51, 4, 8, 51, 51])


# requires: metal
def test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union() raises:
    var labels: List[String] = ["dup", "dup", "z", "unused"]
    var dtype = _dtype(labels)
    var left = offset_layout(
        DataFrame(
            [
                _cat("k", dtype, [0, 1], [True, True]),
                Series("l", Column[Int64]([0, 1])),
            ]
        )
    )
    var same = chunk_layout(
        DataFrame(
            [
                _cat("k", dtype, [1, 0], [True, True]),
                Series("r", Column[Int64]([10, 11])),
            ]
        )
    )
    var other = DataFrame(
        [
            _cat("k", _dtype(["dup", "new"]), [0, 1], [True, True]),
            Series("r", Column[Int64]([10, 11])),
        ]
    )
    for gpu in [False, True]:
        var exact = _collect(
            left.lazy().join(same.lazy(), on="k", how="left"), gpu
        )
        _codes(exact.column("k"), [0, 1], labels)
        _integers(exact, "r", [11, 10])
        var resident = _collect(
            left.lazy()
            .with_columns((col("l") + 1).alias("shift"))
            .select_exprs([col("k").alias("renamed"), col("l")])
            .join(
                same.lazy().with_columns((col("r") + 1).alias("next")),
                left_on=["renamed"],
                right_on=["k"],
                how="left",
            ),
            gpu,
        )
        _codes(resident.column("renamed"), [0, 1], labels)
        _integers(resident, "r", [11, 10])
        var union = _collect(
            left.lazy().join(other.lazy(), on="k", how="left"), gpu
        )
        _codes(union.column("k"), [0, 1], ["dup", "dup", "z", "unused", "new"])
        _integers(union, "r", [-1, 10])
        var strings = DataFrame(
            [Series("k", Column[String](["new", "dup", "new"]))]
        )
        var mixed = _collect(
            strings.lazy().join(same.lazy(), on="k", how="inner"), gpu
        )
        _codes(mixed.column("k"), [1, 1], ["new", "dup", "z", "unused"])


# requires: metal
def test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner() raises:
    var left_dtype = _dtype(["dup", "dup", "unused"])
    var payload_dtype = _dtype(["payload", "payload", "unused-payload"])
    var left = DataFrame(
        [
            _cat("k", left_dtype, [0, 1], [True, True]),
            Series("t", Column[Int64]([1, 1])),
        ]
    )
    var right = chunk_layout(
        DataFrame(
            [
                _cat("k", _dtype(["dup", "right-only"]), [0], [True]),
                Series("t", Column[Int64]([0])),
                _cat("payload", payload_dtype, [1], [True]),
                Series("r", Column[Int64]([10])),
            ]
        )
    )
    for gpu in [False, True]:
        var result = _collect(
            left.lazy().join_asof(right.lazy(), on="t", by=["k"]), gpu
        )
        _codes(result.column("k"), [0, 1], ["dup", "dup", "unused"])
        _codes(
            result.column("payload"),
            [-1, 1],
            ["payload", "payload", "unused-payload"],
        )
        assert_true(same_dictionary(result.column("k").dtype(), left_dtype))
        assert_true(
            same_dictionary(result.column("payload").dtype(), payload_dtype)
        )
        _integers(result, "r", [-1, 10])
        var resident = _collect(
            left.lazy()
            .with_columns((col("t") + 1).alias("left_tmp"))
            .join_asof(
                right.lazy().with_columns((col("r") + 1).alias("right_tmp")),
                on="t",
                by=["k"],
            ),
            gpu,
        )
        _codes(resident.column("k"), [0, 1], ["dup", "dup", "unused"])
        _codes(
            resident.column("payload"),
            [-1, 1],
            ["payload", "payload", "unused-payload"],
        )
        _integers(resident, "r", [-1, 10])


# requires: mojo_gpu
def test_mojo_gpu_categorical_dictionary_budget_owners() raises:
    from std.memory import ArcPointer
    from dataframe.accelerator.gpu.residency import Scope, CategoryLease, Table

    var scope = Scope(budget=64)
    assert_equal(scope.info()[0], 0)
    var first = Table(DataFrame(List[Series]()))
    first.category_leases.append(ArcPointer(CategoryLease(scope, 48)))
    assert_equal(scope.info()[0], 48)
    var shared = first.copy()
    assert_equal(scope.info()[0], 48)
    assert_equal(first.retained_bytes(), 48)
    assert_equal(shared.retained_bytes(), 48)
    _ = first^
    assert_equal(scope.info()[0], 48)
    _ = shared^
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 48)
    assert_false(scope.initialized())


# requires: metal
def test_mojo_gpu_categorical_mixed_join_control_readback_and_budget() raises:
    from std.memory import ArcPointer
    from std.testing import assert_raises
    from dataframe.accelerator.dictionaries import DictionaryCodes, encode_query
    from dataframe.accelerator.code_domains import TextQuery
    from dataframe.accelerator.gpu.dag_runtime import execute
    from dataframe.accelerator.gpu.runtime import capabilities
    from dataframe.accelerator.gpu.joins import METADATA_WORDS

    var left = DataFrame([Series("k", Column[String](["new", "dup", "new"]))])
    var right = DataFrame(
        [
            _cat("k", _dtype(["dup", "unused"]), [0], [True]),
            Series("r", Column[Int64]([10])),
        ]
    )
    var query = left.lazy().join(right.lazy(), on="k", how="inner")
    var dictionary = DictionaryCodes(query)
    var encoded = encode_query(query, dictionary, dynamic_text=True)
    var context = ArcPointer(TextQuery(dictionary.dtype))
    var measured = execute(
        encoded,
        0,
        -1,
        True,
        capabilities(),
        observation_only=True,
        text_context=context,
    )
    # One JOIN shape control, one D-word first-occurrence control, and output
    # errors only. No row value/code/validity readback occurs in this observer.
    var expected_download = (
        METADATA_WORDS * 8
        + len(dictionary.dtype.dictionary()[]) * 4
        + 1 * 2 * 4
    )
    assert_equal(
        Int(measured[1].item(0, "download_bytes").int64()), expected_download
    )
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    assert_true(peak > 0)
    _ = execute(
        encoded,
        0,
        peak,
        True,
        capabilities(),
        observation_only=True,
        text_context=context,
    )
    with assert_raises(contains="memory budget"):
        _ = execute(
            encoded,
            0,
            peak - 1,
            True,
            capabilities(),
            observation_only=True,
            text_context=context,
        )


# requires: metal
def test_mojo_gpu_coded_string_group_publication_drops_original_code_identity() raises:
    from std.memory import ArcPointer
    from dataframe.categorical import decode
    from dataframe.string_column import StringColumn, StringCodes

    var dtype = _dtype(["dup", "dup", "unused"])
    var categorical = _cat("k", dtype, [0, 1, 0], [True, True, True])
    var strings = decode(categorical)
    var original_codes: List[UInt32] = [0, 1, 0]
    strings._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(original_codes^), dtype)
    )
    var source = DataFrame([strings^, Series("v", Column[Int64]([1, 2, 4]))])
    var query = (
        source.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    for gpu in [False, True]:
        var grouped = _collect(query, gpu)
        assert_equal(grouped.height(), 2)
        assert_equal(grouped.column("k").dtype(), DataType.STRING)
        assert_equal(grouped.item(0, "k").string(), "dup")
        assert_equal(grouped.item(1, "k").string(), "dup")
        assert_false(Bool(grouped.column("k")._dictionary_codes()))
        assert_false(Bool(grouped.column("k")._field_metadata))
        _integers(grouped, "v", [5, 2])
        var regrouped = _collect(
            query.group_by("k", maintain_order=True).agg(col("v").sum()), gpu
        )
        assert_equal(regrouped.height(), 1)
        _integers(regrouped, "v", [7])
