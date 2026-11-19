"""Shared allocations do not make different typed row windows one source."""
# contract-group: planner.relational
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    concat,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.planning.graph import plan_canonical_frames, plan_merge_inputs
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.string import StringColumn
from dataframe.storage.string_view import StringViewBuilder
from tests.support.coverage_compare import assert_frame


def _assert_sources(left: DataFrame, right: DataFrame, same: Bool) raises:
    # Check slot import independently of the later shared-subplan pass.
    var imported = left.lazy()
    var other = right.lazy()
    var slots = plan_merge_inputs(imported._plan, other._plan)
    assert_equal(slots[0], 0 if same else 1)
    assert_equal(len(imported._plan._frames), 1 if same else 2)
    # Retain both slots so canonicalization must compare them itself.
    var separate = left.lazy()
    separate._plan._frames.append(right.copy())
    separate._plan._schemas.append(None)
    var canonical = plan_canonical_frames(separate._plan)
    assert_equal(canonical[0], 0)
    assert_equal(canonical[1], 0 if same else 1)


def _assert_collect(query: LazyFrame, expected: DataFrame) raises:
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                query.collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                    engine="cpu",
                ),
                expected,
            )


def test_lazy_concat_preserves_disjoint_source_slices() raises:
    var source = DataFrame([Series("k", Column[Int64]([1, 2, 3, 4]))])
    var left = source.slice(0, 2)
    var right = source.slice(2, 2)
    _assert_sources(left, right, False)
    assert_frame(concat([left.copy(), right.copy()]), source)
    _assert_collect(concat([left.lazy(), right.lazy()]), source)
    # Equal projections over distinct windows must not become one subplan.
    _assert_collect(
        concat(
            [
                left.lazy().select((col("k") * 2).alias("k")),
                right.lazy().select((col("k") * 2).alias("k")),
            ]
        ),
        DataFrame([Series("k", Column[Int64]([2, 4, 6, 8]))]),
    )


def test_source_windows_preserve_null_bits_and_both_string_layouts() raises:
    var valid: List[Bool] = [True, True, False, True, False, True]
    var text = StringColumn(["a", "one", "two", "three", "four", "five"], valid)
    var builder = StringViewBuilder(len(text))
    for row in range(len(text)):
        if valid[row]:
            builder.append(text._get(row))
        else:
            builder.append_null()
    var views = StringColumn(builder^.finish())
    assert_true(views._is_view_storage())
    assert_true(not text._is_view_storage())
    var source = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 2, 3, 4, 5], valid)),
            Series(
                "flag",
                BoolColumn([True, False, True, True, False, False], valid),
            ),
            Series("text", views^),
            Series("legacy", text^),
        ]
    )
    var left = source.slice(1, 2)
    var right = source.slice(3, 2)
    _assert_sources(left, right, False)
    _assert_collect(concat([left.lazy(), right.lazy()]), source.slice(1, 4))
    # Test each storage layout without an integer column masking its error.
    for name in source.columns():
        _assert_sources(
            DataFrame([left.column(name)]),
            DataFrame([right.column(name)]),
            False,
        )


def test_chunked_source_windows_compare_offsets_and_lengths() raises:
    var values = Series("k", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7]))
    var left = DataFrame(
        [Series._from_chunks([values.slice(0, 2), values.slice(4, 2)])]
    )
    var right = DataFrame(
        [Series._from_chunks([values.slice(2, 2), values.slice(6, 2)])]
    )
    assert_true(left.column("k").is_chunked())
    assert_true(right.column("k").is_chunked())
    _assert_sources(left, right, False)
    _assert_collect(
        concat([left.lazy(), right.lazy()]),
        DataFrame([Series("k", Column[Int64]([0, 1, 4, 5, 2, 3, 6, 7]))]),
    )
    # Same chunk starts and total height, different per-chunk lengths.
    var unequal = DataFrame(
        [Series._from_chunks([values.slice(0, 1), values.slice(4, 3)])]
    )
    _assert_sources(left, unequal, False)
    _assert_collect(
        concat([left.lazy(), unequal.lazy()]),
        DataFrame([Series("k", Column[Int64]([0, 1, 4, 5, 0, 4, 5, 6]))]),
    )


def test_typed_aliases_keep_their_values_and_chunked_schema() raises:
    var values = Series("k", Column[Int64]([1, 2, 3, 4]))
    for chunked in [False, True]:
        var numbers = values.slice(0, 2)
        if chunked:
            numbers = Series._from_chunks(
                [values.slice(0, 1), values.slice(2, 1)]
            )
        var days = numbers.with_dtype(DataType.DATE)
        var left = DataFrame([numbers.copy()])
        var right = DataFrame([days^])
        _assert_sources(left, right, False)
        var last = "3" if chunked else "2"
        var date = "1970-01-04" if chunked else "1970-01-03"
        _assert_collect(
            concat(
                [
                    left.lazy().select(col("k").cast("string")),
                    right.lazy().select(col("k").cast("string")),
                ]
            ),
            DataFrame(
                [Series("k", StringColumn(["1", last, "1970-01-02", date]))]
            ),
        )


def test_source_identity_preserves_categorical_dictionary_and_metadata() raises:
    var codes = Series("k", Column[UInt32]([0, 1]))
    var labels = CategoricalDictionary()
    labels.append("red")
    labels.append("blue")
    var categorical = codes.with_dtype(DataType.categorical(labels^))
    _assert_sources(
        DataFrame([categorical.copy()]),
        DataFrame([codes.with_dtype(DataType.CATEGORICAL)]),
        False,
    )
    var reversed = CategoricalDictionary()
    reversed.append("blue")
    reversed.append("red")
    _assert_sources(
        DataFrame([categorical.copy()]),
        DataFrame([codes.with_dtype(DataType.categorical(reversed^))]),
        False,
    )
    _assert_sources(
        DataFrame([categorical.copy()]), DataFrame([categorical.copy()]), True
    )
    var left = codes.copy()
    var right = codes.copy()
    var first: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 97]
    var second: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 98]
    left._field_metadata = ArcPointer(first^)
    right._field_metadata = ArcPointer(second^)
    _assert_sources(DataFrame([left.copy()]), DataFrame([right.copy()]), False)
    _assert_sources(DataFrame([left.copy()]), DataFrame([left.copy()]), True)


def test_identical_source_views_still_share_slots() raises:
    var values = Series("k", Column[Int64]([0, 1, 2, 3, 4, 5]))
    var left = DataFrame([values.slice(1, 2)])
    _assert_sources(left, left.copy(), True)
    _assert_sources(left, DataFrame([values.slice(1, 2)]), True)
    _assert_collect(
        concat([left.lazy(), left.lazy()]),
        DataFrame([Series("k", Column[Int64]([1, 2, 1, 2]))]),
    )
    var chunked = DataFrame(
        [Series._from_chunks([values.slice(0, 2), values.slice(4, 2)])]
    )
    _assert_sources(chunked, chunked.copy(), True)
    var independent = DataFrame(
        [Series._from_chunks([values.slice(0, 2), values.slice(4, 2)])]
    )
    _assert_sources(chunked, independent, True)
