"""Conditional predicates preserve masks, original rows and bitmap windows."""
# contract-group: expr.conditional_predicate_chunks
# contract-group: expr.chunk_windows
# contract-group: expr.null_logic
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    BoolColumn,
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
from dataframe.expr import null
from dataframe.binding import bind
from dataframe.execution import (
    _ConditionalFlags,
    _prepare_conditional_flags,
    _selection_masks,
    _and_validity,
    _whole,
    _whole_masked_branch,
)
from dataframe.series import Storage
from dataframe.storage.buffer import _Buffer
from tests.support.coverage_compare import assert_frame


def _predicate(
    values: List[Bool], valid: List[Bool], first: Int = 13, chunked: Bool = True
) raises -> Series:
    var raw = List[Bool](length=first, fill=False)
    raw.extend(values.copy())
    var flags = List[Bool](length=first, fill=True)
    flags.extend(valid.copy())
    var parent = Series("p", BoolColumn(raw.copy(), flags.copy()))
    var window = parent.slice(first, len(values))
    if not chunked:
        return window^
    var parts = List[Series]()
    parts.append(window.slice(0, 0))
    var at = 0
    for size in [1, 2, 5, len(values)]:
        var count = min(size, len(values) - at)
        parts.append(window.slice(at, count))
        at += count
    return Series._from_chunks(parts)


def _flags(source: Series) raises -> _ConditionalFlags:
    var cold = source.is_chunked() and (
        source._chunked.value()[].cache[].state.load() == 0
    )
    var flags = _prepare_conditional_flags(source, len(source))
    if cold:
        assert_true(flags.stream_mode)
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    return flags^


def _states(bound_dtypes: List[DataType]) raises -> List[Series]:
    var result = List[Series]()
    for dtype in bound_dtypes:
        result.append(Series.full_null("", dtype, 0))
    return result^


def test_selection_masks_truth_states_outer_activity_and_bit_offsets() raises:
    var values = List[Bool]()
    var valid = List[Bool]()
    var active = List[Bool]()
    var expected = List[Bool]()
    var wanted_then = List[Bool]()
    var wanted_other = List[Bool]()
    for state in range(8):
        var raw = state & 1 != 0
        var present = state & 2 != 0
        var seen = state & 4 != 0
        var take = raw and present
        values.append(raw)
        valid.append(present)
        active.append(seen)
        expected.append(take)
        wanted_then.append(seen and take)
        wanted_other.append(seen and not take)
    for first in range(8):
        for chunked in [False, True]:
            var source = _predicate(values, valid, first, chunked)
            var flags = _flags(source)
            var selected = List[Bool](length=8, fill=False)
            var then_mask = List[Bool](length=8, fill=False)
            var other_mask = List[Bool](length=8, fill=False)
            _selection_masks(flags, active, selected, then_mask, other_mask)
            assert_equal(selected, expected)
            assert_equal(then_mask, wanted_then)
            assert_equal(other_mask, wanted_other)
            var none = List[Bool](length=8, fill=False)
            _selection_masks(flags, none, selected, then_mask, other_mask)
            assert_equal(selected, expected)
            assert_equal(then_mask, none)
            assert_equal(other_mask, none)
    var scalar = BoolColumn([True], [False])
    var selected = List[Bool](length=3, fill=True)
    var yes = List[Bool](length=3, fill=True)
    var no = List[Bool](length=3, fill=False)
    _selection_masks(scalar, [True, False, True], selected, yes, no)
    assert_equal(selected, List[Bool]([False, False, False]))
    assert_equal(yes, List[Bool]([False, False, False]))
    assert_equal(no, List[Bool]([True, False, True]))


def test_and_validity_preserves_own_offsets_payload_alias_and_logical_tag() raises:
    var p = _predicate(
        [True, True, False, True, True, False, True, False, True],
        [True, False, True, True, True, True, True, True, True],
    )
    var flags = _flags(p)
    var expected: List[Bool] = [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    for base in range(8):
        for absent in [False, True]:
            var values = List[Int64](length=base, fill=-91)
            for value in range(7, 16):
                values.append(Int64(value))
            var valid = List[Bool](length=base, fill=True)
            valid.extend(
                List[Bool](
                    [True, True, True, True, False, True, True, True, True]
                )
            )
            var parent = Column[Int64](values.copy()) if absent else Column[
                Int64
            ](values.copy(), valid.copy())
            var own = parent.slice(base, 9)
            var masked = own.copy()
            masked._bits = _Buffer(_and_validity(flags, own._bits[], base, 9))
            assert_equal(Int(masked._ptr()), Int(own._ptr()))
            for row in range(9):
                assert_equal(masked._get(row), Int64(7 + row))
                var keep = (
                    row == 0 or row == 3 or row == 4 or row == 6 or row == 8
                ) if absent else expected[row]
                assert_equal(masked._valid(row), keep)
            var tagged = Series("x", masked.copy()).with_dtype(
                DataType.duration("ms")
            )
            assert_true(tagged.dtype() == DataType.duration("ms"))
            assert_equal(
                Int(tagged._data[Column[Int64]]._ptr()), Int(own._ptr())
            )
    var empty = BoolColumn(List[Bool]())
    var own = Column[Int64](List[Int64]())
    assert_equal(len(_and_validity(empty, own._bits[], 0, 0)), 0)


def _public_frame() raises -> DataFrame:
    return DataFrame(
        [
            _predicate([True, False, True, True], [True, True, False, True]),
            Series(
                "x", Column[Int64]([7, 8, 9, 10], [True, True, True, False])
            ),
            Series(
                "b",
                BoolColumn(
                    [True, True, False, True], [True, True, True, False]
                ),
            ),
            Series(
                "s",
                StringColumn(
                    ["A", "B", "C", "hidden"], [True, True, True, False]
                ),
            ),
        ]
    )


def test_public_predicate_paths_scalar_nulls_nested_masks_and_empty() raises:
    for batch in [1, 3, 64]:
        var frame = _public_frame()
        var numeric = (
            frame.lazy()
            .select(
                when(col("p"))
                .then(col("x"))
                .otherwise(lit(Int64(0)))
                .alias("r")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        assert_frame(
            numeric,
            DataFrame(
                [
                    Series(
                        "r",
                        Column[Int64]([7, 0, 0, 0], [True, True, True, False]),
                    )
                ]
            ),
        )
        var booleans = (
            frame.lazy()
            .select(when(col("p")).then(col("b")).end().alias("r"))
            .collect(engine="cpu", batch_size=batch)
        )
        assert_frame(
            booleans,
            DataFrame(
                [
                    Series(
                        "r",
                        BoolColumn(
                            [True, False, False, False],
                            [True, False, False, False],
                        ),
                    )
                ]
            ),
        )
        var strings = (
            frame.lazy()
            .select(
                when(col("p"))
                .then(col("s"))
                .otherwise(null(DataType.STRING))
                .alias("r")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        assert_frame(
            strings,
            DataFrame(
                [
                    Series(
                        "r",
                        StringColumn(
                            ["A", "", "", ""], [True, False, False, False]
                        ),
                    )
                ]
            ),
        )
        for mode in range(3):
            var predicate = (
                lit(True) if mode
                == 0 else lit(False) if mode
                == 1 else null(DataType.BOOL)
            )
            var selected = frame.select(
                when(predicate).then(col("s")).otherwise(lit("else")).alias("r")
            )
            var expected = Series(
                "r",
                StringColumn(["A", "B", "C", ""], [True, True, True, False]),
            ) if mode == 0 else Series(
                "r", StringColumn(["else", "else", "else", "else"])
            )
            assert_frame(selected, DataFrame([expected^]))
        var nested = frame.select(
            when(lit(False))
            .then(when(col("p")).then(col("s")).otherwise(lit("inner")))
            .otherwise(lit("outer"))
            .alias("r")
        )
        assert_frame(
            nested,
            DataFrame(
                [
                    Series(
                        "r", StringColumn(["outer", "outer", "outer", "outer"])
                    )
                ]
            ),
        )
        var empty = frame.clear().select(
            when(col("p")).then(col("s")).end().alias("r")
        )
        assert_frame(
            empty, DataFrame([Series("r", StringColumn(List[String]()))])
        )


def test_whole_predicate_column_computed_controls_preserve_logical_numeric_types() raises:
    var p = _predicate([True, False, True, True], [True, True, False, True])
    var flat = p.rechunk()
    _ = flat^
    for dtype in List[DataType](
        [
            DataType.DATE,
            DataType.datetime("us", "America/New_York"),
            DataType.decimal(18, 2, 64),
        ]
    ):
        var value = Series(
            "x", Column[Int64]([7, 8, 9, 10], [True, True, True, False])
        ).with_dtype(dtype)
        var frame = DataFrame([p.copy(), value.copy()])
        for computed in [False, True]:
            var predicate = col("p").fill_null(lit(False)) if computed else col(
                "p"
            )
            var result = (
                frame.lazy()
                .select(
                    when(predicate)
                    .then(col("x").alias("source_alias"))
                    .end()
                    .alias("r")
                )
                .collect(engine="cpu", batch_size=3)
            )
            var expected = Series(
                "r", Column[Int64]([7, 8, 9, 10], [True, False, False, False])
            ).with_dtype(dtype)
            assert_frame(result, DataFrame([expected^]))
        var bound = bind(col("p").cast(DataType.BOOL), frame._columns)
        var states = _states(bound.dtypes)
        var whole = _whole(
            bound, frame._columns, states, len(bound.expr._nodes) - 1, 4, False
        )
        assert_true(Bool(whole))
        var recognized = whole.take()
        assert_true(recognized.dtype() == DataType.BOOL)
        assert_equal(
            recognized._chunked.value()[].cache[].state.load(), Int64(2)
        )


def _whole_original_rows() raises -> Series:
    var p = _predicate([True, False, True, False], [True, True, True, True])
    var original = p.rechunk()
    _ = original^
    # Fully initialized, local private cache sentinel distinguishes whole-row
    # source slicing (original arrays) from NORMAL cache-consuming preparation.
    p._chunked.value()[].cache[].merged = Storage(
        BoolColumn([False, False, False, False])
    )
    var cached = _flags(p)
    var selected = List[Bool](length=4, fill=True)
    var yes = List[Bool](length=4, fill=True)
    var no = List[Bool](length=4, fill=False)
    _selection_masks(cached, List[Bool](), selected, yes, no)
    assert_equal(selected, List[Bool]([False, False, False, False]))
    var frame = DataFrame([p.copy(), Series("x", Column[Int64]([7, 8, 9, 10]))])
    var expression = when(col("p")).then(col("x")).end().alias("r")
    var bound = bind(expression, frame._columns)
    var states = _states(bound.dtypes)
    var result = _whole_masked_branch[8](
        bound,
        frame._columns,
        frame._columns,
        states,
        len(bound.expr._nodes) - 1,
        4,
        2,
        1,
    )
    assert_true(Bool(result))
    return result.take().renamed("r")


def test_normal_published_cache_and_original_whole_rows_are_distinct_owned_sources() raises:
    var result = _whole_original_rows()
    assert_frame(
        DataFrame([result.copy()]),
        DataFrame(
            [
                Series(
                    "r",
                    Column[Int64]([7, 8, 9, 10], [True, False, True, False]),
                )
            ]
        ),
    )
    var flat = result.rechunk()
    assert_equal(flat._data[Column[Int64]]._get(1), Int64(8))
    assert_equal(flat._data[Column[Int64]]._get(3), Int64(10))


def test_predicate_branch_error_order_binding_and_grouped_streamed_reductions() raises:
    var frame = DataFrame(
        [
            Series("ptext", StringColumn(["true", "bad", "false", "true"])),
            Series("text", StringColumn(["bad", "7", "bad", "9"])),
            _predicate([True, False, True, True], [True, True, False, True]),
            Series("x", Column[Int64]([7, 8, 9, 10])),
            Series("k", Column[Int64]([0, 1, 0, 1])),
        ]
    )
    with assert_raises(contains="to bool failed at row 1"):
        _ = frame.select(
            when(col("ptext").cast(DataType.BOOL))
            .then(col("text").cast(DataType.INT8))
            .otherwise(lit(Int8(0)))
        )
    var masked = frame.select(
        when(lit(False))
        .then(
            when(col("ptext").cast(DataType.BOOL))
            .then(col("text").cast(DataType.INT8))
            .otherwise(lit(Int8(2)))
        )
        .otherwise(lit(Int8(0)))
        .alias("r")
    )
    assert_frame(masked, DataFrame([Series("r", Column[Int8]([0, 0, 0, 0]))]))
    with assert_raises(contains="when requires a bool predicate"):
        _ = frame.select(when(col("x")).then(lit("yes")).otherwise(lit("no")))
    var branch = when(col("p")).then(col("x")).end()
    var total = (
        frame.lazy()
        .select(branch.sum().alias("s"))
        .collect(engine="cpu", batch_size=3)
    )
    assert_frame(total, DataFrame([Series("s", Column[Int64]([17]))]))
    var grouped = (
        frame.lazy()
        .group_by(["k"])
        .agg([branch.sum().alias("s")])
        .sort(["k"])
        .collect(engine="cpu", batch_size=3)
    )
    assert_frame(
        grouped,
        DataFrame(
            [
                Series("k", Column[Int64]([0, 1])),
                Series("s", Column[Int64]([7, 10])),
            ]
        ),
    )
    var overflow = DataFrame(
        [
            _predicate([True, False, False, True], [True, True, True, True]),
            Series("text", StringColumn(["7", "bad", "bad", "999"])),
        ]
    )
    with assert_raises(contains="failed at row 3"):
        _ = overflow.select(
            when(col("p"))
            .then(col("text").cast(DataType.INT8))
            .otherwise(lit(Int8(0)))
        )
