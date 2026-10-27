"""Expression sorts and positional frame conveniences keep eager/lazy parity."""
# contract-group: table.schema
# contract-group: table.row_selection
# contract-group: table.sort
from std.collections import Dict
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("x", Column[Int64]([3, 1, 4, 2, 5])),
            Series("s", Column[String](["B", "a", "A", "b", "c"])),
            Series("__dataframe_sort_key_0", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )


def _renamer(name: String) -> String:
    return "new_" + name


def _duplicate(name: String) -> String:
    _ = name
    return "same"


def _reject(name: String) raises -> String:
    raise Error("callback rejected " + name)


def test_frame_sort_expressions_preserve_columns_collisions_and_key_options() raises:
    var source = _frame()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var keys: List[Expr] = [col("s").str().to_lowercase(), -col("x")]
        var expected = frame.take([Int(2), 1, 0, 3, 4])
        assert_frame(frame.sort(keys, maintain_order=True), expected)
        for optimized in [False, True]:
            assert_frame(
                frame.lazy()
                .sort(keys, maintain_order=True)
                .collect(engine="cpu", optimize=optimized, batch_size=1),
                expected,
            )
            assert_frame(
                frame.lazy()
                .sort(col("s").str().to_lowercase(), maintain_order=True)
                .collect(engine="cpu", optimize=optimized, batch_size=2),
                frame.take([Int(1), 2, 0, 3, 4]),
            )
        var desc: List[Bool] = [True, False]
        var nulls: List[Bool] = [False, True]
        var reverse = frame.take([Int(4), 0, 3, 2, 1])
        assert_frame(
            frame.sort(keys, descending=desc, nulls_last=nulls), reverse
        )
        assert_frame(
            frame.lazy()
            .sort(keys, descending=desc, nulls_last=nulls)
            .collect(engine="cpu", batch_size=1),
            reverse,
        )
        assert_frame(frame.sort(lit(Int64(7)), maintain_order=True), frame)
        assert_frame(
            frame.lazy()
            .sort(lit(Int64(7)), maintain_order=True)
            .collect(engine="cpu", batch_size=1),
            frame,
        )
        assert_frame(frame.clear().sort(keys), frame.clear())
        assert_frame(
            frame.lazy().sort(List[Expr]()).collect(engine="cpu"), frame
        )
        with assert_raises():
            _ = frame.sort(keys, descending=[True], nulls_last=[True])
        with assert_raises():
            _ = frame.lazy().sort(keys, descending=[True], nulls_last=[True])


def test_frame_expression_sort_retains_global_key_axis_and_rejects_bad_lengths() raises:
    var frame = DataFrame([Series("x", Column[Int64]([-10, 1, 2]))])
    var key = col("x").cast(DataType.FLOAT64) * col("x").mean()
    var expected = DataFrame([Series("x", Column[Int64]([2, 1]))])
    assert_frame(frame.sort(key).filter(col("x") > 0), expected)
    for optimized in [False, True]:
        assert_frame(
            frame.lazy()
            .sort(key)
            .filter(col("x") > 0)
            .collect(engine="cpu", optimize=optimized, batch_size=1),
            expected,
        )
        assert_frame(
            frame.lazy()
            .sort(key)
            .head(1)
            .collect(engine="cpu", optimize=optimized, batch_size=1),
            frame.slice(2, 1),
        )
    var nullable = DataFrame(
        [Series("x", Column[Int64]([3, 0, 1], [True, False, True]))]
    )
    assert_frame(
        nullable.sort(-col("x"), nulls_last=False),
        nullable.take([Int(1), 0, 2]),
    )
    assert_frame(
        nullable.lazy().sort(-col("x"), nulls_last=False).collect(engine="cpu"),
        nullable.take([Int(1), 0, 2]),
    )
    with assert_raises():
        _ = nullable.sort(col("x").drop_nulls())
    with assert_raises():
        _ = nullable.lazy().sort(col("x").drop_nulls()).collect(engine="cpu")


def test_frame_insert_column_and_callable_rename_validate_schema_and_height() raises:
    var source = _frame()
    var added = Series("new", Column[Bool]([True, False, True, False, True]))
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        for index in [Int(-3), -1, 0, 1, 3]:
            var position = index + frame.width() if index < 0 else index
            var columns = List[Series]()
            for i in range(frame.width() + 1):
                if i == position:
                    columns.append(added.copy())
                if i < frame.width():
                    columns.append(frame._columns[i].copy())
            var expected = DataFrame(columns^)
            assert_frame(frame.insert_column(index, added), expected)
            for optimized in [False, True]:
                assert_frame(
                    frame.lazy()
                    .insert_column(index, added)
                    .collect(engine="cpu", optimize=optimized, batch_size=1),
                    expected,
                )
        var mapping = Dict[String, String]()
        for name in frame.columns():
            mapping[name] = "new_" + name
        var expected = frame.rename(mapping)
        assert_frame(frame.rename(_renamer), expected)
        assert_frame(
            frame.lazy().rename(_renamer).collect(engine="cpu"), expected
        )
        with assert_raises():
            _ = frame.rename(_duplicate)
        with assert_raises():
            _ = frame.lazy().rename(_duplicate)
        with assert_raises():
            _ = frame.rename(_reject)
        with assert_raises():
            _ = frame.lazy().rename(_reject)
        for index in [Int(-4), 4]:
            with assert_raises():
                _ = frame.insert_column(index, added)
            with assert_raises():
                _ = frame.lazy().insert_column(index, added)
        with assert_raises():
            _ = frame.insert_column(1, added.renamed("x"))
        with assert_raises():
            _ = frame.lazy().insert_column(1, added.renamed("x"))
        var short = added.slice(0, 2)
        with assert_raises():
            _ = frame.insert_column(0, short)
        with assert_raises():
            _ = (
                frame.lazy()
                .insert_column(0, short)
                .select([String("x")])
                .collect(engine="cpu")
            )
    var empty = DataFrame(List[Series]())
    var only = DataFrame([added.copy()])
    assert_frame(empty.insert_column(0, added), only)
    assert_frame(
        empty.lazy().insert_column(0, added).collect(engine="cpu"), only
    )


def test_frame_gather_every_keeps_global_offsets_under_lazy_optimization() raises:
    var source = _frame()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        for n in [Int(1), 2, 3, 8, Int.MAX]:
            for offset in [Int(0), 1, 3, 5, Int.MAX]:
                var rows = List[Int]()
                for i in range(frame.height()):
                    if i >= offset and (i - offset) % n == 0:
                        rows.append(i)
                var expected = frame.take(rows)
                assert_frame(frame.gather_every(n, offset), expected)
                for optimized in [False, True]:
                    assert_frame(
                        frame.lazy()
                        .gather_every(n, offset)
                        .collect(
                            engine="cpu", optimize=optimized, batch_size=1
                        ),
                        expected,
                    )
        var selected = frame.take([Int(1), 3])
        assert_frame(
            frame.lazy()
            .gather_every(2, 1)
            .filter(col("x") > 1)
            .select([String("s")])
            .head(1)
            .collect(engine="cpu", batch_size=1),
            selected.filter(col("x") > 1).select([String("s")]).head(1),
        )
        for n in [Int(0), -1]:
            with assert_raises():
                _ = frame.gather_every(n)
            with assert_raises():
                _ = frame.lazy().gather_every(n)
        with assert_raises():
            _ = frame.gather_every(1, -1)
        with assert_raises():
            _ = frame.lazy().gather_every(1, -1)
    var zero_columns = DataFrame(List[Series](), height=5)
    assert_equal(zero_columns.gather_every(2, 1).height(), 2)
    assert_equal(
        zero_columns.lazy().gather_every(2, 1).collect(engine="cpu").height(), 2
    )
