"""Per-group row edges keep every original column and computed-key hygiene."""
# contract-group: expr.group_sequences
from std.testing import assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def edge_frame(rows: List[Int]) raises -> DataFrame:
    var groups: List[Int64] = [2, 1, 2, 0, 1, 0]
    var columns = List[Series]()
    var names: List[String] = [
        "g",
        "r",
        "x",
        "__group_edge_row",
        "__group_edge_first",
        "__group_edge_count",
        "__group_edge_key",
    ]
    for c in range(len(names)):
        var values = List[Int64](capacity=len(rows))
        var valid = List[Bool](capacity=len(rows))
        for row in rows:
            values.append(
                groups[row] if c
                == 0 else Int64(row) if c
                == 1 else Int64(row + 1000) if c
                == 2 else Int64(row * 10 + c)
            )
            valid.append(c != 2 or row % 3 != 1)
        var series = Series(names[c], Column[Int64](values^, valid))
        if c == 2:
            series = series.with_dtype(DataType.DATE)
        columns.append(series^)
    return DataFrame(columns^)


def test_group_head_tail_keep_all_columns_nulls_types_and_group_input_order() raises:
    var frame = edge_frame([0, 1, 2, 3, 4, 5])
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        for n in [0, 1, 2, 9]:
            var heads: List[Int] = (
                [] if n
                == 0 else List[Int]([0, 1, 3]) if n
                == 1 else List[Int]([0, 2, 1, 4, 3, 5])
            )
            var tails: List[Int] = (
                [] if n
                == 0 else List[Int]([2, 4, 5]) if n
                == 1 else List[Int]([0, 2, 1, 4, 3, 5])
            )
            assert_frame(
                source.group_by("g", maintain_order=True).head(n),
                edge_frame(heads),
            )
            assert_frame(
                source.group_by("g", maintain_order=True).tail(n),
                edge_frame(tails),
            )
            for optimize in [False, True]:
                assert_frame(
                    source.lazy()
                    .group_by("g", maintain_order=True)
                    .head(n)
                    .collect(engine="cpu", optimize=optimize, batch_size=1),
                    edge_frame(heads),
                )
                assert_frame(
                    source.lazy()
                    .group_by("g", maintain_order=True)
                    .tail(n)
                    .collect(engine="cpu", optimize=optimize, batch_size=2),
                    edge_frame(tails),
                )
        assert_frame(
            source.head(0).group_by("g", maintain_order=True).head(),
            source.head(0),
        )
        assert_frame(
            source.head(0)
            .lazy()
            .group_by("g", maintain_order=True)
            .tail()
            .collect(engine="cpu"),
            source.head(0),
        )
    with assert_raises(contains="nonnegative"):
        _ = frame.group_by("g").head(-1)
    with assert_raises(contains="nonnegative"):
        _ = frame.lazy().group_by("g").tail(-1)


def test_group_edges_computed_keys_do_not_replace_payload_or_private_named_columns() raises:
    var frame = edge_frame([0, 1, 2, 3, 4, 5])
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        var key = (col("g") % 2).alias("x")
        assert_frame(
            source.group_by([key.copy()], maintain_order=True).head(2),
            edge_frame([0, 2, 1, 4]),
        )
        assert_frame(
            source.group_by([key.copy()], maintain_order=True).tail(2),
            edge_frame([3, 5, 1, 4]),
        )
        for optimize in [False, True]:
            assert_frame(
                source.lazy()
                .group_by([key.copy()], maintain_order=True)
                .head(2)
                .collect(engine="cpu", optimize=optimize, batch_size=2),
                edge_frame([0, 2, 1, 4]),
            )
            assert_frame(
                source.lazy()
                .group_by([key.copy()], maintain_order=True)
                .tail(2)
                .collect(engine="cpu", optimize=optimize, batch_size=1),
                edge_frame([3, 5, 1, 4]),
            )
