"""String lengths read chunk windows into independently owned Int64 outputs."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, StringColumn, col
from dataframe.string_view import StringViewBuilder
from dataframe.expr import STR_LEN_BYTES, STR_LEN_CHARS, _node
from dataframe.text.kernels import string_op
from tests.support.coverage_compare import assert_frame


def _valid(mode: Int) -> List[Bool]:
    if mode == 0:
        return List[Bool](length=9, fill=True)
    if mode == 2:
        return List[Bool](length=9, fill=False)
    return [True, True, True, True, True, False, True, False, True]


def _source(views: Bool, mode: Int) raises -> Series:
    var words: List[String] = [
        "",
        "é",
        "é",
        "😀",
        "abcdefghijklmnop",
        "終",
        "x",
        "hiddenpayload",
        "z",
    ]
    var values: List[String] = ["pad", "pad", "pad"]
    values.extend(words^)
    values.append("tail")
    var valid: List[Bool] = [True, True, True]
    valid.extend(_valid(mode))
    valid.append(True)
    var source: Series
    if views:
        var builder = StringViewBuilder()
        for row in range(len(values)):
            if valid[row]:
                builder.append(StringSlice(values[row]))
            else:
                builder.append_null()
        source = Series("s", StringColumn(builder^.finish()))
    else:
        source = Series("s", StringColumn(values, valid))
    # Uneven source windows start at non-byte-aligned validity offsets.
    var pieces = List[Series]()
    var start = 3
    for size in [0, 1, 2, 0, 4, 2, 0]:
        pieces.append(source.slice(start, size))
        start += size
    return Series._from_chunks(pieces)


def _expected(mode: Int) raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "bytes",
                Column[Int64]([0, 2, 3, 4, 16, 3, 1, 13, 1], _valid(mode)),
            ),
            Series(
                "chars",
                Column[Int64]([0, 1, 2, 1, 16, 1, 1, 13, 1], _valid(mode)),
            ),
        ]
    )


def test_direct_chunk_lengths_own_outputs_without_merging_source() raises:
    for views in [False, True]:
        for mode in range(3):
            var source = _source(views, mode)
            var bytes = string_op(_node(STR_LEN_BYTES), source)
            var chars = string_op(_node(STR_LEN_CHARS), source)
            assert_true(source.is_chunked())
            assert_equal(source._chunked.value()[].cache[].state.load(), 0)
            assert_equal(bytes.n_chunks(), 1)
            assert_equal(chars.n_chunks(), 1)
            assert_equal(bytes.name(), "")
            assert_equal(chars.name(), "")
            _ = source^
            assert_frame(
                DataFrame([bytes.renamed("bytes"), chars.renamed("chars")]),
                _expected(mode),
            )


def test_public_chunk_lengths_keep_values_schema_and_empty_windows() raises:
    for views in [False, True]:
        for mode in range(3):
            var frame = DataFrame([_source(views, mode)])
            var expressions: List[Expr] = [
                col("s").str().len_bytes().alias("bytes"),
                col("s").str().len_chars().alias("chars"),
            ]
            var expected = _expected(mode)
            assert_frame(frame.select_exprs(expressions), expected)
            assert_frame(
                frame.lazy().select_exprs(expressions).collect(), expected
            )
            assert_frame(
                frame.slice(2, 6).select_exprs(expressions),
                expected.slice(2, 6),
            )
            assert_frame(
                frame.slice(3, 0).select_exprs(expressions),
                expected.slice(3, 0),
            )
