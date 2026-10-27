"""Null-safe comparisons and null predicates match literal value semantics."""
# contract-group: expr.arithmetic
# contract-group: reduce.global
# contract-group: reduce.grouped
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    ListColumn,
    StructColumn,
    col,
    lit,
    null,
    when,
)
from dataframe.bool_column import BoolColumn
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_generators import offset_layout, chunk_layout


def _flags(result: DataFrame, name: String, expected: List[Bool]) raises:
    assert_equal(result.height(), len(expected))
    var column = result.column(name).rechunk()
    assert_equal(column.dtype(), DataType.BOOL)
    assert_equal(column.null_count(), 0)
    for row in range(len(expected)):
        assert_equal(column._data[BoolColumn]._get(row), expected[row])


def _pair(frame: DataFrame, expected: List[Bool]) raises:
    var opposite = List[Bool]()
    for value in expected:
        opposite.append(not value)
    var expressions: List[Expr] = [
        col("a").eq_missing(col("b")).alias("eq"),
        col("a").ne_missing(col("b")).alias("ne"),
    ]
    for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        for result in [
            source.select_exprs(expressions),
            source.lazy().select_exprs(expressions).collect(engine="cpu"),
        ]:
            _flags(result, "eq", expected)
            _flags(result, "ne", opposite)


def _integers[D: DType]() raises:
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    var av = List[Bool]()
    var bv = List[Bool]()
    var expected = List[Bool]()
    for row in range(65):
        a.append(Scalar[D](row % 3))
        b.append(Scalar[D](2 if row % 4 == 0 else row % 3))
        av.append(row % 5 != 0)
        bv.append(row % 7 != 0)
        expected.append(
            (not av[row] and not bv[row])
            or (av[row] and bv[row] and a[row] == b[row])
        )
    _pair(
        DataFrame(
            [
                Series("a", Column[Scalar[D]](a^, av)),
                Series("b", Column[Scalar[D]](b^, bv)),
            ]
        ),
        expected,
    )


def test_missing_comparisons_cover_every_integer_width_and_boolean_null_truth_table() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            _integers[D]()
    _pair(
        DataFrame(
            [
                Series(
                    "a",
                    Column[Bool](
                        [True, False, True, False], [True, True, False, False]
                    ),
                ),
                Series(
                    "b",
                    Column[Bool](
                        [True, True, False, False], [True, True, False, True]
                    ),
                ),
            ]
        ),
        [True, False, True, False],
    )


def _floats[D: DType]() raises where D.is_floating_point():
    var zero = Scalar[D](0)
    var nan = zero / zero
    var inf = Scalar[D](1) / zero
    var ap: List[Scalar[D]] = [
        zero,
        Scalar[D](1),
        Scalar[D](1),
        nan,
        nan,
        zero,
        -zero,
        inf,
        -inf,
    ]
    var bp: List[Scalar[D]] = [
        zero,
        Scalar[D](1),
        zero,
        nan,
        Scalar[D](2),
        -zero,
        zero,
        inf,
        inf,
    ]
    var truth: List[Bool] = [
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
    for n in [0, 1, 7, 8, 9, 17, 65]:
        for offset in range(8):
            var av = List[Scalar[D]](length=offset, fill=0)
            var bv = List[Scalar[D]](length=7 - offset, fill=0)
            var am = List[Bool](length=offset, fill=True)
            var bm = List[Bool](length=7 - offset, fill=True)
            var eq = List[Bool]()
            var ne = List[Bool]()
            var na = List[Bool]()
            var missing = List[Bool]()
            for row in range(n):
                var p = row % 9
                av.append(ap[p])
                bv.append(bp[p])
                am.append(p != 0)
                bm.append(p != 0 and p != 2)
                eq.append(truth[p])
                ne.append(not truth[p])
                na.append(p == 3 or p == 4)
                missing.append(p == 0)
            var a = Column[Scalar[D]](av^, am)
            var b = Column[Scalar[D]](bv^, bm)
            var frame = DataFrame(
                [
                    Series("a", a.slice(offset, n)),
                    Series("b", b.slice(7 - offset, n)),
                ]
            )
            var expressions: List[Expr] = [
                col("a").eq_missing(col("b")).alias("eq"),
                col("a").ne_missing(col("b")).alias("ne"),
                col("a").eq_missing(lit(nan)).alias("nan"),
                col("a").eq_missing(null(DataType.of(D))).alias("null"),
            ]
            for source in [frame.copy(), chunk_layout(frame)]:
                for result in [
                    source.select_exprs(expressions),
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(engine="cpu"),
                ]:
                    _flags(result, "eq", eq)
                    _flags(result, "ne", ne)
                    _flags(result, "nan", na)
                    _flags(result, "null", missing)


def test_missing_float_equality_includes_nan_signed_zero_broadcasts_and_all_bit_offsets() raises:
    _floats[DType.float32]()
    _floats[DType.float64]()
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame([Series("a", Column[Float64]([nan]))])
    _flags(
        frame.select((col("a") == col("a")).alias("ordinary")),
        "ordinary",
        [False],
    )


def _text(
    name: String, values: List[String], valid: List[Bool], mode: Int
) raises -> Series:
    if mode == 3:
        var builder = StringViewBuilder(len(values))
        for row in range(len(values)):
            if valid[row]:
                builder.append(values[row])
            else:
                builder.append_null()
        var views = builder^.finish()
        return Series(name, StringColumn(views^))
    var base = Series(name, StringColumn(values.copy(), valid))
    if mode == 1:
        return base.with_dtype(DataType.BINARY)
    if mode == 2:
        return base.cast("categorical").cast("string")
    if mode == 4:
        return base.cast("categorical")
    return base^


def test_missing_string_literals_binary_views_dictionary_codes_and_lazy_filter() raises:
    var ap: List[String] = [
        "x",
        "",
        "naïve",
        "x\x00y",
        "same-long-value-é",
        "",
        "",
    ]
    var bp: List[String] = [
        "x",
        "",
        "naïve",
        "x\x00z",
        "different-long-value-é",
        "",
        "",
    ]
    var truth: List[Bool] = [True, True, True, False, False, False, False]
    for mode in range(5):
        var a = List[String]()
        var b = List[String]()
        var am = List[Bool]()
        var bm = List[Bool]()
        var expected = List[Bool]()
        var literal = List[Bool]()
        var ids = List[Int64]()
        for row in range(70):
            var p = row % 7
            a.append(ap[p].copy())
            b.append(bp[p].copy())
            am.append(p != 1 and p != 5)
            bm.append(p != 1 and p != 6)
            expected.append(truth[p])
            literal.append(p == 4)
            ids.append(Int64(row))
        var frame = DataFrame(
            [
                _text("a", a, am, mode),
                _text("b", b, bm, mode),
                Series("id", Column[Int64](ids^)),
            ]
        )
        _pair(frame, expected)
        _flags(
            frame.select(
                col("a").eq_missing("same-long-value-é").alias("literal")
            ),
            "literal",
            literal,
        )
        _flags(
            frame.lazy()
            .select(col("a").eq_missing("same-long-value-é").alias("literal"))
            .collect(engine="cpu"),
            "literal",
            literal,
        )
        var changed = (
            frame.lazy()
            .filter(col("a").ne_missing(col("b")))
            .select(["id"])
            .collect(engine="cpu")
        )
        assert_equal(changed.height(), 40)
        var at = 0
        for row in range(70):
            if not truth[row % 7]:
                assert_equal(changed.item(at, "id").int64(), Int64(row))
                at += 1


def test_missing_temporal_decimal_null_and_nested_values_keep_structural_equality() raises:
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("us"),
        DataType.duration("ns"),
    ]:
        _pair(
            DataFrame(
                [
                    Series(
                        "a", Column[Int64]([1, 0, 3], [True, False, True])
                    ).with_dtype(dtype),
                    Series(
                        "b", Column[Int64]([1, 0, 4], [True, False, True])
                    ).with_dtype(dtype),
                ]
            ),
            [True, True, False],
        )
    _pair(
        DataFrame(
            [
                Series(
                    "a", Column[Int32]([1234, 0, 5678], [True, False, True])
                ).with_dtype(DataType.decimal(7, 2, 32)),
                Series(
                    "b",
                    Column[Int128](
                        [Int128(123400), 0, 567900], [True, False, True]
                    ),
                ).with_dtype(DataType.decimal(12, 4, 128)),
            ]
        ),
        [True, True, False],
    )
    _pair(
        DataFrame(
            [
                Series.full_null("a", DataType.NULL, 4),
                Series(
                    "b",
                    StringColumn(
                        ["", "x", "", "y"], [False, True, False, True]
                    ),
                ),
            ]
        ),
        [True, False, True, False],
    )
    var left = Series(
        "a",
        ListColumn(
            [0, 2, 2, 4, 4],
            Series("item", Column[Int64]([1, 2, 1, 2])),
            [UInt8(13)],
        ),
    )
    var right = Series(
        "b",
        ListColumn(
            [0, 2, 2, 3, 3],
            Series("item", Column[Int64]([1, 2, 9])),
            [UInt8(13)],
        ),
    )
    _pair(DataFrame([left^, right^]), [True, True, False, True])
    var nan = Float64(0) / Float64(0)
    var a = StructColumn(
        [
            Series("x", Column[Float64]([nan, 99, 1, 2])),
            Series("s", StringColumn(["a", "poison", "x", "y"])),
        ],
        [UInt8(13)],
    )
    var b = StructColumn(
        [
            Series("x", Column[Float64]([nan, -99, 1, 2])),
            Series("s", StringColumn(["a", "different", "z", "y"])),
        ],
        [UInt8(13)],
    )
    _pair(
        DataFrame([Series("a", a^), Series("b", b^)]), [True, True, False, True]
    )


def test_has_nulls_is_a_non_null_scalar_in_eager_lazy_grouped_and_window_contexts() raises:
    for n in [0, 5]:
        var frame = DataFrame([Series.full_null("n", DataType.NULL, n)])
        _flags(frame.select(col("n").has_nulls().alias("h")), "h", [n > 0])
        _flags(
            frame.lazy()
            .select(col("n").has_nulls().alias("h"))
            .collect(engine="cpu"),
            "h",
            [n > 0],
        )
    var frame = DataFrame(
        [
            Series("g", StringColumn(["a", "a", "b", "b", "c"])),
            Series(
                "a",
                Column[Int64](
                    [0, 1, 2, 3, 0], [False, True, True, True, False]
                ),
            ),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame)]:
        var grouped = (
            source.group_by(["g"])
            .agg([col("a").has_nulls().alias("h")])
            .sort(["g"])
        )
        _flags(grouped, "h", [True, False, True])
        _flags(
            source.lazy()
            .group_by(["g"])
            .agg([col("a").has_nulls().alias("h")])
            .sort(["g"])
            .collect(engine="cpu"),
            "h",
            [True, False, True],
        )
        _flags(
            source.select(col("a").has_nulls().over("g").alias("h")),
            "h",
            [True, True, False, False, True],
        )
        _flags(
            source.lazy()
            .select(col("a").has_nulls().over("g").alias("h"))
            .collect(engine="cpu"),
            "h",
            [True, True, False, False, True],
        )


def test_missing_comparisons_preserve_guarded_errors_and_reject_incompatible_dtypes() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MAX, 1])),
            Series("take", Column[Bool]([False, True])),
        ]
    )
    var guarded = (
        when(col("take"))
        .then((col("a") + 1).eq_missing(2))
        .otherwise(False)
        .alias("h")
    )
    _flags(frame.select(guarded), "h", [False, True])
    _flags(
        frame.lazy().select(guarded).collect(engine="cpu"), "h", [False, True]
    )
    with assert_raises(contains="matching dtypes"):
        _ = frame.select(col("a").eq_missing(lit(Float64(1))))
