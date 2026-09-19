"""String namespace: code point semantics, nulls, broadcasting, validation."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Expr, Series, col, concat_str, lit


def frame() raises -> DataFrame:
    # "é" is e + combining acute: two code points, one grapheme.
    return DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["  Héllo ", "-42", "", "étoile", "日本語", "x"],
                    [True, True, True, True, True, False],
                ),
            ),
            Series("t", Column[String](["a", "b", "c", "d", "e", "f"])),
        ]
    )


def texts(frame: DataFrame, expr: Expr) raises -> List[String]:
    var series = frame.select(expr.alias("r")).column("r")
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def test_lengths_and_case() raises:
    var df = frame()
    assert_equal(
        texts(df, col("s").str().len_chars()),
        [String("8"), "3", "0", "7", "3", "null"],
    )
    assert_equal(
        texts(df, col("s").str().len_bytes()),
        [String("9"), "3", "0", "8", "9", "null"],
    )
    assert_equal(
        texts(df, col("s").str().to_uppercase()),
        [String("  HÉLLO "), "-42", "", "ÉTOILE", "日本語", "null"],
    )
    assert_equal(texts(df, col("s").str().to_lowercase())[0], "  héllo ")


def test_strip_predicates_and_replace() raises:
    var df = frame()
    assert_equal(texts(df, col("s").str().strip_chars())[0], "Héllo")
    assert_equal(texts(df, col("s").str().strip_chars_start())[0], "Héllo ")
    assert_equal(texts(df, col("s").str().strip_chars_end())[0], "  Héllo")
    assert_equal(texts(df, col("s").str().strip_chars(" oH"))[0], "éll")
    assert_equal(texts(df, col("s").str().strip_chars("日語"))[4], "本")
    assert_equal(
        texts(df, col("s").str().starts_with("-")),
        [String("false"), "true", "false", "false", "false", "null"],
    )
    assert_equal(texts(df, col("s").str().ends_with("語"))[4], "true")
    assert_equal(texts(df, col("s").str().contains(""))[2], "true")
    assert_equal(texts(df, col("s").str().contains("́"))[3], "true")
    assert_equal(texts(df, col("s").str().replace("l", "L"))[0], "  HéLlo ")
    assert_equal(texts(df, col("s").str().replace_all("l", "L"))[0], "  HéLLo ")
    assert_equal(texts(df, col("s").str().replace("zz", "!"))[1], "-42")
    assert_equal(texts(df, col("s").str().replace("", "^"))[1], "^-42")
    assert_equal(texts(df, col("s").str().replace_all("", "^"))[1], "-42")


def test_slice_reverse_and_padding() raises:
    var df = frame()
    assert_equal(texts(df, col("s").str().slice(2, 3))[0], "Hél")
    assert_equal(texts(df, col("s").str().slice(-3))[4], "日本語")
    assert_equal(texts(df, col("s").str().slice(-99, 2))[1], "-4")
    assert_equal(texts(df, col("s").str().slice(99))[1], "")
    assert_equal(texts(df, col("s").str().slice(1, 0))[1], "")
    assert_equal(texts(df, col("s").str().head(2))[4], "日本")
    assert_equal(texts(df, col("s").str().tail(2))[4], "本語")
    assert_equal(texts(df, col("s").str().tail(0))[4], "")
    # Reversal is by code point, so the combining mark moves.
    assert_equal(texts(df, col("s").str().reverse())[3], "eliot\u0301e")
    assert_equal(texts(df, col("s").str().pad_start(5, "*"))[1], "**-42")
    assert_equal(texts(df, col("s").str().pad_end(5, "日"))[1], "-42日日")
    assert_equal(texts(df, col("s").str().pad_start(2))[4], "日本語")
    assert_equal(texts(df, col("s").str().zfill(5))[1], "-0042")
    assert_equal(texts(df, col("s").str().zfill(3))[2], "000")
    assert_equal(texts(df, col("s").str().zfill(1))[5], "null")


def test_concat_str_and_broadcasting() raises:
    var df = frame()
    assert_equal(
        texts(df, concat_str([col("t"), lit(String("-")), col("s")])),
        [String("a-  Héllo "), "b--42", "c-", "d-étoile", "e-日本語", "null"],
    )
    assert_equal(texts(df, concat_str([col("t"), col("t")], ", "))[0], "a, a")
    assert_equal(texts(df, concat_str([col("t")]))[5], "f")
    var scalar = DataFrame([], height=3).select(
        lit(String("Mojo")).str().to_uppercase().alias("u")
    )
    assert_equal(scalar.height(), 1)
    assert_equal(scalar.item(0, "u").string(), "MOJO")
    var filtered = df.filter(col("s").str().len_chars() > lit(Int64(2)))
    assert_equal(filtered.height(), 4)
    var e = (
        col("s").str().strip_chars().str().to_lowercase().str().pad_end(6, ".")
    )
    var reference = df.select(e.alias("v"), batch_size=64)
    for size in range(1, 6):
        assert_true(df.select(e.alias("v"), batch_size=size).equals(reference))


def test_validation() raises:
    var df = DataFrame(
        [Series("n", Column[Int64]([1])), Series("s", Column[String](["a"]))]
    )
    with assert_raises(
        contains="str operations require a string expression, found int64"
    ):
        _ = df.select(col("n").str().to_uppercase())
    with assert_raises(contains="pad fill_char must be one character"):
        _ = df.select(col("s").str().pad_start(3, "ab"))
    with assert_raises(contains="pad width must be nonnegative"):
        _ = df.select(col("s").str().pad_end(-1))
    with assert_raises(contains="slice length"):
        _ = df.select(col("s").str().slice(0, -2))
    with assert_raises(contains="concat_str requires string operands"):
        _ = df.select(concat_str([col("s"), col("n")]))
    with assert_raises(contains="at least one"):
        _ = concat_str(List[Expr]())
    assert_equal(df.select(col("s").str().len_bytes()).columns()[0], "s")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
