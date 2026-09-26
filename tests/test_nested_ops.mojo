"""List and struct operations: str.split, explode, the .list namespace,
struct packing, unnest and field access, lazy plans, Arrow round trips,
and the errors unsupported operations raise."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    export_arrow_series,
    import_arrow_series,
    lit,
    write_csv,
)
from dataframe.nested_column import ListColumn, StructColumn


def words() raises -> DataFrame:
    var valid: List[Bool] = [True, True, True, True, False]
    return DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3, 4, 5])),
            Series(
                "s", StringColumn(["a,b", "", ",x,", "solo", "gone"], valid)
            ),
        ]
    )


def test_split_and_explode_match_polars_rules() raises:
    var frame = words().with_columns(col("s").str().split(",").alias("parts"))
    assert_equal(frame.column("parts").dtype(), DataType.list(DataType.STRING))
    assert_equal(String(frame.column("parts").get(0)), "[a, b]")
    # An empty string splits to one empty part; separators at both ends
    # give empty parts at both ends; null stays null.
    assert_equal(String(frame.column("parts").get(1)), "[]")
    assert_equal(len(frame.column("parts").get(1).list()), 1)
    assert_equal(String(frame.column("parts").get(2)), "[, x, ]")
    assert_true(frame.column("parts").get(4).is_null())
    var exploded = frame.explode("parts")
    # 2 + 1 + 3 + 1 + 1(null) rows
    assert_equal(exploded.height(), 8)
    assert_equal(exploded.column("parts").dtype(), DataType.STRING)
    assert_equal(exploded.column("k").get(0).int64(), 1)
    assert_equal(exploded.column("k").get(1).int64(), 1)
    assert_equal(exploded.column("parts").get(1).string(), "b")
    assert_equal(exploded.column("parts").get(2).string(), "")
    assert_equal(exploded.column("parts").get(3).string(), "")
    assert_equal(exploded.column("parts").get(4).string(), "x")
    assert_true(exploded.column("parts").get(7).is_null())
    assert_equal(exploded.column("k").get(7).int64(), 5)
    var inclusive = words().select_exprs(
        [col("s").str().split(",", inclusive=True).alias("p")]
    )
    assert_equal(String(inclusive.column("p").get(0)), "[a,, b]")
    assert_equal(String(inclusive.column("p").get(2)), "[,, x,, ]")


def test_explode_several_columns_and_empty_lists() raises:
    var a = ListColumn.from_lists(
        [
            Series("item", Column[Int64]([1, 2])),
            Series("item", Column[Int64](List[Int64]())),
            Series("item", Column[Int64]([3])),
        ],
        DataType.INT64,
        [True, False, True],
    )
    var b = ListColumn.from_lists(
        [
            Series("item", StringColumn(["x", "y"])),
            Series("item", StringColumn(List[String]())),
            Series("item", StringColumn(["z"])),
        ],
        DataType.STRING,
    )
    var frame = DataFrame(
        [
            Series("a", a^),
            Series("b", b^),
            Series("k", Column[Int64]([7, 8, 9])),
        ]
    )
    var out = frame.explode(["a", "b"])
    assert_equal(out.height(), 4)
    assert_equal(out.column("a").get(1).int64(), 2)
    assert_equal(out.column("b").get(1).string(), "y")
    assert_true(out.column("a").get(2).is_null())
    assert_true(out.column("b").get(2).is_null())
    assert_equal(out.column("k").get(2).int64(), 8)
    assert_equal(out.column("a").get(3).int64(), 3)
    var raised = False
    try:
        _ = frame.explode(["a", "k"])
    except e:
        raised = True
        assert_true("needs list columns" in String(e))
    assert_true(raised)
    var short = ListColumn.from_lists(
        [
            Series("item", Column[Int64]([1])),
            Series("item", Column[Int64]([2])),
            Series("item", Column[Int64]([3])),
        ],
        DataType.INT64,
    )
    var mismatched = frame.with_column(Series("c", short^))
    raised = False
    try:
        _ = mismatched.explode(["a", "c"])
    except e:
        raised = True
        assert_true("different element counts" in String(e))
    assert_true(raised)


def numbers() raises -> DataFrame:
    # xs: [4, 1, null], [], [7, 2], null
    var valid: List[Bool] = [True, True, True, False]
    var element_valid: List[Bool] = [True, True, False, True, True]
    var offsets: List[Int64] = [0, 3, 3, 5, 5]
    var child = Series("item", Column[Int64]([4, 1, 9, 7, 2], element_valid))
    return DataFrame([Series("xs", ListColumn(offsets^, child^, _bits(valid)))])


def _bits(valid: List[Bool]) -> List[UInt8]:
    var bits = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    return bits^


def test_list_namespace() raises:
    # xs: [4, 1, null], [], [7, 2], null
    var frame = numbers()
    var out = frame.select_exprs(
        [
            col("xs").list().len().alias("n"),
            col("xs").list().get(0).alias("first"),
            col("xs").list().last().alias("last"),
            col("xs").list().get(5).alias("far"),
            col("xs").list().contains(Int64(7)).alias("has7"),
            col("xs").list().sum().alias("sum"),
            col("xs").list().min().alias("min"),
            col("xs").list().max().alias("max"),
            col("xs").list().mean().alias("mean"),
        ]
    )
    assert_equal(out.column("n").get(0).int64(), 3)
    assert_equal(out.column("n").get(1).int64(), 0)
    assert_true(out.column("n").get(3).is_null())
    assert_equal(out.column("first").get(0).int64(), 4)
    assert_true(out.column("first").get(1).is_null())
    assert_true(out.column("last").get(0).is_null())
    assert_equal(out.column("last").get(2).int64(), 2)
    assert_true(out.column("far").get(0).is_null())
    assert_equal(out.column("has7").get(2).bool(), True)
    assert_equal(out.column("has7").get(0).bool(), False)
    assert_true(out.column("has7").get(3).is_null())
    assert_equal(out.column("sum").get(0).int64(), 5)
    assert_equal(out.column("sum").get(1).int64(), 0)
    assert_true(out.column("sum").get(3).is_null())
    assert_equal(out.column("min").get(0).int64(), 1)
    assert_equal(out.column("max").get(2).int64(), 7)
    assert_true(out.column("min").get(1).is_null())
    assert_equal(out.column("mean").get(2).float64(), 4.5)
    var texts = words().select_exprs(
        [
            col("s").str().split(",").list().join("-").alias("j"),
            col("s").str().split(",").list().contains("x").alias("hasx"),
        ]
    )
    assert_equal(texts.column("j").get(0).string(), "a-b")
    assert_equal(texts.column("j").get(2).string(), "-x-")
    assert_true(texts.column("j").get(4).is_null())
    assert_equal(texts.column("hasx").get(2).bool(), True)
    assert_equal(texts.column("hasx").get(0).bool(), False)


def test_struct_pack_unnest_and_field() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([1, 2, 3])),
            Series("b", StringColumn(["x", "y", "z"])),
            Series("c", Column[Float64]([0.5, 1.5, 2.5])),
        ]
    )
    var packed = frame.pack_struct("s", ["a", "b"])
    assert_equal(packed.width(), 4)
    assert_equal(
        packed.column("s").dtype().name(), "struct[a: int64, b: string]"
    )
    var fields = packed.select_exprs(
        [col("s").field("a").alias("fa"), col("s").field("b").alias("fb")]
    )
    assert_equal(fields.column("fa").get(2).int64(), 3)
    assert_equal(fields.column("fb").get(0).string(), "x")
    var back = packed.drop(["a", "b"]).unnest("s")
    assert_equal(back.columns()[0], "c")
    assert_equal(back.columns()[1], "a")
    assert_equal(back.columns()[2], "b")
    assert_true(
        back.select(["a", "b", "c"]).equals(frame.select(["a", "b", "c"]))
    )
    var raised = False
    try:
        _ = packed.unnest("s")
    except e:
        raised = True
        assert_true("clashes" in String(e))
    assert_true(raised)
    # Null struct rows null every field.
    var bits = _bits([True, False, True])
    var nullable = DataFrame(
        [
            Series(
                "s",
                StructColumn(
                    [
                        Series("a", Column[Int64]([1, 2, 3])),
                        Series("b", StringColumn(["x", "y", "z"])),
                    ],
                    bits^,
                ),
            )
        ]
    )
    var opened = nullable.unnest("s")
    assert_true(opened.column("a").get(1).is_null())
    assert_true(opened.column("b").get(1).is_null())
    assert_equal(opened.column("a").get(2).int64(), 3)
    var via_field = nullable.select_exprs([col("s").field("a")])
    assert_true(via_field.column("a").get(1).is_null())


def test_lazy_explode_and_unnest() raises:
    var frame = words().with_columns(col("s").str().split(",").alias("parts"))
    var eager = frame.explode("parts").filter(col("k") > lit(Int64(2)))
    var lazy = (
        frame.lazy().explode("parts").filter(col("k") > lit(Int64(2))).collect()
    )
    assert_true(lazy.equals(eager))
    var plan = frame.lazy().explode("parts").select(["parts"]).explain()
    assert_true("EXPLODE parts" in plan)
    var narrow = frame.lazy().explode("parts").select(["parts"]).collect()
    assert_equal(narrow.width(), 1)
    assert_equal(narrow.height(), 8)
    var packed = frame.pack_struct("s", ["k"]).drop(["k"])
    var opened = packed.lazy().unnest("s").select(["k"]).collect()
    assert_equal(opened.height(), 5)
    assert_equal(opened.column("k").get(4).int64(), 5)
    assert_true("UNNEST s" in packed.lazy().unnest("s").explain())


def round_trip(series: Series) raises -> Series:
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(series, array, schema)
    return import_arrow_series(array, schema)


def test_arrow_round_trips_nested() raises:
    var xs = numbers().column("xs")
    var back = round_trip(xs)
    assert_equal(back.dtype(), xs.dtype())
    assert_true(back.equals(xs))
    assert_true(back.get(3).is_null())
    var frame = words().with_columns(col("s").str().split(",").alias("parts"))
    var parts = frame.column("parts")
    assert_true(round_trip(parts).equals(parts))
    var sliced = parts.slice(1, 3)
    assert_true(round_trip(sliced).equals(sliced))
    # A struct holding a list field, with nulls at the struct level.
    var bits = _bits([True, False, True, True, True])
    var s = Series(
        "s",
        StructColumn(
            [Series("k", Column[Int64]([1, 2, 3, 4, 5])), parts.renamed("p")],
            bits^,
        ),
    )
    var again = round_trip(s)
    assert_equal(again.dtype(), s.dtype())
    assert_true(again.equals(s))
    var nested = Series(
        "ll",
        ListColumn.from_lists(
            [parts.slice(0, 2), parts.slice(2, 3)], parts.dtype()
        ),
    )
    assert_equal(nested.dtype().name(), "list[list[string]]")
    assert_true(round_trip(nested).equals(nested))


def test_unsupported_operations_raise_clearly() raises:
    var frame = words().with_columns(col("s").str().split(",").alias("parts"))
    var messages = List[String]()
    try:
        _ = frame.group_by("parts").agg(col("k").sum())
    except e:
        messages.append(String(e))
    try:
        _ = frame.select_exprs([col("parts").sum()])
    except e:
        messages.append(String(e))
    try:
        _ = frame.select_exprs([col("parts").cast(DataType.STRING)])
    except e:
        messages.append(String(e))
    try:
        write_csv(frame, "/tmp/should_not_exist.csv")
    except e:
        messages.append(String(e))
    try:
        _ = frame.select_exprs([col("parts") == col("parts")])
    except e:
        messages.append(String(e))
    assert_equal(len(messages), 5)
    assert_true("cannot be keys" in messages[0])
    assert_true("list or struct" in messages[1])
    assert_true("cast" in messages[2])
    assert_true("CSV" in messages[3])
    assert_true("list or struct" in messages[4])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
