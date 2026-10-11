"""Column names may hold any character (#438): the separator that joins
name lists in expressions (\\x1f) and its escape (\\x1e), commas, Unicode
and empty names come back exactly through windows, selectors, bin labels,
value counts, struct fields, struct group keys and lazy plans."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    as_struct,
    col,
    exclude,
)


def test_window_partition_with_separator_in_name() raises:
    # The issue's reproducer: ascending ordinal ranks per partition.
    var df = DataFrame(
        [
            Series("a\x1fb", Column[Int64]([1, 1, 2])),
            Series("v", Column[Int64]([3, 2, 1])),
        ]
    )
    var ranked = df.select(col("v").rank("ordinal").over("a\x1fb"))
    assert_equal(ranked["v"].get(0).int64(), Int64(2))
    assert_equal(ranked["v"].get(1).int64(), Int64(1))
    assert_equal(ranked["v"].get(2).int64(), Int64(1))


def test_many_partition_names_round_trip() raises:
    # Two partition columns with the separator, its escape and its escaped
    # forms in their names give the same sums as plain names.
    var odd: List[String] = ["x\x1ey\x1f", "\x1e1é, |"]
    var plain: List[String] = ["p", "q"]
    var p = Column[Int64]([1, 1, 2, 2, 1])
    var q = Column[Int64]([0, 1, 0, 0, 0])
    var v = Column[Int64]([10, 20, 30, 40, 50])
    var named = DataFrame(
        [
            Series(odd[0], p.copy()),
            Series(odd[1], q.copy()),
            Series("v", v.copy()),
        ]
    )
    var reference = DataFrame(
        [Series("p", p^), Series("q", q^), Series("v", v^)]
    )
    var got = named.select(col("v").sum().over(odd))
    var want = reference.select(col("v").sum().over(plain))
    assert_true(got.equals(want))


def test_selectors_and_name_prefixes() raises:
    var df = DataFrame(
        [
            Series("a\x1fb", Column[Int64]([1, 2])),
            Series("", Column[Int64]([3, 4])),
            Series("c\x1e0", Column[Int64]([5, 6])),
        ]
    )
    var names: List[String] = ["c\x1e0", "a\x1fb"]
    var picked = df.select(col(names))
    assert_equal(picked.columns(), names)
    var dropped: List[String] = ["a\x1fb"]
    var rest = df.select(exclude(dropped))
    assert_equal(rest.columns(), [String(""), "c\x1e0"])
    var prefixed = df.select(col(names).name_prefix("p\x1f"))
    assert_equal(prefixed.columns(), [String("p\x1fc\x1e0"), "p\x1fa\x1fb"])


def test_bin_labels_value_counts_and_struct_fields() raises:
    var df = DataFrame(
        [
            Series("a\x1fb", Column[Int64]([1, 1, 2])),
            Series("v", Column[Float64]([1.0, 2.0, 3.0])),
        ]
    )
    var labels: List[String] = ["lo\x1fw", "h\x1ei"]
    var binned = df.select(col("v").cut([1.5], labels=labels))
    assert_equal(binned["v"].get(0).string(), "lo\x1fw")
    assert_equal(binned["v"].get(2).string(), "h\x1ei")
    var counts = df.select(col("a\x1fb").value_counts(name="n\x1f"))
    assert_equal(counts.height(), 2)
    var packed = df.select(
        as_struct([col("v").alias("f\x1fx"), col("a\x1fb")], "s\x1f")
    )
    var structs = packed["s\x1f"].struct_column()
    assert_equal(structs.field(0).name(), "f\x1fx")
    assert_equal(structs.field(1).name(), "a\x1fb")


def test_struct_and_plain_group_keys_keep_their_names() raises:
    # A plain key with the separator in its name is not taken for an
    # expanded struct key, and a struct key's name and fields come back.
    var df = DataFrame(
        [
            Series("k\x1fp", Column[Int64]([1, 1, 2, 2])),
            Series("f\x1fx", Column[Int64]([7, 7, 8, 7])),
            Series("v", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    var keyed = df.with_columns([as_struct([col("f\x1fx")], "s\x1fq")])
    var keys: List[String] = ["k\x1fp", "s\x1fq"]
    var grouped = keyed.group_by(keys, maintain_order=True).agg(
        [col("v").sum().alias("t")]
    )
    assert_equal(grouped.columns(), [String("k\x1fp"), "s\x1fq", "t"])
    assert_equal(grouped.height(), 3)
    var structs = grouped["s\x1fq"].struct_column()
    assert_equal(structs.field(0).name(), "f\x1fx")
    assert_equal(grouped["t"].get(0).int64(), Int64(3))


def test_lazy_plans_tell_name_lists_apart() raises:
    # Dropping the column "a, b" and dropping "a" and "b" are different
    # plans; a self-join of the two must not reuse one for the other.
    var base = DataFrame(
        [
            Series("k", Column[Int64]([1, 2])),
            Series("a", Column[Int64]([3, 4])),
            Series("b", Column[Int64]([5, 6])),
            Series("a, b", Column[Int64]([7, 8])),
        ]
    )
    var one: List[String] = ["a, b"]
    var two: List[String] = ["a", "b"]
    # Each side reduces (unique), so the planner would share them if their
    # signatures matched.
    var joined = (
        base.lazy()
        .drop(one)
        .unique()
        .join(base.lazy().drop(two).unique(), on=["k"])
        .collect()
    )
    assert_equal(joined.columns(), [String("k"), "a", "b", "a, b"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
