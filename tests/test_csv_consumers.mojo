"""Public CSV chunked output remains usable by dataframe consumers."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import CsvField, CsvSchema, read_csv, col, lit, to_csv_string


def test_public_csv_chunks_match_contiguous_consumers() raises:
    var path = String("/tmp/dataframe_mojo_csv_consumer_replacement.csv")
    with open(path, "w") as output:
        output.write("id,label,value\n")
        for i in range(5000):
            output.write(
                String(i)
                + ",group"
                + String(i % 13)
                + ","
                + String(i % 97)
                + ".5\n"
            )
    var frame = read_csv(
        path,
        CsvSchema(
            [
                CsvField.int64("id"),
                CsvField.string("label"),
                CsvField.float64("value"),
            ]
        ),
    )
    assert_equal(frame.height(), 5000)
    assert_true(frame.column("id").n_chunks() > 1)
    var flat = frame.rechunk()
    assert_true(frame.equals(flat))
    assert_true(
        frame.select(col("value").sum()).equals(flat.select(col("value").sum()))
    )
    assert_true(
        frame.filter(col("id") > lit(Int64(4980))).equals(
            flat.filter(col("id") > lit(Int64(4980)))
        )
    )
    assert_true(frame.sort(["label", "id"]).equals(flat.sort(["label", "id"])))
    assert_true(
        frame.group_by(["label"], maintain_order=True)
        .agg([col("value").sum()])
        .equals(
            flat.group_by(["label"], maintain_order=True).agg(
                [col("value").sum()]
            )
        )
    )
    var right = flat.head(3)
    assert_true(
        frame.join(right, on=["id"], how="inner").equals(
            flat.join(right, on=["id"], how="inner")
        )
    )
    assert_equal(to_csv_string(frame), to_csv_string(flat))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
