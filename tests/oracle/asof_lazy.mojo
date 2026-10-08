"""Lazy as-of oracle, separate from embedded Python's native loader."""
from std.collections import Optional
from std.sys import argv
from std.testing import assert_true
from dataframe import (
    AnyValue,
    DataType,
    read_parquet,
    scan_parquet,
    write_parquet,
    col,
)
from dataframe.json_value import JsonValue


def main() raises:
    var directory = String(argv()[1])
    var text: String
    with open(directory + "/cases.json", "r") as file:
        text = file.read()
    var scenarios = JsonValue(text).items()
    for i in range(len(scenarios)):
        ref spec = scenarios[i]
        var path = directory + "/" + String(i)
        var l = read_parquet(path + "-left.parquet")
        var r = read_parquet(path + "-right.parquet")
        var groups = List[String]()
        for name in spec.get("by").items():
            groups.append(name.string())
        var tol = Optional[AnyValue]()
        if spec.get("tolerance").text != "null":
            var ticks = Int64(Int(spec.get("tolerance").text))
            if spec.get("unit").text == "null":
                tol = AnyValue(ticks)
            else:
                tol = AnyValue.temporal(
                    DataType.duration(spec.get("unit").string()), ticks
                )
        var right_key = "rk" if spec.get("different").text == "true" else "k"
        var plan = scan_parquet(path + "-left.parquet").join_asof(
            scan_parquet(path + "-right.parquet"),
            left_on="k",
            right_on=right_key,
            by=groups,
            strategy=spec.get("strategy").string(),
            tolerance=tol,
            suffix="_r",
            allow_exact_matches=spec.get("exact").text == "true",
        )
        var result = plan.collect(
            streaming=i % 2 == 0, optimize=i % 4 != 0, batch_size=3
        )
        var expected = l.join_asof(
            r,
            left_on="k",
            right_on=right_key,
            by=groups,
            strategy=spec.get("strategy").string(),
            tolerance=tol,
            suffix="_r",
            allow_exact_matches=spec.get("exact").text == "true",
        )
        assert_true(result.equals(expected))
        write_parquet(result, path + "-out.parquet")
        assert_true(
            plan.select(["k"])
            .head(3)
            .collect(batch_size=2)
            .equals(expected.select(["k"]).head(3))
        )
    print("Lazy as-of Parquet oracle passed:", len(scenarios), "cases")
