"""All as-of strategies, dtypes and null/tolerance/group combinations vs Polars."""
from std.collections import Optional
from std.python import Python
from std.testing import assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    AnyValue,
    DataType,
    import_arrow,
    export_arrow,
)
from dataframe.arrow import _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("asof_fixtures")
    var pa = Python.import_module("pyarrow")
    var cases = fixtures.cases()
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    var count = Int(py=cases.__len__())
    for index in range(count):
        var scenario = cases[index]
        scenario["left"]._export_to_c(a, s)
        var left = import_arrow(a, s)
        scenario["right"]._export_to_c(a, s)
        var right = import_arrow(a, s)
        var groups = List[String]()
        for i in range(Int(py=scenario["by"].__len__())):
            groups.append(String(py=scenario["by"][i]))
        var strategy = String(py=scenario["strategy"])
        var exact = Bool(py=scenario["exact"])
        var rk = "rk" if Bool(py=scenario["different"]) else "k"
        var tol = Optional[AnyValue]()
        if Bool(py=scenario["has_tolerance"]):
            var value = Int64(Int(py=scenario["tolerance"]))
            if Bool(py=scenario["unit"]):
                tol = AnyValue.temporal(
                    DataType.duration(String(py=scenario["unit"])), value
                )
            else:
                tol = AnyValue(value)
        var result = left.join_asof(
            right,
            left_on="k",
            right_on=rk,
            by=groups,
            strategy=strategy,
            tolerance=tol,
            suffix="_r",
            allow_exact_matches=exact,
        )
        export_arrow(result, a, s)
        var actual = pa.RecordBatch._import_from_c(a, s)
        try:
            assert_true(Bool(py=fixtures.check(scenario, actual)))
        except e:
            print("Failed as-of oracle scenario", index, strategy)
            print(scenario)
            raise e
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print("Polars as-of oracle passed:", count, "cases")
