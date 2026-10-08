"""Time and row windows against independent Polars outputs."""
from std.python import Python
from std.collections import Optional
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Expr,
    col,
    import_arrow,
    export_arrow,
)
from dataframe.arrow import _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("time_window_fixtures")
    var pa = Python.import_module("pyarrow")
    var cases = fixtures.cases()
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    var count = Int(py=cases.__len__())
    for i in range(count):
        var scenario = cases[i]
        scenario["data"]._export_to_c(a, s)
        var df = import_arrow(a, s)
        var mode = String(py=scenario["mode"])
        var period = String(py=scenario["period"])
        var closed = String(py=scenario["closed"])
        var grouped = Bool(py=scenario["grouped"])
        var needed = Int(py=scenario["needed"])
        var by = List[String](["g"]) if grouped else List[String]()
        var exprs: List[Expr]
        if mode == "row":
            var size = Int(py=scenario["size"])
            var ddof = Int(py=scenario["ddof"])
            exprs = [
                col("v").rolling_std(size, needed, ddof).alias("std"),
                col("v").rolling_var(size, needed, ddof).alias("var"),
            ]
        elif mode == "by":
            exprs = [
                col("v")
                .rolling_sum_by("t", period, closed, needed)
                .alias("sum"),
                col("v")
                .rolling_mean_by("t", period, closed, needed)
                .alias("mean"),
                col("v")
                .rolling_min_by("t", period, closed, needed)
                .alias("min"),
                col("v")
                .rolling_max_by("t", period, closed, needed)
                .alias("max"),
                col("v")
                .rolling_std_by("t", period, closed, needed)
                .alias("std"),
                col("v")
                .rolling_var_by("t", period, closed, needed)
                .alias("var"),
            ]
        else:
            exprs = [
                col("v").sum().alias("sum"),
                col("v").mean().alias("mean"),
                col("v").min().alias("min"),
                col("v").max().alias("max"),
                col("v").std().alias("std"),
                col("v").var().alias("var"),
            ]
        var result = df.head(0)
        if mode == "row" or mode == "by":
            if grouped:
                for j in range(len(exprs)):
                    exprs[j] = exprs[j].over("g")
            result = df.select_exprs(exprs)
        else:
            var off = String(py=scenario["offset"])
            var offset = Optional[String]()
            if off != "default":
                offset = off
            if mode == "rolling":
                result = df.rolling("t", period, offset, closed, by).agg(exprs)
            else:
                result = df.group_by_dynamic(
                    "t",
                    String(py=scenario["every"]),
                    period,
                    offset,
                    closed,
                    String(py=scenario["label"]),
                    by,
                ).agg(exprs)
        export_arrow(result, a, s)
        var actual = pa.RecordBatch._import_from_c(a, s)
        try:
            _ = fixtures.check(scenario, actual)
        except e:
            print("Failed time-window scenario", i, mode, period, closed)
            print(scenario)
            raise e
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print("Polars time-window oracle passed:", count, "cases")
