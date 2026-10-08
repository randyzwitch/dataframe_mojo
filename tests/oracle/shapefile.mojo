"""Shapefile geometry, attributes and CRS versus independent GIS readers."""
from std.python import Python
from std.sys import argv
from std.testing import assert_raises
from dataframe import read_shapefile, export_arrow, ArrowArray, ArrowSchema
from dataframe.arrow import _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("shapefile_fixtures")
    var pa = Python.import_module("pyarrow")
    var args = argv()
    var cases = fixtures.cases(String(args[1]))
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    var count = Int(py=cases.__len__())
    for i in range(count):
        var scenario = cases[i]
        var path = String(py=scenario["path"])
        print("Checking", path)
        var frame = read_shapefile(
            path, encoding=String(py=scenario["encoding"])
        )
        export_arrow(frame, a, s)
        _ = fixtures.check(scenario, pa.RecordBatch._import_from_c(a, s))
    var bad = fixtures.malformed(String(args[1]))
    for i in range(Int(py=bad.__len__())):
        with assert_raises(contains=String(py=bad[i]["message"])):
            _ = read_shapefile(String(py=bad[i]["path"]))
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print(
        "Shapefile oracle passed:",
        count,
        "datasets and",
        Int(py=bad.__len__()),
        "invalid inputs",
    )
