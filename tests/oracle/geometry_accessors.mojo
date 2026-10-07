"""Independent GeoPandas/GEOS validation of WKB types, bounds and CRS."""
from std.python import Python
from std.testing import assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    DataFrame,
    import_arrow,
    export_arrow,
)
from dataframe.arrow import _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("geometry_accessors_fixtures")
    var pa = Python.import_module("pyarrow")
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    for order in range(2):
        var original = fixtures.make(order)
        original._export_to_c(a, s)
        var frame = import_arrow(a, s)
        var geo = frame.column("geometry")
        var total = geo.total_bounds().value().copy()
        export_arrow(frame, a, s)
        var back = pa.RecordBatch._import_from_c(a, s)
        export_arrow(geo.bounding_box(), a, s)
        var bounds = pa.RecordBatch._import_from_c(a, s)
        export_arrow(DataFrame([geo.geometry_type()]), a, s)
        var types = pa.RecordBatch._import_from_c(a, s)
        assert_true(
            Bool(
                py=fixtures.check(
                    original,
                    back,
                    bounds,
                    types,
                    Python.list(total.xmin, total.ymin, total.xmax, total.ymax),
                )
            )
        )
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print("GeoPandas/GEOS geometry types, bounds, CRS and WKB oracle passed")
