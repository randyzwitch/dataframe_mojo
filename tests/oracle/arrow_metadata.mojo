"""Registered and unknown extension types round-trip through PyArrow."""
from std.python import Python
from std.testing import assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    export_arrow,
    import_arrow,
    DataFrame,
    concat,
)
from dataframe.arrow import _at, _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("arrow_metadata_fixtures")
    var pa = Python.import_module("pyarrow")
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    for registered in range(2):
        var original = fixtures.make(registered)
        original._export_to_c(a, s)
        var frame = import_arrow(a, s)
        # Release all producer references before export.
        original = Python.none()
        export_arrow(frame, a, s)
        var back = pa.RecordBatch._import_from_c(a, s)
        assert_true(
            Bool(py=fixtures.check(back, registered, Python.list(0, 1, 2, 3)))
        )
        var taken = DataFrame([frame.column("x").take([3, 1, 0])])
        export_arrow(taken, a, s)
        back = pa.RecordBatch._import_from_c(a, s)
        assert_true(
            Bool(py=fixtures.check(back, registered, Python.list(3, 1, 0)))
        )
        var sliced = concat([frame.slice(2, 2), frame.slice(0, 2)])
        export_arrow(sliced, a, s)
        back = pa.RecordBatch._import_from_c(a, s)
        assert_true(
            Bool(py=fixtures.check(back, registered, Python.list(2, 3, 0, 1)))
        )
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print("Arrow field metadata and registered extension PyArrow oracle passed")
