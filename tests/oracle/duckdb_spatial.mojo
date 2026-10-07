"""DuckDB Spatial <-> Arrow C <-> Mojo geometry interoperability (#122)."""
from std.python import Python
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    DataFrame,
    import_arrow,
    export_arrow,
    from_geojson,
    to_wkb,
)
from dataframe.arrow import _at, _leak, _reclaim


def main() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/oracle")
    var fixtures = Python.import_module("duckdb_spatial_fixtures")
    var pa = Python.import_module("pyarrow")
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    # Unknown CRS and two PROJJSON CRSs, each with 32- and 64-bit offsets.
    for scenario in range(6):
        var original = fixtures.make(scenario)
        original._export_to_c(a, s)
        var frame = import_arrow(a, s)
        assert_equal(_at[ArrowArray](a)[].release, 0)
        assert_equal(_at[ArrowSchema](s)[].release, 0)
        assert_true(frame.column("geometry").dtype().is_geometry())
        assert_equal(frame.column("geometry").null_count(), 1)
        if scenario < 2:
            assert_equal(frame.column("geometry").crs(), "null")
        var rows = Python.list()
        for i in range(frame.height()):
            rows.append(i)
        export_arrow(frame, a, s)
        var back = pa.RecordBatch._import_from_c(a, s)
        assert_true(Bool(py=fixtures.check(original, back, scenario, rows)))
        # The result includes an empty geometry and a null, with reordered rows.
        var selected = DataFrame(
            [frame.column("geometry").take([2, 1, frame.height() - 1, 0])]
        )
        export_arrow(selected, a, s)
        back = pa.RecordBatch._import_from_c(a, s)
        assert_true(
            Bool(
                py=fixtures.check(
                    original,
                    back,
                    scenario,
                    Python.list(2, 1, frame.height() - 1, 0),
                )
            )
        )
    # Also consume geometry originating in Mojo, not just DuckDB-produced WKB.
    var native = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,2]}},{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[]}},{"type":"Feature","properties":{},"geometry":null}]}'
    )
    export_arrow(native, a, s)
    var native_back = pa.RecordBatch._import_from_c(a, s)
    var duckdb_back = fixtures.check_native(native_back)
    duckdb_back._export_to_c(a, s)
    var restored = import_arrow(a, s)
    assert_true(
        to_wkb(restored.column("geometry")).equals(
            to_wkb(native.column("geometry"))
        )
    )
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print(
        "DuckDB Spatial Arrow C WKB, CRS, null/empty and dimension oracle passed (6 cases)"
    )
