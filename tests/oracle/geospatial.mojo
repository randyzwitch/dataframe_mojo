"""Independent PyArrow producer/consumer checks for geospatial interchange."""
from std.python import Python
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    import_arrow,
    export_arrow,
    read_geojson,
    read_parquet,
    write_geoparquet,
    DataType,
)
from dataframe.arrow import _at, _leak, _reclaim


def main() raises:
    var pa = Python.import_module("pyarrow")
    var binary_struct = Python.import_module("struct")
    var a = _leak(ArrowArray())
    var s = _leak(ArrowSchema())
    var wkb = binary_struct.pack("<BIdd", 1, 1, 1.0, 2.0)
    var none = Python.none()
    # No Mojo exporter participates in this producer. Both Arrow offset widths
    # and sliced arrays carry independently constructed extension metadata.
    for large in range(2):
        var dtype = pa.large_binary() if large else pa.binary()
        var field = pa.field("geometry", dtype, metadata=Python.dict())
        var metadata = Python.dict()
        metadata["ARROW:extension:name"] = "geoarrow.wkb"
        metadata["ARROW:extension:metadata"] = '{"crs":"OGC:CRS84"}'
        metadata["unrelated:binary"] = Python.import_module("builtins").bytes(
            Python.list(255, 0, 128)
        )
        field = field.with_metadata(metadata)
        var schema = pa.schema(Python.list(field))
        var data = pa.array(Python.list(wkb, none, wkb), type=dtype)
        var batch = pa.RecordBatch.from_arrays(
            Python.list(data), schema=schema
        ).slice(1, 2)
        batch._export_to_c(a, s)
        var frame = import_arrow(a, s)
        assert_true(frame.column("geometry").dtype().is_geometry())
        assert_equal(frame.column("geometry").null_count(), 1)
        assert_true(frame.column("geometry").get(0).is_null())
        export_arrow(frame, a, s)
        var back = pa.RecordBatch._import_from_c(a, s)
        back.validate(full=True)
        assert_true(
            Bool(py=back.column(0).to_pylist() == Python.list(none, wkb))
        )
        assert_equal(
            String(
                back.schema.field(0)
                .metadata.get(
                    Python.import_module("builtins").bytes(
                        "ARROW:extension:name", "utf-8"
                    )
                )
                .decode()
            ),
            "geoarrow.wkb",
        )
    _ = _reclaim[ArrowArray](a)
    _ = _reclaim[ArrowSchema](s)
    print("GeoArrow PyArrow producer/consumer oracle passed")
