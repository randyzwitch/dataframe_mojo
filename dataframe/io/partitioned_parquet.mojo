"""Atomic eager Hive-partitioned Parquet datasets, retaining partition fields."""
from std.collections import Dict
from std.ffi import external_call
from std.os import listdir, mkdir
from std.os.path import exists, isdir
from dataframe.categorical import decode
from dataframe.frame import DataFrame
from dataframe.kernels.hashing import encode_rows
from dataframe.series import Series
from dataframe.temporal import zone_of
from .arrow import (
    ArrowArray,
    ArrowSchema,
    _c_string,
    _read_c_string,
    _release_imported,
    export_arrow,
)
from .csv import _cell_text, _cell_valid
from .dataset_options import HIVE_NULL, hive_encode
from .native_library import _Library, _load_library
from .parquet import _write_parquet
from .parquet_writer import _validate_path, _validate_schema


def _empty_destination(path: String) raises:
    if exists(path):
        if not isdir(path) or len(listdir(path)) != 0:
            raise Error(
                "partitioned Parquet destination must be absent or an empty directory: "
                + path
            )


struct _DatasetOutput(Movable):
    """Track only paths created inside our exclusive staging directory."""

    var destination: String
    var temporary: String
    var directories: List[String]
    var files: List[String]
    var pending: Bool

    def __init__(out self, destination: String) raises:
        self.destination = destination
        self.temporary = ""
        self.directories = List[String]()
        self.files = List[String]()
        self.pending = False
        var pattern = _c_string(destination + ".dataframe-XXXXXX")
        var address = external_call["mkdtemp", Int](pattern.unsafe_ptr())
        if address == 0:
            raise Error("cannot create staging directory beside " + destination)
        self.temporary = _read_c_string(address)
        _ = pattern^
        self.pending = True

    def __deinit__(deinit self):
        if self.pending:
            for file in self.files:
                var path = _c_string(file)
                _ = external_call["unlink", Int32](path.unsafe_ptr())
            for k in range(len(self.directories) - 1, -1, -1):
                var path = _c_string(self.directories[k])
                _ = external_call["rmdir", Int32](path.unsafe_ptr())
            var root = _c_string(self.temporary)
            _ = external_call["rmdir", Int32](root.unsafe_ptr())

    def file_path(mut self, components: List[String]) raises -> String:
        var directory = self.temporary
        for component in components:
            directory += "/" + component
            if not exists(directory):
                mkdir(directory)
                self.directories.append(directory)
        var path = directory + "/part-00000.parquet"
        # Register before writing so even an incomplete native file is removed.
        self.files.append(path)
        return path

    def publish(mut self) raises:
        _empty_destination(self.destination)
        var source = _c_string(self.temporary)
        var destination = _c_string(self.destination)
        var status = external_call["rename", Int32](
            source.unsafe_ptr(), destination.unsafe_ptr()
        )
        _ = source^
        _ = destination^
        if status != 0:
            raise Error(
                "cannot publish partitioned Parquet directory: "
                + self.destination
            )
        self.pending = False


def write_partitioned_parquet(
    frame: DataFrame,
    path: String,
    *,
    partition_by: List[String],
    compression: String = "zstd",
    row_group_size: Int = 1_000_000,
) raises:
    """Write one part-00000.parquet per Hive key tuple, atomically.

    Partition fields remain in the files, and rows retain input order within
    each partition. Names and scalar values use reversible UTF-8 percent
    escapes; null uses the Hive sentinel. An empty input creates an empty
    root without files. Existing nonempty destinations are rejected.
    """
    _validate_path(path)
    _validate_schema(frame)
    if len(partition_by) == 0:
        raise Error("partition_by must contain at least one column")
    if compression not in ["zstd", "snappy", "uncompressed"]:
        raise Error("write_parquet: unsupported compression: " + compression)
    if row_group_size <= 0:
        raise Error("write_parquet: row_group_size must be positive")
    var destination = path
    while destination.byte_length() > 1 and destination.endswith("/"):
        var trimmed = String(
            destination[byte = 0 : destination.byte_length() - 1]
        )
        destination = trimmed^
    _empty_destination(destination)
    var keys = List[Series]()
    var names = List[String]()
    for name in partition_by:
        if name == "" or name in names:
            raise Error("partition_by requires unique nonempty column names")
        var key = frame.column(name)
        if key.dtype().is_nested() or key.dtype()._is_bytes():
            raise Error("unsupported partition column dtype: " + name)
        if key.dtype().is_categorical():
            key = decode(key)
        keys.append(key^)
        names.append(name)
    var groups = encode_rows(keys, nulls_equal=True)
    var paths = List[List[String]]()
    var seen = Dict[String, Int]()
    # Render all paths first: distinct logical keys must not silently merge.
    for representative in groups.representatives:
        var components = List[String]()
        var relative = String()
        for k in range(len(keys)):
            var value = HIVE_NULL
            if _cell_valid(keys[k], representative):
                value = hive_encode(
                    _cell_text(
                        keys[k], representative, zone_of(keys[k].dtype())
                    )
                )
            var component = hive_encode(names[k]) + "=" + value
            components.append(component)
            relative += "/" + component
        if relative in seen:
            raise Error(
                "partition keys render to the same Hive path: " + relative
            )
        seen[relative] = len(paths)
        paths.append(components^)
    var rows = List[List[Int]](capacity=groups.count())
    for _ in range(groups.count()):
        rows.append(List[Int]())
    for row in range(frame.height()):
        rows[groups.ids[row]].append(row)
    # Validate Arrow types and library availability before creating any paths.
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(frame.clear(), array, schema)
    _release_imported(array, schema)
    var library = _load_library()
    _ = _Library._symbol(library.handle, "dfq_write_parquet")
    var output = _DatasetOutput(destination)
    for group in range(groups.count()):
        var file = output.file_path(paths[group])
        _write_parquet(
            frame.take(rows[group]), file, compression, row_group_size, ""
        )
    output.publish()
