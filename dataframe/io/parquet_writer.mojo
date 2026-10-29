"""Incremental Parquet ownership and atomic local output publication."""
from std.ffi import external_call
from std.memory import Pointer
from dataframe.frame import DataFrame
from .arrow import (
    ArrowArray,
    ArrowSchema,
    _c_string,
    _read_c_string,
    _release_imported,
    export_arrow,
)
from .native_library import _Library, _load_library, _raise_backend_error


def _validate_path(path: String) raises:
    if path == "" or "\x00" in path:
        raise Error("output path must be nonempty and contain no NUL bytes")


def _validate_schema(frame: DataFrame) raises:
    for name in frame.columns():
        if "\x00" in name:
            raise Error("output column names must not contain NUL bytes")


def _check_schema(expected: DataFrame, actual: DataFrame) raises:
    if expected.columns() != actual.columns():
        raise Error("output batch column names changed")
    for k in range(expected.width()):
        var expected_type = expected._columns[k].dtype()
        var actual_type = actual._columns[k].dtype()
        # Dictionary labels and code order belong to each batch's data;
        # Arrow's dictionary schema describes only index and value types.
        if expected_type.is_categorical() and actual_type.is_categorical():
            continue
        if expected_type != actual_type:
            raise Error(
                "output batch dtype changed: " + expected._columns[k].name()
            )


struct _AtomicOutput(Movable):
    """Own one exclusive temporary file beside the destination until rename."""

    var destination: String
    var temporary: String
    var pending: Bool

    def __init__(out self, destination: String) raises:
        _validate_path(destination)
        self.destination = destination
        self.temporary = ""
        self.pending = False
        var pattern = _c_string(destination + ".dataframe-XXXXXX")
        var fd = external_call["mkstemp", Int32](pattern.unsafe_ptr())
        if fd < 0:
            raise Error("cannot create temporary output beside " + destination)
        self.temporary = _read_c_string(Int(pattern.unsafe_ptr()))
        self.pending = True
        var closed = external_call["close", Int32](fd)
        _ = pattern ^
        if closed != 0:
            raise Error("cannot close temporary output: " + self.temporary)

    def __deinit__(deinit self):
        if self.pending:
            var path = _c_string(self.temporary)
            _ = external_call["unlink", Int32](path.unsafe_ptr())

    def publish(mut self) raises:
        var source = _c_string(self.temporary)
        var destination = _c_string(self.destination)
        if (
            external_call["rename", Int32](
                source.unsafe_ptr(), destination.unsafe_ptr()
            )
            != 0
        ):
            raise Error("cannot replace output: " + self.destination)
        self.pending = False


comptime _Open = def(Int, Int, Int, Int64, Int, Int) thin abi("C") -> Int32
comptime _Write = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime _Close = def(Int, Int) thin abi("C") -> Int32
comptime _Abort = def(Int) thin abi("C") -> None


def _open_writer(
    entry: Int,
    path: String,
    compression: String,
    row_group_size: Int,
    mut schema: ArrowSchema,
    mut handle: Int,
    mut error: Int,
) -> Int32:
    var filename = _c_string(path)
    var codec = _c_string(compression)
    var status = Pointer(to=entry).unsafe_bitcast[_Open]()[](
        Int(filename.unsafe_ptr()),
        Int(Pointer(to=schema)),
        Int(codec.unsafe_ptr()),
        Int64(row_group_size),
        Int(Pointer(to=handle)),
        Int(Pointer(to=error)),
    )
    _ = filename ^
    _ = codec ^
    return status


def _write_batch(
    entry: Int,
    handle: Int,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) -> Int32:
    # This mut boundary makes the consuming C import visible to Mojo's
    # ownership analysis; both release fields can change through raw pointers.
    return Pointer(to=entry).unsafe_bitcast[_Write]()[](
        handle,
        Int(Pointer(to=array)),
        Int(Pointer(to=schema)),
        Int(Pointer(to=error)),
    )


def _close_writer(entry: Int, handle: Int, mut error: Int) -> Int32:
    return Pointer(to=entry).unsafe_bitcast[_Close]()[](
        handle, Int(Pointer(to=error))
    )


struct _ParquetWriter(Movable):
    var library: _Library
    var output: _AtomicOutput
    var handle: Int
    var write: Int
    var close: Int
    var abort: Int
    var schema: DataFrame

    def __init__(
        out self,
        frame: DataFrame,
        path: String,
        compression: String = "zstd",
        row_group_size: Int = 1_000_000,
    ) raises:
        _validate_path(path)
        _validate_schema(frame)
        if compression not in ["zstd", "snappy", "uncompressed"]:
            raise Error(
                "write_parquet: unsupported compression: " + compression
            )
        if row_group_size <= 0:
            raise Error("write_parquet: row_group_size must be positive")
        self.library = _load_library()
        self.handle = 0
        self.write = _Library._symbol(
            self.library.handle, "dfq_parquet_writer_write"
        )
        self.close = _Library._symbol(
            self.library.handle, "dfq_parquet_writer_close"
        )
        self.abort = _Library._symbol(
            self.library.handle, "dfq_parquet_writer_abort"
        )
        var entry = _Library._symbol(
            self.library.handle, "dfq_parquet_writer_open"
        )
        self.schema = frame.clear()
        self.output = _AtomicOutput(path)
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(self.schema, array, schema)
        try:
            var error = 0
            var status = _open_writer(
                entry,
                self.output.temporary,
                compression,
                row_group_size,
                schema,
                self.handle,
                error,
            )
            _release_imported(array, schema)
            if status != 0:
                _raise_backend_error(self.library, error, "write_parquet")
        except caught:
            _release_imported(array, schema)
            raise caught ^

    def __deinit__(deinit self):
        if self.handle != 0:
            var entry = self.abort
            Pointer(to=entry).unsafe_bitcast[_Abort]()[](self.handle)

    def append(mut self, frame: DataFrame) raises:
        if self.handle == 0:
            raise Error("Parquet writer is closed")
        _check_schema(self.schema, frame)
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(frame, array, schema)
        try:
            var error = 0
            var entry = self.write
            var status = _write_batch(
                entry,
                self.handle,
                array,
                schema,
                error,
            )
            _release_imported(array, schema)
            if status != 0:
                _raise_backend_error(self.library, error, "write_parquet")
        except caught:
            _release_imported(array, schema)
            raise caught ^

    def finish(mut self) raises:
        if self.handle == 0:
            raise Error("Parquet writer is closed")
        var handle = self.handle
        self.handle = 0
        var error = 0
        var entry = self.close
        if _close_writer(entry, handle, error) != 0:
            _raise_backend_error(self.library, error, "write_parquet")
        self.output.publish()
