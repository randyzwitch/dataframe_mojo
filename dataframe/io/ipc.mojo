"""Arrow IPC/Feather v2 files and IPC streams through libdfparquet.

Readers import one record batch at a time and retain batches as column
chunks. Import currently copies foreign buffers, including when memory_map
selects a mapped native input. Lazy scans and foreign-buffer ownership are
separate features. Writers export aligned chunk windows instead of flattening
the complete frame. No Python dependency is required at runtime.
"""
from std.collections import Optional
from std.memory import Pointer
from ..frame import DataFrame, Field, concat, _shared_dictionaries
from ..series import Series
from ..nested_column import (
    ListColumn,
    StructColumn,
    _hold,
    _hold_many,
    _fields_at,
)
from ..parallel import Pool
from .arrow import (
    ArrowArray,
    ArrowSchema,
    _c_string,
    _release_imported,
    _read_c_string,
    _import_arrow_with_pool,
    _arrow_import_workers,
    import_arrow,
    export_arrow,
)
from .parquet import (
    _Library,
    _load_library,
    _raise_backend_error,
    _ArrowArrayStream,
    _release_stream,
    _stream_next,
    _stream_schema,
    _StreamError,
)

comptime _Read = def(Int, Int32, Int32, Int, Int32, Int64, Int, Int) thin abi(
    "C"
) -> Int32
comptime _Schema = def(Int, Int32, Int, Int, Int) thin abi("C") -> Int32
comptime _Open = def(Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32
comptime _Write = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime _Close = def(Int, Int) thin abi("C") -> Int32
comptime _Abort = def(Int) thin abi("C") -> None


def _path(path: String) raises:
    if path == "" or "\x00" in path:
        raise Error("IPC requires a nonempty path without NUL bytes")


def _error(mut stream: _ArrowArrayStream) raises:
    var entry = stream.get_last_error
    var message = _read_c_string(
        Pointer(to=entry).unsafe_bitcast[_StreamError]()[](
            Int(Pointer(to=stream))
        )
    )
    raise Error("read_ipc stream: " + message)


def _align_children(parts: List[Series]) raises -> List[Series]:
    """Unify dictionary values recursively without flattening batch arrays."""
    var same = True
    for part in parts:
        same = same and part.dtype() == parts[0].dtype()
    # Nested dtype equality describes child types, not their dictionaries.
    # Inspect the actual children even when parent schemas compare equal.
    if same and not parts[0].dtype().is_nested():
        return parts.copy()
    if parts[0].dtype().is_categorical():
        var frames = List[DataFrame]()
        for part in parts:
            frames.append(DataFrame([part.copy()]))
        var unified = _shared_dictionaries(frames, "vertical")
        var result = List[Series]()
        for frame in unified:
            result.append(frame._columns[0].copy())
        return result^
    var result = List[Series]()
    if parts[0].dtype().is_list():
        var children = List[Series]()
        for part in parts:
            children.append(part._data[ListColumn].child())
        var aligned = _align_children(children)
        for i in range(len(parts)):
            var column = parts[i]._data[ListColumn].copy()
            column._child = _hold(aligned[i].copy())
            var updated = Series(parts[i].name(), column^)
            updated._field_metadata = parts[i]._field_metadata
            result.append(updated^)
    elif parts[0].dtype().is_struct():
        var fields = List[List[Series]]()
        for k in range(parts[0]._data[StructColumn].field_count()):
            var children = List[Series]()
            for part in parts:
                children.append(
                    _fields_at(part._data[StructColumn]._fields)[k].copy()
                )
            fields.append(_align_children(children))
        for i in range(len(parts)):
            var columns = List[Series]()
            for field in fields:
                columns.append(field[i].copy())
            var column = parts[i]._data[StructColumn].copy()
            column._fields = _hold_many(columns^)
            var updated = Series(parts[i].name(), column^)
            updated._field_metadata = parts[i]._field_metadata
            result.append(updated^)
    else:
        raise Error("IPC record batches have incompatible field types")
    return result^


def _align_nested(frames: List[DataFrame]) raises -> List[DataFrame]:
    var result = frames.copy()
    if len(frames) <= 1:
        return result^
    for k in range(frames[0].width()):
        if not frames[0]._columns[k].dtype().is_nested():
            continue
        var parts = List[Series]()
        for frame in frames:
            parts.append(frame._columns[k].copy())
        var aligned = _align_children(parts)
        for i in range(len(result)):
            result[i]._columns[k] = aligned[i].copy()
    return result^


def _collect(mut stream: _ArrowArrayStream) raises -> DataFrame:
    var frames = List[DataFrame]()
    var pool = Pool(1)
    var initialized = False
    try:
        while True:
            var array = ArrowArray()
            var schema = ArrowSchema()
            try:
                if _stream_next(stream, array) != 0:
                    _error(stream)
                if array.release == 0:
                    break
                if _stream_schema(stream, schema) != 0:
                    _error(stream)
                if not initialized:
                    pool = Pool(
                        _arrow_import_workers(
                            Int(array.length), Int(array.n_children)
                        )
                    )
                    initialized = True
                frames.append(_import_arrow_with_pool(array, schema, pool))
            except error:
                _release_imported(array, schema)
                raise error^
        # concat unifies differing categorical dictionaries and retains
        # the imported arrays as chunks. IPC preserves declared decimals;
        # Parquet's smallest-physical-decimal adapter does not apply here.
        var result = concat(_align_nested(frames))._with_statistics()
        _release_stream(stream)
        return result^
    except error:
        _release_stream(stream)
        raise error^


def _read_call(
    entry: Int,
    path: String,
    stream_format: Bool,
    memory_map: Bool,
    columns: List[String],
    count: Int,
    rows: Int,
    mut stream: _ArrowArrayStream,
    mut error: Int,
) -> Int32:
    var filename = _c_string(path)
    var names = List[List[UInt8]]()
    for name in columns:
        names.append(_c_string(name))
    var addresses = List[Int]()
    for i in range(len(names)):
        addresses.append(Int(names[i].unsafe_ptr()))
    var status = Pointer(to=entry).unsafe_bitcast[_Read]()[](
        Int(filename.unsafe_ptr()),
        Int32(stream_format),
        Int32(memory_map),
        Int(addresses.unsafe_ptr()),
        Int32(count),
        Int64(rows),
        Int(Pointer(to=stream)),
        Int(Pointer(to=error)),
    )
    _ = addresses^
    _ = names^
    _ = filename^
    return status


def _read(
    path: String,
    columns: Optional[List[String]],
    n_rows: Optional[Int],
    memory_map: Bool,
    stream_format: Bool,
) raises -> DataFrame:
    _path(path)
    var selected = List[String]()
    var count = -1
    if columns:
        selected = columns.value().copy()
        count = len(selected)
        for i in range(count):
            if "\x00" in selected[i]:
                raise Error("IPC column names must not contain NUL bytes")
            for j in range(i):
                if selected[i] == selected[j]:
                    raise Error("IPC column requested twice: " + selected[i])
    var limit = -1
    if n_rows:
        limit = n_rows.value()
        if limit < 0:
            raise Error("IPC n_rows must be nonnegative")
    var library = _load_library()
    var entry = _Library._symbol(library.handle, "dfq_read_ipc")
    var stream = _ArrowArrayStream()
    var error = 0
    var status = _read_call(
        entry,
        path,
        stream_format,
        memory_map,
        selected,
        count,
        limit,
        stream,
        error,
    )
    if status != 0:
        _release_stream(stream)
        _raise_backend_error(library, error, "read_ipc")
    return _collect(stream)


def read_ipc(
    path: String,
    *,
    columns: Optional[List[String]] = None,
    n_rows: Optional[Int] = None,
    memory_map: Bool = True,
) raises -> DataFrame:
    """Read an Arrow IPC file (Feather v2), preserving record-batch chunks.

    None selects all columns; an empty column list retains only row count.
    Selection follows caller order. n_rows=0 reads an empty typed result.
    memory_map chooses native mapped input; imported payloads still copy.
    Requires libdfparquet built with IPC support.
    """
    return _read(path, columns, n_rows, memory_map, False)


def read_ipc_stream(
    path: String,
    *,
    columns: Optional[List[String]] = None,
    n_rows: Optional[Int] = None,
    memory_map: Bool = False,
) raises -> DataFrame:
    """Read a local IPC stream file, with the same selection rules as read_ipc."""
    return _read(path, columns, n_rows, memory_map, True)


def read_ipc(
    path: String,
    *,
    columns: List[String],
    n_rows: Optional[Int] = None,
    memory_map: Bool = True,
) raises -> DataFrame:
    """Read selected IPC file columns, including an empty selection."""
    return _read(path, Optional(columns.copy()), n_rows, memory_map, False)


def read_ipc_stream(
    path: String,
    *,
    columns: List[String],
    n_rows: Optional[Int] = None,
    memory_map: Bool = False,
) raises -> DataFrame:
    """Read selected IPC stream columns in the requested order."""
    return _read(path, Optional(columns.copy()), n_rows, memory_map, True)


def _schema_call(
    entry: Int,
    path: String,
    stream: Bool,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) -> Int32:
    var filename = _c_string(path)
    var status = Pointer(to=entry).unsafe_bitcast[_Schema]()[](
        Int(filename.unsafe_ptr()),
        Int32(stream),
        Int(Pointer(to=array)),
        Int(Pointer(to=schema)),
        Int(Pointer(to=error)),
    )
    _ = filename^
    return status


def read_ipc_schema(
    path: String, *, stream: Bool = False
) raises -> List[Field]:
    """Read field names/types without reading record-batch bodies.

    stream=True selects the IPC stream variant. Dictionary labels reside
    in data messages, so categorical schema entries have an empty dictionary.
    """
    _path(path)
    var library = _load_library()
    var entry = _Library._symbol(library.handle, "dfq_read_ipc_schema")
    var array = ArrowArray()
    var schema = ArrowSchema()
    var error = 0
    try:
        if _schema_call(entry, path, stream, array, schema, error) != 0:
            _raise_backend_error(library, error, "read_ipc_schema")
        return import_arrow(array, schema).schema()
    except caught:
        _release_imported(array, schema)
        raise caught^


def _open_writer(
    entry: Int,
    path: String,
    compression: String,
    stream: Bool,
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
        Int32(stream),
        Int(Pointer(to=handle)),
        Int(Pointer(to=error)),
    )
    _ = codec^
    _ = filename^
    return status


def _write_batch(
    entry: Int,
    handle: Int,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) -> Int32:
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


struct _Writer(Movable):
    var library: _Library
    var handle: Int
    var write: Int
    var close: Int
    var abort: Int

    def __init__(
        out self, frame: DataFrame, path: String, codec: String, stream: Bool
    ) raises:
        self.library = _load_library()
        self.handle = 0
        self.write = _Library._symbol(
            self.library.handle, "dfq_ipc_writer_write"
        )
        self.close = _Library._symbol(
            self.library.handle, "dfq_ipc_writer_close"
        )
        self.abort = _Library._symbol(
            self.library.handle, "dfq_ipc_writer_abort"
        )
        var entry = _Library._symbol(self.library.handle, "dfq_ipc_writer_open")
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(frame.clear(), array, schema)
        try:
            var error = 0
            var status = _open_writer(
                entry, path, codec, stream, schema, self.handle, error
            )
            _release_imported(array, schema)
            if status != 0:
                _raise_backend_error(self.library, error, "write_ipc")
        except caught:
            _release_imported(array, schema)
            raise caught^

    def __deinit__(deinit self):
        if self.handle != 0:
            var entry = self.abort
            Pointer(to=entry).unsafe_bitcast[_Abort]()[](self.handle)

    def append(mut self, frame: DataFrame) raises:
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(frame, array, schema)
        try:
            var error = 0
            var status = _write_batch(
                self.write, self.handle, array, schema, error
            )
            _release_imported(array, schema)
            if status != 0:
                _raise_backend_error(self.library, error, "write_ipc")
        except caught:
            _release_imported(array, schema)
            raise caught^

    def finish(mut self) raises:
        var handle = self.handle
        self.handle = 0
        var error = 0
        if _close_writer(self.close, handle, error) != 0:
            _raise_backend_error(self.library, error, "write_ipc")


def _write(
    frame: DataFrame, path: String, codec: String, size: Int, stream: Bool
) raises:
    _path(path)
    if codec != "uncompressed" and codec != "lz4" and codec != "zstd":
        raise Error("write_ipc: unsupported compression: " + codec)
    if size <= 0:
        raise Error("write_ipc: batch_size must be positive")
    for column in frame._columns:
        if "\x00" in column.name():
            raise Error("IPC column names must not contain NUL bytes")
    var writer = _Writer(frame, path, codec, stream)
    var offset = 0
    if frame.height() == 0:
        # Even an empty categorical batch carries its dictionary values.
        writer.append(frame)
    while offset < frame.height():
        var count = min(size, frame.height() - offset)
        for column in frame._columns:
            if column.is_chunked():
                var located = column._chunk_at(offset)
                count = min(count, len(located[0]) - located[1])
        var columns = List[Series]()
        for column in frame._columns:
            columns.append(column.slice(offset, count))
        writer.append(DataFrame(columns^, height=count))
        offset += count
    writer.finish()


def write_ipc(
    frame: DataFrame,
    path: String,
    *,
    compression: String = "uncompressed",
    batch_size: Int = 65536,
) raises:
    """Write an IPC/Feather v2 file with uncompressed, lz4 or zstd buffers.

    Output batches end at an input chunk boundary or batch_size, whichever
    comes first. The file replaces path; an I/O failure can leave a partial
    file. Requires libdfparquet built with IPC support.
    """
    _write(frame, path, compression, batch_size, False)


def write_ipc_stream(
    frame: DataFrame,
    path: String,
    *,
    compression: String = "uncompressed",
    batch_size: Int = 65536,
) raises:
    """Write a local IPC stream file, preserving input chunk boundaries."""
    _write(frame, path, compression, batch_size, True)
