"""Parquet reading.

`read_parquet` is the public entry point. It is served today by
`_read_with_dfparquet`, which loads `libdfparquet` (Arrow C++'s Parquet
reader built with nothing else, behind three C symbols; see
`native/dfparquet/`) and brings the result in through the Arrow C Data
Interface importer this package already has. The backend's whole contract
is "a path and column names in, a DataFrame out", so a Mojo-native reader
replaces that one function and nothing above it changes.

The library is looked for in this order:

1. `DATAFRAME_PARQUET_LIBRARY`, a full path to the shared library.
2. `$CONDA_PREFIX/lib/libdfparquet.so` (`.dylib` on macOS).
3. `build/dfparquet/libdfparquet.so` under the working directory, which is
   where `pixi run -e native build-dfparquet` puts it.

Nothing is linked at build time, so a program that never reads Parquet
never needs the library. Once opened it stays open for the process: the
exported batch's release callback lives in the library, and Arrow's thread
pool would otherwise be torn down and rebuilt on every read. The standard
library's `OwnedDLHandle` closes on destruction, so the library is opened
with `dlopen` directly and its handle is never closed; repeat opens only
bump the loader's reference count.
"""
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget

from .arrow import (
    ArrowArray,
    ArrowSchema,
    _c_string,
    _read_c_string,
    import_arrow,
)
from .frame import DataFrame

comptime PARQUET_LIBRARY_ENV = "DATAFRAME_PARQUET_LIBRARY"

# dlopen mode: resolve every symbol now, so a broken library fails at open.
comptime _RTLD_NOW = Int32(2)

# int dfq_read_parquet(const char *path, int use_threads,
#     const char **columns, int n_columns, struct ArrowArray *out_array,
#     struct ArrowSchema *out_schema, char **error_out)
comptime _ReadFn = def(Int, Int32, Int, Int32, Int, Int, Int) thin abi(
    "C"
) -> Int32
# void dfq_free(void *pointer)
comptime _FreeFn = def(Int) thin abi("C") -> None
# const char *dfq_arrow_version(void)
comptime _VersionFn = def() thin abi("C") -> Int


def read_parquet(
    path: String,
    *,
    columns: List[String] = List[String](),
    use_threads: Bool = True,
) raises -> DataFrame:
    """Read a local Parquet file into a DataFrame.

    `columns` selects fields by name, in the order given; the default reads
    every column. Column types map as the Arrow importer maps them: integer
    and float widths are kept, `date32` becomes Date, timestamps keep their
    unit, and strings, booleans and nulls carry over. Nested columns and
    decimals are not supported yet.

    Raises when the file cannot be read, a requested column is missing, or
    the reader library is not installed (see the module notes for where it
    is looked for).
    """
    if path == "":
        raise Error("read_parquet requires a path")
    for i in range(len(columns)):
        if columns[i] == "":
            raise Error("read_parquet: column names must not be empty")
        for j in range(i):
            if columns[j] == columns[i]:
                raise Error(
                    "read_parquet: column requested twice: " + columns[i]
                )
    return _read_with_dfparquet(path, columns, use_threads)


def parquet_library_candidates() -> List[String]:
    """Where `read_parquet` looks for its reader library, in order."""
    var name = String(
        "libdfparquet.dylib"
    ) if CompilationTarget.is_macos() else String("libdfparquet.so")
    var candidates = List[String]()
    var explicit = getenv(PARQUET_LIBRARY_ENV)
    if explicit != "":
        candidates.append(explicit)
    var prefix = getenv("CONDA_PREFIX")
    if prefix != "":
        candidates.append(prefix + "/lib/" + name)
    candidates.append("build/dfparquet/" + name)
    return candidates^


def parquet_backend_version() raises -> String:
    """The reader library's Arrow version, for diagnostics."""
    var library = _load_library()
    return _read_c_string(
        Pointer(to=library.version).unsafe_bitcast[_VersionFn]()[]()
    )


struct _Library(Copyable, Movable):
    """The opened reader library: its handle and the three entry points."""

    var read: Int
    var free: Int
    var version: Int

    def __init__(out self, path: String) raises:
        var c_path = _c_string(path)
        var handle = external_call["dlopen", Int](
            c_path.unsafe_ptr(), _RTLD_NOW
        )
        if handle == 0 or len(c_path) == 0:
            raise Error(
                "read_parquet could not load "
                + path
                + ": "
                + _read_c_string(external_call["dlerror", Int]())
            )
        self.read = Self._symbol(handle, "dfq_read_parquet")
        self.free = Self._symbol(handle, "dfq_free")
        self.version = Self._symbol(handle, "dfq_arrow_version")

    @staticmethod
    def _symbol(handle: Int, name: String) raises -> Int:
        var c_name = _c_string(name)
        var address = external_call["dlsym", Int](handle, c_name.unsafe_ptr())
        if address == 0 or len(c_name) == 0:
            raise Error("read_parquet: reader library lacks " + name)
        return address


def _load_library() raises -> _Library:
    var candidates = parquet_library_candidates()
    for candidate in candidates:
        if exists(candidate):
            return _Library(candidate)
    var looked = String()
    for candidate in candidates:
        if looked != "":
            looked += ", "
        looked += candidate
    raise Error(
        "read_parquet needs libdfparquet; looked for "
        + looked
        + ". Build it with `pixi run -e native build-dfparquet` or set "
        + PARQUET_LIBRARY_ENV
        + " to its path."
    )


def _read_with_dfparquet(
    path: String, columns: List[String], use_threads: Bool
) raises -> DataFrame:
    """The FFI backend: one call into libdfparquet, then the Arrow import.

    `dfq_read_parquet` reads the file (or the named columns) into one Arrow
    record batch and exports it through the C Data Interface. `import_arrow`
    copies the buffers into our own and calls the exported release, so no
    foreign memory outlives this function.
    """
    var library = _load_library()
    var c_path = _c_string(path)
    var c_columns = List[List[UInt8]](capacity=len(columns))
    for name in columns:
        c_columns.append(_c_string(name))
    var column_pointers = List[Int](capacity=len(c_columns))
    for i in range(len(c_columns)):
        column_pointers.append(Int(c_columns[i].unsafe_ptr()))
    var array = ArrowArray()
    var schema = ArrowSchema()
    var error = 0
    var status = _call_reader(
        library,
        c_path,
        c_columns,
        column_pointers,
        use_threads,
        array,
        schema,
        error,
    )
    if status != 0:
        var message = _read_c_string(error)
        Pointer(to=library.free).unsafe_bitcast[_FreeFn]()[](error)
        raise Error("read_parquet: " + message)
    return import_arrow(array, schema)


def _call_reader(
    library: _Library,
    c_path: List[UInt8],
    c_columns: List[List[UInt8]],
    column_pointers: List[Int],
    use_threads: Bool,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) raises -> Int32:
    """Call `dfq_read_parquet`. The C strings and the pointer table are
    borrowed parameters, so they outlive the call; as locals their last use
    would be the argument expression, and Mojo may free them before the
    callee runs."""
    if len(c_columns) != len(column_pointers):
        raise Error("read_parquet: column pointer table is inconsistent")
    var columns_address = (
        Int(column_pointers.unsafe_ptr()) if len(column_pointers) > 0 else 0
    )
    return Pointer(to=library.read).unsafe_bitcast[_ReadFn]()[](
        Int(c_path.unsafe_ptr()),
        Int32(1) if use_threads else Int32(0),
        columns_address,
        Int32(len(column_pointers)),
        Int(Pointer(to=array)),
        Int(Pointer(to=schema)),
        Int(Pointer(to=error)),
    )
