"""Shared optional native I/O library loading, independent of frames/Arrow."""
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget

comptime PARQUET_LIBRARY_ENV = "DATAFRAME_PARQUET_LIBRARY"
comptime _RTLD_NOW = Int32(2)
comptime _FreeFn = def(Int) thin abi("C") -> None


def _c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8](capacity=text.byte_length() + 1)
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def _read_c_string(address: Int) -> String:
    if address == 0:
        return ""
    var pointer = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var length = 0
    while pointer[unsafe_offset=length] != 0:
        length += 1
    var bytes = List[UInt8](capacity=length)
    for i in range(length):
        bytes.append(pointer[unsafe_offset=i])
    return String(unsafe_from_utf8=bytes)


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


struct _Library(Copyable, Movable):
    """The opened reader library: its entry points."""

    var handle: Int
    var read: Int
    # dfq_read_parquet_stream_dict, or 0 in a library built before it.
    var read_dict: Int
    var statistics: Int
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
        self.handle = handle
        self.read = Self._symbol(handle, "dfq_read_parquet_stream")
        var c_name = _c_string("dfq_read_parquet_stream_dict")
        self.read_dict = external_call["dlsym", Int](
            handle, c_name.unsafe_ptr()
        )
        self.statistics = Self._symbol(handle, "dfq_row_group_statistics")
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


def _raise_backend_error(library: _Library, error: Int, what: String) raises:
    var message = _read_c_string(error)
    Pointer(to=library.free).unsafe_bitcast[_FreeFn]()[](error)
    raise Error(what + ": " + message)
