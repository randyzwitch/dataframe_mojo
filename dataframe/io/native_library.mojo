"""Shared optional native I/O library loading, independent of frames/Arrow."""
from std.ffi import OwnedDLHandle
from std.memory import OwnedPointer, Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget

comptime PARQUET_LIBRARY_ENV = "DATAFRAME_PARQUET_LIBRARY"
comptime _FreeFn = def(Int) thin abi("C") -> None


def _read_c_string(address: Int) -> String:
    if address == 0:
        return ""
    return String(
        unsafe_from_utf8_ptr=Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=address
        )
    )


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
        var library: OwnedDLHandle
        try:
            library = OwnedDLHandle(path, flags=2)
        except error:
            raise Error(
                "read_parquet could not load " + path + ": " + String(error)
            )
        self.read = Self._entry(library, "dfq_read_parquet_stream")
        var optional_read = library.get_symbol[NoneType](
            "dfq_read_parquet_stream_dict"
        )
        self.read_dict = Int(optional_read.value()) if optional_read else 0
        self.statistics = Self._entry(library, "dfq_row_group_statistics")
        self.free = Self._entry(library, "dfq_free")
        self.version = Self._entry(library, "dfq_arrow_version")
        # Returned Arrow streams retain callbacks after this loader returns.
        # Keep their library loaded for the process, as before. Failed symbol
        # resolution above still closes the local owner automatically.
        var owner = OwnedPointer(library^)
        self.handle = Int(owner^.unsafe_take_allocation().unsafe_leak())

    @staticmethod
    def _entry(library: OwnedDLHandle, name: String) raises -> Int:
        var symbol = library.get_symbol[NoneType](name)
        if not symbol:
            raise Error("read_parquet: reader library lacks " + name)
        return Int(symbol.value())

    @staticmethod
    def _symbol(handle: Int, name: String) raises -> Int:
        return Self._entry(
            Pointer[OwnedDLHandle, ImmutAnyOrigin](
                unsafe_from_address=handle
            )[],
            name,
        )


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
