"""Parquet reading.

`read_parquet` is the public entry point. It is served today by
`_read_with_dfparquet`, which loads `libdfparquet` (Arrow C++'s Parquet
reader built with nothing else, behind four C symbols; see
`native/dfparquet/`) and brings the result in through the Arrow C Data
Interface importer this package already has. The backend's whole contract
is "a path, column names and row groups in, a DataFrame out" plus a
statistics frame per file, so a Mojo-native reader replaces the two
`_..._with_dfparquet` functions and nothing above them changes.

Column types the importer cannot hold are coerced by the reader: dictionary
columns are decoded to their value type, float16 widens to float32, string
views become strings, and a timestamp with a time zone loses the zone. Its
values are UTC instants, so the result is the same moment as a naive UTC
datetime; the zone name is not kept. Decimal, binary and nested columns
still raise.

Row-group pruning: `parquet_row_group_statistics` returns one row per row
group with each column's minimum, maximum and null count from the footer,
and `_pruned_row_groups` turns a filter on constant bounds (`col > lit`,
combined with `&` and `|`) into the row groups that can contain a match.
The lazy `scan_parquet` uses both, so a filter decodes only those groups.
A null bound means unknown and never prunes.

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
from std.collections import Optional
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
from .expr import (
    AND,
    COL,
    EQ,
    GE,
    GT,
    LE,
    LIT_FLOAT,
    LIT_INT,
    LIT_STRING,
    LT,
    OR,
    Expr,
    Node,
    col,
)
from .frame import DataFrame

comptime PARQUET_LIBRARY_ENV = "DATAFRAME_PARQUET_LIBRARY"

# dlopen mode: resolve every symbol now, so a broken library fails at open.
comptime _RTLD_NOW = Int32(2)

# int dfq_read_parquet(const char *path, int use_threads,
#     const char **columns, int n_columns, const int *row_groups,
#     int n_row_groups, struct ArrowArray *out_array,
#     struct ArrowSchema *out_schema, char **error_out)
comptime _ReadFn = def(
    Int, Int32, Int, Int32, Int, Int32, Int, Int, Int
) thin abi("C") -> Int32
# int dfq_row_group_statistics(const char *path, struct ArrowArray *out,
#     struct ArrowSchema *out_schema, char **error_out)
comptime _StatisticsFn = def(Int, Int, Int, Int) thin abi("C") -> Int32
# void dfq_free(void *pointer)
comptime _FreeFn = def(Int) thin abi("C") -> None
# const char *dfq_arrow_version(void)
comptime _VersionFn = def() thin abi("C") -> Int


def read_parquet(
    path: String,
    *,
    columns: List[String] = List[String](),
    row_groups: Optional[List[Int]] = None,
    use_threads: Bool = True,
) raises -> DataFrame:
    """Read a local Parquet file into a DataFrame.

    `columns` selects fields by name, in the order given; the default reads
    every column. `row_groups` selects row groups by index, in file order;
    the default reads them all, and an empty list reads none and returns the
    schema with no rows. Column types map as the Arrow importer maps them:
    integer and float widths are kept, `date32` becomes Date, timestamps
    keep their unit (a time zone is dropped; values stay UTC instants), and
    strings, booleans and nulls carry over. Dictionary columns arrive as
    plain strings and float16 as float32. Nested columns, decimals and
    binary are not supported yet.

    Raises when the file cannot be read, a requested column or row group is
    missing, or the reader library is not installed (see the module notes
    for where it is looked for).
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
    var groups = List[Int32]()
    var group_count = -1
    if row_groups:
        group_count = len(row_groups.value())
        for g in row_groups.value():
            if g < 0:
                raise Error("read_parquet: row group indices must be >= 0")
            groups.append(Int32(g))
    return _read_with_dfparquet(path, columns, groups, group_count, use_threads)


def parquet_row_group_statistics(path: String) raises -> DataFrame:
    """One row per row group of `path`, without decoding any data.

    Columns: `row_group`, `rows`, then `min:<name>`, `max:<name>` (in the
    column's type; null when the footer has no bound or a text bound is not
    exact) and `nulls:<name>` for every column of a flat schema.
    """
    if path == "":
        raise Error("parquet_row_group_statistics requires a path")
    return _statistics_with_dfparquet(path)


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


# --- row-group pruning ---------------------------------------------------


def _pruned_row_groups(
    statistics: DataFrame, predicate: Expr
) raises -> Optional[List[Int]]:
    """Row groups that may hold a row passing `predicate`, or None when the
    predicate gives no usable bound and every group must be read.

    Handles `col <op> lit` and `lit <op> col` for <, <=, >, >=, == on
    numbers and strings, joined with `&` and `|`. Anything else, including a
    column without statistics, keeps every group. The bounds are compared
    with the package's own expressions on the statistics frame, so type
    rules match the filter's.
    """
    var mask = _bounds_mask(
        statistics, predicate._nodes, len(predicate._nodes) - 1
    )
    if not mask:
        return None
    var groups = List[Int]()
    for g in range(len(mask.value())):
        if mask.value()[g]:
            groups.append(Int(statistics.column("row_group").get(g).int32()))
    if len(groups) == statistics.height():
        return None
    return groups^


def _bounds_mask(
    statistics: DataFrame, nodes: List[Node], index: Int
) raises -> Optional[List[Bool]]:
    ref node = nodes[index]
    if node.op == AND or node.op == OR:
        var left = _bounds_mask(statistics, nodes, node.left)
        var right = _bounds_mask(statistics, nodes, node.right)
        if node.op == AND:
            # A group survives an AND only if it survives both sides; a
            # side without bounds constrains nothing.
            if not left:
                return right^
            if not right:
                return left^
            var out = List[Bool](capacity=len(left.value()))
            for g in range(len(left.value())):
                out.append(left.value()[g] and right.value()[g])
            return out^
        if not left or not right:
            return None
        var out = List[Bool](capacity=len(left.value()))
        for g in range(len(left.value())):
            out.append(left.value()[g] or right.value()[g])
        return out^
    if not (
        node.op == GT
        or node.op == GE
        or node.op == LT
        or node.op == LE
        or node.op == EQ
    ):
        return None
    ref left = nodes[node.left]
    ref right = nodes[node.right]
    var op = node.op
    var name: String
    var literal: Node
    if left.op == COL and _is_literal(right):
        name = left.text
        literal = right.copy()
    elif right.op == COL and _is_literal(left):
        # lit < col is col > lit, and so on.
        name = right.text
        literal = left.copy()
        if op == GT:
            op = LT
        elif op == GE:
            op = LE
        elif op == LT:
            op = GT
        elif op == LE:
            op = GE
    else:
        return None
    var value = Expr([literal.copy()], "literal")
    var low = col("min:" + name)
    var high = col("max:" + name)
    # An unknown bound (null) must keep the group.
    var keep: Expr
    if op == GT:
        keep = (high > value) | high.is_null()
    elif op == GE:
        keep = (high >= value) | high.is_null()
    elif op == LT:
        keep = (low < value) | low.is_null()
    elif op == LE:
        keep = (low <= value) | low.is_null()
    else:
        keep = ((low <= value) | low.is_null()) & (
            (high >= value) | high.is_null()
        )
    try:
        var flags = statistics.select(keep.alias("keep")).column("keep")
        var out = List[Bool](capacity=len(flags))
        for g in range(len(flags)):
            var cell = flags.get(g)
            out.append(True if cell.is_null() else cell.bool())
        return out^
    except:
        # No statistics for this column, or a type the comparison rejects:
        # the filter itself will report a real type error when it runs.
        return None


def _is_literal(node: Node) -> Bool:
    return node.op == LIT_INT or node.op == LIT_FLOAT or node.op == LIT_STRING


# --- the dfparquet backend -----------------------------------------------


struct _Library(Copyable, Movable):
    """The opened reader library: its entry points."""

    var read: Int
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
        self.read = Self._symbol(handle, "dfq_read_parquet")
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


def _read_with_dfparquet(
    path: String,
    columns: List[String],
    row_groups: List[Int32],
    group_count: Int,
    use_threads: Bool,
) raises -> DataFrame:
    """The FFI backend: one call into libdfparquet, then the Arrow import.

    `dfq_read_parquet` reads the selection into one Arrow record batch and
    exports it through the C Data Interface. `import_arrow` copies the
    buffers into our own and calls the exported release, so no foreign
    memory outlives this function.
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
        row_groups,
        group_count,
        use_threads,
        array,
        schema,
        error,
    )
    if status != 0:
        _raise_backend_error(library, error, "read_parquet")
    return import_arrow(array, schema)


def _statistics_with_dfparquet(path: String) raises -> DataFrame:
    var library = _load_library()
    var c_path = _c_string(path)
    var array = ArrowArray()
    var schema = ArrowSchema()
    var error = 0
    var status = _call_statistics(library, c_path, array, schema, error)
    if status != 0:
        _raise_backend_error(library, error, "parquet_row_group_statistics")
    return import_arrow(array, schema)


# The C entry points are called from helpers whose argument buffers are
# borrowed parameters, so they outlive the call; as locals their last use
# would be the argument expression, and Mojo may free them before the
# callee runs.


def _call_reader(
    library: _Library,
    c_path: List[UInt8],
    c_columns: List[List[UInt8]],
    column_pointers: List[Int],
    row_groups: List[Int32],
    group_count: Int,
    use_threads: Bool,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) raises -> Int32:
    if len(c_columns) != len(column_pointers):
        raise Error("read_parquet: column pointer table is inconsistent")
    var columns_address = (
        Int(column_pointers.unsafe_ptr()) if len(column_pointers) > 0 else 0
    )
    var groups_address = (
        Int(row_groups.unsafe_ptr()) if len(row_groups) > 0 else 0
    )
    return Pointer(to=library.read).unsafe_bitcast[_ReadFn]()[](
        Int(c_path.unsafe_ptr()),
        Int32(1) if use_threads else Int32(0),
        columns_address,
        Int32(len(column_pointers)),
        groups_address,
        Int32(group_count),
        Int(Pointer(to=array)),
        Int(Pointer(to=schema)),
        Int(Pointer(to=error)),
    )


def _call_statistics(
    library: _Library,
    c_path: List[UInt8],
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    mut error: Int,
) raises -> Int32:
    return Pointer(to=library.statistics).unsafe_bitcast[_StatisticsFn]()[](
        Int(c_path.unsafe_ptr()),
        Int(Pointer(to=array)),
        Int(Pointer(to=schema)),
        Int(Pointer(to=error)),
    )
