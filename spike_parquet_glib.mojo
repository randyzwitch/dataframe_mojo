"""Spike: read Parquet through Arrow GLib over FFI and import each record
batch via the C Data Interface. Not part of the package.

Usage: spike_parquet_glib FILE.parquet [REPETITIONS]

Libraries are loaded from $CONDA_PREFIX/lib so the spike runs under
`pixi run` without LD_LIBRARY_PATH.
"""
from std.ffi import OwnedDLHandle
from std.memory import Pointer
from std.os import getenv
from std.sys import argv
from std.time import monotonic

from dataframe import DataFrame, col, concat
from dataframe.arrow import _read_c_string, import_arrow


struct ArrowGlib:
    """The handful of C entry points a table read needs."""

    var glib: OwnedDLHandle
    var gobject: OwnedDLHandle
    var arrow: OwnedDLHandle
    var parquet: OwnedDLHandle

    def __init__(out self) raises:
        var prefix = getenv("CONDA_PREFIX")
        if prefix == "":
            raise Error("CONDA_PREFIX is not set; run under pixi")
        self.glib = OwnedDLHandle(prefix + "/lib/libglib-2.0.so.0")
        self.gobject = OwnedDLHandle(prefix + "/lib/libgobject-2.0.so.0")
        self.arrow = OwnedDLHandle(prefix + "/lib/libarrow-glib.so")
        self.parquet = OwnedDLHandle(prefix + "/lib/libparquet-glib.so")

    def check(self, error: Int, what: String) raises:
        """Raise with the GError message, freeing the GError."""
        if error == 0:
            return
        # GError: guint32 domain, gint code, gchar *message.
        var message_address = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=error + 8
        )[]
        var text = _read_c_string(message_address)
        self.glib.get_function[NoneType]("g_error_free")(error)
        raise Error(what + ": " + text)

    def unref(self, object: Int) raises:
        self.gobject.get_function[NoneType]("g_object_unref")(object)

    def free(self, address: Int) raises:
        self.glib.get_function[NoneType]("g_free")(address)


struct Phases(Movable):
    var read: Int
    var import_: Int
    var concat: Int
    var batches: Int

    def __init__(out self):
        self.read = 0
        self.import_ = 0
        self.concat = 0
        self.batches = 0


def read_parquet(
    lib: ArrowGlib, path: String, combine: Bool, mut phases: Phases
) raises -> DataFrame:
    var c_path = List[UInt8](capacity=path.byte_length() + 1)
    c_path.extend(path.as_bytes())
    c_path.append(0)
    var error = 0
    var start = monotonic()
    var reader = lib.parquet.get_function[Int](
        "gparquet_arrow_file_reader_new_path"
    )(c_path.unsafe_ptr(), Pointer(to=error))
    lib.check(error, "open")
    # The GLib reader defaults to single-threaded column decoding.
    lib.parquet.get_function[NoneType](
        "gparquet_arrow_file_reader_set_use_threads"
    )(reader, Int32(1))
    var table = lib.parquet.get_function[Int](
        "gparquet_arrow_file_reader_read_table"
    )(reader, Pointer(to=error))
    lib.check(error, "read_table")
    lib.unref(reader)
    if combine:
        # Let Arrow merge the row-group chunks so there is one batch to import.
        var combined = lib.arrow.get_function[Int](
            "garrow_table_combine_chunks"
        )(table, Pointer(to=error))
        lib.check(error, "combine_chunks")
        lib.unref(table)
        table = combined
    phases.read += monotonic() - start

    # One batch per table chunk (Arrow reads one chunk per row group).
    var batches = lib.arrow.get_function[Int]("garrow_table_batch_reader_new")(
        table
    )
    var frames = List[DataFrame]()
    while True:
        start = monotonic()
        var batch = lib.arrow.get_function[Int](
            "garrow_record_batch_reader_read_next"
        )(batches, Pointer(to=error))
        lib.check(error, "read_next")
        if batch == 0:
            break
        # gboolean garrow_record_batch_export(batch, gpointer *c_abi_array,
        #     gpointer *c_abi_schema, GError **error): fills both structs.
        var c_array = 0
        var c_schema = 0
        var ok = lib.arrow.get_function[Int32]("garrow_record_batch_export")(
            batch, Pointer(to=c_array), Pointer(to=c_schema), Pointer(to=error)
        )
        lib.check(error, "export batch")
        if ok == 0 or c_array == 0 or c_schema == 0:
            raise Error("export batch returned no structs")
        # import_arrow copies the buffers and calls the producer's release;
        # the struct shells themselves are g_malloc'ed and ours to free.
        frames.append(import_arrow(c_array, c_schema))
        lib.free(c_array)
        lib.free(c_schema)
        lib.unref(batch)
        phases.import_ += monotonic() - start
    lib.unref(batches)
    lib.unref(table)
    phases.batches = len(frames)
    if len(frames) == 1:
        return frames[0].copy()
    start = monotonic()
    var result = concat(frames)
    phases.concat += monotonic() - start
    return result^


def main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 4:
        raise Error(
            "usage: spike_parquet_glib FILE.parquet [REPETITIONS] [combine]"
        )
    var path = String(args[1])
    var repetitions = Int(String(args[2])) if len(args) >= 3 else 0
    var combine = len(args) == 4 and String(args[3]) == "combine"
    var lib = ArrowGlib()
    var phases = Phases()
    var frame = read_parquet(lib, path, combine, phases)
    if repetitions == 0:
        print(frame)
        for name in frame.columns():
            print(name, frame.column(name).dtype(), sep="\t")
        return
    print(frame.height(), "rows,", frame.width(), "columns")
    # Checksums to compare with pyarrow on the same file.
    print(
        "x_sum", frame.select(col("x").sum()).item(),
        "x_nulls", frame.select(col("x").null_count()).item(),
        "n_sum", frame.select(col("n").sum()).item(),
        "jk_sum", frame.select(col("jk").sum()).item(),
        "str_bytes",
        frame.select(col("key_str").str().len_bytes().sum()).item(),
    )
    var best = Int.MAX
    for _ in range(repetitions):
        var timed = Phases()
        var start = monotonic()
        var again = read_parquet(lib, path, combine, timed)
        var total = monotonic() - start
        if again.height() != frame.height():
            raise Error("row count changed between reads")
        if total < best:
            best = total
            phases = timed^
    print(
        "best of", repetitions, ":", Float64(best) / 1e6, "ms",
        "(arrow read", Float64(phases.read) / 1e6,
        "import", Float64(phases.import_) / 1e6,
        "concat", Float64(phases.concat) / 1e6, "batches", phases.batches, ")",
    )
