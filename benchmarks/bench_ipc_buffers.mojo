"""Read latency, complete payload traversal, and actual allocation requests."""
from std.ffi import external_call
from std.memory import Pointer, bitcast
from std.sys import argv, CompilationTarget
from std.time import perf_counter_ns
from dataframe import (
    Column,
    BoolColumn,
    StringColumn,
    ListColumn,
    StructColumn,
    Series,
    DataFrame,
    ArrowArray,
    ArrowSchema,
    export_arrow,
    import_arrow,
    read_ipc,
    scan_ipc,
)
from dataframe.io.arrow import _c_string

comptime Reset = def() thin abi("C") -> None
comptime Counter = def() thin abi("C") -> UInt64


def symbol(name: String) raises -> Int:
    var c = _c_string(name)
    var address = external_call["dlsym", Int](
        Int(-2) if CompilationTarget.is_macos() else Int(0), c.unsafe_ptr()
    )
    _ = c^
    return address


def touch(series: Series) raises -> UInt64:
    var total = UInt64(0)
    if series.is_chunked():
        for chunk in series.chunks():
            total += touch(chunk)
        return total
    if series._data.isa[Column[Int64]]():
        ref column = series._data[Column[Int64]]
        for i in range(len(column)):
            if column._valid(i):
                total += UInt64(column._get(i)) + 1
    elif series._data.isa[Column[Float64]]():
        ref column = series._data[Column[Float64]]
        for i in range(len(column)):
            if column._valid(i):
                total += bitcast[DType.uint64](column._get(i)) + 1
    elif series._data.isa[Column[UInt32]]():
        ref column = series._data[Column[UInt32]]
        var dictionary = series.dtype().dictionary()
        for i in range(len(column)):
            if column._valid(i):
                total += 1
                for byte in dictionary[].get(Int(column._get(i))).as_bytes():
                    total += UInt64(byte)
    elif series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        for i in range(len(column)):
            if column._valid(i):
                total += UInt64(column._get(i)) + 1
    elif series._data.isa[StringColumn]():
        ref column = series._data[StringColumn]
        for i in range(len(column)):
            if column._valid(i):
                total += 1
                for byte in column._row_bytes(i):
                    total += UInt64(byte)
    elif series._data.isa[ListColumn]():
        ref column = series._data[ListColumn]
        total += UInt64(len(column) - column.null_count())
        total += touch(column.child())
    elif series._data.isa[StructColumn]():
        ref column = series._data[StructColumn]
        total += UInt64(len(column) - column.null_count())
        for i in range(column.field_count()):
            total += touch(column.field(i))
    else:
        raise Error("Unmeasured physical type")
    return total


def report(
    frame: DataFrame, begin: Int, read_end: Int, bytes: UInt64, calls: UInt64
) raises:
    var checksum = UInt64(0)
    for column in frame._columns:
        checksum += touch(column)
    var end = Int(perf_counter_ns())
    print(
        '{"read_ns":',
        read_end - begin,
        ',"total_ns":',
        end - begin,
        ',"allocated_bytes":',
        bytes,
        ',"allocation_calls":',
        calls,
        ',"rows":',
        frame.height(),
        ',"checksum":',
        checksum,
        "}",
    )


def main() raises:
    var args = argv()
    var reset = symbol("df_bench_alloc_reset")
    var bytes_counter = symbol("df_bench_alloc_bytes")
    var calls_counter = symbol("df_bench_alloc_calls")
    if args[2] == "owned":
        var rows = Int(String(args[1]))
        var columns = List[Series]()
        for k in range(8):
            var values = List[Int64](unsafe_uninit_length=rows)
            for i in range(rows):
                values[i] = Int64(i + k)
            columns.append(Series("c" + String(k), Column[Int64](values^)))
        var source = DataFrame(columns^)
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(source, array, schema)
        if reset != 0:
            Pointer(to=reset).unsafe_bitcast[Reset]()[]()
        var begin = Int(perf_counter_ns())
        var frame = import_arrow(array, schema)
        var read_end = Int(perf_counter_ns())
        var bytes = Pointer(to=bytes_counter).unsafe_bitcast[
            Counter
        ]()[]() if bytes_counter != 0 else UInt64(0)
        var calls = Pointer(to=calls_counter).unsafe_bitcast[
            Counter
        ]()[]() if calls_counter != 0 else UInt64(0)
        report(frame, begin, read_end, bytes, calls)
        _ = source^
        return
    if reset != 0:
        Pointer(to=reset).unsafe_bitcast[Reset]()[]()
    var begin = Int(perf_counter_ns())
    var frame = scan_ipc(String(args[1])).collect() if args[
        2
    ] == "scan" else read_ipc(String(args[1]))
    var read_end = Int(perf_counter_ns())
    var bytes = Pointer(to=bytes_counter).unsafe_bitcast[
        Counter
    ]()[]() if bytes_counter != 0 else UInt64(0)
    var calls = Pointer(to=calls_counter).unsafe_bitcast[
        Counter
    ]()[]() if calls_counter != 0 else UInt64(0)
    report(frame, begin, read_end, bytes, calls)
