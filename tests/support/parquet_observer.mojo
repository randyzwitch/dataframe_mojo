"""Thin C-ABI observer forwarding the real native Parquet backend.

No Parquet parsing or expected values live here. Counters observe actual reader
requests, native decoded batches, and native release callback invocation.
Build with mojo build --emit shared-lib -o <path> this_file.mojo.
"""
from std.ffi import external_call, _get_global
from std.memory import Pointer
from std.os import getenv
from std.sys import size_of


@fieldwise_init
struct CStream(Copyable, Movable):
    var get_schema: Int
    var get_next: Int
    var get_last_error: Int
    var release: Int
    var private_data: Int


@fieldwise_init
struct CArray(Copyable, Movable):
    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: Int
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


@fieldwise_init
struct CSchema(Copyable, Movable):
    var format: Int
    var name: Int
    var metadata: Int
    var flags: Int64
    var n_children: Int64
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


comptime ReadFn = def(Int, Int32, Int, Int32, Int, Int32, Int, Int) thin abi(
    "C"
) -> Int32
comptime GetFn = def(Int, Int) thin abi("C") -> Int32
comptime ReleaseFn = def(Int) thin abi("C") -> None
comptime ErrorFn = def(Int) thin abi("C") -> Int
comptime StatisticsFn = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime FreeFn = def(Int) thin abi("C") -> None
comptime VersionFn = def() thin abi("C") -> Int
comptime WriteFn = def(Int, Int, Int, Int, Int64, Int) thin abi("C") -> Int32


def at[T: AnyType](address: Int) -> Pointer[T, MutAnyOrigin]:
    return Pointer[T, MutAnyOrigin](unsafe_from_address=address)


def counters_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    return Pointer[NoneType, MutUntrackedOrigin](
        unsafe_from_address=external_call["calloc", Int](1024, 8)
    )


def counters_keep(pointer: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    pass


def counts() -> Pointer[Int64, MutAnyOrigin]:
    return at[Int64](
        Int(
            _get_global[
                "DF_TEST_PARQUET_OBSERVER", counters_init, counters_keep
            ]().value()
        )
    )


def add(index: Int, value: Int64 = 1):
    counts()[unsafe_offset=index] += value


def native_entry(name: String) -> Int:
    if counts()[unsafe_offset=1000] == 0:
        var path = getenv("DATAFRAME_TEST_PARQUET_DELEGATE") + "\0"
        counts()[unsafe_offset=1000] = Int64(
            external_call["dlopen", Int](path.unsafe_ptr(), Int32(2))
        )
        if counts()[unsafe_offset=1000] == 0:
            # A failed delegate open must never turn dlsym(NULL, ...) into a
            # recursive call to this observer's own public symbols.
            _ = external_call["puts", Int32](path.unsafe_ptr())
            external_call["abort", NoneType]()
        _ = path^
    var symbol = name + "\0"
    var entry = external_call["dlsym", Int](
        Int(counts()[unsafe_offset=1000]), symbol.unsafe_ptr()
    )
    if entry == 0:
        _ = external_call["puts", Int32](symbol.unsafe_ptr())
        external_call["abort", NoneType]()
    return entry


@export
def df_test_reset() abi("C"):
    for index in range(900):
        counts()[unsafe_offset=index] = 0


@export
def df_test_get(index: Int32) abi("C") -> Int64:
    return counts()[unsafe_offset=Int(index)]


def observed_array_release(address: Int) abi("C"):
    ref array = at[CArray](address)[]
    var wrapped = array.private_data
    var original = Int(at[Int64](wrapped)[unsafe_offset=0])
    array.private_data = Int(at[Int64](wrapped)[unsafe_offset=1])
    array.release = original
    add(7)
    Pointer(to=original).unsafe_bitcast[ReleaseFn]()[](address)
    external_call["free", NoneType](wrapped)


def observed_schema_release(address: Int) abi("C"):
    ref schema = at[CSchema](address)[]
    var wrapped = schema.private_data
    var original = Int(at[Int64](wrapped)[unsafe_offset=0])
    schema.private_data = Int(at[Int64](wrapped)[unsafe_offset=1])
    schema.release = original
    add(9)
    Pointer(to=original).unsafe_bitcast[ReleaseFn]()[](address)
    external_call["free", NoneType](wrapped)


def observe_array(address: Int):
    ref array = at[CArray](address)[]
    if array.release == 0:
        return
    var wrapped = external_call["calloc", Int](2, 8)
    at[Int64](wrapped)[unsafe_offset=0] = Int64(array.release)
    at[Int64](wrapped)[unsafe_offset=1] = Int64(array.private_data)
    var callback: ReleaseFn = observed_array_release
    array.private_data = wrapped
    array.release = Pointer(to=callback).unsafe_bitcast[Int]()[]
    add(6)


def observe_schema(address: Int):
    ref schema = at[CSchema](address)[]
    if schema.release == 0:
        return
    var wrapped = external_call["calloc", Int](2, 8)
    at[Int64](wrapped)[unsafe_offset=0] = Int64(schema.release)
    at[Int64](wrapped)[unsafe_offset=1] = Int64(schema.private_data)
    var callback: ReleaseFn = observed_schema_release
    schema.private_data = wrapped
    schema.release = Pointer(to=callback).unsafe_bitcast[Int]()[]
    add(8)


def observed_next(stream_address: Int, array_address: Int) abi("C") -> Int32:
    var native = at[CStream](at[CStream](stream_address)[].private_data)
    var record = 256 + 16 * Int(at[Int64](Int(native))[unsafe_offset=5])
    var callback = native[].get_next
    add(2)
    add(record + 4)
    var status = Pointer(to=callback).unsafe_bitcast[GetFn]()[](
        Int(native), array_address
    )
    if status != 0:
        add(12)
        add(record + 6)
    elif at[CArray](array_address)[].release != 0:
        add(3)
        add(4, at[CArray](array_address)[].length)
        add(record + 2)
        add(record + 3, at[CArray](array_address)[].length)
    observe_array(array_address)
    return status


def observed_schema(stream_address: Int, schema_address: Int) abi("C") -> Int32:
    var native = at[CStream](at[CStream](stream_address)[].private_data)
    var callback = native[].get_schema
    var status = Pointer(to=callback).unsafe_bitcast[GetFn]()[](
        Int(native), schema_address
    )
    observe_schema(schema_address)
    return status


def observed_error(stream_address: Int) abi("C") -> Int:
    var native = at[CStream](at[CStream](stream_address)[].private_data)
    var callback = native[].get_last_error
    return Pointer(to=callback).unsafe_bitcast[ErrorFn]()[](Int(native))


def observed_stream_release(stream_address: Int) abi("C"):
    ref stream = at[CStream](stream_address)[]
    var address = stream.private_data
    ref native = at[CStream](address)[]
    var callback = native.release
    add(5)
    add(256 + 16 * Int(at[Int64](address)[unsafe_offset=5]) + 5)
    if callback:
        Pointer(to=callback).unsafe_bitcast[ReleaseFn]()[](address)
    stream.release = 0
    external_call["free", NoneType](address)


def observed_read(
    entry_name: String,
    path: Int,
    threads: Int32,
    columns: Int,
    n_columns: Int32,
    groups: Int,
    n_groups: Int32,
    output: Int,
    error: Int,
) -> Int32:
    add(0)
    counts()[unsafe_offset=10] = Int64(n_columns)
    counts()[unsafe_offset=11] = Int64(n_groups)
    for group in range(max(0, Int(n_groups))):
        counts()[unsafe_offset=16 + group] = Int64(
            at[Int32](groups)[unsafe_offset=group]
        )
    for column in range(Int(n_columns)):
        var name = at[Int](columns)[unsafe_offset=column]
        for byte in range(31):
            var character = at[UInt8](name)[unsafe_offset=byte]
            counts()[unsafe_offset=64 + 32 * column + byte] = Int64(character)
            if character == 0:
                break
    var entry = native_entry(entry_name)
    var status = Pointer(to=entry).unsafe_bitcast[ReadFn]()[](
        path, threads, columns, n_columns, groups, n_groups, output, error
    )
    if status != 0 or at[CStream](output)[].release == 0:
        return status
    var owner = external_call["calloc", Int](1, size_of[CStream]() + 8)
    at[CStream](owner)[] = at[CStream](output)[].copy()
    var stream_id = Int(counts()[unsafe_offset=128])
    at[Int64](owner)[unsafe_offset=5] = Int64(stream_id)
    counts()[unsafe_offset=256 + 16 * stream_id] = Int64(n_columns)
    counts()[unsafe_offset=257 + 16 * stream_id] = Int64(n_groups)
    add(128)
    var next: GetFn = observed_next
    var schema: GetFn = observed_schema
    var last_error: ErrorFn = observed_error
    var release: ReleaseFn = observed_stream_release
    at[CStream](output)[] = CStream(
        Pointer(to=schema).unsafe_bitcast[Int]()[],
        Pointer(to=next).unsafe_bitcast[Int]()[],
        Pointer(to=last_error).unsafe_bitcast[Int]()[],
        Pointer(to=release).unsafe_bitcast[Int]()[],
        owner,
    )
    return status


@export
def dfq_read_parquet_stream(
    path: Int,
    threads: Int32,
    columns: Int,
    n_columns: Int32,
    groups: Int,
    n_groups: Int32,
    output: Int,
    error: Int,
) abi("C") -> Int32:
    return observed_read(
        "dfq_read_parquet_stream",
        path,
        threads,
        columns,
        n_columns,
        groups,
        n_groups,
        output,
        error,
    )


@export
def dfq_read_parquet_stream_dict(
    path: Int,
    threads: Int32,
    columns: Int,
    n_columns: Int32,
    groups: Int,
    n_groups: Int32,
    output: Int,
    error: Int,
) abi("C") -> Int32:
    return observed_read(
        "dfq_read_parquet_stream_dict",
        path,
        threads,
        columns,
        n_columns,
        groups,
        n_groups,
        output,
        error,
    )


@export
def dfq_row_group_statistics(
    path: Int, array: Int, schema: Int, error: Int
) abi("C") -> Int32:
    add(1)
    var entry = native_entry("dfq_row_group_statistics")
    var status = Pointer(to=entry).unsafe_bitcast[StatisticsFn]()[](
        path, array, schema, error
    )
    observe_array(array)
    observe_schema(schema)
    return status


@export
def dfq_free(address: Int) abi("C"):
    var entry = native_entry("dfq_free")
    Pointer(to=entry).unsafe_bitcast[FreeFn]()[](address)


@export
def dfq_arrow_version() abi("C") -> Int:
    var entry = native_entry("dfq_arrow_version")
    return Pointer(to=entry).unsafe_bitcast[VersionFn]()[]()


@export
def dfq_write_parquet(
    path: Int,
    array: Int,
    schema: Int,
    compression: Int,
    size: Int64,
    error: Int,
) abi("C") -> Int32:
    observe_array(array)
    observe_schema(schema)
    var entry = native_entry("dfq_write_parquet")
    return Pointer(to=entry).unsafe_bitcast[WriteFn]()[](
        path, array, schema, compression, size, error
    )
