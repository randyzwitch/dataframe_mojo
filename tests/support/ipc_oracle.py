"""Independent PyArrow IPC fixtures and consumers of actual Mojo outputs."""
import argparse
import ctypes as c
import os
import sys
from pathlib import Path
# Native IPC checks load a second Arrow library; use compatible allocators.
if sys.platform == "darwin":
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import pyarrow as pa
import pyarrow.ipc as ipc
from parquet_oracle import core_table, typed_table

FIXTURES = Path(__file__).parents[1] / "fixtures/ipc"


def write(path, batches, schema, stream=False, compression=None, deltas=False):
    options = ipc.IpcWriteOptions(compression=compression, emit_dictionary_deltas=deltas)
    with pa.OSFile(str(path), "wb") as file:
        factory = ipc.new_stream if stream else ipc.new_file
        with factory(file, schema, options=options) as writer:
            for batch in batches:
                writer.write_batch(batch)


def dictionary_batches(delta=False):
    labels = [["red", "blue"], ["red", "blue", "green"]] if delta else [["red", "blue"], ["blue", "green", "red"]]
    codes = [[0, 1, None], [0, 2, 1]] if delta else [[0, 1, None], [2, 1, 0]]
    return [pa.record_batch([
        pa.array([3*i, 3*i+1, 3*i+2], type=pa.int64()),
        pa.DictionaryArray.from_arrays(pa.array(codes[i], type=pa.int8()), pa.array(labels[i], type=pa.large_string())),
    ], names=["x", "category"]) for i in range(2)]


def corrupt_later_batch(path, stream, target=2):
    data = bytearray(path.read_bytes())
    with pa.BufferReader(data) as source:
        source.seek(0 if stream else 8)
        records = 0
        while True:
            message = ipc.read_message(source)
            if message.type != "record batch":
                continue
            records += 1
            if records == target:
                body = source.tell() - message.body.size
                # Zstd-compressed IPC bodies start with an eight-byte
                # uncompressed length followed by the compressed frame.
                data[body+8:body+16] = b"\x00" * 8
                path.write_bytes(data)
                return


def generate():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    core = core_table()
    typed = typed_table().append_column("narrow_precision_wide_storage", pa.array([123, None, -45], type=pa.decimal128(9, 2)))
    for stream, extension in [(False, "arrow"), (True, "stream")]:
        for codec in [None, "lz4", "zstd"]:
            name = codec or "uncompressed"
            write(FIXTURES / f"core-{name}.{extension}", core.to_batches(max_chunksize=3), core.schema, stream, codec)
        write(FIXTURES / f"typed.{extension}", typed.to_batches(max_chunksize=2), typed.schema, stream)
        write(FIXTURES / f"empty.{extension}", [], core.schema, stream)
        zero = core.to_batches()[0].select([])
        write(FIXTURES / f"zero-columns.{extension}", [zero], zero.schema, stream)
        write(FIXTURES / f"late-corrupt.{extension}", core.to_batches(max_chunksize=3), core.schema, stream, "zstd")
        corrupt_later_batch(FIXTURES / f"late-corrupt.{extension}", stream)
        batches = dictionary_batches(delta=True)
        write(FIXTURES / f"dictionary-delta.{extension}", batches, batches[0].schema, stream, deltas=True)
    write(FIXTURES / "early-corrupt.arrow", core.to_batches(max_chunksize=3), core.schema, False, "zstd")
    corrupt_later_batch(FIXTURES / "early-corrupt.arrow", False, target=1)
    nested_delta = []
    for i, batch in enumerate(dictionary_batches(delta=True)):
        values = batch.column(1)
        lists = pa.LargeListArray.from_arrays(pa.array([0, 1, 3], type=pa.int64()), values)
        records = pa.StructArray.from_arrays([values.slice(0, 2)], names=["category"], mask=pa.array([False, i == 0]))
        nested_delta.append(pa.record_batch([lists, records], names=["items", "record"]))
    write(FIXTURES / "nested-dictionary.arrow", nested_delta, nested_delta[0].schema, deltas=True)
    batches = dictionary_batches()
    write(FIXTURES / "dictionary-replacement.stream", batches, batches[0].schema, True)
    nested = []
    for i, batch in enumerate(batches):
        values = batch.column(1)
        lists = pa.LargeListArray.from_arrays(pa.array([0, 1, 3], type=pa.int64()), values)
        records = pa.StructArray.from_arrays([values.slice(0, 2)], names=["category"], mask=pa.array([False, i == 0]))
        nested.append(pa.record_batch([lists, records], names=["items", "record"]))
    write(FIXTURES / "nested-dictionary.stream", nested, nested[0].schema, True)
    views = pa.record_batch([
        pa.array(["long external string", None, "é"], type=pa.string_view()),
        pa.array([b"\xff\x00", None, b""], type=pa.binary_view()),
    ], names=["text", "binary"])
    write(FIXTURES / "views.arrow", [views], views.schema)
    (FIXTURES / "invalid.arrow").write_bytes(b"not an Arrow IPC file")
    (FIXTURES / "truncated.stream").write_bytes(b"\xff\xff\xff\xff\x80\x00\x00\x00")


def check_written(directory):
    root = Path(directory)
    for kind, opening in [("arrow", ipc.open_file), ("stream", ipc.open_stream)]:
        for codec in ["uncompressed", "lz4", "zstd"]:
            path = root / f"ipc-written-{codec}.{kind}"
            with pa.memory_map(str(path)) as source:
                reader = opening(source)
                if kind == "arrow":
                    batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
                else:
                    batches = list(reader)
                actual = pa.Table.from_batches(batches)
                assert actual.column_names == ["x", "s"]
                assert actual.to_pydict() == {"x":[0, 1, None, 3, 4, 5], "s":["red", "", "λ", None, "green", "last"]}
                assert actual.schema.field("x").type == pa.int64()
                assert actual.schema.field("s").type == pa.large_string()
                # Input boundaries [2, 6] and [1, 4, 6], plus size=3,
                # must yield their union rather than one rechunked batch.
                assert [b.num_rows for b in batches] == [1, 1, 2, 2]
        with pa.memory_map(str(root / f"ipc-written-empty.{kind}")) as source:
            actual = opening(source).read_all()
            assert actual.num_rows == 0 and actual.column_names == ["x", "s"]
        with pa.memory_map(str(root / f"ipc-written-zero-columns.{kind}")) as source:
            actual = opening(source).read_all()
            assert actual.num_rows == 7 and actual.num_columns == 0
        with pa.memory_map(str(root / f"ipc-written-typed.{kind}")) as source:
            actual = opening(source).read_all()
            expected = typed_table().append_column("narrow_precision_wide_storage", pa.array([123, None, -45], type=pa.decimal128(9, 2)))
            assert actual.to_pydict() == expected.to_pydict()
            for got, want in zip(actual.schema, expected.schema):
                assert got.type == want.type, (got, want)
                assert got.metadata == want.metadata, (got, want)
        with pa.memory_map(str(root / f"ipc-written-dictionary.{kind}")) as source:
            actual = opening(source).read_all()
            assert pa.types.is_dictionary(actual.schema.field("category").type)
            assert actual.to_pydict() == {"x":list(range(6)), "category":["red", "blue", None, "red", "green", "blue"]}
        with pa.memory_map(str(root / f"ipc-written-nested-dictionary.{kind}")) as source:
            actual = opening(source).read_all()
            assert actual.to_pydict() == {"items":[["red"], ["blue", None], ["red"], ["green", "blue"]],
                                        "record":[{"category":"red"}, None, {"category":"red"}, {"category":"green"}]}
    print("Independent IPC consumers checked actual Mojo file/stream outputs")


def check_scan_library(path):
    """Use PyArrow's C stream consumer against the actual native scan ABI."""
    class Stream(c.Structure):
        _fields_ = [(name, c.c_void_p) for name in
                    ("get_schema", "get_next", "get_last_error", "release", "private_data")]

    library = c.CDLL(str(path))
    library.dfq_free.argtypes = [c.c_void_p]
    library.dfq_scan_ipc.argtypes = [c.c_char_p, c.c_int, c.POINTER(c.c_char_p), c.c_int,
                                    c.c_int64, c.c_int64, c.POINTER(Stream), c.POINTER(c.c_void_p)]
    library.dfq_scan_ipc.restype = c.c_int

    def scan(name, columns=None, offset=0, rows=-1, mapped=True):
        stream, error = Stream(), c.c_void_p()
        encoded = [column.encode() for column in columns or []]
        addresses = (c.c_char_p * len(encoded))(*encoded)
        status = library.dfq_scan_ipc(os.fsencode(FIXTURES / name), mapped, addresses,
                                     -1 if columns is None else len(columns), offset, rows,
                                     c.byref(stream), c.byref(error))
        if status:
            message = c.string_at(error).decode()
            library.dfq_free(error)
            assert not stream.release
            raise ValueError(message)
        reader = pa.RecordBatchReader._import_from_c(c.addressof(stream))
        assert not stream.release, "the external consumer must take stream ownership"
        return reader

    for mapped in [False, True]:
        with scan("core-zstd.arrow", ["s", "x"], 4, 3, mapped) as reader:
            assert reader.read_all().to_pydict() == {"s":["red", "green", "red"], "x":[100, 101, 102]}
        with scan("early-corrupt.arrow", offset=3, rows=3, mapped=mapped) as reader:
            assert reader.read_all().column("x").to_pylist() == [3, 100, 101]
        with scan("late-corrupt.arrow", offset=6, rows=2, mapped=mapped) as reader:
            assert reader.read_all().column("x").to_pylist() == [102, 200]
        with scan("early-corrupt.arrow", ["s"], mapped=mapped) as reader:
            assert reader.read_all().column("s").to_pylist() == core_table().column("s").to_pylist()
        with scan("late-corrupt.arrow", rows=3, mapped=mapped) as reader:
            assert reader.read_all().num_rows == 3
        with scan("early-corrupt.arrow", offset=2**63-1, rows=0, mapped=mapped) as reader:
            assert reader.read_all().num_rows == 0
        with scan("core-zstd.arrow", [], 5, 2, mapped) as reader:
            table = reader.read_all()
            assert table.num_columns == 0 and table.num_rows == 2
        with scan("early-corrupt.arrow", [], mapped=mapped) as reader:
            table = reader.read_all()
            assert table.num_columns == 0 and table.num_rows == 9
    with scan("dictionary-delta.arrow") as reader:
        batches = list(reader)
        assert len(batches) == 2
        dictionaries = [batch.column(1).dictionary.to_pylist() for batch in batches]
        assert dictionaries == [["red", "blue", "green"]] * 2
        assert pa.Table.from_batches(batches).column("category").to_pylist() == ["red", "blue", None, "red", "green", "blue"]
    with scan("nested-dictionary.arrow", offset=2) as reader:
        assert reader.read_all().to_pydict() == {"items":[["red"], ["green", "blue"]],
                                                "record":[{"category":"red"}, {"category":"green"}]}
    for name in ["early-corrupt.arrow", "late-corrupt.arrow"]:
        with scan(name) as reader:
            try:
                reader.read_all()
            except (pa.ArrowInvalid, OSError) as error:
                assert "ZSTD" in str(error), error
            else:
                raise AssertionError("full native scan must report the corrupt body")
    # Exercise early close without fetching the later corrupt batch.
    reader = scan("late-corrupt.arrow")
    assert reader.read_next_batch().num_rows == 3
    reader.close()
    reader.close()
    for offset, rows in [(-1, 3), (0, -2)]:
        try:
            scan("core-zstd.arrow", offset=offset, rows=rows)
        except ValueError:
            pass
        else:
            raise AssertionError("native scan must reject invalid bounds")
    print("Independent native IPC scan consumer passed projection, metadata-only skips, limits, dictionaries, errors and early close")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-written", type=Path)
    parser.add_argument("--check-scan-library", type=Path)
    args = parser.parse_args()
    if args.check_scan_library:
        check_scan_library(args.check_scan_library)
    elif args.check_written:
        check_written(args.check_written)
    else:
        generate()
