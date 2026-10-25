"""Independent PyArrow IPC fixtures and consumers of actual Mojo outputs."""
import argparse
from pathlib import Path
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


def corrupt_later_batch(path, stream):
    data = bytearray(path.read_bytes())
    with pa.BufferReader(data) as source:
        source.seek(0 if stream else 8)
        records = 0
        while True:
            message = ipc.read_message(source)
            if message.type != "record batch":
                continue
            records += 1
            if records == 2:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-written", type=Path)
    args = parser.parse_args()
    if args.check_written:
        check_written(args.check_written)
    else:
        generate()
