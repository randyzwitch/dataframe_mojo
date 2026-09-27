#!/usr/bin/env python3
"""Exercise the native Arrow C stream: projection, EOF, close and decode errors.

Run with pixi run -e oracle python3 scripts/check_parquet_stream.py LIBRARY.
The Mojo lifecycle tests separately count importer releases on error paths.
"""
import ctypes as c
from pathlib import Path
import sys
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq


class Stream(c.Structure):
    _fields_ = [
        (name, c.c_void_p)
        for name in (
            "get_schema",
            "get_next",
            "get_last_error",
            "release",
            "private_data",
        )
    ]


class Array(c.Structure):
    _fields_ = [
        (name, c.c_int64)
        for name in (
            "length",
            "null_count",
            "offset",
            "n_buffers",
            "n_children",
        )
    ]
    _fields_ += [
        (name, c.c_void_p)
        for name in (
            "buffers",
            "children",
            "dictionary",
            "release",
            "private_data",
        )
    ]


class Schema(c.Structure):
    _fields_ = [(name, c.c_void_p) for name in ("format", "name", "metadata")]
    _fields_ += [("flags", c.c_int64), ("n_children", c.c_int64)]
    _fields_ += [
        (name, c.c_void_p)
        for name in ("children", "dictionary", "release", "private_data")
    ]


GET = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)
RELEASE = c.CFUNCTYPE(None, c.c_void_p)
ERROR = c.CFUNCTYPE(c.c_char_p, c.c_void_p)
lib = c.CDLL(sys.argv[1])
lib.dfq_read_parquet_stream.argtypes = [
    c.c_char_p,
    c.c_int,
    c.c_void_p,
    c.c_int,
    c.c_void_p,
    c.c_int,
    c.POINTER(Stream),
    c.POINTER(c.c_void_p),
]
lib.dfq_free.argtypes = [c.c_void_p]


def opened(path, columns=(), groups=None):
    stream = Stream()
    error = c.c_void_p()
    names = (c.c_char_p * len(columns))(*(n.encode() for n in columns))
    indices = (c.c_int * len(groups))(*groups) if groups is not None else None
    status = lib.dfq_read_parquet_stream(
        str(path).encode(),
        0,
        names,
        len(names),
        indices,
        -1 if groups is None else len(groups),
        c.byref(stream),
        c.byref(error),
    )
    if status:
        message = c.string_at(error).decode()
        lib.dfq_free(error)
        raise RuntimeError(message)
    return stream


def close(stream):
    assert stream.release
    RELEASE(stream.release)(c.byref(stream))
    assert not stream.release


def next_batch(stream):
    array, schema = Array(), Schema()
    status = GET(stream.get_next)(c.byref(stream), c.byref(array))
    try:
        if status:
            raise RuntimeError(
                ERROR(stream.get_last_error)(c.byref(stream)).decode()
            )
        if not array.release:
            return None
        assert GET(stream.get_schema)(c.byref(stream), c.byref(schema)) == 0
        return pa.RecordBatch._import_from_c(
            c.addressof(array), c.addressof(schema)
        )
    finally:
        if array.release:
            RELEASE(array.release)(c.byref(array))
        if schema.release:
            RELEASE(schema.release)(c.byref(schema))


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "groups.parquet"
    table = pa.table(
        {"id": range(3000), "text": [f'row-{i}' for i in range(3000)]}
    )
    pq.write_table(
        table,
        path,
        row_group_size=1000,
        compression="NONE",
        use_dictionary=False,
    )
    for columns, groups in [
        ((), None),
        (("text", "id"), [2, 0, 2]),
        (("id",), []),
    ]:
        stream = opened(path, columns, groups)
        try:
            batches = []
            while (batch := next_batch(stream)) is not None:
                batches.append(batch)
            assert next_batch(stream) is None  # repeated EOF
            selected = list(range(3)) if groups is None else groups
            assert len(batches) == max(1, len(selected))
            expected = pa.concat_tables(
                [table.slice(g * 1000, 1000) for g in selected]
            ) if selected else table.slice(0, 0)
            if columns:
                expected = expected.select(columns)
            assert pa.Table.from_batches(batches).equals(expected)
        finally:
            close(stream)
    stream = opened(path)
    first = next_batch(stream)
    close(stream)  # consumer stops early; exported batch remains valid
    assert first.column(0).to_pylist() == list(range(1000))
    metadata = pq.read_metadata(path)
    offset = metadata.row_group(1).column(0).data_page_offset
    with path.open("r+b") as file:
        file.seek(offset)
        file.write(b"\xff" * 32)
    stream = opened(path)
    try:
        assert next_batch(stream).num_rows == 1000
        try:
            next_batch(stream)
        except RuntimeError as error:
            assert str(error)
        else:
            raise AssertionError("corrupt second row group did not fail")
    finally:
        close(stream)
print(
    "native stream: projection, ordering, empty selection, EOF, early close and mid-stream error passed"
)
