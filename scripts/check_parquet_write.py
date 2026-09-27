#!/usr/bin/env python3
"""Read files produced by the Mojo API with an independent PyArrow reader."""
from pathlib import Path
import subprocess
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq


def normalized(dtype):
    if pa.types.is_string(dtype):
        return pa.large_string()
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.large_list(normalized(dtype.value_type))
    if pa.types.is_struct(dtype):
        return pa.struct([(f.name, normalized(f.type)) for f in dtype])
    return dtype


with tempfile.TemporaryDirectory() as directory:
    binary = str(Path(directory) / "writer")
    subprocess.run(
        [
            "mojo",
            "build",
            "-I",
            ".",
            "tests/oracle/parquet_write.mojo",
            "-o",
            binary,
        ],
        check=True,
    )
    subprocess.run([binary, directory], check=True)
    expected = {}
    for fixture in (
        "types.parquet",
        "list_int.parquet",
        "struct.parquet",
        "empty.parquet",
    ):
        table = pq.read_table("tests/fixtures/" + fixture)
        schema = pa.schema([(f.name, normalized(f.type)) for f in table.schema])
        expected[fixture] = table.cast(schema)
    columns = {}
    for unit in ("ns", "us", "ms"):
        columns["duration_" + unit] = pa.array(
            [0, None, 456], type=pa.duration(unit)
        )
        columns["timestamp_" + unit] = pa.array(
            [0, -123, 456], type=pa.timestamp(unit)
        )
    expected["temporal.parquet"] = pa.table(columns)
    for codec in ("zstd", "snappy", "uncompressed"):
        for fixture, source in expected.items():
            path = str(Path(directory) / (codec + "-" + fixture))
            actual = pq.read_table(path)
            actual.validate(full=True)
            assert actual.equals(source), (
                codec,
                fixture,
                actual.schema,
                source.schema,
            )
            metadata = pq.read_metadata(path)
            for group in range(metadata.num_row_groups):
                assert metadata.row_group(group).num_rows <= 2
                assert (
                    metadata.row_group(group).column(0).compression
                    == codec.upper()
                )
print(
    "PyArrow verified Mojo writes: types, nested fields, nulls, codecs, row groups and temporal units"
)
