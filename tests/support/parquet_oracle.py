"""Independent PyArrow Parquet integration adapter; runtime contracts are Mojo.

Generate: .pixi/envs/oracle/bin/python tests/support/parquet_oracle.py
Verify public Mojo writer outputs: same command --check-written <directory>
"""
import argparse
from decimal import Decimal
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

FIXTURES = Path(__file__).parents[1] / "fixtures/parquet"


def core_table():
    return pa.table(
        {
            "x": pa.array(
                [0, 1, 2, 3, 100, 101, 102, 200, 201], type=pa.int64()
            ),
            "s": pa.array(
                ["red", "", "blue", None, "red", "green", "red", "blue", "last"]
            ),
            "score": pa.array([1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        }
    )


def typed_table():
    arrays = [
        pa.array([-2, None, 7], type=pa.int8()),
        pa.array([2**64 - 1, None, 0], type=pa.uint64()),
        pa.array([1.25, None, -2.5], type=pa.float32()),
        pa.array([True, None, False]),
        pa.array(["", None, 'café\nquoted "x"'], type=pa.large_string()),
        pa.array([b"\xff\x00", None, b""], type=pa.large_binary()),
        pa.array([-1, None, 19782], type=pa.date32()),
        pa.array([-1, None, 42], type=pa.timestamp("ns", "America/New_York")),
        pa.array([-1, None, 42], type=pa.duration("us")),
        pa.array([1, None, 42], type=pa.time64("ns")),
        pa.array(
            [Decimal("-12.34"), None, Decimal("56.78")], type=pa.decimal32(9, 2)
        ),
        pa.array([[1, None, 3], [], None], type=pa.large_list(pa.int64())),
        pa.array(
            [{"id": 7, "name": "a"}, None, {"id": 9, "name": None}],
            type=pa.struct([("id", pa.int64()), ("name", pa.large_string())]),
        ),
    ]
    names = [
        "i",
        "u",
        "f",
        "flag",
        "text",
        "binary",
        "day",
        "stamp",
        "duration",
        "time",
        "decimal",
        "list",
        "struct",
    ]
    fields = [
        pa.field(name, array.type, metadata={b"application": b"independent"})
        for name, array in zip(names, arrays)
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def generate():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    core = core_table()
    for name, dictionary, statistics in [
        ("dictionary", ["s"], True),
        ("plain", False, True),
        ("unknown-statistics", False, False),
    ]:
        path = FIXTURES / f"{name}.parquet"
        with pq.ParquetWriter(
            path,
            core.schema,
            compression="NONE",
            use_dictionary=dictionary,
            write_statistics=statistics,
        ) as writer:
            for start, size in [(0, 4), (4, 0), (4, 3), (7, 2)]:
                writer.write_table(core.slice(start, size))
        file = pq.ParquetFile(path)
        assert [file.metadata.row_group(i).num_rows for i in range(4)] == [
            4,
            0,
            3,
            2,
        ]
        assert file.read().equals(core)
    good = FIXTURES / "plain.parquet"
    file = pq.ParquetFile(good)
    damaged = bytearray(good.read_bytes())
    chunk = file.metadata.row_group(3).column(0)
    start = chunk.data_page_offset
    damaged[start : start + 16] = b"\x00" * 16
    corrupt = FIXTURES / "late-corrupt.parquet"
    corrupt.write_bytes(damaged)
    check = pq.ParquetFile(corrupt)
    assert check.read_row_group(0).equals(core.slice(0, 4))
    assert check.read(columns=["s"]).equals(core.select(["s"]))
    try:
        check.read_row_group(3, columns=["x"])
    except (pa.ArrowInvalid, OSError):
        pass
    else:
        raise AssertionError(
            "independently damaged final column page must reject"
        )
    typed = typed_table()
    pq.write_table(
        typed, FIXTURES / "typed.parquet", compression="NONE", row_group_size=2
    )
    assert pq.read_table(FIXTURES / "typed.parquet").equals(typed)
    pq.write_table(
        typed.slice(0, 0), FIXTURES / "empty.parquet", compression="NONE"
    )
    print(
        "Verified independent fixtures: four row groups including empty group, dictionary/plain/no statistics, late corrupt column page, typed/nested metadata, empty typed file"
    )


def check_written(directory):
    expected = typed_table()
    for codec in ["zstd", "snappy", "uncompressed"]:
        for size in [1, 2, 17]:
            file = pq.ParquetFile(
                directory / f"parquet-written-{codec}-{size}.parquet"
            )
            actual = file.read()
            assert actual.equals(expected), (
                codec,
                size,
                actual.schema,
                expected.schema,
            )
            for actual_field, expected_field in zip(
                actual.schema, expected.schema
            ):
                assert actual_field.metadata == expected_field.metadata, (
                    codec,
                    size,
                    actual_field,
                    expected_field,
                )
            for group in range(file.metadata.num_row_groups):
                assert file.metadata.row_group(group).num_rows <= size
                for column in range(file.metadata.num_columns):
                    declared = (
                        file.metadata.row_group(group)
                        .column(column)
                        .compression
                    )
                    assert (
                        declared
                        == {
                            "zstd": "ZSTD",
                            "snappy": "SNAPPY",
                            "uncompressed": "UNCOMPRESSED",
                        }[codec]
                    )
    empty = pq.read_table(directory / "parquet-written-empty.parquet")
    assert empty.equals(expected.slice(0, 0))
    print(
        "Independent PyArrow consumer verified all public Mojo writer outputs, exact typed/nested cells, schema, field metadata, codecs and row-group bounds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-written", type=Path)
    args = parser.parse_args()
    if args.check_written:
        check_written(args.check_written)
    else:
        generate()
