"""Read actual incremental/partitioned writer artifacts with PyArrow."""
import argparse
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq


def check_written(root: Path):
    source = {"i": [1, 2, 3], "s": ["a,b", "", 'quoted"value']}
    path = root / "stream-writer-groups.parquet"
    file = pq.ParquetFile(path)
    assert file.read().to_pydict() == source
    assert file.schema_arrow.field("i").type == pa.int64()
    assert [file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)] == [2, 1]
    assert file.metadata.row_group(0).column(0).compression == "ZSTD"
    empty = pq.ParquetFile(root / "stream-writer-empty.parquet")
    assert empty.read().to_pydict() == {"i": [], "s": []}
    assert empty.schema_arrow == file.schema_arrow
    dictionary = pq.ParquetFile(root / "stream-writer-dictionary-batches.parquet")
    assert dictionary.read().to_pydict() == {"s": ["red", "blue", "blue", "red", "green"]}

    # Literal values are independent of the Mojo reader and Hive encoder.
    expected = {
        ("a/b", 1): [0, 1], ("", 2): [2],
        ("__HIVE_DEFAULT_PARTITION__", 2): [3],
        (None, 3): [4], ("é", 3): [5], ("a=b", 4): [6], ("a/b", 5): [7],
    }
    found = {}
    for path in (root / "partitioned-parquet-escaped").rglob("*.parquet"):
        parts = path.relative_to(root / "partitioned-parquet-escaped").parts
        key, raw = parts[0].split("=", 1)
        assert unquote(key) == "group/name"
        value = None if raw == "__HIVE_DEFAULT_PARTITION__" else unquote(raw)
        j = int(parts[1].split("=", 1)[1])
        table = pq.ParquetFile(path).read()
        rows = expected[(value, j)]
        assert table.to_pydict() == {
            "group/name": [value] * len(rows), "j": [j] * len(rows), "v": rows,
        }
        found[(value, j)] = rows
    assert found == expected
    print("Independent PyArrow incremental and partitioned output checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-written", type=Path, required=True)
    check_written(parser.parse_args().check_written)
