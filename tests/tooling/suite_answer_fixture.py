"""Typed synthetic answer files for collector/report contracts, outside git."""
import json
from pathlib import Path
import tempfile
import unittest


_FILES = tempfile.TemporaryDirectory(prefix="suite-answer-fixtures-")


def export_fixture(
    columns=None, *, query="q1", repetition=0, oracle=False, directory=None
):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise unittest.SkipTest("complete answer fixtures require PyArrow")
    import suite_answers

    directory = Path(directory) if directory is not None else Path(
        tempfile.mkdtemp(dir=_FILES.name)
    )
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns or {"id1": ["a", "b"], "v1": [1, 2]})
    path = directory / f"{query}-{repetition}.parquet"
    schema_path = directory / f"{query}-{repetition}.schema.json"
    pq.write_table(table, path)
    descriptor = {
        "format": "suite-answer-v1",
        "query": query,
        "repetition": repetition,
        "rows": table.num_rows,
        "columns": [
            {"name": f.name, "arrow": str(f.type)} for f in table.schema
        ],
    }
    if oracle:
        descriptor["oracle"] = {
            "width": table.num_columns,
            "order": [],
            "limit": None,
            "offset": 0,
        }
    schema_path.write_text(json.dumps(descriptor))
    return suite_answers.pin_export(
        {
            "path": str(path),
            "schema_path": str(schema_path),
            "repetition": repetition,
        },
        directory,
        query,
    )
