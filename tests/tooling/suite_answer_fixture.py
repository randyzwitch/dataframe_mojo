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


def arrow_type(label):
    """The Arrow type for a logical type label used in these contracts."""
    import re
    import pyarrow as pa

    match = re.fullmatch(r"decimal(?:128)?\((\d+), (\d+)\)", label)
    if match:
        return pa.decimal128(int(match.group(1)), int(match.group(2)))
    if label == "integer_sum":
        return pa.decimal128(38, 0)
    named = {
        "string": pa.string(),
        "binary": pa.binary(),
        "bool": pa.bool_(),
        "float": pa.float32(),
        "double": pa.float64(),
    }
    return named.get(label) or pa.type_for_alias(label)


def rows_answer(rows, names=("a",), types=("int64",), original_schema=None):
    """A columnar suite answer built from Python rows."""
    import pyarrow as pa
    import suite_answers

    names, types = list(names), list(types)
    columns = [
        pa.array([row[k] for row in rows], type=arrow_type(t))
        for k, t in enumerate(types)
    ]
    table = pa.Table.from_arrays(columns, names=names)
    if not columns:
        table = pa.table({"_": pa.nulls(len(rows))}).select([])
    return suite_answers.Answer(names, types, table, original_schema or {})
