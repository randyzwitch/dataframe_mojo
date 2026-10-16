"""Verify optional answer exports without changing query correctness status."""
import hashlib
import json
from pathlib import Path

ENGINES = {"cpu", "metal", "cpu_repeat"}


def parse_export(payload):
    record = json.loads(payload)
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("engine"), str)
        or record.get("engine") not in ENGINES
        or type(record.get("repetition")) is not int
        or record["repetition"] < 0
        or not isinstance(record.get("status"), str)
        or record.get("status") not in {"written", "unavailable"}
        or any(
            not isinstance(record.get(k), str) for k in ("path", "schema_path")
        )
        or (
            record["status"] == "unavailable"
            and not isinstance(record.get("error"), str)
        )
    ):
        raise ValueError("Malformed answer export protocol")
    return record


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_exports(records, prefix, repetitions):
    """Read only driver-owned paths, pin complete files, retain missing evidence."""
    prefix = Path(prefix)
    expected = (
        [("cpu", 0)]
        + [("metal", i) for i in range(repetitions)]
        + [("cpu_repeat", 0)]
    )
    selected = {}
    for record in records:
        key = record["engine"], record["repetition"]
        if key not in expected or key in selected:
            raise ValueError("Duplicate or unexpected answer export")
        stem = str(prefix) + "-" + key[0] + "-" + str(key[1])
        paths = {
            "path": Path(stem + ".parquet"),
            "schema_path": Path(stem + ".schema.json"),
        }
        if any(
            Path(record[k]) != p
            or p.resolve().parent != prefix.parent.resolve()
            for k, p in paths.items()
        ):
            raise ValueError("Answer export path is not owned by this case")
        result = dict(record)
        if result["status"] == "written":
            try:
                schema = json.loads(paths["schema_path"].read_text())
                columns = schema.get("columns")
                if (
                    schema.get("format") != "dataframe-answer-schema-v1"
                    or schema.get("engine") != key[0]
                    or schema.get("repetition") != key[1]
                    or type(schema.get("rows")) is not int
                    or schema["rows"] < 0
                    or not isinstance(columns, list)
                    or any(
                        not isinstance(c, dict)
                        or c.get("index") != i
                        or not isinstance(c.get("name"), str)
                        or not isinstance(c.get("dtype"), dict)
                        or any(
                            not isinstance(c["dtype"].get(k), str)
                            for k in ("name", "encoded")
                        )
                        for i, c in enumerate(columns)
                    )
                    or len({c["name"] for c in columns}) != len(columns)
                ):
                    raise ValueError("Invalid original answer schema receipt")
                result.update(
                    sha256=file_sha256(paths["path"]),
                    schema_sha256=file_sha256(paths["schema_path"]),
                    original_schema=schema,
                )
            except (OSError, ValueError, TypeError, AttributeError) as error:
                result.update(status="unavailable", error=str(error))
        selected[key] = result
    return [
        selected.get(
            key,
            dict(
                engine=key[0],
                repetition=key[1],
                status="not_observed",
                reason="Query phase was not observed; CPU repeat is only executed after a mismatch.",
            ),
        )
        for key in expected
    ]
