"""Polars as-of validation: eager C Data and lazy Parquet plans."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "oracle"))
from asof_fixtures import cases, check


def main():
    subprocess.run(
        ["mojo", "run", "-I", ".", "tests/oracle/asof.mojo"],
        cwd=ROOT,
        check=True,
    )
    selected = cases()[::7]
    with tempfile.TemporaryDirectory(prefix="asof-oracle-") as name:
        directory = Path(name)
        for i, case in enumerate(selected):
            for side in ("left", "right"):
                pq.write_table(
                    pa.Table.from_batches([case[side]]),
                    directory / f"{i}-{side}.parquet",
                    row_group_size=4,
                )
        (directory / "cases.json").write_text(
            json.dumps(
                [
                    {
                        k: v
                        for k, v in case.items()
                        if k not in ("left", "right")
                    }
                    for case in selected
                ]
            )
        )
        subprocess.run(
            ["mojo", "run", "-I", ".", "tests/oracle/asof_lazy.mojo", name],
            cwd=ROOT,
            check=True,
        )
        for i, case in enumerate(selected):
            check(case, pq.read_table(directory / f"{i}-out.parquet"))
        print(
            f"Polars lazy as-of oracle passed: {len(selected)} cases",
            flush=True,
        )


if __name__ == "__main__":
    main()
