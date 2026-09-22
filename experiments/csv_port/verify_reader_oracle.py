"""Compare actual clean-reader outputs with pinned Polars (no timing).

Build verify_reader.mojo first, then pass its binary path. Serialization is
outside the read path; rereading uses Polars' expected schema to preserve types.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile

import polars as pl
from polars.testing import assert_frame_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    assert pl.__version__ == "1.44.2", pl.__version__
    with tempfile.TemporaryDirectory(prefix="csv-reader-oracle-") as directory:
        source = Path(directory) / "input.csv"
        output = Path(directory) / "output.csv"
        with source.open("w", newline="", encoding="utf8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["id", "value", "active", "label"])
            for i in range(6000):
                label = f'group,{i}: "café 日本"'
                if i % 101 == 0:
                    label += "\nsecond line"
                if i % 23 == 0:
                    label = "short"
                writer.writerow([i - 3000, "" if i % 11 == 0 else i / 8,
                                 "true" if i % 2 else "false", label])
        schema = {"id": pl.Int64, "value": pl.Float64,
                  "active": pl.Boolean, "label": pl.String}
        for threads in (1, 4):
            for mode in ("explicit", "inferred"):
                for scenario in ("full", "projected"):
                    columns = None if scenario == "full" else ["id", "label"]
                    expected = pl.read_csv(source, schema=schema, columns=columns)
                    subprocess.run([str(args.binary.resolve()), str(source), str(output),
                                    mode, scenario], check=True,
                                   env={**os.environ, "DATAFRAME_THREADS": str(threads)})
                    actual = pl.read_csv(output, schema=expected.schema)
                    assert_frame_equal(actual, expected)
                    print(f"PASS {mode} {scenario} threads={threads}: {actual.shape}")


if __name__ == "__main__":
    main()
