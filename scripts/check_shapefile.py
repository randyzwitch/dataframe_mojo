"""Run the independent shapefile oracle in a temporary fixture directory."""
from pathlib import Path
import subprocess
import tempfile

with tempfile.TemporaryDirectory(prefix="shapefile-oracle-") as directory:
    subprocess.run(
        ["mojo", "run", "-I", ".", "tests/oracle/shapefile.mojo", directory],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
