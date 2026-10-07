"""Development-only #122 oracle: pixi run -e oracle oracle-duckdb-spatial."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "oracle"))
from duckdb_spatial_fixtures import install_spatial, check_mutation_detection


def main():
    install_spatial()
    check_mutation_detection()
    subprocess.run(
        ["mojo", "run", "-I", ".", "tests/oracle/duckdb_spatial.mojo"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
