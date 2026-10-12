"""Independent PyArrow producer for native coercion and dictionary contracts."""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    root = Path(__file__).parents[1] / "fixtures/parquet"
    half = pa.array(
        np.asarray([1.5, 0, -2.25], dtype=np.float16),
        mask=np.asarray([False, True, False]),
    )
    table = pa.table(
        {
            "half": half,
            "view": pa.array(
                ["long external string", None, "é"], type=pa.string_view()
            ),
        }
    )
    path = root / "coercions.parquet"
    pq.write_table(
        table, path, compression="NONE", use_dictionary=False, row_group_size=2
    )
    actual = pq.read_table(path)
    actual.validate(full=True)
    assert actual.equals(table)
    print("Independent PyArrow float16 and utf8_view fixture verified")


if __name__ == "__main__":
    main()
