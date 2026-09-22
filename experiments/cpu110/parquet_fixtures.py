"""Generate comparable Parquet encodings for the format/Arrow yardstick."""
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq

root = Path('build/cpu110_parquet')
root.mkdir(parents=True, exist_ok=True)
for rows in [1_000_000, 10_000_000]:
    table = pl.read_csv(f'build/bench_polars/left_{rows}.csv').to_arrow()
    for codec in ['NONE', 'SNAPPY', 'ZSTD', 'LZ4']:
        for dictionary in [False, True]:
            # Full codec/encoding matrix at 1M; larger-file scaling on the
            # common dictionary/Snappy combination.
            if rows == 10_000_000 and (codec != 'SNAPPY' or not dictionary):
                continue
            path = root / f'left_{rows}_{codec}_{int(dictionary)}.parquet'
            pq.write_table(table, path, compression=codec, use_dictionary=dictionary, row_group_size=1_000_000)
            back = pq.read_table(path)
            if not back.equals(table):
                raise AssertionError(path)
            print(path, path.stat().st_size, flush=True)
