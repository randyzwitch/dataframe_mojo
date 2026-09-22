"""Polars yardstick for the PyArrow/deep-copy Mojo bridge experiment."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter_ns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('paths', nargs='+', type=Path)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--reps', type=int, default=7)
args = parser.parse_args()
os.environ['POLARS_MAX_THREADS'] = str(args.threads)
import polars as pl

for path in args.paths:
    warm = pl.read_parquet(path)
    expected = warm['x'].sum()
    best = None
    for _ in range(args.reps):
        start = perf_counter_ns()
        frame = pl.read_parquet(path)
        elapsed = perf_counter_ns() - start
        assert frame.shape == warm.shape
        assert abs(frame['x'].sum() - expected) <= 1e-10 * max(1, abs(expected))
        best = elapsed if best is None else min(best, elapsed)
    print(json.dumps({'file': str(path), 'rows': warm.height, 'bytes': path.stat().st_size,
                      'best_ns': best, 'x_sum': expected, 'polars': pl.__version__,
                      'threads': args.threads}), flush=True)
