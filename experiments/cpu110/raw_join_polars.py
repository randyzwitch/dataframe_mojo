"""Matched unique-build-key Polars comparator for raw_partitioned_join_matrix_bench.

Fixture construction and exact output validation are excluded from timing.
"""
import argparse
import os
import time
import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['int64', 'string'])
    p.add_argument('rows', type=int)
    p.add_argument('reps', type=int)
    p.add_argument('cardinality', type=int)
    p.add_argument('shape', choices=['repeat', 'skew', 'sparse'])
    a = p.parse_args()
    if min(a.rows, a.reps, a.cardinality) <= 0:
        raise ValueError('positive dimensions required')
    def key(i):
        if a.kind == 'string':
            return f'key_{i}'
        return -(1 << 63) + i * 1_000_003 if a.shape == 'sparse' else i
    def index(i):
        return 0 if a.shape == 'skew' and i % 10 else i % a.cardinality
    dtype = pl.String if a.kind == 'string' else pl.Int64
    left = pl.DataFrame({
        'k': pl.Series([key(index(i)) if i % 127 else None for i in range(a.rows)], dtype=dtype),
        'left_payload': pl.Series(range(a.rows), dtype=pl.Int64),
    })
    right = pl.DataFrame({
        'k': pl.Series([key(i) for i in range(a.cardinality)], dtype=dtype),
        'right_payload': pl.Series(range(a.cardinality), dtype=pl.Int64),
    })
    def operation():
        return left.join(right, on='k', how='inner', maintain_order='left_right')
    result = operation()
    expected_rows = [i for i in range(a.rows) if i % 127]
    if result['left_payload'].to_list() != expected_rows:
        raise AssertionError('left order or null matching differs')
    if result['right_payload'].to_list() != [index(i) for i in expected_rows]:
        raise AssertionError('right payload differs')
    best = float('inf')
    for _ in range(a.reps):
        start = time.perf_counter_ns()
        result = operation()
        best = min(best, time.perf_counter_ns() - start)
    print('kind,shape,rows,cardinality,threads,output_rows,polars_join_ns')
    print(a.kind, a.shape, a.rows, a.cardinality, os.getenv('POLARS_MAX_THREADS'), result.height, best, sep=',')


if __name__ == '__main__':
    main()
