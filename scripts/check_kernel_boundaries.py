#!/usr/bin/env python3
"""Keep computational implementation below expression/query execution."""
from pathlib import Path
import re
import sys
from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {
    'dataframe.' + name for name in (
        'expr_kernels', 'float_ops', 'fusion', 'gather', 'mask_filter',
        'list_kernels', 'temporal_kernels', 'hashing', 'partition', 'sampling',
        'row_encode', 'packed_sort', 'row_sort', 'comparison_sort', 'top_k',
        'rank', 'runtime_filter', 'join_fuse', 'join_hash', 'distinct',
        'distinct_statistics', 'approx_distinct', 'horizontal', 'cast',
        'replace', 'canonical_labels',
    )
}


def check():
    failures = []
    for path in sorted((ROOT / 'dataframe/kernels').rglob('*.mojo')):
        name = module_name(path)
        for dependency in imports(path):
            if dependency in {'dataframe', 'dataframe.execution', 'dataframe.binding.bind'} or dependency.startswith((
                'dataframe.execution.', 'dataframe.frame', 'dataframe.lazy',
                'dataframe.planning', 'dataframe.optimizer', 'dataframe.query_execution',
                'dataframe.streaming', 'dataframe.fused_compare', 'dataframe.sorted_range',
                'dataframe_provider', 'dataframe._accel_provider',
            )):
                failures.append(f'{name} imports expression/query execution: {dependency}')
            if dependency.startswith('dataframe.binding.') and dependency != 'dataframe.binding.BoundExpr':
                failures.append(f'{name} imports binding behavior rather than its value model: {dependency}')
    for tree in ('dataframe', 'dataframe_metal', 'dataframe_provider', 'providers'):
        for path in sorted((ROOT / tree).rglob('*.mojo')):
            name = module_name(path)
            if name in LEGACY:
                source = re.sub(r'"""[\s\S]*?"""|#[^\n]*', '', path.read_text())
                if re.search(r'^(?:def|struct|trait|comptime)\s+', source, re.M):
                    failures.append(f'{name} defines implementation in a compatibility facade')
                continue
            for dependency in sorted(set(imports(path)) & LEGACY):
                failures.append(f'{name} imports legacy kernel facade: {dependency}')
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print('Computational kernel boundaries hold.')
    return 0


if __name__ == '__main__':
    sys.exit(check())
