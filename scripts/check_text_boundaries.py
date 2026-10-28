#!/usr/bin/env python3
"""Keep lexical/text processing separate from query and file orchestration."""
from pathlib import Path
import re
import sys
from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {'dataframe.' + name for name in (
    'str_kernels', 'string_bytes', 'string_lengths', 'string_predicates',
    'string_transform', 'text_functions', 'byte_encoding', 'parse',
    'regex_kernels', 'regex_cache',
)}


def check():
    failures = []
    for path in sorted((ROOT / 'dataframe/text').rglob('*.mojo')):
        name = module_name(path)
        for dependency in imports(path):
            if dependency in {'dataframe', 'dataframe.execution', 'dataframe.binding'} or dependency.startswith((
                'dataframe.execution.', 'dataframe.binding.', 'dataframe.frame',
                'dataframe.lazy', 'dataframe.planning', 'dataframe.optimizer',
                'dataframe.query_execution', 'dataframe.streaming',
                'dataframe_provider', 'dataframe._accel_provider',
            )):
                failures.append(f'{name} imports query/expression execution: {dependency}')
            # Legacy lexical floating conversion shares only this numeric
            # primitive with the separate CSV decoder; it does not read files.
            if dependency.startswith('dataframe.io.') and dependency not in {
                'dataframe.io.csv_numeric', 'dataframe.io.csv_numeric._pow10_f64',
            }:
                failures.append(f'{name} imports file orchestration: {dependency}')
    for tree in ('dataframe', 'dataframe_metal', 'dataframe_provider', 'providers'):
        for path in sorted((ROOT / tree).rglob('*.mojo')):
            name = module_name(path)
            if name in LEGACY:
                source = re.sub(r'"""[\s\S]*?"""|#[^\n]*', '', path.read_text())
                if re.search(r'^(?:def|struct|trait|comptime)\s+', source, re.M):
                    failures.append(f'{name} defines implementation in a compatibility facade')
                continue
            for dependency in sorted(set(imports(path)) & LEGACY):
                failures.append(f'{name} imports legacy text facade: {dependency}')
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print('Text processing boundaries hold.')
    return 0


if __name__ == '__main__':
    sys.exit(check())
