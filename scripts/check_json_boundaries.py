#!/usr/bin/env python3
"""Keep JSON value/lexical semantics separate from table I/O and query execution."""
from pathlib import Path
import re
import sys
from check_lazy_boundaries import imports, module_name

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {'dataframe.json_' + name for name in (
    'decode', 'number', 'path', 'path_number', 'reader', 'strings', 'value',
)} | {'dataframe.io.json_wire'}


def check():
    failures = []
    for path in sorted((ROOT / 'dataframe/json').rglob('*.mojo')):
        name = module_name(path)
        for dependency in imports(path):
            if dependency in {'dataframe', 'dataframe.execution', 'dataframe.binding'} or dependency.startswith((
                'dataframe.execution.', 'dataframe.binding.', 'dataframe.frame',
                'dataframe.lazy', 'dataframe.planning', 'dataframe.optimizer',
                'dataframe.query_execution', 'dataframe.streaming', 'dataframe.io.',
                'dataframe_provider', 'dataframe._accel_provider',
            )):
                failures.append(f'{name} imports table/query orchestration: {dependency}')
    for tree in ('dataframe', 'dataframe_metal', 'dataframe_provider', 'providers'):
        for path in sorted((ROOT / tree).rglob('*.mojo')):
            name = module_name(path)
            if name in LEGACY:
                source = re.sub(r'"""[\s\S]*?"""|#[^\n]*', '', path.read_text())
                if re.search(r'^(?:def|struct|trait|comptime)\s+', source, re.M):
                    failures.append(f'{name} defines implementation in a compatibility facade')
                continue
            for dependency in sorted(set(imports(path)) & LEGACY):
                failures.append(f'{name} imports legacy JSON facade: {dependency}')
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print('JSON value boundaries hold.')
    return 0


if __name__ == '__main__':
    sys.exit(check())
