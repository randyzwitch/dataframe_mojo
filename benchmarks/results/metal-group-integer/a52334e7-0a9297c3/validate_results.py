"""Validate paired coverage, raw outputs, build attribution and grouped routes."""
import importlib.util
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
ATTRIBUTION = json.loads((ROOT / 'attribution.json').read_text())


def load_parser():
    spec = importlib.util.spec_from_file_location('archived_harness', ROOT / 'harness/bench_metal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(folder, reference):
    path = ROOT / folder / 'results.json'
    data = json.loads(path.read_text())
    expected = set(itertools.product(range(PLAN['rounds']), ('baseline', 'mojo'), PLAN['rows'], PLAN['candidates'][0]['workloads'], PLAN['variants']))
    observed = set()
    parser = load_parser()
    for run in data['runs']:
        key = (run['round'], run['runtime'], run['rows'], run['workload'], run['variant'])
        assert key not in observed, ('duplicate', key)
        observed.add(key)
        assert run['status'] == 'ok' and run['validated'] is True, key
        assert len(run['samples']) == PLAN['reps'], key
        assert all(s['cpu_ns'] > 0 and s['metal_ns'] > 0 for s in run['samples']), key
        assert isinstance(run['output_rows'], int) and 0 <= run['output_rows'] <= run['rows'], key
        parsed = parser.parse_run((path.parent / 'results' / run['log']).read_text(), PLAN['reps'])
        for field, value in parsed.items():
            assert run[field] == value, ('raw log mismatch', key, field)
        for field in ('upload_bytes', 'download_bytes', 'host_result_bytes', 'peak_requested_bytes', 'kernel_launches', 'synchronizations'):
            assert isinstance(run['profile'][field], int) and run['profile'][field] >= 0, (key, field)
        if run['runtime'] == 'mojo':
            route = 'mojo_gpu_grouped_int_partials' if run['rows'] == 1048576 and run['workload'] in ('group_hash','group_skew') else 'legacy'
            assert run['profile']['region_0_grouped_int_algorithm'] == route, key
    assert observed == expected, 'Missing or unexpected workers'
    receipts = data['metadata']['binary_receipts']
    for role, source in [('baseline', reference), ('mojo', 'candidate')]:
        r = receipts[role]
        assert r['status'] == 'verified' and not r['engine_status'] and not r['git_status']
        assert r['git_commit'] == ATTRIBUTION[source]['measurement_revision']
        assert r['engine_content_sha256'] == ATTRIBUTION[source]['engine_content_sha256']
    for field in ('benchmark_source_sha256','driver_source_sha256','pixi_lock_sha256','compiler_identity','compiler_executable_sha256','build_sdk','build_xcode','compile_flags'):
        assert receipts['baseline'][field] == receipts['mojo'][field], field
    assert receipts['mojo']['benchmark_source_sha256'] == PLAN['benchmark_file_sha256']['benchmarks/bench_metal.mojo']
    assert receipts['mojo']['driver_source_sha256'] == PLAN['benchmark_file_sha256']['benchmarks/tools/bench_metal.py']
    print('PASS', folder, len(observed), 'validated workers; raw logs, source/build identity and routes match')


def validate_smoke(role):
    path = ROOT / (role + '-smoke') / 'results.json'
    data = json.loads(path.read_text())
    expected = set(itertools.product(PLAN['candidates'][0]['workloads'], PLAN['variants']))
    observed = set()
    parser = load_parser()
    for run in data['runs']:
        key = (run['workload'], run['variant'])
        assert key not in observed and run['rows'] == 4097 and run['runtime'] == 'mojo' and run['round'] == 0
        observed.add(key)
        assert run['status'] == 'ok' and run['validated'] is True and len(run['samples']) == 1
        parsed = parser.parse_run((path.parent / 'results' / run['log']).read_text(), 1)
        for field, value in parsed.items():
            assert run[field] == value
    assert observed == expected
    receipt = data['metadata']['binary_receipts']['mojo']
    assert receipt['status'] == 'verified' and receipt['git_commit'] == ATTRIBUTION[role]['measurement_revision']
    assert receipt['engine_content_sha256'] == ATTRIBUTION[role]['engine_content_sha256']
    print('PASS', role + '-smoke', len(observed), 'validated workers and raw logs')


if __name__ == '__main__':
    for name, digest in PLAN['benchmark_file_sha256'].items():
        assert hashlib.sha256((ROOT / 'harness' / Path(name).name).read_bytes()).hexdigest() == digest
    gate = json.loads((ROOT / 'validation/group-restack-receipt.json').read_text())['gates'][0]
    assert gate['revision'] == ATTRIBUTION['candidate']['engine_revision'] and gate['clean'] is True and gate['build_exit'] == 0 and gate['test_exit'] == 0
    for name, digest in gate['files'].items():
        if name.endswith('.log'):
            assert hashlib.sha256((ROOT / 'validation' / Path(name).name).read_bytes()).hexdigest() == digest
    validate_smoke('baseline')
    validate_smoke('intermediate')
    validate('baseline-pair', 'baseline')
    validate('intermediate-pair', 'intermediate')
