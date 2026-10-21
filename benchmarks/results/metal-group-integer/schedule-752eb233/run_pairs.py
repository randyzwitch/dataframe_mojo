"""Compare the qualified schedule fix with three pinned prior engine binaries."""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
TREES = Path('/Users/randyzwitch/GitHub')
CANDIDATE = TREES / PLAN['candidate_remote']
PIXI = '/Users/randyzwitch/.pixi/bin/pixi'
MANIFEST = '/Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(tree, *args):
    return subprocess.check_output(['git', *args], cwd=tree, text=True).strip()


def parser():
    spec = importlib.util.spec_from_file_location('measurement_harness', CANDIDATE / 'benchmarks/tools/bench_metal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_tree(tree, pin):
    assert git(tree, 'rev-parse', 'HEAD') == pin
    assert not git(tree, 'status', '--porcelain')
    for name, digest in PLAN['benchmark_file_sha256'].items():
        assert sha(tree / name) == digest, name


def check_receipt(receipt, pin, fingerprint):
    assert receipt['status'] == 'verified'
    assert receipt['git_commit'] == pin
    assert receipt['engine_content_sha256'] == fingerprint
    assert not receipt['git_status'] and not receipt['engine_status']
    assert receipt['benchmark_source_sha256'] == PLAN['benchmark_file_sha256']['benchmarks/bench_metal.mojo']
    assert receipt['driver_source_sha256'] == PLAN['benchmark_file_sha256']['benchmarks/tools/bench_metal.py']


def expected_route(rows, workload, variant):
    parallel = rows == 1048576 and (workload == 'group_skew' or (workload == 'group_hash' and variant == 'nulls'))
    return 'mojo_gpu_grouped_int_partials' if parallel else 'legacy'


def audit(path, reference=None):
    data = json.loads(path.read_text())
    harness = parser()
    rounds, reps = (PLAN['rounds'], PLAN['reps']) if reference else (1, 1)
    rows = PLAN['rows'] if reference else [4097]
    runtimes = ['mojo', 'baseline'] if reference else ['mojo']
    expected = set(itertools.product(range(rounds), runtimes, rows, PLAN['workloads'], PLAN['variants']))
    observed = set()
    for run in data['runs']:
        key = (run['round'], run['runtime'], run['rows'], run['workload'], run['variant'])
        assert key not in observed
        observed.add(key)
        assert run['status'] == 'ok' and run['validated'] is True
        assert len(run['samples']) == reps
        assert all(s['metal_ns'] > 0 and s['cpu_ns'] > 0 for s in run['samples'])
        assert isinstance(run['output_rows'], int) and 0 <= run['output_rows'] <= run['rows']
        parsed = harness.parse_run((path.parent / 'results' / run['log']).read_text(), reps)
        for name, value in parsed.items():
            assert run[name] == value, ('Raw log mismatch', key, name)
        for name in ['peak_requested_bytes', 'kernel_launches', 'upload_bytes', 'download_bytes', 'synchronizations']:
            assert isinstance(run['profile'][name], int) and run['profile'][name] >= 0
        if run['runtime'] == 'mojo':
            assert run['profile']['region_0_grouped_int_algorithm'] == expected_route(run['rows'], run['workload'], run['variant']), key
    assert observed == expected
    receipts = data['metadata']['binary_receipts']
    check_receipt(receipts['mojo'], PLAN['candidate_pin'], harness.source_fingerprint(CANDIDATE / 'dataframe'))
    if reference:
        check_receipt(receipts['baseline'], reference['measurement_pin'], reference['engine_content_sha256'])
        assert receipts['baseline']['binary_sha256'] == reference['binary_sha256']
        harness.validate_paired_build_receipts(receipts['baseline'], receipts['mojo'])
    print('AUDIT', path.parent.name, len(observed), 'validated workers', flush=True)


def run(label, reference=None):
    output = CANDIDATE / 'build' / label / 'results.json'
    assert not output.exists()
    rows = PLAN['rows'] if reference else [4097]
    args = [PIXI, 'run', '--manifest-path', MANIFEST, '--locked', '-e', 'oracle', 'python3', str(CANDIDATE / 'benchmarks/tools/bench_metal.py'), '--binary', str(CANDIDATE / 'build/bench_metal'), '--rows', ','.join(map(str, rows)), '--workloads', ','.join(PLAN['workloads']), '--variants', ','.join(PLAN['variants']), '--rounds', str(PLAN['rounds'] if reference else 1), '--reps', str(PLAN['reps'] if reference else 1), '--warmups', str(PLAN['warmups'] if reference else 0), '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(output)]
    if reference:
        args += ['--skip-build', '--baseline-binary', str(TREES / reference['remote'] / 'build/bench_metal')]
    print('START', label, flush=True)
    with (ROOT / (label + '.log')).open('x') as log:
        result = subprocess.run(args, cwd=CANDIDATE, stdout=log, stderr=subprocess.STDOUT)
    print('STATUS', label, result.returncode, flush=True)
    assert result.returncode == 0
    audit(output, reference)


def main():
    # Refuse overlap even if the caller has overlooked the correctness process.
    for path in ['/tmp/metal-native-join-paired-oct20/run.pid', '/tmp/metal-group-schedule-gate-oct20-v2.pid']:
        pid = int(Path(path).read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError('Other Mac work still live: ' + str(pid))
    gate_bytes = Path(PLAN['gate_receipt']).read_bytes()
    gate = json.loads(gate_bytes)['gates'][0]
    assert gate['revision'] == PLAN['candidate_engine'] and gate['clean'] is True
    assert gate['build_exit'] == 0 and gate['test_exit'] == 0
    for path, digest in gate['files'].items():
        assert sha(path) == digest
    assert not CANDIDATE.exists()
    subprocess.run(['git', 'clone', '-q', '/tmp/metal-oct20-group-schedule-benchmark-v2.bundle', str(CANDIDATE)], check=True)
    check_tree(CANDIDATE, PLAN['candidate_pin'])
    assert not git(CANDIDATE, 'diff', PLAN['candidate_engine'], 'HEAD', '--', 'dataframe')
    assert git(CANDIDATE, 'rev-parse', 'HEAD:dataframe') == gate['engine_tree']
    harness = parser()
    for ref in PLAN['references']:
        tree = TREES / ref['remote']
        check_tree(tree, ref['measurement_pin'])
        assert sha(tree / 'build/bench_metal') == ref['binary_sha256']
        assert harness.source_fingerprint(tree / 'dataframe') == ref['engine_content_sha256']
        check_receipt(harness.load_build_receipt(tree / 'build/bench_metal'), ref['measurement_pin'], ref['engine_content_sha256'])
    with (ROOT / 'qualification.json').open('x') as output:
        json.dump({'gate_receipt_sha256': hashlib.sha256(gate_bytes).hexdigest(), 'gate': gate}, output, indent=2)
    run('schedule-smoke')
    for ref in PLAN['references']:
        run('schedule-vs-' + ref['label'], ref)


if __name__ == '__main__':
    main()
