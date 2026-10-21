"""Pinned Float32 comparisons; invoke manually only after correctness queues finish."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
TREES = Path('/Users/randyzwitch/GitHub')
PIXI = '/Users/randyzwitch/.pixi/bin/pixi'
MANIFEST = '/Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml'
GUARDS = (74693, 72762, 72793, 72905, 72928, 73144, 73292, 73991, 74147, 74190)


def check_tree(tree, pin):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip() == pin
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree)
    for file, digest in PLAN['benchmark_file_sha256'].items():
        assert hashlib.sha256((tree / file).read_bytes()).hexdigest() == digest


def audit(output, rows, rounds, reps, runtimes, native):
    data = json.loads(output.read_text())
    expected = set(itertools.product(range(rounds), runtimes, rows, PLAN['workloads'], PLAN['variants']))
    keys = [(r['round'], r['runtime'], r['rows'], r['workload'], r['variant']) for r in data['runs']]
    assert len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate workers'
    failures = [r for r in data['runs'] if r['status'] != 'ok' or not r.get('validated') or len(r.get('samples', [])) != reps]
    print('AUDIT', str(output), len(keys), 'workers;', len(failures), 'failed', flush=True)
    if failures:
        raise RuntimeError('Validation or measurement failure; retain original output')
    if native:
        for run in data['runs']:
            if run['runtime'] == 'mojo':
                assert run.get('profile', {}).get('region_0_algorithm') == 'mojo_gpu_fused_float32_sum', 'Candidate did not use the fused Float32 path'


def run(tree, label, rows, rounds, reps, warmups, native=False, baseline=None):
    output = tree / 'build' / label / 'results.json'
    assert not output.exists(), 'Refusing to overwrite results'
    args = [PIXI, 'run', '--manifest-path', MANIFEST, '--locked', '-e', 'oracle', 'python3', str(tree / 'benchmarks/tools/bench_metal.py'), '--binary', str(tree / 'build/bench_metal'), '--rows', ','.join(map(str, rows)), '--workloads', ','.join(PLAN['workloads']), '--variants', ','.join(PLAN['variants']), '--rounds', str(rounds), '--reps', str(reps), '--warmups', str(warmups), '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(output)]
    if baseline:
        args += ['--skip-build', '--baseline-binary', str(baseline / 'build/bench_metal')]
    print('START', label, flush=True)
    with (ROOT / (label + '.log')).open('x') as log:
        result = subprocess.run(args, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
    print('STATUS', label, result.returncode, flush=True)
    if result.returncode or not output.exists():
        raise RuntimeError('Benchmark failed; retain logs and do not automatically restart')
    audit(output, rows, rounds, reps, ['mojo', 'baseline'] if baseline else ['mojo'], native)


def main():
    for pid in GUARDS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError('Correctness job still live: ' + str(pid))
    gates = Path(PLAN['gate_log']).read_text().splitlines()
    for status in ('BUILD_STATUS', 'TEST_STATUS'):
        assert f"{status} {PLAN['gate_label']} 0" in gates, 'Final correctness gate missing or failed'
    gate = TREES / PLAN['gate_remote']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=gate, text=True).strip() == PLAN['candidate_engine']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=gate)
    base, candidate = TREES / PLAN['baseline_remote'], TREES / PLAN['candidate_remote']
    for tree, pin, engine in [(base, PLAN['baseline_pin'], PLAN['baseline_engine']), (candidate, PLAN['candidate_pin'], PLAN['candidate_engine'])]:
        check_tree(tree, pin)
        assert not subprocess.check_output(['git', 'diff', engine, 'HEAD', '--', 'dataframe'], cwd=tree), 'Engine identity changed'
    run(base, 'float32-baseline-smoke', [4097], 1, 1, 0)
    run(candidate, 'float32-candidate-smoke', [4097], 1, 1, 0, native=True)
    run(candidate, 'float32-paired', PLAN['rows'], PLAN['rounds'], PLAN['reps'], PLAN['warmups'], native=True, baseline=base)


if __name__ == '__main__':
    main()
