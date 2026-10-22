"""Run only after the serialized correctness queue has completed."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
HOME_TREES = Path('/Users/randyzwitch/GitHub')
BASE = HOME_TREES / 'dataframe_mojo-metal-group-benchmark-v2'
PIXI = '/Users/randyzwitch/.pixi/bin/pixi'
MANIFEST = '/Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml'


def check_tree(tree, pin):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
    if actual != pin or subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree):
        raise RuntimeError('Source pin or cleanliness changed: ' + str(tree))
    for name, digest in PLAN['benchmark_file_sha256'].items():
        if hashlib.sha256((tree / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Benchmark source mismatch: ' + str(tree / name))


def audit_results(path, workloads, rows, variants, rounds, reps, builds):
    data = json.loads(path.read_text())
    expected = set(itertools.product(range(rounds), builds, rows, workloads, variants))
    observed = [(r['round'], r['runtime'], r['rows'], r['workload'], r['variant']) for r in data['runs']]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError('Missing or duplicate workers in ' + str(path))
    failures = [r for r in data['runs'] if r['status'] != 'ok' or not r.get('validated') or len(r.get('samples', [])) != reps]
    print('AUDIT', str(path), len(observed), 'workers;', len(failures), 'failed', flush=True)
    routes = {}
    for run in data['runs']:
        route = run.get('profile', {}).get('region_0_grouped_int_algorithm', 'missing')
        key = (run['runtime'], run['rows'], run['workload'], run['variant'], route)
        routes[key] = routes.get(key, 0) + 1
    print('GROUP_ROUTES', json.dumps([dict(runtime=k[0], rows=k[1], workload=k[2], variant=k[3], route=k[4], workers=v) for k, v in sorted(routes.items())]), flush=True)
    if path.parent.name in ('operator-paired', 'group-vs-intermediate'):
        candidate = [r for r in data['runs'] if r['runtime'] == 'mojo' and r['status'] == 'ok']
        if any(r.get('profile', {}).get('region_0_grouped_int_algorithm') not in ('legacy', 'mojo_gpu_grouped_int_partials') for r in candidate):
            raise RuntimeError('Candidate grouped route diagnostics missing')
    return not failures


def run_group_intermediate():
    intermediate = json.loads((ROOT / 'group-intermediate.json').read_text())
    tree = HOME_TREES / intermediate['remote']
    check_tree(tree, intermediate['measurement_pin'])
    if subprocess.check_output(['git', 'diff', intermediate['engine_pin'], 'HEAD', '--', 'dataframe'], cwd=tree):
        raise RuntimeError('Intermediate engine changed')
    case = next(c for c in PLAN['candidates'] if c['label'] == 'group_integer')
    candidate = HOME_TREES / case['remote']
    check_tree(candidate, case['measurement_pin'])
    prior = candidate / 'build/operator-paired/results.json'
    if not prior.exists() or not audit_results(prior, case['workloads'], PLAN['rows'], PLAN['variants'], PLAN['rounds'], PLAN['reps'], ['mojo', 'baseline']):
        print('SKIP intermediate comparison: candidate baseline pair incomplete or failed', flush=True)
        return False
    output = tree / 'build/intermediate-smoke/results.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite intermediate smoke results')
    common = [PIXI, 'run', '--manifest-path', MANIFEST, '--locked', '-e', 'oracle', 'python3']
    smoke = common + [str(tree / 'benchmarks/tools/bench_metal.py'), '--binary', str(tree / 'build/bench_metal'), '--rows', '4097', '--workloads', ','.join(intermediate['workloads']), '--variants', ','.join(PLAN['variants']), '--rounds', '1', '--reps', '1', '--warmups', '0', '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(output)]
    print('START intermediate smoke', intermediate['measurement_pin'], flush=True)
    with (ROOT / 'group-intermediate-smoke.log').open('x') as log:
        status = subprocess.call(smoke, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
    if status or not output.exists() or not audit_results(output, intermediate['workloads'], [4097], PLAN['variants'], 1, 1, ['mojo']):
        print('FAIL intermediate smoke', status, flush=True)
        return False
    output = candidate / 'build/group-vs-intermediate/results.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite intermediate paired results')
    args = common + [str(candidate / 'benchmarks/tools/bench_metal.py'), '--skip-build', '--binary', str(candidate / 'build/bench_metal'), '--baseline-binary', str(tree / 'build/bench_metal'), '--rows', ','.join(map(str, PLAN['rows'])), '--workloads', ','.join(case['workloads']), '--variants', ','.join(PLAN['variants']), '--rounds', str(PLAN['rounds']), '--reps', str(PLAN['reps']), '--warmups', str(PLAN['warmups']), '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(output)]
    print('START candidate versus #865 intermediate', flush=True)
    with (ROOT / 'group-vs-intermediate-run.log').open('x') as log:
        status = subprocess.call(args, cwd=candidate, stdout=log, stderr=subprocess.STDOUT)
    if status or not output.exists():
        print('FAIL intermediate pair', status, flush=True)
        return False
    return audit_results(output, case['workloads'], PLAN['rows'], PLAN['variants'], PLAN['rounds'], PLAN['reps'], ['mojo', 'baseline'])


def main():
    # Do not overlap the current baseline or any serialized validation job.
    for pid in (70730, 70777, 70866, 71113, 71854, 72145, 72578, 72762, 72793, 72905, 72928, 73144, 73292, 73991, 74147, 74190):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError('Validation job still live: ' + str(pid))
    check_tree(BASE, PLAN['benchmark_pin'])
    smoke_cases = ['group_hash', 'group_high', 'group_skew']
    smoke_output = BASE / 'build/group-smoke/results.json'
    if smoke_output.exists():
        raise RuntimeError('Refusing to overwrite baseline smoke evidence')
    args = [PIXI, 'run', '--manifest-path', MANIFEST, '--locked', '-e', 'oracle', 'python3', str(BASE / 'benchmarks/tools/bench_metal.py'), '--binary', str(BASE / 'build/bench_metal'), '--rows', '4097', '--workloads', ','.join(smoke_cases), '--variants', ','.join(PLAN['variants']), '--rounds', '1', '--reps', '1', '--warmups', '0', '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(smoke_output)]
    with (ROOT / 'baseline-smoke.log').open('x') as log:
        status = subprocess.call(args, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
    if status or not smoke_output.exists() or not audit_results(smoke_output, smoke_cases, [4097], PLAN['variants'], 1, 1, ['mojo']):
        raise RuntimeError('Baseline benchmark smoke has failures; retain and resolve before pairs')
    gates = Path('/tmp/metal-group-restack-oct20-queue.log').read_text().splitlines()
    all_ok = True
    for case in PLAN['candidates']:
        label = case['label']
        gate_label = case['gate_label']
        if f'BUILD_STATUS {gate_label} 0' not in gates or f'TEST_STATUS {gate_label} 0' not in gates:
            print('SKIP failed or missing correctness gate:', label, flush=True)
            all_ok = False
            continue
        gate_tree = HOME_TREES / case['gate_remote']
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=gate_tree, text=True).strip() != case['engine_pin'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=gate_tree):
            raise RuntimeError('Correctness gate source changed: ' + label)
        tree = HOME_TREES / case['remote']
        check_tree(tree, case['measurement_pin'])
        if subprocess.check_output(['git', 'diff', case['engine_pin'], 'HEAD', '--', 'dataframe'], cwd=tree):
            raise RuntimeError('Measurement engine changed: ' + label)
        output = tree / 'build/operator-paired/results.json'
        if output.exists():
            raise RuntimeError('Refusing to overwrite results: ' + str(output))
        args = [PIXI, 'run', '--manifest-path', MANIFEST, '--locked', '-e', 'oracle', 'python3', str(tree / 'benchmarks/tools/bench_metal.py'), '--binary', str(tree / 'build/bench_metal'), '--baseline-binary', str(BASE / 'build/bench_metal'), '--rows', ','.join(map(str, PLAN['rows'])), '--workloads', ','.join(case['workloads']), '--variants', ','.join(PLAN['variants']), '--rounds', str(PLAN['rounds']), '--reps', str(PLAN['reps']), '--warmups', str(PLAN['warmups']), '--threads', '8', '--timeout-seconds', str(PLAN['timeout_seconds']), '--output', str(output)]
        print('START', label, case['measurement_pin'], flush=True)
        with (ROOT / (label + '-run.log')).open('x') as log:
            status = subprocess.call(args, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
        print('RUN_STATUS', label, status, flush=True)
        if status or not output.exists():
            all_ok = False
            continue
        all_ok = audit_results(output, case['workloads'], PLAN['rows'], PLAN['variants'], PLAN['rounds'], PLAN['reps'], ['mojo', 'baseline']) and all_ok
    all_ok = run_group_intermediate() and all_ok
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
