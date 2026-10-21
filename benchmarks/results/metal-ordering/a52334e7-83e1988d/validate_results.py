"""Replay ordering logs; verify coverage, source receipts and inferred routes."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    auditor = load('ordering_auditor', ROOT / 'run_pairs.py')
    harness = load('ordering_harness', ROOT / 'harness/bench_metal.py')
    for name, digest in plan['harness_file_sha256'].items():
        assert hashlib.sha256((ROOT / 'harness' / Path(name).name).read_bytes()).hexdigest() == digest
    for role, filename, label in [('candidate', 'topk-restack-receipt.json', 'topk-restack'), ('intermediate', 'radix-restack-receipt.json', 'radix-restack')]:
        path = ROOT / 'validation' / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['gate_receipts'][role]['sha256']
        gates = [g for g in json.loads(path.read_text())['gates'] if g['label'] == label]
        assert len(gates) == 1
        gate = gates[0]
        assert gate['revision'] == plan['source_roles'][role]['engine']
        assert gate['engine_tree'] == plan['source_roles'][role]['tree']
        assert gate['clean'] is True and gate['build_exit'] == gate['test_exit'] == 0
        assert len(gate['files']) == 3
        for file, digest in gate['files'].items():
            if file.endswith('.log'):
                assert hashlib.sha256((ROOT / 'validation' / Path(file).name).read_bytes()).hexdigest() == digest
    smoke_receipts = {}
    paired_receipts = []
    total = 0
    for label, role, baseline_role in [('smoke-frozen','frozen',None), ('smoke-intermediate','intermediate',None), ('smoke-candidate','candidate',None), ('frozen-pair','candidate','frozen'), ('radix-pair','candidate','intermediate')]:
        path = ROOT / label / 'results.json'
        data = json.loads(path.read_text())
        config = plan if baseline_role else plan['smoke']
        reps = config['reps']
        runtimes = ['mojo', 'baseline'] if baseline_role else ['mojo']
        count = auditor.audit(data, role, auditor.workload_set(label), config['variants'], config['rows'], config['rounds'], reps, runtimes, baseline_role)
        receipts = data['metadata']['binary_receipts']
        if baseline_role:
            harness.validate_paired_build_receipts(receipts['baseline'], receipts['mojo'])
            paired_receipts.append((baseline_role, receipts))
        else:
            smoke_receipts[role] = receipts['mojo']
        for run in data['runs']:
            parsed = harness.parse_run((path.parent / 'results' / run['log']).read_text(), reps)
            for field, value in parsed.items():
                assert run[field] == value, (label, run['log'], field)
            assert all(s['metal_ns'] > 0 and s['cpu_ns'] > 0 for s in run['samples'])
        total += count
        print('PASS', label, count, 'workers; raw logs, coverage and pinned receipts')
    for baseline_role, receipts in paired_receipts:
        assert smoke_receipts[baseline_role] == receipts['baseline']
        assert smoke_receipts['candidate'] == receipts['mojo']
    assert total == plan['workers']['total'] == 204
    print('PASS 204 workers; route attribution remains inference from plans and launch counts')


if __name__ == '__main__':
    main()
