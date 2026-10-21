"""Replay archived join outputs and verify qualification, coverage and attribution."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for name, digest in PLAN['harness_file_sha256'].items():
        assert hashlib.sha256((ROOT / 'harness' / Path(name).name).read_bytes()).hexdigest() == digest
    receipt_path = ROOT / 'validation/hash-receipt.json'
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == PLAN['gate_receipt_sha256']
    gates = json.loads(receipt_path.read_text())['gates']
    for required in PLAN['required_gates']:
        gate = next(g for g in gates if g['label'] == required['label'])
        for field in ['revision','engine_tree','build_exit','test_exit','clean']:
            assert gate[field] == required[field]
        for name, digest in required['artifacts'].items():
            assert gate['files'][name] == digest
            if name.endswith('.log'):
                assert hashlib.sha256((ROOT / 'validation' / Path(name).name).read_bytes()).hexdigest() == digest
    auditor = load('auditor', ROOT / 'run_pairs.py')
    harness = load('harness', ROOT / 'harness/bench_metal.py')
    paired_receipts = None
    smoke_receipts = {}
    for folder, role, paired in [('baseline-smoke','baseline',False),('candidate-smoke','candidate',False),('paired','candidate',True)]:
        path = ROOT / folder / 'results.json'
        data = json.loads(path.read_text())
        if paired:
            count = auditor.audit_payload(data, PLAN['rows'], PLAN['rounds'], PLAN['reps'], ['mojo','baseline'], True)
            paired_receipts = data['metadata']['binary_receipts']
            harness.validate_paired_build_receipts(paired_receipts['baseline'], paired_receipts['mojo'])
        else:
            count = auditor.audit_payload(data,[4097],1,1,['mojo'],role == 'candidate',workloads=['join_hash'],variants=['base'],mojo_role=role)
            smoke_receipts[role] = data['metadata']['binary_receipts']['mojo']
        for run in data['runs']:
            parsed = harness.parse_run((path.parent / 'results' / run['log']).read_text(), PLAN['reps'] if paired else 1)
            for field, value in parsed.items():
                assert run[field] == value, (folder,run['log'],field)
            assert all(s['metal_ns'] > 0 and s['cpu_ns'] > 0 for s in run['samples'])
        print('PASS',folder,count,'validated workers; raw logs, coverage, receipts and routes')
    for role, runtime in [('baseline','baseline'),('candidate','mojo')]:
        assert smoke_receipts[role] == paired_receipts[runtime]
    print('PASS qualification hashes, equal paired build identity, smoke-to-pair binary receipts')


if __name__ == '__main__':
    main()
