"""Retain a predeclared longer check of the small sorted-skew regression."""
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path('/tmp/metal-group-schedule-paired-oct20-v2')
try:
    os.kill(77706,0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError('Main comparison still live')
assert 'AUDIT schedule-vs-previous 108 validated workers' in (ROOT/'run.log').read_text()
spec=importlib.util.spec_from_file_location('paired', ROOT/'run_pairs.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ref=next(r for r in m.PLAN['references'] if r['label']=='previous')
m.check_tree(m.CANDIDATE,m.PLAN['candidate_pin'])
m.check_tree(m.TREES/ref['remote'],ref['measurement_pin'])
assert m.sha(m.TREES/ref['remote']/'build/bench_metal')==ref['binary_sha256']
h=m.parser()
original=json.loads((m.CANDIDATE/'build/schedule-smoke/results.json').read_text())['metadata']['binary_receipts']['mojo']
assert h.load_build_receipt(m.CANDIDATE/'build/bench_metal')==original
m.PLAN.update(rows=[4097],workloads=['group_skew'],variants=['sorted'],rounds=10,reps=5,warmups=1)
with (ROOT/'skew-followup-plan.json').open('x') as f:
    json.dump({'reason':'Original 3-round previous-candidate comparison showed a 1.7% warm slowdown for small sorted skew; retain original and followup separately.','plan':m.PLAN,'reference':ref},f,indent=2)
m.run('schedule-skew-followup',ref)
