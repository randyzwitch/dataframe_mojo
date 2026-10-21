"""Validate the archived primitive without invoking Metal or rebuilding."""
from pathlib import Path
import hashlib,importlib.util,json,re
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((root/'identity.json').read_text())
for key,name in [('source_sha256','bench_wrapped_host_readback.mojo'),('runner_sha256','run_probe.py'),('build_log_sha256','build.log')]:
 assert sha(root/name)==identity[key]
assert re.fullmatch('[0-9a-f]{64}',identity['binary_sha256'])
assert identity['compiler_identity']=='Mojo 1.2.0.dev2026100905 (3b71af75)'
assert identity['compiler_sha256']=='ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b'
assert identity['pixi_lock_sha256']=='098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277'
assert identity['compile_flags']==['-O3','-I','.']
assert json.loads((root/'status.json').read_text())=={'build_exit':0,'run_exit':0}
spec=importlib.util.spec_from_file_location('probe',root/'run_probe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=module.parse_samples((root/'samples.log').read_text())
data['samples_sha256']=sha(root/'samples.log')
assert data==json.loads((root/'summary.json').read_text())
print('PASS 30 primitive samples; raw replay, byte/phase arithmetic and source/build receipts')
