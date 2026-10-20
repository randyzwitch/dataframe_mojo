"""Run the reviewed primitive serially; retain build identity and raw outputs."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'bench_wrapped_host_readback.mojo'
SOURCE_SHA = 'ea2c71432fc61f2e6ae6d8773dd89e261b5480963fccba137cc9561618fe9412'
MANIFEST = Path('/Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml')
LOCK_SHA = '098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277'
PIXI = '/Users/randyzwitch/.pixi/bin/pixi'
COMPILER_VERSION = 'Mojo 1.2.0.dev2026100905 (3b71af75)'
BINARY = ROOT / 'probe'
OUTPUTS = ['probe','build.log','samples.log','identity.json','summary.json','status.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def check_idle():
    patterns = [r'\bmojo\s+(?:build|test)\b', r'\bbench_metal\.py\b', r'\brun_pairs\.py\b', r'metal-[^\s]*(?:queue|gate)\.py']
    busy = []
    for line in capture(['ps','-Ao','pid=,command=']).splitlines():
        m = re.match(r'\s*(\d+)\s+(.*)',line)
        if m and int(m[1]) != os.getpid() and any(re.search(p,m[2]) for p in patterns):
            busy.append(line)
    if busy:
        raise RuntimeError('Mac has active benchmark/compiler jobs: '+repr(busy))


def parse_samples(text):
    contexts = [line for line in text.splitlines() if line.startswith('READBACK_CONTEXT ')]
    assert len(contexts) == 1 and re.fullmatch(r'READBACK_CONTEXT api metal device Apple M1 init_ns [0-9]+ repetitions 5 timed_synchronizations separate_source_seed_raw_host_wrapped_transfer_and_wrapped_unmap', contexts[0])
    samples = []
    for line in text.splitlines():
        if not line.startswith('READBACK_SAMPLE '):
            continue
        fields = line.split()[1:]
        assert len(fields) % 2 == 0
        record = dict(zip(fields[::2],fields[1::2]))
        assert len(record)*2 == len(fields), 'duplicate sample field'
        assert record['validated'] == 'True'
        for key in list(record):
            if key not in ['path','sample_kind','validated']:
                record[key] = int(record[key])
        rows, rep, route = record['rows'],record['rep'],record['path']
        assert record['sample_kind'] == ('first' if rep == 0 else 'warm')
        padded = ((rows+4095)//4096)*4096
        assert record['logical_bytes'] == 4*rows
        assert record['transfer_bytes'] == 4*(padded if route=='wrapped_host' else rows)
        assert record['owned_result_bytes'] == 4*rows
        assert record['shared_seed_device_bytes'] == 4*(rows+padded)
        assert record['temporary_host_bytes'] == (4*padded if route=='wrapped_host' else 4*rows if route=='host_buffer' else 0)
        phases = ['target_alloc_ns','wrap_ns','copy_enqueue_ns','transfer_wait_ns','owned_list_copy_ns','release_unmap_wait_ns']
        assert all(record[k] >= 0 for k in phases)
        assert record['total_owned_ready_ns'] > 0
        assert sum(record[k] for k in phases) <= record['total_owned_ready_ns']
        samples.append(record)
    routes = ['raw_list','host_buffer','wrapped_host']
    expected_order = [(rows,routes[(rep+slot)%3],rep) for rows in [4097,1048576] for rep in range(5) for slot in range(3)]
    assert [(r['rows'],r['path'],r['rep']) for r in samples] == expected_order
    summary = []
    for rows in [4097,1048576]:
        for route in routes:
            group = [r for r in samples if r['rows']==rows and r['path']==route]
            warm = group[1:]
            metrics = [k for k in group[0] if k.endswith('_ns')]
            summary.append({'rows':rows,'path':route,'first':group[0], 'warm_sample_count':len(warm), 'warm_ns':{k:{'median':statistics.median(r[k] for r in warm),'min':min(r[k] for r in warm),'max':max(r[k] for r in warm)} for k in metrics}})
    return {'samples':samples,'summary':summary,'scope':'Primitive copy routes including owned-storage construction; not complete-query or device-bandwidth measurements.'}


def main():
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    assert sha(SOURCE) == SOURCE_SHA
    assert sha(MANIFEST.with_name('pixi.lock')) == LOCK_SHA
    assert not any((ROOT/name).exists() for name in OUTPUTS), 'Refusing to overwrite prior probe artifacts'
    check_idle()
    prefix = [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','-e','gpu']
    version = capture(prefix+['mojo','--version'])
    assert version == COMPILER_VERSION, version
    compiler = capture(prefix+['which','mojo'])
    assert sha(compiler) == 'ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b', 'Compiler executable differs from prior qualified Mac runs'
    identity = {'source_sha256':sha(SOURCE),'runner_sha256':sha(Path(__file__)), 'compiler_identity':version,'compiler_path':compiler,'compiler_sha256':sha(compiler),'pixi_lock_sha256':LOCK_SHA,'xcode':capture(['xcodebuild','-version']),'sdk_version':capture(['xcrun','--sdk','macosx','--show-sdk-version']),'os':capture(['sw_vers']),'hardware_model':capture(['sysctl','-n','hw.model']),'memory_bytes':capture(['sysctl','-n','hw.memsize']),'compile_flags':['-O3','-I','.'],'primitive_only':True}
    with (ROOT/'identity.json').open('x') as f:
        json.dump(identity,f,indent=2)
    build_cmd = prefix+['mojo','build','-O3','-I','.',str(SOURCE),'-o',str(BINARY)]
    with (ROOT/'build.log').open('x') as f:
        build = subprocess.run(build_cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    status = {'build_exit':build.returncode,'run_exit':None}
    (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert build.returncode == 0, 'Build failed; artifacts retained'
    assert sha(SOURCE) == SOURCE_SHA
    identity.update(binary_sha256=sha(BINARY),build_log_sha256=sha(ROOT/'build.log'),compile_command=build_cmd)
    (ROOT/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    with (ROOT/'samples.log').open('x') as f:
        result = subprocess.run([str(BINARY)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    status['run_exit'] = result.returncode
    (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert result.returncode == 0, 'Probe failed; artifacts retained'
    data = parse_samples((ROOT/'samples.log').read_text())
    data['samples_sha256'] = sha(ROOT/'samples.log')
    (ROOT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print('PASS 30 validated primitive samples; performance conclusions require review',flush=True)


if __name__ == '__main__':
    main()
