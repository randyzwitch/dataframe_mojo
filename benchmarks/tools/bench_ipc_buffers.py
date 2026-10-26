"""Run identical IPC programs against the pre-change and candidate libraries.

Timing/RSS runs have no allocator interposer. Separate instrumented runs count
actual Mojo aligned-allocation requests. Every run
checks an independently generated checksum over all non-null payload bytes.
"""
import argparse
import contextlib
import fcntl
import json
import hashlib
import os
import platform
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def compiling():
    output = subprocess.check_output(['ps', '-eo', 'comm='], text=True)
    names = {Path(line.strip()).name for line in output.splitlines()}
    return bool(names & {'mojo', 'mojo-compiler', 'clang', 'clang++', 'cc1', 'cc1plus', 'ld.lld'})


@contextlib.contextmanager
def quiet():
    root = Path.home() / '.dataframe-agents'
    lock = root / 'lock'
    if not lock.exists():
        while compiling():
            print('Waiting for existing host compilers', flush=True)
            time.sleep(10)
        yield
        return
    flags = Path.home() / '.dataframe-timing-quiet.d'
    flags.mkdir(exist_ok=True)
    flag = flags / str(os.getpid())
    flag.write_text('issue795 IPC measurements\n')
    try:
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            while compiling():
                print('Waiting for existing host compilers', flush=True)
                time.sleep(10)
            yield
    finally:
        flag.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-revision', required=True)
    parser.add_argument('--candidate-revision', required=True)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline-library', required=True, type=Path)
    parser.add_argument('--candidate-library', required=True, type=Path)
    parser.add_argument('--observer', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for name, item in manifest.items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != item['sha256']:
            raise AssertionError('Benchmark input digest differs: ' + name)
    provenance = dict(baseline_revision=args.baseline_revision, candidate_revision=args.candidate_revision, architecture=platform.machine(), cpu_count=os.cpu_count(), dataframe_threads=4, flags='-O3 -I providers/cpu -I .', timing='Fresh process; warmed input files; total_ns includes full checksum traversal, before destruction; no interposer in timing/RSS runs', hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.baseline, args.candidate, args.baseline_library, args.candidate_library, args.observer)})
    records = []
    variants = [('baseline', args.baseline, args.baseline_library), ('candidate', args.candidate, args.candidate_library)]
    cases = [(name, mode, item) for name, item in manifest.items() for mode in ('read', 'scan')]
    cases.insert(0, ('1000000', 'owned', {'rows': 1000000, 'checksum': 8 * (1000000 * 999999 // 2) + 28 * 1000000 + 8 * 1000000}))

    def run(variant, binary, library, name, mode, expected, observed):
        env = dict(os.environ, DATAFRAME_PARQUET_LIBRARY=str(library.resolve()), DATAFRAME_THREADS='4')
        env.pop('LD_PRELOAD', None)
        env.pop('DYLD_INSERT_LIBRARIES', None)
        if observed:
            env['DYLD_INSERT_LIBRARIES' if sys.platform == 'darwin' else 'LD_PRELOAD'] = str(args.observer.resolve())
        filename = name if mode == 'owned' else str((args.data / name).resolve())
        if observed:
            command = [str(binary.resolve()), filename, mode]
        elif sys.platform == 'darwin':
            command = ['/usr/bin/time', '-l', str(binary.resolve()), filename, mode]
        else:
            command = ['/usr/bin/time', '-f', 'maximum_rss_kib=%M', str(binary.resolve()), filename, mode]
        for attempt in range(3):
            while compiling():
                time.sleep(10)
            process = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            if not compiling():
                break
            print('Discarding sample overlapping a host compiler', flush=True)
        else:
            raise RuntimeError('Unable to obtain a compiler-free timing sample')
        result = json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{')))
        if result['checksum'] != expected['checksum'] or result['rows'] != expected['rows']:
            raise AssertionError((variant, name, mode, result, expected))
        if observed and (result['allocated_bytes'] == 0 or result['allocation_calls'] == 0):
            raise AssertionError('Allocation interposer did not observe the read')
        pattern = r'(\d+)\s+maximum resident set size' if sys.platform == 'darwin' else r'maximum_rss_kib=(\d+)'
        rss = int(re.search(pattern, process.stderr).group(1)) if not observed else None
        result.update(case=name, mode=mode, variant=variant, instrumented=observed, rss_kib=rss / 1024 if sys.platform == 'darwin' and rss is not None else rss)
        if observed and mode == 'owned' and result['allocated_bytes'] < expected['rows'] * 8 * 8:
            raise AssertionError('Allocation observer does not cover Mojo numeric payload allocations')
        records.append(result)
        args.output.write_text(json.dumps({'platform': sys.platform, 'allocation_scope': 'Mojo aligned-allocation requests at KGEN_CompilerRT_AlignedAlloc; excludes native allocators; RSS includes the complete process', 'provenance': provenance, 'runs': records}, indent=2) + '\n')

    with quiet():
        for name, mode, expected in cases:
            for repetition in range(args.repetitions):
                order = variants if repetition % 2 == 0 else list(reversed(variants))
                for variant, binary, library in order:
                    run(variant, binary, library, name, mode, expected, False)
            for variant, binary, library in variants:
                run(variant, binary, library, name, mode, expected, True)
            medians = {}
            for variant, _, _ in variants:
                rows = [row for row in records if row['case'] == name and row['mode'] == mode and row['variant'] == variant and not row['instrumented']]
                medians[variant] = {key: statistics.median(row[key] for row in rows) for key in ('read_ns', 'total_ns', 'rss_kib')}
            print(json.dumps(dict(case=name, mode=mode, medians=medians)), flush=True)


if __name__ == '__main__':
    main()
