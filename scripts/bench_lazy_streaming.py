#!/usr/bin/env python3
"""Fresh-process time/RSS comparison for a bounded CSV pipeline."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/lazy-streaming'


def compiling():
    return os.environ.get('BENCH_WAIT_FOR_COMPILERS') == '1' and subprocess.run(['pgrep', '-x', 'mojo'], stdout=subprocess.DEVNULL).returncode == 0


def generate(rows):
    path = OUT / f'{rows}.csv'
    if path.exists():
        return path
    with path.open('w') as file:
        file.write('k,v,padding\n')
        for start in range(0, rows, 10000):
            file.writelines(f'{i % 100},{i},' + 'x' * 96 + '\n' for i in range(start, min(rows, start + 10000)))
    return path


def run():
    with (OUT / 'results.csv').open('w') as report:
        writer = csv.writer(report)
        writer.writerow(['rows', 'bytes', 'threads', 'batch_size', 'round', 'streaming', 'ns', 'peak_rss_bytes'])
        for rows in (1000000, 10000000):
            path = generate(rows)
            for threads in (4, 8, 16):
                env = dict(os.environ, DATAFRAME_THREADS=str(threads))
                for round_ in range(3):
                    for streaming in ((0, 1) if round_ % 2 == 0 else (1, 0)):
                        while True:
                            while compiling():
                                time.sleep(5)
                            with (OUT / 'process.log').open('w+') as log:
                                child = subprocess.Popen([str(OUT / 'bench'), str(path), str(streaming), '65536', str(rows)], stdout=log, stderr=subprocess.STDOUT, env=env)
                                _, status, usage = os.wait4(child.pid, 0)
                                child.returncode = os.waitstatus_to_exitcode(status)
                                log.seek(0)
                                output = log.read()
                            if child.returncode:
                                raise RuntimeError(output)
                            if compiling():
                                continue
                            rss = usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
                            writer.writerow([rows, path.stat().st_size, threads, 65536, round_, streaming, int(output.split()[0]), rss])
                            report.flush()
                            print(rows, threads, round_, streaming, output.strip(), rss, flush=True)
                            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.build:
        subprocess.run(['mojo', 'build', '-I', '.', 'benchmarks/bench_lazy_streaming.mojo', '-o', str(OUT / 'bench')], check=True)
    if args.run:
        run()
