"""Repeat the established CSV body ten times for the capped streaming test."""
from pathlib import Path
import shutil

source = Path('build/cpu110_streaming_1gb.csv')
target = Path('build/cpu110_streaming_11gb.csv')
with source.open('rb') as inp, target.open('wb') as out:
    header = inp.readline()
    offset = inp.tell()
    out.write(header)
    for _ in range(10):
        inp.seek(offset)
        shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
print(f'path={target},bytes={target.stat().st_size},repetitions=10')
