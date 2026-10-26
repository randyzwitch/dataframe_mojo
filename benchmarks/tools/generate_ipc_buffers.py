"""Independent IPC inputs and checksums for every non-null payload byte."""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'build/ipc-buffers/data')
OUT = parser.parse_args().output
OUT.mkdir(exist_ok=True, parents=True)
MASK = (1 << 64) - 1

def checksum(array):
    if isinstance(array, pa.ChunkedArray):
        return sum(checksum(chunk) for chunk in array.chunks) & MASK
    t = array.type
    valid = len(array) - array.null_count
    if pa.types.is_dictionary(t):
        return checksum(array.dictionary_decode())
    if pa.types.is_integer(t) or pa.types.is_boolean(t):
        return (int(pc.sum(array).as_py() or 0) + valid) & MASK
    if pa.types.is_large_list(t) or pa.types.is_list(t):
        return (valid + checksum(array.values)) & MASK
    if pa.types.is_struct(t):
        return (valid + sum(checksum(array.field(i)) for i in range(t.num_fields))) & MASK
    # Only a few distinct text/binary values; compute exact byte sums once.
    counts = pc.value_counts(array.cast(pa.large_binary()))
    result = valid
    for row in counts.to_pylist():
        if row['values'] is not None:
            result += sum(row['values']) * row['counts']
    return result & MASK

def write(name, table):
    expected = sum(checksum(column) for column in table.columns) & MASK
    for codec in ('uncompressed', 'lz4', 'zstd'):
        path = OUT / f'{name}-{codec}.arrow'
        options = ipc.IpcWriteOptions(compression=None if codec == 'uncompressed' else codec)
        with ipc.new_file(path, table.schema, options=options) as writer:
            for batch in table.to_batches(max_chunksize=65536):
                writer.write_batch(batch)
        manifest[path.name] = dict(rows=table.num_rows, columns=table.num_columns, checksum=expected, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

manifest = {}
n = 1000000
ids = np.arange(n, dtype=np.int64)
write('numeric', pa.table({f'c{k}': pa.array(ids+k) for k in range(8)}))
n = 500000
ids = np.arange(n, dtype=np.int64)
texts = np.take(np.array(['', 'é', 'long external string payload 0123456789', 'a\0b'], dtype=object), ids % 4)
binaries = np.take(np.array([b'\xff\0', b'0123456789abcdef0123456789', b''], dtype=object), ids % 3)
text = pa.array(texts, mask=ids % 13 == 0, type=pa.large_string())
category = pa.DictionaryArray.from_arrays(pa.array((ids % 5).astype(np.uint32), mask=ids % 19 == 0), pa.array(['red', 'blue', 'green', 'é', ''], type=pa.large_string()))
child = np.arange(2*n, dtype=np.int64)
lists = pa.LargeListArray.from_arrays(pa.array(2*np.arange(n+1,dtype=np.int64)), pa.array(child, mask=child % 7 == 0), mask=pa.array(ids % 23 == 0))
record = pa.StructArray.from_arrays([pa.array(7*ids), text], names=['number', 'text'], mask=pa.array(ids % 29 == 0))
write('mixed', pa.table({'x': pa.array(ids, mask=ids % 11 == 0), 'text': text, 'binary': pa.array(binaries, mask=ids % 31 == 0, type=pa.large_binary()), 'flag': pa.array(ids % 3 == 0, mask=ids % 17 == 0), 'category': category, 'items': lists, 'record': record}))
write('views', pa.table({'x': pa.array(ids), 'text': text.cast(pa.string_view()), 'binary': pa.array(binaries, mask=ids % 31 == 0, type=pa.binary_view())}))
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
