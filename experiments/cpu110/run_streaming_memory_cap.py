"""Run the streaming prototype with an address-space cap below input size.

This verifies bounded execution under a per-process allocation limit; it does
not disable the operating system page cache or constrain total host memory.
Usage: python3 run_streaming_memory_cap.py BINARY CSV EXPECTED_SUM
"""
import os
import resource
import sys

binary, csv, expected = sys.argv[1:]
cap = int(os.environ.get("STREAMING_CAP_MIB", "8192")) * 1024 * 1024
size = os.stat(csv).st_size
if size <= cap:
    raise ValueError('input must exceed the configured address-space cap')
print(f'input_bytes={size},address_space_cap_bytes={cap},threads=1', flush=True)
resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
os.environ['DATAFRAME_THREADS'] = '1'
os.execv(binary, [binary, csv, 'parallel', '4194304', '--rss', expected])
