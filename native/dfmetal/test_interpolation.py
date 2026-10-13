"""Independent partition/gap references for native linear and nearest windows."""
from pathlib import Path
import ctypes as C
import math
import random
import struct
import sys

# Reuse only the ABI declarations and metadata-only preflights, never GPU tests.
source = Path(__file__).with_name("test_native.py").read_text()
exec(compile(source.split("probe_ctx =", 1)[0], "test_native.py", "exec"))
err = P()
ctx = lib.dfm_context_create(0, 0, C.byref(err))
if not ctx:
    check(1, err)
libm = C.CDLL(None)
libm.fma.argtypes = [C.c_double] * 3
libm.fma.restype = C.c_double
rng = random.Random(0x1A7E9)
widths = {1: 4, 2: 4, 3: 8, 5: 1, 6: 2, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8, 12: 16}
fields = 0


def bits(value):
    return struct.pack("=d", value)


def run(dtype, method, n, pattern):
    global fields
    valid = [rng.randrange(5) > 1 for _ in range(n)]
    if pattern == 1:
        valid = [False] * n
    elif pattern == 2:
        valid = [True] * n
    elif pattern == 3:
        valid = [i in (1, n - 2) for i in range(n)]
    keys = [rng.randrange(5) for _ in range(n)]
    kv = [i % 13 != 7 for i in range(n)]
    offset = 5
    validity, key_valid = packed(valid, offset), packed(kv, offset)
    if method == 13:
        values = [rng.uniform(-1e12, 1e12) for _ in range(n)]
        edges = [-0., 0., math.inf, -math.inf, math.nan, 5e-324, 1e308, -1e308]
        for i in range(0, n, 11):
            values[i] = edges[(i // 11) % len(edges)]
        payloads = [bits(value) for value in values]
    elif dtype == 4:
        payloads = [bytes([rng.randrange(2)]) for _ in range(n)]
    else:
        payloads = [rng.getrandbits(widths[dtype] * 8).to_bytes(widths[dtype], "little") for _ in range(n)]
    if dtype == 4:
        data = packed([bool(value[0]) for value in payloads], offset)
        result = (C.c_uint8 * max(1, (n + 7) // 8))()
    else:
        data = C.create_string_buffer(b"".join(payloads) or b"\0")
        result = (C.c_uint8 * max(1, n * widths[dtype]))()
    key_data = (C.c_int32 * max(1, n))(*keys)
    output_valid = (C.c_uint8 * max(1, (n + 7) // 8))()
    inputs = (Input * 2)(Input(ptr(data), ptr(validity), dtype, offset, 1),
                          Input(ptr(key_data), ptr(key_valid), 2, offset, 1))
    code = (I * 8)(214, 0, 0, method, 200, 1, 0, 1)
    literals = (I * 2)(0, 0)
    nodes, slots = (I * 2)(dtype, 2), (I * 3)(dtype, 2, dtype)
    steps = (Step * 1)(Step(0, 2, 2, 5, 0, 0))
    outputs = (Output * 1)(Output(ptr(result), ptr(output_valid), dtype, 2, -1, 0))
    request = Request(2, n, 0, 3, 2, 8, 2, 1, 0, 1, -1, 0, 1, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    request.budget = memory.peak
    stats = Stats()
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
    assert stats.memory.peak == memory.peak and stats.rows == n
    groups = {}
    for row in range(n):
        groups.setdefault(keys[row] if kv[row] else None, []).append(row)
    expected = [payloads[i] if valid[i] else None for i in range(n)]
    for rows in groups.values():
        anchors = [j for j, row in enumerate(rows) if valid[row]]
        for left, right in zip(anchors, anchors[1:]):
            for j in range(left + 1, right):
                row = rows[j]
                if method == 14:
                    anchor = left if j - left < right - j else right
                    expected[row] = payloads[rows[anchor]]
                else:
                    before, after = values[rows[left]], values[rows[right]]
                    expected[row] = bits(libm.fma(after - before, (j - left) / (right - left), before))
    for row in range(n):
        assert bool(output_valid[row // 8] & (1 << (row % 8))) == (expected[row] is not None)
        if expected[row] is None:
            continue
        if dtype == 4:
            actual = bytes([int(bool(result[row // 8] & (1 << (row % 8))))])
        else:
            actual = bytes(result[row * widths[dtype]:(row + 1) * widths[dtype]])
        if method == 13 and math.isnan(struct.unpack("=d", expected[row])[0]):
            assert math.isnan(struct.unpack("=d", actual)[0])
        else:
            assert actual == expected[row], (dtype, method, n, pattern, row, actual.hex(), expected[row].hex())
        fields += 1
    if n == 513 and pattern == 0:
        request.budget = memory.peak - 1
        assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
        assert b"budget" in C.string_at(err)
        lib.dfm_free(err)


try:
    for n in (0, 1, 2, 31, 257, 513, 2049):
        for pattern in range(4):
            for dtype in range(1, 13):
                run(dtype, 14, n, pattern)
            run(11, 13, n, pattern)
finally:
    lib.dfm_context_release(ctx)
print(f"PASS {fields} native interpolation fields: all physical types, null partitions, NaN/zero payloads, gap edges, exact budgets")
