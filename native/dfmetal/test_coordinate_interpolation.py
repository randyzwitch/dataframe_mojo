"""Independent numeric-coordinate, null-anchor, and partition interpolation references."""
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
fields = 0


def bits(value):
    return struct.pack("=d", value)


def run(n, pattern):
    global fields
    dtype, method = 11, 15
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
    values = [rng.uniform(-1e12, 1e12) for _ in range(n)]
    edges = [-0., 0., math.inf, -math.inf, math.nan, 5e-324, 1e308, -1e308]
    for i in range(0, n, 11):
        values[i] = edges[(i // 11) % len(edges)]
    payloads = [bits(value) for value in values]
    data = C.create_string_buffer(b"".join(payloads) or b"\0")
    result = (C.c_uint8 * max(1, n * 8))()
    axis = [rng.uniform(-100, 100) for _ in range(n)]
    av = [i % 7 != 3 for i in range(n)]
    if pattern == 4:
        axis = [1.] * n
    if pattern == 5:
        for i in range(0, n, 3):
            axis[i] = (math.inf, -math.inf, math.nan)[(i // 3) % 3]
    axis_data = (C.c_double * max(1, n))(*axis)
    axis_valid = packed(av, offset)
    key_data = (C.c_int32 * max(1, n))(*keys)
    output_valid = (C.c_uint8 * max(1, (n + 7) // 8))()
    inputs = (Input * 3)(Input(ptr(data), ptr(validity), dtype, offset, 1),
                          Input(ptr(axis_data), ptr(axis_valid), 11, offset, 1),
                          Input(ptr(key_data), ptr(key_valid), 2, offset, 1))
    code = (I * 8)(214, 0, 1, method, 200, 2, 0, 1)
    literals = (I * 2)(0, 0)
    nodes, slots = (I * 2)(dtype, 2), (I * 4)(dtype, 11, 2, dtype)
    steps = (Step * 1)(Step(0, 2, 3, 5, 0, 0))
    outputs = (Output * 1)(Output(ptr(result), ptr(output_valid), dtype, 3, -1, 0))
    request = Request(2, n, 0, 4, 3, 8, 2, 1, 0, 1, -1, 0, 1, -1,
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
        anchors = [j for j, row in enumerate(rows) if valid[row] and av[row]]
        for left, right in zip(anchors, anchors[1:]):
            for j in range(left + 1, right):
                row = rows[j]
                if valid[row] or not av[row]:
                    continue
                span = axis[rows[right]] - axis[rows[left]]
                if span == 0:
                    continue
                before, after = values[rows[left]], values[rows[right]]
                fraction = (axis[row] - axis[rows[left]]) / span
                expected[row] = bits(libm.fma(after - before, fraction, before))
    for row in range(n):
        assert bool(output_valid[row // 8] & (1 << (row % 8))) == (expected[row] is not None)
        if expected[row] is None:
            continue
        actual = bytes(result[row * 8:(row + 1) * 8])
        if math.isnan(struct.unpack("=d", expected[row])[0]):
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
        for pattern in range(6):
            run(n, pattern)
finally:
    lib.dfm_context_release(ctx)
print(f"PASS {fields} native coordinate interpolation fields: independent validity, unsorted/duplicate/nonfinite axes, partitions, gaps and budgets")
