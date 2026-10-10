"""Exercise the optional native ABI on an Apple Silicon host after build.sh."""
import ctypes as C
import sys
import math
import random

I = C.c_int64
P = C.c_void_p


class Input(C.Structure):
    _fields_ = [
        ("values", P),
        ("validity", P),
        ("dtype", I),
        ("offset", I),
        ("has_validity", I),
    ]


class Step(C.Structure):
    _fields_ = [
        (x, I)
        for x in [
            "start",
            "nodes",
            "slot",
            "filter",
            "gather_start",
            "gather_count",
        ]
    ]


class Output(C.Structure):
    _fields_ = [("values", P), ("validity", P)] + [
        (x, I) for x in ["dtype", "slot", "reduction", "min_count"]
    ]


class Request(C.Structure):
    _fields_ = [
        (x, I)
        for x in [
            "abi",
            "rows",
            "dtype",
            "slots",
            "input_count",
            "code_words",
            "literal_count",
            "step_count",
            "gather_count",
            "output_count",
            "limit",
            "reductions",
            "profiling",
            "budget",
        ]
    ] + [
        (x, P)
        for x in ["inputs", "code", "literals", "steps", "gathers", "outputs"]
    ]


class Memory(C.Structure):
    _fields_ = [
        (x, I)
        for x in [
            "shared",
            "result",
            "peak",
            "upload",
            "launches",
            "blocks",
            "capacity",
            "packed",
        ]
    ]


class Stats(C.Structure):
    _fields_ = [("memory", Memory)] + [
        (x, I)
        for x in [
            "rows",
            "download",
            "waits",
            "cache_hit",
            "compile_ns",
            "stage_ns",
            "submit_ns",
            "copy_ns",
            "gpu_ns",
        ]
    ]


def ptr(x):
    return C.cast(x, P)


def packed(values, offset=0):
    out = (C.c_uint8 * max(1, (len(values) + offset + 7) // 8))()
    for i, v in enumerate(values):
        if v:
            out[(i + offset) // 8] |= 1 << ((i + offset) % 8)
    return out


def unpack(bits, n):
    return [bool(bits[i // 8] & (1 << (i % 8))) for i in range(n)]


lib = C.CDLL(
    sys.argv[1] if len(sys.argv) > 1 else "build/dfmetal/libdfmetal.dylib"
)
lib.dfm_context_create.argtypes = [I, I, C.POINTER(P)]
lib.dfm_context_create.restype = P
lib.dfm_context_release.argtypes = [P]
lib.dfm_execute.argtypes = [
    P,
    C.POINTER(Request),
    C.POINTER(Stats),
    C.POINTER(P),
]
lib.dfm_estimate.argtypes = [
    C.POINTER(Request),
    C.POINTER(Memory),
    C.POINTER(P),
]
lib.dfm_plan_cached.argtypes = [P, C.POINTER(Request), C.POINTER(P)]
lib.dfm_free.argtypes = [P]


def check(status, err):
    if status:
        message = C.string_at(err).decode() if err else "missing native error"
        lib.dfm_free(err)
        raise RuntimeError(message)


assert C.sizeof(Input) == 40 and C.sizeof(Step) == 48 and C.sizeof(Output) == 48
assert (
    C.sizeof(Request) == 160
    and C.sizeof(Memory) == 64
    and C.sizeof(Stats) == 136
)
lib.dfm_abi_version.restype = I
lib.dfm_device_count.restype = I
assert lib.dfm_abi_version() == 1
bad = Request()
memory = Memory()
err = P()
assert lib.dfm_estimate(C.byref(bad), C.byref(memory), C.byref(err)) == 1
assert b"ABI version mismatch" in C.string_at(err)
lib.dfm_free(err)
# Pure fusion preflight must run even on CI VMs without a Metal device.
# Descriptor-only estimates never dereference caller data or output storage.
def check_fusion_preflight_without_device():
    count = 32
    inputs = (Input * 1)(Input(None, None, 2, 0, 0))
    programs = []
    steps = []
    for i in range(count):
        programs += [(0, -1, -1, i), (1, -1, -1, -1), (5, 0, 1, -1)]
        steps.append(Step(3 * i, 3, i + 1, 0, 0, 0))
    code = (I * (4 * len(programs)))(*[x for node in programs for x in node])
    literals = (I * len(programs))()
    step_array = (Step * count)(*steps)
    outputs = (Output * 1)(Output(None, None, 2, count, -1, 0))
    request = Request(
        1,
        0,
        2,
        count + 1,
        1,
        len(code),
        len(literals),
        count,
        0,
        1,
        -1,
        0,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(literals),
        ptr(step_array),
        None,
        ptr(outputs),
    )
    for rows in [0, 1, 8388608]:
        request.rows = rows
        memory, err = Memory(), P()
        check(
            lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)),
            err,
        )
        assert memory.launches == 3
        assert memory.capacity == max(1, rows)
        # A long chain reserves input and final output, not 33 full columns.
        assert memory.shared < 11 * max(1, rows) + 1024
        assert memory.peak == memory.shared + memory.result
    step_array[0].slot = 2
    code[3] = 1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"undefined slot" in C.string_at(err)
    lib.dfm_free(err)


check_fusion_preflight_without_device()
print("Native Metal ABI and pure request validation: PASS")
if lib.dfm_device_count() == 0:
    print("Native Metal execution: SKIP (no GPU exposed on this host)")
    sys.exit(0)
err = P()
ctx = lib.dfm_context_create(0, 0, C.byref(err))
if not ctx:
    check(1, err)

# All request-owned buffers stay alive until the synchronous call returns.
SCALARS = {
    1: C.c_float,
    2: C.c_int32,
    3: I,
    4: C.c_uint8,
    5: C.c_int8,
    6: C.c_int16,
    7: C.c_uint8,
    8: C.c_uint16,
    9: C.c_uint32,
    10: C.c_uint64,
}


def execute(
    values,
    dtype,
    nodes,
    literals,
    *,
    valid=None,
    offset=0,
    filter_nodes=None,
    output_bool=False,
    reduction=-1,
    minimum=0,
    limit=-1,
    budget=-1,
    expect_error=None,
    repeats=1
):
    scalar = SCALARS[dtype]
    n = len(values)
    data = packed(values, offset) if dtype == 4 else (scalar * max(1, n))(
        *values
    )
    validity = packed(valid, offset) if valid is not None else None
    inputs = (Input * 1)(
        Input(
            ptr(data),
            ptr(validity) if validity is not None else None,
            dtype,
            offset,
            valid is not None,
        )
    )
    programs = []
    steps = []
    gathers = []
    lit = []
    slot = 1
    if filter_nodes:
        programs += filter_nodes
        lit += [0.0] * len(filter_nodes)
        steps.append(Step(0, len(filter_nodes), slot, 1, 0, 1))
        gathers.append(0)
        slot += 1
    start = len(lit)
    programs += nodes
    lit += literals
    steps.append(Step(start, len(nodes), slot, 0, len(gathers), 0))
    code = (I * (4 * len(programs)))(*[x for node in programs for x in node])
    # Int literals are supplied as Python integers and use raw signed bits.
    words = []
    for node, value in zip(programs, lit):
        words.append(
            C.c_uint64(int(value)).value if node[0]
            == 1 else C.cast(
                C.pointer(C.c_double(value)), C.POINTER(C.c_uint64)
            ).contents.value
        )
    literal = (C.c_uint64 * len(words))(*words)
    step_array = (Step * len(steps))(*steps)
    gather_array = (I * max(1, len(gathers)))(*gathers)
    out_type = (
        (dtype if dtype in (2, 9) else 3) if reduction == 10 else 3
    ) if reduction >= 0 else (4 if output_bool else dtype)
    out_scalar = SCALARS[out_type]
    result = (out_scalar * max(1, n))()
    bits = (C.c_uint8 * max(1, (n + 7) // 8))()
    outputs = (Output * 1)(
        Output(ptr(result), ptr(bits), out_type, slot, reduction, minimum)
    )
    request = Request(
        1,
        n,
        1 if dtype == 4 else dtype,
        slot + 1,
        1,
        len(code),
        len(words),
        len(steps),
        len(gathers),
        1,
        limit,
        reduction >= 0,
        1,
        budget,
        ptr(inputs),
        ptr(code),
        ptr(literal),
        ptr(step_array),
        ptr(gather_array),
        ptr(outputs),
    )
    memory = Memory()
    err = P()
    status = lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    if status and expect_error:
        message = C.string_at(err).decode()
        lib.dfm_free(err)
        assert expect_error in message, message
        return
    check(status, err)
    for repeat in range(repeats):
        stats = Stats()
        err = P()
        status = lib.dfm_execute(
            ctx, C.byref(request), C.byref(stats), C.byref(err)
        )
        if expect_error:
            assert status, expect_error
            message = C.string_at(err).decode()
            lib.dfm_free(err)
            assert expect_error in message, message
            return
        check(status, err)
        assert stats.memory.peak == memory.peak
        assert stats.memory.launches == memory.launches, (
            stats.memory.launches,
            memory.launches,
        )
        assert stats.waits == 1
        assert lib.dfm_plan_cached(ctx, C.byref(request), C.byref(err)) == 1
        if repeat:
            assert stats.cache_hit and stats.compile_ns == 0
    return (
        unpack(result, stats.rows) if out_type
        == 4 else list(result)[: stats.rows],
        unpack(bits, stats.rows),
        stats,
    )


col = (0, -1, -1, 0)
for n in [0, 1, 7, 8, 9, 255, 256, 257, 1023, 4097]:
    vals = [float(i % 37 - 18) for i in range(n)]
    valid = [i % 5 != 0 for i in range(n)]
    result, bits, stats = execute(
        vals,
        1,
        [col, (2, -1, -1, -1), (5, 0, 1, -1)],
        [0.0, 1.25, 0.0],
        valid=valid,
        offset=3,
        repeats=2,
    )
    assert bits == valid
    assert all(a == b + 1.25 for a, b, v in zip(result, vals, valid) if v)
    pred = [col, (2, -1, -1, -1), (8, 0, 1, -1)]
    result, bits, stats = execute(
        vals,
        1,
        [col],
        [0.0],
        valid=valid,
        offset=5,
        filter_nodes=pred,
        limit=19,
    )
    expected = [x for x, v in zip(vals, valid) if v and x > 0][:19]
    assert result == expected and bits == [True] * len(expected), (
        n,
        result,
        expected,
    )

# Packed Boolean source and null logic use non-byte-aligned slice windows.
values = [True, False, True, False, False, True, True, False, True] * 33
valid = [i % 3 != 0 for i in range(len(values))]
result, bits, _ = execute(
    values,
    4,
    [col, (3, -1, -1, -1), (30, 0, 1, -1)],
    [0.0, 0.0, 0.0],
    valid=valid,
    offset=7,
    output_bool=True,
)
assert result == [False] * len(values) and all(bits)
result, bits, _ = execute(
    values,
    4,
    [col, (3, -1, -1, -1), (31, 0, 1, -1)],
    [0.0, 1.0, 0.0],
    valid=valid,
    offset=1,
    output_bool=True,
)
assert result == [True] * len(values) and all(bits)

for dtype, lo, hi in [
    (2, -(2**31), 2**31 - 1),
    (3, -(2**63), 2**63 - 1),
]:
    for op, a, b in [(5, hi, 1), (6, lo, 1), (7, lo, -1)]:
        execute(
            [a],
            dtype,
            [col, (1, -1, -1, -1), (op, 0, 1, -1)],
            [0, b, 0],
            expect_error="overflow",
        )
    execute(
        [lo], dtype, [col, (50, 0, -1, -1)], [0, 0], expect_error="overflow"
    )
    for op, a, b, expected in [
        (5, hi, -1, hi - 1),
        (6, lo, -1, lo + 1),
        (7, lo, 1, lo),
        (7, hi, 0, 0),
    ]:
        result, bits, _ = execute(
            [a], dtype, [col, (1, -1, -1, -1), (op, 0, 1, -1)], [0, b, 0]
        )
        assert result == [expected] and bits == [True]
    # Invalid rows must never trigger checked overflow.
    result, bits, _ = execute(
        [hi],
        dtype,
        [col, (1, -1, -1, -1), (5, 0, 1, -1)],
        [0, 1, 0],
        valid=[False],
    )
    assert bits == [False]

for n in [0, 1, 257, 10001]:
    values = [i % 100 - 50 for i in range(n)]
    valid = [i % 7 != 0 for i in range(n)]
    for reduction in [10, 11, 90]:
        result, bits, _ = execute(
            values, 2, [col], [0], valid=valid, reduction=reduction
        )
        expected = (
            sum(x for x, v in zip(values, valid) if v) if reduction
            == 10 else sum(valid) if reduction
            == 11 else n
        )
        assert result == [expected] and bits == [True]
result, bits, _ = execute(
    [1, 2], 2, [col], [0], valid=[False, False], reduction=10, minimum=1
)
assert bits == [False]
execute([2**31 - 1, 1], 2, [col], [0], reduction=10, expect_error="overflow")
result, bits, _ = execute([2**31 - 1, 1, -1], 2, [col], [0], reduction=10)
assert result == [2**31 - 1]
execute([1.0], 1, [col], [0.0], budget=0, expect_error="memory budget")
execute(
    [1.0], 1, [col], [0.0], reduction=10, expect_error="accumulator precision"
)
# Separate Float32 multiply/add rounding must survive source specialization.
def f32(x):
    return C.c_float(x).value


rng = random.Random(815)
values = [f32(rng.uniform(-10000, 10000)) for _ in range(4097)]
a = f32(1.0001)
b = f32(0.0001)
result, bits, _ = execute(
    values,
    1,
    [col, (2, -1, -1, -1), (7, 0, 1, -1), (2, -1, -1, -1), (5, 2, 3, -1)],
    [0.0, a, 0.0, b, 0.0],
)
expected = [f32(f32(x * a) + b) for x in values]
assert result == expected, "Float32 multiply/add contraction changed rounding"
# Safe arithmetic must preserve exceptional values and signed zero.
result, bits, _ = execute(
    [math.nan, math.inf, -math.inf, -0.0], 1, [col, (50, 0, -1, -1)], [0.0, 0.0]
)
assert (
    math.isnan(result[0]) and result[1] == -math.inf and result[2] == math.inf
)
assert math.copysign(1, result[3]) == 1
tiny = 2.0**-149
result, bits, _ = execute([tiny, -tiny, 0.0], 1, [col], [0.0])
assert result == [tiny, -tiny, 0.0]
result, bits, _ = execute(
    [tiny, -tiny, 0.0], 1, [col, (50, 0, -1, -1)], [0.0, 0.0]
)
assert result == [-tiny, tiny, -0.0]
for op, a, b in [
    (5, tiny, tiny),
    (6, 2.0**-126, f32(2.0**-126 + tiny)),
    (7, 1e-20, 1e-20),
    (8, tiny, 0.0),
]:
    execute(
        [a],
        1,
        [col, (2, -1, -1, -1), (op, 0, 1, -1)],
        [0.0, b, 0.0],
        output_bool=op == 8,
        expect_error="subnormal",
    )
# Two filter boundaries retain an early branch and the original input while
# pruning the other 30 temporary projections from shared storage and gathers.
def check_fused_boundaries():
    n = 4097
    data = (C.c_int32 * n)(*[i % 71 - 35 for i in range(n)])
    valid = [i % 13 != 0 for i in range(n)]
    validity = packed(valid)
    inputs = (Input * 1)(Input(ptr(data), ptr(validity), 2, 0, 1))
    programs, literals, steps, gathers = [], [], [], []
    slots = 1
    current = 0
    live_columns = [0]
    expected = [(data[i], data[i], valid[i]) for i in range(n)]
    early = -1
    for stage in range(32):
        start = len(programs)
        programs += [(0, -1, -1, current), (1, -1, -1, -1), (5, 0, 1, -1)]
        literals += [0, 1, 0]
        steps.append(Step(start, 3, slots, 0, len(gathers), 0))
        current = slots
        live_columns.append(slots)
        slots += 1
        expected = [(x, y + 1, v) for x, y, v in expected]
        if stage == 9:
            early = current
        if stage in (15, 31):
            start = len(programs)
            programs += [(0, -1, -1, current), (1, -1, -1, -1), (8, 0, 1, -1)]
            literals += [0, 5 if stage == 15 else 25, 0]
            steps.append(
                Step(start, 3, slots, 1, len(gathers), len(live_columns))
            )
            gathers += live_columns
            slots += 1
            expected = [
                (x, y, v)
                for x, y, v in expected
                if v and y > (5 if stage == 15 else 25)
            ]
    arrays = [(C.c_int32 * n)() for _ in range(3)]
    bitmaps = [(C.c_uint8 * ((n + 7) // 8))() for _ in range(3)]
    outputs = (Output * 3)(
        *[
            Output(ptr(a), ptr(b), 2, slot, -1, 0)
            for a, b, slot in zip(arrays, bitmaps, [0, early, current])
        ]
    )
    code = (I * (4 * len(programs)))(*[x for node in programs for x in node])
    words = (I * len(literals))(*literals)
    step_array = (Step * len(steps))(*steps)
    gather_array = (I * len(gathers))(*gathers)
    request = Request(
        1,
        n,
        2,
        slots,
        1,
        len(code),
        len(words),
        len(steps),
        len(gathers),
        3,
        -1,
        0,
        1,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(words),
        ptr(step_array),
        ptr(gather_array),
        ptr(outputs),
    )
    memory, stats, err = Memory(), Stats(), P()
    check(
        lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err
    )
    for _ in range(2):
        check(
            lib.dfm_execute(
                ctx, C.byref(request), C.byref(stats), C.byref(err)
            ),
            err,
        )
        assert stats.rows == len(expected)
        assert list(arrays[0])[: stats.rows] == [x for x, y, v in expected]
        assert list(arrays[1])[: stats.rows] == [x + 10 for x, y, v in expected]
        assert list(arrays[2])[: stats.rows] == [y for x, y, v in expected]
        assert all(all(unpack(b, stats.rows)) for b in bitmaps)
        assert stats.memory.shared == memory.shared
        assert stats.memory.launches == memory.launches
        assert memory.launches < len(steps)
        assert memory.shared < n * slots * 5

    # A pruned temporary still contributes checked-arithmetic errors.
    data[1] = 2147483647
    err = P()
    assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
    assert b"Integer overflow" in C.string_at(err)
    lib.dfm_free(err)
    data[1] = 0
    # Undefined logical slots cannot become negative physical addresses.
    step_array[0].slot = 2
    code[3] = 1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"undefined slot" in C.string_at(err)
    lib.dfm_free(err)


check_fused_boundaries()
# Every native width exercises exact literals, checked arithmetic, stable
# compaction with offset validity, comparisons, and reduction storage.
for dtype, lo, hi in [
    (5, -128, 127),
    (6, -32768, 32767),
    (7, 0, 255),
    (8, 0, 65535),
    (9, 0, 2**32 - 1),
    (10, 0, 2**64 - 1),
]:
    arithmetic = [col, (1, -1, -1, -1), (5, 0, 1, -1)]
    result, bits, _ = execute([hi - 1], dtype, arithmetic, [0, 1, 0])
    assert result == [hi] and bits == [True], (dtype, result)
    execute([hi], dtype, arithmetic, [0, 1, 0], expect_error="overflow")
    execute(
        [lo],
        dtype,
        [col, (1, -1, -1, -1), (6, 0, 1, -1)],
        [0, 1, 0],
        expect_error="overflow",
    )
    execute(
        [hi],
        dtype,
        [col, (1, -1, -1, -1), (7, 0, 1, -1)],
        [0, 2, 0],
        expect_error="overflow",
    )
    execute(
        [lo if lo else 1],
        dtype,
        [col, (50, 0, -1, -1)],
        [0, 0],
        expect_error="overflow",
    )
    result, bits, _ = execute([0], dtype, arithmetic, [0, hi, 0])
    assert result == [hi] and bits == [True]
    for op in (5, 6, 7):
        samples = [(0, 0), (hi, 0), (hi, 1), (hi, hi), (lo, 1), (lo, lo)]
        samples += [
            (random.randint(lo, hi), random.randint(lo, hi)) for _ in range(8)
        ]
        for a, b in samples:
            expected = a + b if op == 5 else a - b if op == 6 else a * b
            if lo <= expected <= hi:
                result, bits, _ = execute(
                    [a],
                    dtype,
                    [col, (1, -1, -1, -1), (op, 0, 1, -1)],
                    [0, b, 0],
                )
                assert result == [expected] and bits == [True], (
                    dtype,
                    op,
                    a,
                    b,
                    result,
                )
            else:
                execute(
                    [a],
                    dtype,
                    [col, (1, -1, -1, -1), (op, 0, 1, -1)],
                    [0, b, 0],
                    expect_error="overflow",
                )
    values = [0, 1, hi] * 257
    valid = [i % 7 != 0 for i in range(len(values))]
    result, bits, _ = execute(
        values,
        dtype,
        [col],
        [0],
        valid=valid,
        offset=5,
        filter_nodes=[col, (1, -1, -1, -1), (8, 0, 1, -1)],
    )
    assert result == [x for x, v in zip(values, valid) if v and x > 0]
    assert all(bits)
    result, bits, _ = execute(
        [hi, hi - 1],
        dtype,
        [col, (1, -1, -1, -1), (9, 0, 1, -1)],
        [0, hi, 0],
        output_bool=True,
    )
    assert result == [True, False] and all(bits)
    result, bits, _ = execute(
        values, dtype, [col], [0], valid=valid, reduction=11
    )
    assert result == [sum(valid)] and bits == [True]
    if dtype != 10:
        result, bits, _ = execute([1, 2, 3], dtype, [col], [0], reduction=10)
        assert result == [6] and bits == [True]
    else:
        execute(
            [1],
            dtype,
            [col],
            [0],
            reduction=10,
            expect_error="accumulator precision",
        )
execute([2**32 - 1, 1], 9, [col], [0], reduction=10, expect_error="overflow")
lib.dfm_context_release(ctx)
print(
    "Native Metal ABI, boundaries, filters, bitmaps, integer checks, reductions and cache: PASS"
)
