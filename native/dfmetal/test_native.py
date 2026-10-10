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
        for x in [
            "inputs",
            "code",
            "literals",
            "steps",
            "gathers",
            "outputs",
            "node_types",
            "slot_types",
        ]
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


lib = C.CDLL(sys.argv[1] if len(sys.argv) > 1 else "build/dfmetal/libdfmetal.dylib")
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
assert C.sizeof(Request) == 176 and C.sizeof(Memory) == 64 and C.sizeof(Stats) == 136
lib.dfm_abi_version.restype = I
lib.dfm_device_count.restype = I
assert lib.dfm_abi_version() == 2
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
        2,
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


def check_sort_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 2, 0, 0))
    code = (I * 4)(200, 0, 0, 1)
    literals = (I * 1)()
    steps = (Step * 1)(Step(0, 1, 0, 2, 0, 1))
    gathers = (I * 1)(0)
    outputs = (Output * 1)(Output(None, None, 2, 0, -1, 0))
    node_types, slot_types = (I * 1)(2), (I * 1)(2)
    request = Request(
        2,
        513,
        0,
        1,
        1,
        4,
        1,
        1,
        1,
        1,
        -1,
        0,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(literals),
        ptr(steps),
        ptr(gathers),
        ptr(outputs),
        ptr(node_types),
        ptr(slot_types),
    )
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.launches == 14
    assert memory.shared >= 30 * 513
    for field, value, message in ((2, 2, b"sort key"), (1, 1, b"sort key")):
        old = code[field]
        code[field] = value
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert message in C.string_at(err)
        lib.dfm_free(err)
        code[field] = old
    node_types[0] = 1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"sort key dtype" in C.string_at(err)
    lib.dfm_free(err)


check_sort_preflight_without_device()


def check_group_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 2, 0, 0))
    programs = [(201, 1, 2, 3), (202, 0, 0, 0), (203, 0, 4, 10)]
    code = (I * 12)(*[x for node in programs for x in node])
    literals = (I * 3)()
    nt = (I * 3)(3, 2, 2)
    st = (I * 5)(2, 3, 3, 3, 2)
    steps = (Step * 1)(Step(0, 3, 1, 3, 0, 0))
    outputs = (Output * 1)(Output(None, None, 2, 4, -1, 0))
    request = Request(
        2,
        0,
        0,
        5,
        1,
        12,
        3,
        1,
        0,
        1,
        -1,
        0,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(literals),
        ptr(steps),
        None,
        ptr(outputs),
        ptr(nt),
        ptr(st),
    )
    for n in (0, 1, 8388608):
        request.rows = n
        memory, err = Memory(), P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        assert memory.result == 4 * max(1, n) + (max(1, n) + 7) // 8
        assert memory.launches == 8 + (max(1, n) - 1).bit_length()
    for index, value in [(2, 0), (3, 2), (9, 1), (10, 3), (11, 11)]:
        before = code[index]
        code[index] = value
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)
        code[index] = before


check_group_preflight_without_device()

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
    data = packed(values, offset) if dtype == 4 else (scalar * max(1, n))(*values)
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
            C.c_uint64(int(value)).value
            if node[0] == 1
            else C.cast(
                C.pointer(C.c_double(value)), C.POINTER(C.c_uint64)
            ).contents.value
        )
    literal = (C.c_uint64 * len(words))(*words)
    step_array = (Step * len(steps))(*steps)
    gather_array = (I * max(1, len(gathers)))(*gathers)
    out_type = (
        ((dtype if dtype in (2, 9) else 3) if reduction == 10 else 3)
        if reduction >= 0
        else (4 if output_bool else dtype)
    )
    out_scalar = SCALARS[out_type]
    result = (out_scalar * max(1, n))()
    bits = (C.c_uint8 * max(1, (n + 7) // 8))()
    outputs = (Output * 1)(
        Output(ptr(result), ptr(bits), out_type, slot, reduction, minimum)
    )
    request = Request(
        2,
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
        status = lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
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
        unpack(result, stats.rows) if out_type == 4 else list(result)[: stats.rows],
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
    execute([lo], dtype, [col, (50, 0, -1, -1)], [0, 0], expect_error="overflow")
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
            sum(x for x, v in zip(values, valid) if v)
            if reduction == 10
            else sum(valid) if reduction == 11 else n
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
execute([1.0], 1, [col], [0.0], reduction=10, expect_error="accumulator precision")


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
assert math.isnan(result[0]) and result[1] == -math.inf and result[2] == math.inf
assert math.copysign(1, result[3]) == 1
tiny = 2.0**-149
result, bits, _ = execute([tiny, -tiny, 0.0], 1, [col], [0.0])
assert result == [tiny, -tiny, 0.0]
result, bits, _ = execute([tiny, -tiny, 0.0], 1, [col, (50, 0, -1, -1)], [0.0, 0.0])
assert result == [-tiny, tiny, -0.0]
for op, a, b in [
    (5, tiny, tiny),
    (6, 2.0**-126, f32(2.0**-126 + tiny)),
    (7, 1e-20, 1e-20),
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
            steps.append(Step(start, 3, slots, 1, len(gathers), len(live_columns)))
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
        2,
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
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for _ in range(2):
        check(
            lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
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
        samples += [(random.randint(lo, hi), random.randint(lo, hi)) for _ in range(8)]
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
    result, bits, _ = execute(values, dtype, [col], [0], valid=valid, reduction=11)
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


def test_typed_rows():
    # Each input retains its native width; a UInt64 predicate compacts all
    # columns, including Float32, signed minima, UInt64 maxima and Bool bits.
    n = 259
    offset = 5
    types = list(SCALARS)
    values = {}
    arrays = []
    validities = []
    inputs_list = []
    for t in types:
        if t == 1:
            vs = [float(i % 19) / 4 for i in range(n)]
        elif t == 10:
            vs = [2**64 - 1 if i % 3 else 2**63 for i in range(n)]
        elif t == 3:
            vs = [-(2**63) + i for i in range(n)]
        elif t == 4:
            vs = [bool(i % 2) for i in range(n)]
        else:
            vs = [i % 67 for i in range(n)]
        values[t] = vs
        data = packed(vs, offset) if t == 4 else (SCALARS[t] * n)(*vs)
        valid = packed([i % 13 != 0 for i in range(n)], offset)
        arrays.append(data)
        validities.append(valid)
        inputs_list.append(Input(ptr(data), ptr(valid), t, offset, 1))
    inputs = (Input * len(types))(*inputs_list)
    programs = [(0, -1, -1, types.index(10)), (1, -1, -1, -1), (8, 0, 1, -1)]
    node_types = [10, 10, 4]
    words = [0, 2**63, 0]
    slots = types + [4]
    steps_list = [Step(0, 3, len(types), 1, 0, len(types))]
    gather = (I * len(types))(*range(len(types)))
    outputs_list, results, bitmaps = [], [], []
    for col_index, t in enumerate(types):
        start = len(words)
        # Preserve the two 64-bit boundary columns and packed Boolean.
        increment = t not in [3, 4, 10]
        programs += [(0, -1, -1, col_index)]
        node_types += [t]
        words += [0]
        if increment:
            programs += [(2 if t == 1 else 1, -1, -1, -1), (5, 0, 1, -1)]
            node_types += [t, t]
            words += [
                (
                    C.cast(C.pointer(C.c_double(1.0)), C.POINTER(C.c_uint64))[0]
                    if t == 1
                    else 1
                ),
                0,
            ]
        slot = len(slots)
        slots.append(t)
        steps_list.append(Step(start, 3 if increment else 1, slot, 0, 0, 0))
        result = (SCALARS[t] * n)()
        bitmap = (C.c_uint8 * ((n + 7) // 8))()
        results.append(result)
        bitmaps.append(bitmap)
        outputs_list.append(Output(ptr(result), ptr(bitmap), t, slot, -1, 0))
    code = (I * (4 * len(words)))(*[v for node in programs for v in node])
    literals = (C.c_uint64 * len(words))(*words)
    steps = (Step * len(steps_list))(*steps_list)
    outputs = (Output * len(outputs_list))(*outputs_list)
    nt = (I * len(node_types))(*node_types)
    st = (I * len(slots))(*slots)
    request = Request(
        2,
        n,
        0,
        len(slots),
        len(types),
        len(code),
        len(words),
        len(steps),
        len(gather),
        len(outputs),
        -1,
        0,
        1,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(literals),
        ptr(steps),
        ptr(gather),
        ptr(outputs),
        ptr(nt),
        ptr(st),
    )
    memory, stats, err = Memory(), Stats(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    check(
        lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
        err,
    )
    selected = [i for i in range(n) if i % 13 != 0 and i % 3 != 0]
    assert stats.rows == len(selected)
    assert stats.memory.peak == memory.peak
    for j, t in enumerate(types):
        result = (
            unpack(results[j], stats.rows) if t == 4 else list(results[j])[: stats.rows]
        )
        expected = [values[t][i] + (t not in [3, 4, 10]) for i in selected]
        assert result == expected, (t, result[:10], expected[:10])
        assert unpack(bitmaps[j], stats.rows) == [True] * stats.rows
    # ABI validation rejects mismatched metadata before creating pipelines.
    nt[0] = 2
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"dtype mismatch" in C.string_at(err)
    lib.dfm_free(err)
    # Mixed scans have no expression nodes and need no node type array.
    request.step_count = request.literal_count = request.code_words = 0
    request.node_types = None
    request.slots = len(types)
    for j, t in enumerate(types):
        outputs[j].slot = j
    err = P()
    check(
        lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
        err,
    )
    assert stats.rows == n
    for j, t in enumerate(types):
        result = unpack(results[j], n) if t == 4 else list(results[j])
        validity = unpack(bitmaps[j], n)
        assert validity == [i % 13 != 0 for i in range(n)]
        assert all(result[i] == values[t][i] for i in range(n) if validity[i])


test_typed_rows()


def test_numeric_casts():
    source_values = {
        1: [
            -float("inf"),
            -129.5,
            -128.5,
            -1.5,
            -0.0,
            0.0,
            1e-45,
            0.5,
            1.5,
            127.75,
            255.0,
            2**31,
            2**63,
            2**64,
            float("inf"),
            float("nan"),
        ],
        2: [-(2**31), -129, -128, -1, 0, 1, 127, 128, 255, 256, 2**31 - 1],
        3: [
            -(2**63),
            -(2**53) - 1,
            -129,
            -1,
            0,
            1,
            2**53 + 1,
            2**63 - 1,
        ],
        4: [False, True],
        5: [-128, -1, 0, 1, 127],
        6: [-32768, -129, -1, 0, 255, 32767],
        7: [0, 1, 127, 128, 255],
        8: [0, 1, 255, 256, 65535],
        9: [0, 1, 65535, 2**31, 2**32 - 1],
        10: [0, 1, 2**53 + 1, 2**63, 2**64 - 1],
    }
    for source in SCALARS:
        values = source_values[source]
        n = len(values)
        data = packed(values) if source == 4 else (SCALARS[source] * n)(*values)
        # Compare the exactly representable input, rather than Python doubles.
        actual = values if source == 4 else list(data)
        for target in SCALARS:
            inputs = (Input * 1)(Input(ptr(data), None, source, 0, 0))
            code = (I * 8)(0, -1, -1, 0, 79, 0, -1, 0)
            literals = (C.c_uint64 * 2)()
            steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
            result = (SCALARS[target] * n)()
            valid = (C.c_uint8 * ((n + 7) // 8))()
            outputs = (Output * 1)(Output(ptr(result), ptr(valid), target, 1, -1, 0))
            nt, st = (I * 2)(source, target), (I * 2)(source, target)
            request = Request(
                2,
                n,
                0,
                2,
                1,
                8,
                2,
                1,
                0,
                1,
                -1,
                0,
                0,
                -1,
                ptr(inputs),
                ptr(code),
                ptr(literals),
                ptr(steps),
                None,
                ptr(outputs),
                ptr(nt),
                ptr(st),
            )
            stats, err = Stats(), P()
            check(
                lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
                err,
            )
            got = unpack(result, n) if target == 4 else list(result)
            validity = unpack(valid, n)
            expected_valid = []
            for i, x in enumerate(actual):
                if target == 4:
                    fits = not (source == 1 and math.isnan(x))
                    expected = bool(x)
                elif target == 1:
                    fits = True
                    expected = C.c_float(float(x)).value
                else:
                    width = C.sizeof(SCALARS[target]) * 8
                    unsigned = target in [7, 8, 9, 10]
                    lo, hi = (
                        (0, 2**width - 1)
                        if unsigned
                        else (
                            -(2 ** (width - 1)),
                            2 ** (width - 1) - 1,
                        )
                    )
                    fits = not (source == 1 and not math.isfinite(x))
                    expected = math.trunc(x) if fits else 0
                    fits = fits and lo <= expected <= hi
                expected_valid.append(fits)
                if fits:
                    assert got[i] == expected or (
                        target == 1 and math.isnan(got[i]) and math.isnan(expected)
                    ), (source, target, i, x, got[i], expected)
            assert validity == expected_valid, (
                source,
                target,
                validity,
                expected_valid,
            )
            if not all(expected_valid):
                code[7] = 1
                err = P()
                assert lib.dfm_execute(
                    ctx, C.byref(request), C.byref(stats), C.byref(err)
                )
                assert b"strict cast failed" in C.string_at(err), (
                    source,
                    target,
                    C.string_at(err),
                )
                lib.dfm_free(err)
    # A value just below a Float32 halfway point can round via Float64 to
    # that point. This is a native precision boundary, never a silent mismatch.
    source, target = 10, 1
    data = (C.c_uint64 * 1)(2**63 + 2**39 - 1)
    inputs = (Input * 1)(Input(ptr(data), None, source, 0, 0))
    code = (I * 8)(0, -1, -1, 0, 79, 0, -1, 0)
    literals = (C.c_uint64 * 2)()
    steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
    result, valid = (C.c_float * 1)(), (C.c_uint8 * 1)()
    outputs = (Output * 1)(Output(ptr(result), ptr(valid), target, 1, -1, 0))
    nt, st = (I * 2)(source, target), (I * 2)(source, target)
    request = Request(
        2,
        1,
        0,
        2,
        1,
        8,
        2,
        1,
        0,
        1,
        -1,
        0,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(literals),
        ptr(steps),
        None,
        ptr(outputs),
        ptr(nt),
        ptr(st),
    )
    stats, err = Stats(), P()
    assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
    assert b"double-rounding boundary" in C.string_at(err), C.string_at(err)
    lib.dfm_free(err)


test_numeric_casts()


def test_extrema():
    types = {
        1: C.c_float,
        2: C.c_int32,
        3: C.c_int64,
        4: C.c_uint8,
        5: C.c_int8,
        6: C.c_int16,
        7: C.c_uint8,
        8: C.c_uint16,
        9: C.c_uint32,
        10: C.c_uint64,
    }

    def check(dtype, values, valid=None, offset=5):
        ctype = types[dtype]
        n = len(values)
        if valid is None:
            valid = [True] * n
        bitmap = (C.c_uint8 * ((n + offset + 7) // 8))()
        for i, yes in enumerate(valid):
            if yes:
                bitmap[(i + offset) // 8] |= 1 << ((i + offset) % 8)
        if dtype == 4:
            data = (C.c_uint8 * ((n + offset + 7) // 8))()
            for i, x in enumerate(values):
                if x:
                    data[(i + offset) // 8] |= 1 << ((i + offset) % 8)
        else:
            data = (ctype * n)(*values)
        inputs = (Input * 1)(Input(ptr(data), ptr(bitmap), dtype, offset, 1))
        # Padding detects writes wider than the physical output dtype.
        results = [(C.c_uint8 * 24)(*([0xA5] * 24)) for _ in range(2)]
        bits = [(C.c_uint8 * 1)() for _ in range(2)]
        outputs = (Output * 2)(
            *[
                Output(C.addressof(results[i]) + 8, ptr(bits[i]), dtype, 0, 80 + i, 0)
                for i in range(2)
            ]
        )
        slots = (I * 1)(dtype)
        request = Request(
            2,
            n,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            2,
            -1,
            1,
            0,
            -1,
            ptr(inputs),
            None,
            None,
            None,
            None,
            ptr(outputs),
            None,
            ptr(slots),
        )
        stats, err = Stats(), P()
        rc = lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
        assert rc == 0, C.string_at(err) if err else rc
        chosen = [x for x, yes in zip(values, valid) if yes]
        for i in range(2):
            assert bool(bits[i][0] & 1) == bool(chosen), (dtype, i, chosen)
            size = C.sizeof(ctype)
            assert list(results[i][:8]) == [0xA5] * 8
            assert list(results[i][8 + size :]) == [0xA5] * (16 - size), (
                dtype,
                i,
                list(results[i]),
            )
            if not chosen:
                continue
            key = lambda x: (
                (math.isnan(x), 0 if math.isnan(x) else x) if dtype == 1 else x
            )
            expected = (min if i == 0 else max)(chosen, key=key)
            got = ctype.from_address(C.addressof(results[i]) + 8).value
            if dtype == 1 and math.isnan(expected):
                assert math.isnan(got)
            elif dtype == 1:
                assert C.string_at(C.byref(ctype(got)), size) == C.string_at(
                    C.byref(ctype(expected)), size
                ), (got, expected)
            else:
                assert got == expected, (dtype, i, got, expected)

    rng = random.Random(811)
    for dtype, ctype in types.items():
        if dtype == 1:
            values = [rng.uniform(-100, 100) for _ in range(521)]
            values = [ctype(x).value for x in values]
        elif dtype == 4:
            values = [bool(rng.randrange(2)) for _ in range(521)]
        else:
            bits = C.sizeof(ctype) * 8
            signed = dtype in (2, 3, 5, 6)
            lo = -(1 << (bits - 1)) if signed else 0
            hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
            values = [lo, hi] + [rng.randint(lo, hi) for _ in range(519)]
        check(dtype, values, [i % 7 != 3 for i in range(521)])
        check(dtype, [])
        check(dtype, values, [False] * 521)
    float_bits = lambda u: C.cast(C.pointer(C.c_uint32(u)), C.POINTER(C.c_float))[0]
    for values in (
        [float("nan"), 3.0, -2.0],
        [float("nan")] * 521,
        [0.0, -0.0] * 300,
        [-0.0, 0.0] * 300,
        [float_bits(1), float_bits(0x80000001), 0.0],
        [float("inf"), float("-inf")],
    ):
        check(1, values)
    # Exercise strided partials after the 1024-group dispatch cap.
    values = [0.0] * 262401
    values[256] = -0.0
    values[0] = -0.0
    check(1, values)


test_extrema()


def test_additional_row_ops():
    for dtype in (2, 3, 5, 6, 7, 8, 9, 10):
        signed = dtype in (2, 3, 5, 6)
        values = ([-7, -1, 0, 1, 7] if signed else [0, 1, 2, 7, 15]) * 53
        valid = [i % 7 != 2 for i in range(len(values))]
        for denominator in ([3, -3] if signed else [3]):
            for op in (25, 26):
                result, bits, _ = execute(
                    values,
                    dtype,
                    [col, (1, -1, -1, -1), (op, 0, 1, -1)],
                    [0, denominator, 0],
                    valid=valid,
                    offset=5,
                )
                expected = [
                    x // denominator if op == 25 else x % denominator for x in values
                ]
                assert bits == valid
                assert all(
                    not yes or x == y for x, y, yes in zip(result, expected, valid)
                ), (dtype, op, result, expected)
        _, bits, _ = execute(
            values, dtype, [col, (1, -1, -1, -1), (25, 0, 1, -1)], [0, 0, 0]
        )
        assert not any(bits)
        small = [-3, -2, 0, 2, 3] if signed else [0, 1, 2, 3, 4]
        result, bits, _ = execute(
            small, dtype, [col, (1, -1, -1, -1), (27, 0, 1, -1)], [0, 2, 0]
        )
        assert result == [x * x for x in small]
        for op in (28, 29):
            result, _, _ = execute(
                values, dtype, [col, (1, -1, -1, -1), (op, 0, 1, -1)], [0, 2, 0]
            )
            assert result == [(max if op == 28 else min)(x, 2) for x in values]
        for op in (51, 55, 56, 57):
            result, _, _ = execute(values, dtype, [col, (op, 0, -1, 0)], [0, 0])
            assert result == [abs(x) if op == 51 else x for x in values]
        if signed:
            minimum = -(1 << (C.sizeof(SCALARS[dtype]) * 8 - 1))
            execute(
                [minimum],
                dtype,
                [col, (51, 0, -1, -1)],
                [0, 0],
                expect_error="overflow",
            )
            execute(
                [minimum],
                dtype,
                [col, (1, -1, -1, -1), (25, 0, 1, -1)],
                [0, -1, 0],
                expect_error="overflow",
            )
            result, _, _ = execute(
                [minimum], dtype, [col, (1, -1, -1, -1), (26, 0, 1, -1)], [0, -1, 0]
            )
            assert result == [0]
            execute(
                [2],
                dtype,
                [col, (1, -1, -1, -1), (27, 0, 1, -1)],
                [0, -1, 0],
                expect_error="nonnegative exponent",
            )
            # Conditional masks suppress an otherwise observable ABS overflow.
            result, _, _ = execute(
                [minimum, -1, 0, 1],
                dtype,
                [
                    col,
                    (1, -1, -1, -1),
                    (8, 0, 1, -1),
                    col,
                    (51, 3, -1, -1),
                    col,
                    (100, 2, 4, 5),
                ],
                [0, 0, 0, 0, 0, 0, 0],
            )
            assert result == [minimum, -1, 0, 1]
    rng = random.Random(918)
    from_bits = lambda x: C.cast(C.pointer(C.c_uint32(x)), C.POINTER(C.c_float))[0]
    values = [
        0.0,
        -0.0,
        2**-149,
        -(2**-149),
        math.inf,
        -math.inf,
        math.nan,
        0.5,
        -0.5,
        1.5,
        -1.5,
    ] + [from_bits(rng.getrandbits(32)) for _ in range(521)]
    operations = {
        8: lambda x, y: x > y,
        9: lambda x, y: x == y,
        20: lambda x, y: x < y,
        21: lambda x, y: x >= y,
        22: lambda x, y: x <= y,
        23: lambda x, y: x != y,
    }
    for bound in (0.0, 2**-149, -(2**-149), math.nan, math.inf, -math.inf):
        for op, compare in operations.items():
            result, _, _ = execute(
                values,
                1,
                [col, (2, -1, -1, -1), (op, 0, 1, -1)],
                [0, bound, 0],
                output_bool=True,
            )
            assert result == [compare(x, bound) for x in values], (op, bound)
    for op in (51, 55, 56, 57):
        result, _, _ = execute(values, 1, [col, (op, 0, -1, 0)], [0, 0])
        for x, y in zip(values, result):
            if math.isnan(x):
                assert math.isnan(y)
                continue
            if op == 51:
                expected = abs(x)
            elif math.isinf(x):
                expected = x
            else:
                expected = float(
                    math.floor(x)
                    if op == 55
                    else (
                        math.ceil(x)
                        if op == 56
                        else math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
                    )
                )
                if expected == 0:
                    expected = math.copysign(0.0, x)
                expected = f32(expected)
            assert C.string_at(C.byref(C.c_float(y)), 4) == C.string_at(
                C.byref(C.c_float(expected)), 4
            ), (op, x, y, expected)
    for op in (63, 64, 65, 66):
        result, _, _ = execute(
            values, 1, [col, (op, 0, -1, -1)], [0, 0], output_bool=True
        )
        expected = [
            (
                math.isnan(x)
                if op == 63
                else (
                    not math.isnan(x)
                    if op == 64
                    else math.isfinite(x) if op == 65 else math.isinf(x)
                )
            )
            for x in values
        ]
        assert result == expected, op
    result, _, _ = execute(
        values, 1, [col, (2, -1, -1, -1), (34, 0, 1, -1)], [0, 3.5, 0]
    )
    assert all(y == 3.5 if math.isnan(x) else y == x for x, y in zip(values, result))
    _, bits, _ = execute(values, 1, [col, (12, -1, -1, -1), (34, 0, 1, -1)], [0, 0, 0])
    assert bits == [not math.isnan(x) for x in values]
    for op in (28, 29):
        result, _, _ = execute(
            values, 1, [col, (2, -1, -1, -1), (op, 0, 1, -1)], [0, 0, 0]
        )
        for x, y in zip(values, result):
            if math.isnan(x):
                assert math.isnan(y)
            else:
                expected = 0.0 if (x < 0 if op == 28 else x > 0) else x
                assert C.string_at(C.byref(C.c_float(y)), 4) == C.string_at(
                    C.byref(C.c_float(expected)), 4
                )
    result, bits, _ = execute(
        [1.0, 2.0, 3.0],
        1,
        [col, (2, -1, -1, -1), (35, 0, 1, -1)],
        [0, 4.0, 0],
        valid=[True, False, True],
    )
    assert bits == [True, False, True] and result[0] == result[2] == 4.0


test_additional_row_ops()


def test_stable_sort():
    from functools import cmp_to_key

    def run(dtype, values, valid, descending, nulls_last):
        n = len(values)
        offset = 5
        scalar = SCALARS[dtype]
        data = packed(values, offset) if dtype == 4 else (scalar * max(1, n))(*values)
        bitmap = packed(valid, offset)
        ids = (I * max(1, n))(*range(n))
        inputs = (Input * 2)(
            Input(ptr(data), ptr(bitmap), dtype, offset, 1),
            Input(ptr(ids), None, 3, 0, 0),
        )
        code = (I * 4)(200, 0, descending, nulls_last)
        literals = (I * 1)()
        steps = (Step * 1)(Step(0, 1, 0, 2, 0, 2))
        gathers = (I * 2)(0, 1)
        key = (scalar * max(1, n))()
        order = (I * max(1, n))()
        keybits, orderbits = packed([False] * n), packed([False] * n)
        outputs = (Output * 2)(
            Output(ptr(key), ptr(keybits), dtype, 0, -1, 0),
            Output(ptr(order), ptr(orderbits), 3, 1, -1, 0),
        )
        nt, st = (I * 1)(dtype), (I * 2)(dtype, 3)
        request = Request(
            2,
            n,
            0,
            2,
            2,
            4,
            1,
            1,
            2,
            2,
            -1,
            0,
            1,
            -1,
            ptr(inputs),
            ptr(code),
            ptr(literals),
            ptr(steps),
            ptr(gathers),
            ptr(outputs),
            ptr(nt),
            ptr(st),
        )
        memory, stats, err = Memory(), Stats(), P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
        assert stats.rows == n and stats.waits == 1
        assert (
            stats.memory.peak == memory.peak
            and stats.memory.launches == memory.launches
        )

        def compare(a, b):
            if valid[a] != valid[b]:
                return (-1 if valid[a] else 1) * (1 if nulls_last else -1)
            if not valid[a]:
                return 0
            x, y = values[a], values[b]
            if dtype == 1 and (math.isnan(x) or math.isnan(y)):
                c = 0 if math.isnan(x) and math.isnan(y) else 1 if math.isnan(x) else -1
                return c  # NaNs follow numeric values in either direction.
            else:
                c = (x > y) - (x < y)
            return -c if descending else c

        expected = sorted(range(n), key=cmp_to_key(compare))
        assert list(order)[:n] == expected, (
            dtype,
            n,
            descending,
            nulls_last,
            list(order)[:n],
            expected,
        )
        assert unpack(keybits, n) == [valid[i] for i in expected]
        assert all(unpack(orderbits, n))
        if dtype == 4:
            assert unpack(key, n) == [bool(values[i]) for i in expected]
        else:
            size = C.sizeof(scalar)
            for i, source in enumerate(expected):
                assert C.string_at(C.addressof(key) + i * size, size) == C.string_at(
                    C.addressof(data) + source * size, size
                ), (dtype, i, source)

    rng = random.Random(10001)
    for dtype, scalar in SCALARS.items():
        bits = C.sizeof(scalar) * 8
        if dtype == 1:
            from_bits = lambda x: C.cast(
                C.pointer(C.c_uint32(x)), C.POINTER(C.c_float)
            )[0]
            values = [
                0.0,
                -0.0,
                2**-149,
                -(2**-149),
                math.inf,
                -math.inf,
                math.nan,
            ] * 75
            values += [from_bits(rng.getrandbits(32)) for _ in range(80)]
        elif dtype == 4:
            values = [bool(rng.randrange(2)) for _ in range(605)]
        elif dtype in (2, 3, 5, 6):
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            values = [lo, hi] + [
                rng.randint(-min(33, hi), min(33, hi)) for _ in range(603)
            ]
        else:
            hi = (1 << bits) - 1
            values = [0, hi] + [rng.randrange(17) for _ in range(603)]
        for descending in (False, True):
            for nulls_last in (False, True):
                run(
                    dtype,
                    values,
                    [i % 7 != 3 for i in range(len(values))],
                    descending,
                    nulls_last,
                )
        for n in (0, 1, 7, 8, 255, 256, 257, 513):
            run(dtype, values[:n], [True] * n, False, True)
        run(dtype, values[:257], [False] * 257, True, True)


test_stable_sort()


def test_grouped_reductions():
    def run(dtype, n, key_count=2, unique=False):
        scalar = SCALARS[dtype]
        if dtype == 1:
            pool = [0.0, -0.0, 1.0, float("nan"), 1.401298464324817e-45]
            vals = [3.0, -0.0, 0.0, float("nan"), -1.401298464324817e-45]
        elif dtype == 4:
            pool, vals = [False, True], [True, False]
        elif dtype in (7, 8, 9, 10):
            high = (1 << (C.sizeof(scalar) * 8)) - 1
            pool, vals = [0, 1, high], [0, 1, high, 2]
        else:
            high = (1 << (C.sizeof(scalar) * 8 - 1)) - 1
            pool, vals = [0, 1, high, -high - 1], [-high - 1, high, 0, 2]
        key = [i if unique else pool[(i * 7) % len(pool)] for i in range(n)]
        value = [vals[(i // 3) % len(vals)] for i in range(n)]
        second = [i % 3 == 0 for i in range(n)]
        kv, vv, bv = (
            [i % 11 != 2 for i in range(n)],
            [i % 7 != 1 for i in range(n)],
            [i % 13 != 3 for i in range(n)],
        )
        ka = packed(key, 5) if dtype == 4 else (scalar * max(1, n))(*key)
        va = packed(value, 5) if dtype == 4 else (scalar * max(1, n))(*value)
        ba = packed(second, 5)
        kb, vb, bb = packed(kv, 5), packed(vv, 5), packed(bv, 5)
        inputs = (Input * 3)(
            Input(ptr(ka), ptr(kb), dtype, 5, 1),
            Input(ptr(ba), ptr(bb), 4, 5, 1),
            Input(ptr(va), ptr(vb), dtype, 5, 1),
        )
        slots = [dtype, 4, dtype, 3, 3, 3]
        nodes = [(201, key_count, 4, 5)] + (
            [(202, 0, 0, 0), (202, 1, 0, 0)] if key_count else []
        )
        nt = [3] + ([dtype, 4] if key_count else [])
        words = [0] * len(nodes)
        specs = [
            (2, 11, 3, 0),
            (2, 90, 3, 0),
            (2, 80, dtype, 0),
            (2, 81, dtype, 0),
            (1, 11, 3, 0),
        ]
        if dtype not in (1, 3, 4, 10):
            specs.append((2, 10, dtype if dtype in (2, 9) else 3, 3))
        outs = [(0, dtype), (1, 4)] if key_count else []
        for source, kind, outtype, minimum in specs:
            target = len(slots)
            slots.append(outtype)
            outs.append((target, outtype))
            nodes.append((203, source, target, kind))
            nt.append(outtype)
            words.append(minimum)
        stepnodes = len(nodes)
        # Preserve first appearance order after grouping.
        nodes.append((200, 5, 0, 0))
        nt.append(3)
        words.append(0)
        gathers = (I * len(outs))(*[slot for slot, t in outs])
        steps = (Step * 2)(
            Step(0, stepnodes, 3, 3, 0, 0), Step(stepnodes, 1, 0, 2, 0, len(outs))
        )
        code = (I * (4 * len(nodes)))(*[x for node in nodes for x in node])
        literals = (I * len(words))(*words)
        types = (I * len(nt))(*nt)
        st = (I * len(slots))(*slots)
        arrays = [
            packed([False] * max(1, n)) if t == 4 else (SCALARS[t] * max(1, n))()
            for _, t in outs
        ]
        bits = [packed([False] * max(1, n)) for _ in outs]
        output = (Output * len(outs))(
            *[
                Output(ptr(a), ptr(b), t, slot, -1, 0)
                for (slot, t), a, b in zip(outs, arrays, bits)
            ]
        )
        req = Request(
            2,
            n,
            0,
            len(slots),
            3,
            len(code),
            len(words),
            2,
            len(outs),
            len(outs),
            -1,
            0,
            1,
            -1,
            ptr(inputs),
            ptr(code),
            ptr(literals),
            ptr(steps),
            ptr(gathers),
            ptr(output),
            ptr(types),
            ptr(st),
        )
        groups = {}
        for i in range(n):
            raw = key[i] if kv[i] else None
            canon = "nan" if dtype == 1 and raw is not None and math.isnan(raw) else raw
            k = (canon, second[i] if bv[i] else None) if key_count else ()
            groups.setdefault(k, []).append(i)
        if not key_count and not n:
            groups[()] = []
        expected = []
        for rows in groups.values():
            row = rows[0] if rows else 0
            good = [i for i in rows if vv[i]]
            rank_value = lambda i: (
                (math.isnan(value[i]), value[i]) if dtype == 1 else value[i]
            )
            smallest = min(good, key=rank_value) if good else 0
            largest = max(good, key=rank_value) if good else 0
            record = [(key[row], kv[row]), (second[row], bv[row])] if key_count else []
            record += [
                (len(good), True),
                (len(rows), True),
                (value[smallest] if good else 0, bool(good)),
                (value[largest] if good else 0, bool(good)),
                (sum(bv[i] for i in rows), True),
            ]
            if len(specs) == 6:
                record.append((sum(value[i] for i in good), len(good) >= 3))
            expected.append(record)
        stats, memory, err = Stats(), Memory(), P()
        check(lib.dfm_estimate(C.byref(req), C.byref(memory), C.byref(err)), err)
        # Large-width sample sums can intentionally overflow the narrow result.
        overflow = False
        if len(specs) == 6 and outs[-1][1] in (2, 9):
            signed = outs[-1][1] == 2
            lo, hi = (-(2**31), 2**31 - 1) if signed else (0, 2**32 - 1)
            overflow = any(
                valid and not lo <= v <= hi
                for record in expected
                for v, valid in [record[-1]]
            )
        status = lib.dfm_execute(ctx, C.byref(req), C.byref(stats), C.byref(err))
        if overflow:
            assert status and b"overflow" in C.string_at(err)
            lib.dfm_free(err)
            return
        check(status, err)
        assert stats.rows == len(expected), (dtype, n, stats.rows, len(expected))
        assert stats.waits == 1 and stats.memory.shared == memory.shared
        assert stats.memory.launches == memory.launches, (
            stats.memory.launches,
            memory.launches,
        )
        for c, ((slot, t), a, b) in enumerate(zip(outs, arrays, bits)):
            actual = unpack(a, stats.rows) if t == 4 else list(a)[: stats.rows]
            validity = unpack(b, stats.rows)
            for row, record in enumerate(expected):
                value0, valid = record[c]
                assert validity[row] == valid, (dtype, n, c, row, validity[row], valid)
                if valid:
                    assert (
                        t == 1 and math.isnan(value0) and math.isnan(actual[row])
                    ) or actual[row] == value0, (dtype, n, c, row, actual[row], value0)
                    if t == 1 and value0 == 0:
                        assert C.string_at(
                            C.byref(C.c_float(actual[row])), 4
                        ) == C.string_at(C.byref(C.c_float(value0)), 4), (
                            dtype,
                            n,
                            c,
                            row,
                            actual[row],
                            value0,
                        )

    for t in range(1, 11):
        for n in (0, 1, 7, 257, 1025):
            run(t, n)
    run(5, 0, 0)
    run(5, 262401, 0)
    run(3, 1025, 2, True)


test_grouped_reductions()


def test_grouped_errors_preserve_row_fault_priority():
    n = 2
    keys = (C.c_int32 * n)(0, 0)
    bad = (C.c_float * n)(float("nan"), 0)
    sums = (C.c_int32 * n)(2**31 - 1, 1)
    inputs = (Input * 3)(
        Input(ptr(keys), None, 2, 0, 0),
        Input(ptr(bad), None, 1, 0, 0),
        Input(ptr(sums), None, 2, 0, 0),
    )
    programs = [
        (0, -1, -1, 1),
        (79, 0, -1, 1),
        (201, 1, 5, 6),
        (202, 0, 0, 0),
        (203, 3, 7, 10),
        (203, 2, 8, 10),
    ]
    code = (I * 24)(*[v for node in programs for v in node])
    words = (I * 6)()
    nt = (I * 6)(1, 2, 3, 2, 2, 2)
    st = (I * 9)(2, 1, 2, 2, 3, 3, 3, 2, 2)
    steps = (Step * 2)(Step(0, 2, 3, 0, 0, 0), Step(2, 4, 4, 3, 0, 0))
    arrays = [(C.c_int32 * n)(77, 77) for _ in range(2)]
    bits = [packed([True] * n) for _ in range(2)]
    outputs = (Output * 2)(
        *[
            Output(ptr(a), ptr(b), 2, slot, -1, 0)
            for a, b, slot in zip(arrays, bits, [7, 8])
        ]
    )
    req = Request(
        2,
        n,
        0,
        9,
        3,
        24,
        6,
        2,
        0,
        2,
        -1,
        0,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(words),
        ptr(steps),
        None,
        ptr(outputs),
        ptr(nt),
        ptr(st),
    )
    stats, err = Stats(), P()
    assert lib.dfm_execute(ctx, C.byref(req), C.byref(stats), C.byref(err))
    assert b"strict cast" in C.string_at(err), C.string_at(err)
    lib.dfm_free(err)
    assert all(list(a) == [77, 77] for a in arrays)
    assert all(unpack(b, n) == [True, True] for b in bits)


test_grouped_errors_preserve_row_fault_priority()

lib.dfm_context_release(ctx)
print(
    "Native Metal ABI, boundaries, filters, bitmaps, integer checks, reductions and cache: PASS"
)
