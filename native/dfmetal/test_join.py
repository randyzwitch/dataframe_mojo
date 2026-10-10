"""Native two-phase join ABI parity; run on Apple Silicon after build.sh."""
import ctypes as C
import math
import random
import struct
import sys

I, P = C.c_int64, C.c_void_p


class Input(C.Structure):
    _fields_ = [("values", P), ("validity", P)] + [
        (name, I) for name in ("dtype", "offset", "has_validity")
    ]


class JoinInput(C.Structure):
    _fields_ = [("input", Input), ("side", I)]


class Key(C.Structure):
    _fields_ = [("left", I), ("right", I)]


class Output(C.Structure):
    _fields_ = [("values", P), ("validity", P)] + [
        (name, I) for name in ("dtype", "left", "right", "prefer_right")
    ]


class Request(C.Structure):
    _fields_ = [
        (name, I)
        for name in (
            "abi",
            "left_rows",
            "right_rows",
            "inputs_count",
            "keys_count",
            "outputs_count",
            "how",
            "limit",
            "budget",
            "profiling",
        )
    ] + [(name, P) for name in ("inputs", "keys", "outputs")]


class Memory(C.Structure):
    _fields_ = [
        (name, I)
        for name in (
            "shared",
            "result",
            "peak",
            "upload",
            "launches",
            "blocks",
            "capacity",
            "packed",
        )
    ]


class Stats(C.Structure):
    _fields_ = [("memory", Memory)] + [
        (name, I)
        for name in (
            "rows",
            "download",
            "waits",
            "cache_hit",
            "compile_ns",
            "stage_ns",
            "submit_ns",
            "copy_ns",
            "gpu_ns",
        )
    ]


assert C.sizeof(JoinInput) == 48 and C.sizeof(Key) == 16
assert C.sizeof(Output) == 48 and C.sizeof(Request) == 104
assert Request.inputs.offset == 80 and C.sizeof(Stats) == 136
lib = C.CDLL(
    sys.argv[1] if len(sys.argv) > 1 else "build/dfmetal/libdfmetal.dylib"
)
lib.dfm_context_create.argtypes = [I, I, C.POINTER(P)]
lib.dfm_context_create.restype = P
lib.dfm_context_release.argtypes = [P]
lib.dfm_free.argtypes = [P]
lib.dfm_device_count.restype = I
lib.dfm_join_estimate.argtypes = [
    C.POINTER(Request),
    C.POINTER(Memory),
    C.POINTER(P),
]
lib.dfm_join_prepare.argtypes = [
    P,
    C.POINTER(Request),
    C.POINTER(P),
    C.POINTER(Stats),
    C.POINTER(P),
]
lib.dfm_join_finish.argtypes = [
    P,
    C.POINTER(Output),
    I,
    C.POINTER(Stats),
    C.POINTER(P),
]
lib.dfm_join_release.argtypes = [P]
SCALARS = {
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


def ptr(value):
    return C.cast(value, P)


def packed(values, offset=0):
    out = (C.c_uint8 * max(1, (len(values) + offset + 7) // 8))()
    for i, value in enumerate(values):
        if value:
            out[(i + offset) // 8] |= 1 << ((i + offset) % 8)
    return out


def checked(status, err):
    if status:
        message = C.string_at(err).decode() if err else "missing native error"
        lib.dfm_free(err)
        raise AssertionError(message)


def rejected(call, expected):
    err = P()
    assert call(err) != 0
    message = C.string_at(err)
    lib.dfm_free(err)
    assert expected in message, message


# Metadata validation must work without a device and never read storage.
inputs = (JoinInput * 2)(
    JoinInput(Input(None, None, 2, 5, 1), 0),
    JoinInput(Input(None, None, 2, 5, 1), 1),
)
keys = (Key * 1)(Key(0, 1))
outputs = (Output * 1)(Output(None, None, 2, 0, 1, 0))
request = Request(
    1, 7, 9, 2, 1, 1, 0, -1, -1, 0, ptr(inputs), ptr(keys), ptr(outputs)
)
memory, err = Memory(), P()
checked(
    lib.dfm_join_estimate(C.byref(request), C.byref(memory), C.byref(err)), err
)
assert memory.peak == memory.shared and memory.result == 0
assert memory.upload == 68
for field, value in (
    ("abi", 2),
    ("left_rows", -1),
    ("right_rows", 2**31),
    ("how", 7),
    ("limit", -2),
    ("budget", -2),
    ("profiling", 2),
):
    previous = getattr(request, field)
    setattr(request, field, value)
    rejected(
        lambda e: lib.dfm_join_estimate(
            C.byref(request), C.byref(memory), C.byref(e)
        ),
        b"Invalid",
    )
    setattr(request, field, previous)
inputs[1].input.dtype = 3
rejected(
    lambda e: lib.dfm_join_estimate(
        C.byref(request), C.byref(memory), C.byref(e)
    ),
    b"dtype mismatch",
)
inputs[1].input.dtype = 2
keys[0].left = 1
rejected(
    lambda e: lib.dfm_join_estimate(
        C.byref(request), C.byref(memory), C.byref(e)
    ),
    b"source index",
)
keys[0].left = 0
request.how = 4
outputs[0].right = -1
checked(
    lib.dfm_join_estimate(C.byref(request), C.byref(memory), C.byref(err)), err
)
outputs[0].right = 1
rejected(
    lambda e: lib.dfm_join_estimate(
        C.byref(request), C.byref(memory), C.byref(e)
    ),
    b"output",
)
print("Metal join pure ABI validation passed")
if lib.dfm_device_count() == 0:
    print("No Metal device; skipped join GPU fixtures")
    sys.exit(0)
err = P()
context = lib.dfm_context_create(0, 0, C.byref(err))
assert context, C.string_at(err)


def key_token(value):
    return "nan" if isinstance(value, float) and math.isnan(value) else value


def run(
    dtype,
    n,
    m,
    how,
    limit=-1,
    second=True,
    all_null=False,
    budget=-1,
    bad_finish=False,
    seed=17,
    validity_absent=False,
):
    rng = random.Random(seed)
    if dtype == 1:
        pattern = [
            -float("inf"),
            -0.0,
            0.0,
            1.0,
            float("inf"),
            float("nan"),
            C.c_float(1e-45).value,
            C.c_float(-1e-45).value,
            struct.unpack("<f", struct.pack("<I", 0x7FC00011))[0],
            struct.unpack("<f", struct.pack("<I", 0xFFC00077))[0],
        ]
    elif dtype == 4:
        pattern = [False, True]
    elif dtype in (7, 8, 9, 10):
        maximum = (1 << (8 * C.sizeof(SCALARS[dtype]))) - 1
        pattern = [0, 1, maximum, maximum - 1, 7, 31]
    else:
        bits = 8 * C.sizeof(SCALARS[dtype])
        pattern = [0, -1, 1, -(1 << (bits - 1)), (1 << (bits - 1)) - 1, 7, 31]
    left = [rng.choice(pattern) for _ in range(n)]
    right = [rng.choice(pattern) for _ in range(m)]
    lk, rk = [i % 3 for i in range(n)], [i % 3 for i in range(m)]
    lv = [not all_null and i % 11 != 2 for i in range(n)]
    rv = [not all_null and i % 13 != 5 for i in range(m)]
    lvalid, rvalid = [i % 17 != 7 for i in range(n)], [
        i % 19 != 8 for i in range(m)
    ]
    if validity_absent:
        lv, rv, lvalid, rvalid = [True] * n, [True] * m, [True] * n, [True] * m
    # Separate nullable payload columns exercise full/right padding and bitwise copies.
    owned, input_list, logical = [], [], []

    def source(values, valid, typ, side):
        data = packed(values, 5) if typ == 4 else (
            SCALARS[typ] * max(1, len(values))
        )(*values)
        bitmap = packed(valid, 5)
        owned.extend([data, bitmap])
        input_list.append(
            JoinInput(
                Input(
                    ptr(data),
                    None if validity_absent else ptr(bitmap),
                    typ,
                    5,
                    int(not validity_absent),
                ),
                side,
            )
        )
        # Canonicalize Float32 before building the independent expected result.
        logical.append(
            (
                list(values) if typ
                == 4 else [data[i] for i in range(len(values))],
                valid,
                data,
                typ,
            )
        )

    source(left, lv, dtype, 0)  # 0: left key
    source(right, rv, dtype, 1)  # 1: right key
    source(lk, [True] * n, 5, 0)  # 2: left second key
    source(rk, [True] * m, 5, 1)  # 3: right second key
    source(list(range(n)), [True] * n, 3, 0)  # 4: left row identity
    source(list(range(m)), [True] * m, 3, 1)  # 5: right row identity
    source(left, lvalid, dtype, 0)  # 6: left nullable payload
    source(right, rvalid, dtype, 1)  # 7: right nullable payload
    source([i % 3 == 0 for i in range(n)], lvalid, 4, 0)  # 8: packed bool
    left, right = logical[0][0], logical[1][0]

    def token(side, i):
        if not (lv if side == 0 else rv)[i]:
            return None
        return (
            key_token((left if side == 0 else right)[i]),
            (lk if side == 0 else rk)[i] if second else 0,
        )

    build_side = 0 if how == 2 else 1
    build_n, probe_n = (n, m) if how == 2 else (m, n)
    lookup = {}
    for row in range(build_n):
        t = token(build_side, row)
        if t is not None:
            lookup.setdefault(t, []).append(row)
    expected, matched = [], set()
    for row in range(probe_n):
        t = token(1 - build_side, row)
        matches = list(range(build_n)) if how == 6 else (
            lookup.get(t, []) if t is not None else []
        )
        if how == 4:
            if matches:
                expected.append((row, None))
        elif how == 5:
            if not matches:
                expected.append((row, None))
        else:
            for other in matches:
                expected.append((other, row) if how == 2 else (row, other))
                matched.add(other)
            if not matches and how in (1, 2, 3):
                expected.append((None, row) if how == 2 else (row, None))
        if limit >= 0 and len(expected) >= limit:
            break
    if how == 3:
        expected += [(None, row) for row in range(m) if row not in matched]
    if limit >= 0:
        expected = expected[:limit]
    specs = [
        (dtype, 0, 1, int(how == 2)),
        (3, 4, -1, 0),
        (3, -1, 5, 0),
        (dtype, 6, -1, 0),
        (dtype, -1, 7, 0),
        (4, 8, -1, 0),
    ]
    if how in (4, 5):
        specs = [
            (dtype, 0, -1, 0),
            (3, 4, -1, 0),
            (dtype, 6, -1, 0),
            (4, 8, -1, 0),
        ]
    ia = (JoinInput * len(input_list))(*input_list)
    key_list = [] if how == 6 else [Key(0, 1)] + ([Key(2, 3)] if second else [])
    ka = (Key * max(1, len(key_list)))(*key_list)
    oa = (Output * len(specs))(*(Output(None, None, *spec) for spec in specs))
    req = Request(
        1,
        n,
        m,
        len(ia),
        len(key_list),
        len(oa),
        how,
        limit,
        -1 if budget == -2 else budget,
        1,
        ptr(ia),
        ptr(ka),
        ptr(oa),
    )
    stats, err, session = Stats(), P(), P()
    if budget == -2:
        memory, err = Memory(), P()
        checked(
            lib.dfm_join_estimate(C.byref(req), C.byref(memory), C.byref(err)),
            err,
        )
        req.budget = memory.shared
    if budget in (0, -2):
        rejected(
            lambda e: lib.dfm_join_prepare(
                context,
                C.byref(req),
                C.byref(session),
                C.byref(stats),
                C.byref(e),
            ),
            b"memory budget",
        )
        assert not session
        return
    checked(
        lib.dfm_join_prepare(
            context,
            C.byref(req),
            C.byref(session),
            C.byref(stats),
            C.byref(err),
        ),
        err,
    )
    try:
        assert stats.rows == len(expected), (
            dtype,
            n,
            m,
            how,
            stats.rows,
            len(expected),
        )
        assert (
            stats.waits == 1
            and stats.memory.peak == stats.memory.shared + stats.memory.result
        )
        # Prepare owns all metadata, including descriptors from temporary requests.
        ia[0].side = 99
        if key_list:
            ka[0].left = 999
        req.how, req.limit = 99, 0
        output_owned = []
        for o in oa:
            size = (
                stats.rows + 7
            ) // 8 if o.dtype == 4 else stats.rows * C.sizeof(SCALARS[o.dtype])
            data, bits = (C.c_uint8 * max(1, size))(), (
                C.c_uint8 * max(1, (stats.rows + 7) // 8)
            )()
            C.memset(data, 0xA5, C.sizeof(data))
            C.memset(bits, 0xA5, C.sizeof(bits))
            o.values, o.validity = ptr(data), ptr(bits)
            output_owned.append((data, bits))
        if bad_finish:
            oa[0].prefer_right = 1 - oa[0].prefer_right
            rejected(
                lambda e: lib.dfm_join_finish(
                    session, oa, len(oa), C.byref(stats), C.byref(e)
                ),
                b"metadata mismatch",
            )
            assert all(
                bytes(a) == b"\xA5" * len(a) and bytes(b) == b"\xA5" * len(b)
                for a, b in output_owned
            )
            rejected(
                lambda e: lib.dfm_join_finish(
                    session, oa, len(oa), C.byref(stats), C.byref(e)
                ),
                b"already consumed",
            )
            return
        checked(
            lib.dfm_join_finish(
                session, oa, len(oa), C.byref(stats), C.byref(err)
            ),
            err,
        )
        assert stats.waits == 2
        assert stats.download == 8 + stats.memory.result
        expected_upload = 0
        for values, valid, data, typ in logical:
            expected_upload += (
                len(values) + 5 + 7
            ) // 8 if typ == 4 and values else (
                len(values) * C.sizeof(SCALARS[typ]) if typ != 4 else 0
            )
            if not validity_absent and values:
                expected_upload += (len(values) + 5 + 7) // 8
        assert stats.memory.upload == expected_upload
        assert stats.memory.capacity == max(1, stats.rows)
        for c, (o, (data, bitmap)) in enumerate(zip(oa, output_owned)):
            for row, (li, ri) in enumerate(expected):
                if o.prefer_right and ri is not None:
                    source_index, source_row = o.right, ri
                elif o.left >= 0 and li is not None:
                    source_index, source_row = o.left, li
                elif o.right >= 0 and ri is not None:
                    source_index, source_row = o.right, ri
                else:
                    source_index, source_row = -1, None
                valid = (
                    source_index >= 0 and logical[source_index][1][source_row]
                )
                actual_valid = bool(bitmap[row // 8] & (1 << (row % 8)))
                assert actual_valid == valid, (dtype, how, c, row, li, ri)
                if o.dtype == 4:
                    value = bool(data[row // 8] & (1 << (row % 8)))
                    wanted = bool(
                        logical[source_index][0][source_row]
                    ) if valid else False
                    assert value == wanted, (dtype, how, c, row)
                else:
                    width = C.sizeof(SCALARS[o.dtype])
                    got = bytes(data[row * width : (row + 1) * width])
                    wanted = C.string_at(
                        C.addressof(logical[source_index][2])
                        + source_row * width,
                        width,
                    ) if valid else bytes(width)
                    assert got == wanted, (
                        dtype,
                        how,
                        c,
                        row,
                        li,
                        ri,
                        got,
                        wanted,
                    )
        rejected(
            lambda e: lib.dfm_join_finish(
                session, oa, len(oa), C.byref(stats), C.byref(e)
            ),
            b"already consumed",
        )
    finally:
        lib.dfm_join_release(session)


try:
    for dtype in SCALARS:
        for how in range(7):
            for n, m in [(0, 0), (0, 9), (7, 0), (17, 23), (521, 263)]:
                run(dtype, n, m, how)
            run(dtype, 17, 23, how, all_null=True)
            run(dtype, 17, 23, how, validity_absent=True)
            for limit in [0, 1, 29]:
                run(dtype, 33, 41, how, limit=limit, second=False)
    # More than 1024 scan blocks and fanout much larger than the head result.
    run(10, 262401, 17, 3, limit=400000)
    run(1, 20000, 20000, 0, limit=257, second=False)
    # Total fanout exceeds Int32.MAX; the bounded result still fits the ABI.
    run(4, 50000, 50000, 6, limit=513)
    run(2, 17, 23, 0, budget=0)
    run(2, 17, 23, 0, budget=-2)
    run(2, 17, 23, 3, bad_finish=True)
    print(
        "Metal join GPU parity passed: seven joins, ten types, nulls, stable order, slicing, large scans, limits, lifecycle"
    )
finally:
    lib.dfm_context_release(context)
