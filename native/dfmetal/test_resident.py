"""Native resident DAG ownership, parity, traffic and whole-query budget checks."""
import ctypes as C
import math
import sys

I, P = C.c_int64, C.c_void_p


def structure(name, pointers, integers):
    return type(
        name,
        (C.Structure,),
        {
            "_fields_": [(field, P) for field in pointers]
            + [(field, I) for field in integers]
        },
    )


Input = structure("Input", ["values", "validity"], ["dtype", "offset", "valid"])
Output = structure(
    "Output", ["values", "validity"], ["dtype", "slot", "reduction", "minimum"]
)
Step = structure(
    "Step", [], ["start", "nodes", "slot", "filter", "gather", "count"]
)
ResidentInput = structure("ResidentInput", ["table"], ["column"])
ResidentInfo = structure("ResidentInfo", [], ["rows", "columns", "bytes"])
QueryInfo = structure("QueryInfo", [], ["live", "peak", "budget"])
Memory = structure(
    "Memory",
    [],
    [
        "shared",
        "result",
        "peak",
        "upload",
        "launches",
        "blocks",
        "capacity",
        "packed",
    ],
)


class Request(C.Structure):
    _fields_ = [
        (name, I)
        for name in (
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
        )
    ] + [
        (name, P)
        for name in (
            "inputs",
            "code",
            "literals",
            "steps",
            "gathers",
            "outputs",
            "node_types",
            "slot_types",
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


class JoinInput(C.Structure):
    _fields_ = [("input", Input), ("side", I)]


JoinKey = structure("JoinKey", [], ["left", "right"])
JoinOutput = structure(
    "JoinOutput", ["values", "validity"], ["dtype", "left", "right", "prefer"]
)


class JoinRequest(C.Structure):
    _fields_ = [
        (name, I)
        for name in (
            "abi",
            "left_rows",
            "right_rows",
            "input_count",
            "key_count",
            "output_count",
            "how",
            "limit",
            "budget",
            "profiling",
        )
    ] + [(name, P) for name in ("inputs", "keys", "outputs")]


lib = C.CDLL(
    sys.argv[1] if len(sys.argv) > 1 else "build/dfmetal/libdfmetal.dylib"
)
lib.dfm_context_create.argtypes = [I, I, C.POINTER(P)]
lib.dfm_context_create.restype = P
lib.dfm_context_release.argtypes = [P]
lib.dfm_query_create.argtypes = [P, I, C.POINTER(P)]
lib.dfm_query_create.restype = P
lib.dfm_query_release.argtypes = [P]
lib.dfm_resident_release.argtypes = [P]
lib.dfm_resident_info.argtypes = [P, C.POINTER(ResidentInfo), C.POINTER(P)]
lib.dfm_query_info.argtypes = [P, C.POINTER(QueryInfo), C.POINTER(P)]
lib.dfm_resident_rows.argtypes = [
    P,
    C.POINTER(Request),
    P,
    I,
    I,
    C.POINTER(P),
    C.POINTER(Stats),
    C.POINTER(P),
]
lib.dfm_resident_join.argtypes = [
    P,
    C.POINTER(JoinRequest),
    P,
    I,
    I,
    C.POINTER(P),
    C.POINTER(Stats),
    C.POINTER(P),
]
lib.dfm_resident_export.argtypes = [P, P, I, C.POINTER(Stats), C.POINTER(P)]
lib.dfm_estimate.argtypes = [
    C.POINTER(Request),
    C.POINTER(Memory),
    C.POINTER(P),
]
lib.dfm_free.argtypes = [P]
lib.dfm_device_count.restype = I
lib.dfm_resident_abi_version.restype = I
assert lib.dfm_resident_abi_version() == 1
assert C.sizeof(Request) == 176 and C.sizeof(ResidentInput) == 16
assert C.sizeof(JoinRequest) == 104 and C.sizeof(Stats) == 136


def ptr(value):
    return C.cast(value, P)


def check(status, err):
    if status:
        message = C.string_at(err).decode() if err else "missing native error"
        lib.dfm_free(err)
        raise RuntimeError(message)


def bits(values, offset=0):
    data = (C.c_uint8 * max(1, (len(values) + offset + 7) // 8))()
    for i, value in enumerate(values):
        if value:
            data[(i + offset) // 8] |= 1 << ((i + offset) % 8)
    return data


SCALAR = {
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


class Rows:
    def __init__(
        self,
        columns,
        *,
        rows=None,
        outputs=None,
        nodes=(),
        steps=(),
        slot_types=None,
        literals=None,
        reductions=False,
        offset=0
    ):
        # A column is (dtype, values, validity), or (dtype, table, column_index).
        self.owners, inputs, refs = [], [], []
        for dtype, source, valid in columns:
            if isinstance(source, list):
                rows = len(source) if rows is None else rows
                data = bits(source, offset) if dtype == 4 else (
                    SCALAR[dtype] * max(1, len(source))
                )(*source)
                bitmap = bits(valid, offset) if valid is not None else None
                self.owners.extend([data, bitmap])
                inputs.append(
                    Input(
                        ptr(data),
                        ptr(bitmap) if bitmap is not None else None,
                        dtype,
                        offset,
                        valid is not None,
                    )
                )
                refs.append(ResidentInput(None, 0))
            else:
                inputs.append(Input(None, None, dtype, 0, 0))
                refs.append(ResidentInput(source, valid))
        self.inputs = (Input * len(inputs))(*inputs)
        self.refs = (ResidentInput * len(refs))(*refs)
        slot_types = slot_types or [column[0] for column in columns]
        self.types = (I * len(slot_types))(*slot_types)
        self.code = (I * (len(nodes) * 4))(*[x for node in nodes for x in node])
        self.literals = (C.c_uint64 * len(nodes))(
            *(literals or [0] * len(nodes))
        )
        self.node_types = (I * len(nodes))(
            *[slot_types[step[2]] for step in steps for _ in range(step[1])]
        )
        self.steps = (Step * len(steps))(*(Step(*step) for step in steps))
        outputs = outputs or [(t, i, -1, 0) for i, t in enumerate(slot_types)]
        self.outputs = (Output * len(outputs))(
            *(Output(None, None, *o) for o in outputs)
        )
        self.request = Request(
            2,
            rows,
            0,
            len(slot_types),
            len(inputs),
            len(self.code),
            len(nodes),
            len(steps),
            0,
            len(outputs),
            -1,
            reductions,
            0,
            -1,
            ptr(self.inputs),
            ptr(self.code),
            ptr(self.literals),
            ptr(self.steps),
            None,
            ptr(self.outputs),
            ptr(self.node_types),
            ptr(self.types),
        )

    def run(self, query, final=False, failure=None):
        result, stats, err = P(), Stats(), P()
        status = lib.dfm_resident_rows(
            query,
            C.byref(self.request),
            self.refs,
            len(self.refs),
            final,
            C.byref(result),
            C.byref(stats),
            C.byref(err),
        )
        if failure:
            assert status and not result
            message = C.string_at(err).decode()
            lib.dfm_free(err)
            assert failure in message, message
            return None, stats
        check(status, err)
        assert result and stats.waits == 1
        assert stats.copy_ns == 0
        return result, stats


def query(context, budget=-1):
    err = P()
    result = lib.dfm_query_create(context, budget, C.byref(err))
    check(not result, err)
    return result


def information(handle, query=False):
    result, err = (QueryInfo() if query else ResidentInfo()), P()
    callback = lib.dfm_query_info if query else lib.dfm_resident_info
    check(callback(handle, C.byref(result), C.byref(err)), err)
    return result


def export(table, types, invalid=False):
    n = information(table).rows
    arrays, bitmaps, outputs = [], [], []
    for i, dtype in enumerate(types):
        data = (SCALAR[dtype] * max(1, n))() if dtype != 4 else bits(
            [False] * n
        )
        bitmap = bits([False] * n)
        C.memset(data, 0xA5, C.sizeof(data))
        C.memset(bitmap, 0xA5, C.sizeof(bitmap))
        arrays.append(data)
        bitmaps.append(bitmap)
        outputs.append(Output(ptr(data), ptr(bitmap), dtype, i, -1, 0))
    descriptors = (Output * len(outputs))(*outputs)
    before = [bytes(a) for a in arrays + bitmaps]
    if invalid:
        descriptors[-1].dtype = 10 if types[-1] != 10 else 2
    stats, err = Stats(), P()
    status = lib.dfm_resident_export(
        table, descriptors, len(outputs), C.byref(stats), C.byref(err)
    )
    if invalid:
        assert status and [bytes(a) for a in arrays + bitmaps] == before
        assert b"metadata mismatch" in C.string_at(err)
        lib.dfm_free(err)
        return
    check(status, err)
    values = [
        [bool(a[i // 8] & (1 << (i % 8))) for i in range(n)] if dtype
        == 4 else list(a)[:n]
        for a, dtype in zip(arrays, types)
    ]
    validity = [
        [bool(b[i // 8] & (1 << (i % 8))) for i in range(n)] for b in bitmaps
    ]
    return values, validity, stats


def joined(q, left, right, n, m, how=0, final=False):
    inputs = (JoinInput * 4)(
        *(JoinInput(Input(None, None, 2, 0, 0), i // 2) for i in range(4))
    )
    refs = (ResidentInput * 4)(
        ResidentInput(left, 0),
        ResidentInput(left, 1),
        ResidentInput(right, 0),
        ResidentInput(right, 1),
    )
    keys = (JoinKey * 1)(JoinKey(0, 2))
    outputs = (JoinOutput * 3)(
        JoinOutput(None, None, 2, 0, 2, 0),
        JoinOutput(None, None, 2, 1, -1, 0),
        JoinOutput(None, None, 2, -1, 3, 0),
    )
    request = JoinRequest(
        1, n, m, 4, 1, 3, how, -1, -1, 0, ptr(inputs), ptr(keys), ptr(outputs)
    )
    table, stats, err = P(), Stats(), P()
    check(
        lib.dfm_resident_join(
            q,
            C.byref(request),
            refs,
            4,
            final,
            C.byref(table),
            C.byref(stats),
            C.byref(err),
        ),
        err,
    )
    assert (
        stats.memory.upload == 0 and stats.download == 8 and stats.copy_ns == 0
    )
    assert stats.waits == 2
    return table, stats


if lib.dfm_device_count() == 0:
    print("Metal resident ABI layouts passed; GPU unavailable")
    sys.exit(0)

err = P()
context = lib.dfm_context_create(0, 0, C.byref(err))
check(not context, err)
q = query(context)
left, _ = Rows(
    [
        (2, [1, 2, 2, 9, 4], [True, True, True, False, True]),
        (2, [10, 20, 21, 90, 40], None),
    ]
).run(q)
right, _ = Rows([(2, [2, 2, 3], None), (2, [200, 201, 300], None)]).run(q)
table, _ = joined(q, left, right, 5, 3)
assert information(table).rows == 4
# A second consumer must see the same immutable source, including null keys.
outer, _ = joined(q, left, right, 5, 3, how=3, final=True)
values, valid, _ = export(outer, [2, 2, 2])
assert values[0] == [1, 2, 2, 2, 2, 0, 4, 3]
assert valid[0] == [True] * 5 + [False, True, True]
lib.dfm_resident_release(outer)
lib.dfm_resident_release(left)
lib.dfm_resident_release(right)
post = Rows(
    [(2, table, i) for i in range(3)],
    rows=4,
    outputs=[(2, 0, -1, 0), (2, 3, -1, 0)],
    nodes=[(0, -1, -1, 1), (0, -1, -1, 2), (5, 0, 1, -1)],
    steps=[(0, 3, 3, 0, 0, 0)],
    slot_types=[2] * 4,
)
result, stats = post.run(q, final=True)
assert stats.memory.upload == 24  # Literal descriptors, no column payload.
lib.dfm_resident_release(table)
export(result, [2, 2], invalid=True)
values, valid, stats = export(result, [2, 2])
assert values == [[2] * 4, [220, 221, 221, 222]] and all(all(v) for v in valid)
assert stats.download == 8 + 2 * (4 * 4 + 1)
lib.dfm_resident_release(result)
assert information(q, query=True).live == 0

# All native types, non-byte-aligned CPU sources, empty/all-null inputs, exact
# UInt64 payloads, Float32 signed zeros, NaNs and subnormal bit patterns.
for dtype in range(1, 11):
    for n in (0, 1, 9, 257):
        source = [i % 17 for i in range(n)]
        if dtype == 1:
            source = (
                [0.0, -0.0, math.nan, math.inf, -math.inf, 1e-40]
                * ((n + 5) // 6)
            )[:n]
        if dtype == 10:
            source = [(1 << 64) - 1 - i for i in range(n)]
        if dtype == 4:
            source = [i % 2 for i in range(n)]
        for all_null in (False, True):
            valid = [False if all_null else i % 3 != 0 for i in range(n)]
            first, _ = Rows([(dtype, source, valid)], offset=3).run(q)
            second, stats = Rows([(dtype, first, 0)], rows=n).run(q, final=True)
            assert stats.memory.upload == 0
            lib.dfm_resident_release(first)
            actual, bitmap, _ = export(second, [dtype])
            assert bitmap[0] == valid
            for i in range(n):
                if valid[i]:
                    expected = SCALAR[dtype](source[i]).value
                    assert (
                        dtype == 1
                        and math.isnan(expected)
                        and math.isnan(actual[0][i])
                    ) or actual[0][i] == expected
                    if dtype == 1 and expected == 0:
                        assert bytes(C.c_float(actual[0][i])) == bytes(
                            C.c_float(expected)
                        )
            lib.dfm_resident_release(second)

# Scalar reductions can feed another native region without host payload copies.
for kind, dtype, expected in (
    (10, 2, 6),
    (11, 3, 3),
    (90, 3, 4),
    (80, 2, 1),
    (81, 2, 3),
):
    first, _ = Rows(
        [(2, [1, 2, 3, 99], [True, True, True, False])],
        outputs=[(dtype, 0, kind, 1)],
        reductions=True,
    ).run(q)
    second, _ = Rows([(dtype, first, 0)], rows=1).run(q, final=True)
    lib.dfm_resident_release(first)
    values, valid, _ = export(second, [dtype])
    assert values == [[expected]] and valid == [[True]]
    lib.dfm_resident_release(second)

# Whole-query budget includes retained siblings. Releasing a sibling restores
# capacity, and rejection never creates a table or leaks an allocation owner.
cpu = Rows([(2, list(range(521)), None)])
first, stats = cpu.run(q)
budget = stats.memory.peak
lib.dfm_resident_release(first)
bounded = query(context, budget)
first, _ = cpu.run(bounded)
cpu.run(bounded, failure="memory budget")
assert information(bounded, query=True).live == information(first).bytes
lib.dfm_resident_release(first)
first, _ = cpu.run(bounded)
lib.dfm_resident_release(first)
assert information(bounded, query=True).live == 0
lib.dfm_query_release(bounded)

# Reject foreign scopes and dtype/row/column mismatches before submission.
first, _ = cpu.run(q)
other = query(context)
Rows([(2, first, 0)], rows=521).run(other, failure="mismatch")
for dtype, n, column in ((3, 521, 0), (2, 520, 0), (2, 521, 1)):
    Rows([(dtype, first, column)], rows=n).run(q, failure="mismatch")
lib.dfm_query_release(other)
lib.dfm_resident_release(first)

# Failed checked arithmetic drains work and does not poison the next consumer.
Rows(
    [(2, [2147483647] * 9, None)],
    nodes=[(0, -1, -1, 0), (1, -1, -1, -1), (5, 0, 1, -1)],
    literals=[0, 1, 0],
    steps=[(0, 3, 1, 0, 0, 0)],
    slot_types=[2, 2],
    outputs=[(2, 1, -1, 0)],
).run(q, failure="overflow")
assert information(q, query=True).live == 0
last, _ = Rows([(2, [42], None)]).run(q, final=True)
# Tables own their query and context even after both caller handles are dropped.
lib.dfm_query_release(q)
lib.dfm_context_release(context)
assert export(last, [2])[0] == [[42]]
lib.dfm_resident_release(last)
print(
    "Metal resident DAG parity, traffic, budgets, failure recovery and ownership: PASS"
)
