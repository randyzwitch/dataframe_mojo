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


class ProbeInfo(C.Structure):
    _fields_ = [(x, I) for x in ("registry", "recommended", "allocated", "unified", "cached", "reclaimable", "max_buffer")] + [("host", C.c_char_p)]


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
    C.sizeof(Request) == 176
    and C.sizeof(Memory) == 64
    and C.sizeof(Stats) == 136
)
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
    check(
        lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err
    )
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
        check(
            lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)),
            err,
        )
        assert memory.result == 4 * max(1, n) + (max(1, n) + 7) // 8
        expected_launches = 8 + (max(1, n) - 1).bit_length()
        if n >= 4096:
            passes = ((2 * n - 1).bit_length() + 3) // 4
            expected_launches += 6 + 3 * passes
        assert memory.launches == expected_launches
    code[0] = 209
    for workers in (1, 8, 2**31 - 1):
        literals[0] = workers
        err = P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for workers in (0, -1, 2**31):
        literals[0] = workers
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"window worker metadata" in C.string_at(err)
        lib.dfm_free(err)
    code[0], literals[0] = 201, 0
    for index, value in [(2, 0), (3, 2), (9, 1), (10, 3), (11, 11)]:
        before = code[index]
        code[index] = value
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)
        code[index] = before

    literals[2] = -1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    lib.dfm_free(err)
    literals[2] = 0


    # Decimal means carry an integer scale in the grouped literal payload.
    # Validate this path even on CI machines without a Metal device.
    code[11], nt[2], st[4], outputs[0].dtype = 212, 11, 11, 11
    for dtype, maximum in ((2, 9), (3, 18)):
        inputs[0].dtype = st[0] = nt[1] = dtype
        for scale in (0, maximum):
            literals[2] = scale
            err = P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        for scale in (-1, maximum + 1):
            literals[2] = scale
            err = P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"decimal mean scale" in C.string_at(err)
            lib.dfm_free(err)

    # Group outputs use the same packed metadata in the node literal.
    inputs[0].dtype = st[0] = nt[1] = 12
    request.rows = 131072
    metadata = 1 | (38 << 32) | (2 << 38) | (8 << 44)
    for opcode, result_type in ((215, 12), (216, 11)):
        code[11] = opcode
        nt[2] = st[4] = outputs[0].dtype = result_type
        literals[2] = metadata
        err = P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for invalid in (0, (39 << 32) | (8 << 44), metadata | (1 << 60), -1):
        literals[2] = invalid
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Decimal128 reduction metadata" in C.string_at(err)
        lib.dfm_free(err)


check_group_preflight_without_device()


def check_additional_reduction_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 1, 0, 0))
    slots = (I * 1)(1)
    outputs = (Output * 1)(Output(None, None, 9, 0, 95, 0))
    request = Request(
        2,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
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
    for op, outtype in [(83, 1), (84, 1), (93, 3), (95, 9), (96, 9)]:
        outputs[0].reduction, outputs[0].dtype = op, outtype
        memory, err = Memory(), P()
        check(
            lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)),
            err,
        )
        assert memory.result == ({1: 4, 3: 8, 9: 4}[outtype]) + 1
    outputs[0].dtype = 3
    outputs[0].reduction = 95
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    lib.dfm_free(err)
    inputs[0].dtype = slots[0] = 4
    outputs[0].dtype = 4
    for op in (91, 92):
        outputs[0].reduction = op
        for ignore in (0, 1):
            outputs[0].min_count = ignore
            err = P()
            check(
                lib.dfm_estimate(
                    C.byref(request), C.byref(memory), C.byref(err)
                ),
                err,
            )
        outputs[0].min_count = 2
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)


check_additional_reduction_preflight_without_device()


def check_predicate_reduction_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 2, 0, 0))
    code = (I * 16)(0, -1, -1, 0, 1, -1, -1, -1,
                     8, 0, 1, -1, 0, -1, -1, 0)
    literals = (I * 4)()
    steps = (Step * 2)(Step(0, 3, 1, 1, 0, 1), Step(3, 1, 2, 0, 1, 0))
    gathers = (I * 1)(0)
    outputs = (Output * 1)(Output(None, None, 3, 2, 11, 0))
    request = Request(2, 4097, 2, 3, 1, 16, 4, 2, 1, 1, -1, 1, 0, -1,
        ptr(inputs), ptr(code), ptr(literals), ptr(steps), ptr(gathers), ptr(outputs))
    lib.dfm_plan_features.argtypes = [C.POINTER(Request), C.POINTER(P)]
    lib.dfm_plan_features.restype = I
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.launches == 5
    assert lib.dfm_plan_features(C.byref(request), C.byref(err)) == 1
    request.reductions = 0
    outputs[0].dtype, outputs[0].reduction = 2, -1
    rows = Memory()
    check(lib.dfm_estimate(C.byref(request), C.byref(rows), C.byref(err)), err)
    assert lib.dfm_plan_features(C.byref(request), C.byref(err)) == 0
    assert rows.launches == 7 and memory.shared < rows.shared


check_predicate_reduction_preflight_without_device()

def check_sqrt_preflight_without_device():
    # Descriptor validation must not depend on a Metal device being exposed.
    inputs = (Input * 1)(Input(None, None, 11, 0, 0))
    code = (I * 8)(0, -1, -1, 0, 52, 0, -1, -1)
    literals = (I * 2)()
    steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 11, 1, -1, 0))
    nt, st = (I * 2)(11, 11), (I * 2)(11, 11)
    request = Request(2, 0, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nt), ptr(st))
    memory, err = Memory(), P()
    for dtype in (11, 3, 10):
        inputs[0].dtype = nt[0] = st[0] = dtype
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    inputs[0].dtype = nt[0] = st[0] = 4
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"Invalid Metal square root dtype" in C.string_at(err)
    lib.dfm_free(err)
    inputs[0].dtype = nt[0] = st[0] = 1
    outputs[0].dtype = nt[1] = st[1] = 1
    err = P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    outputs[0].dtype = nt[1] = st[1] = 3
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"Invalid Metal square root dtype" in C.string_at(err)
    lib.dfm_free(err)


check_sqrt_preflight_without_device()

def check_float32_floor_division_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 1, 0, 0))
    code = (I * 8)(0, -1, -1, 0, 25, 0, 0, -1)
    literals = (I * 2)()
    steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 1, 1, -1, 0))
    nt, st = (I * 2)(1, 1), (I * 2)(1, 1)
    request = Request(2, 0, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nt), ptr(st))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    code[4] = 26
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for opcode in (27,):
        code[4] = opcode
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Metal floating operator requires CPU execution" in C.string_at(err)
        lib.dfm_free(err)


check_float32_floor_division_preflight_without_device()

def check_rank_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 1, 0, 0))
    code = (I * 4)(207, 0, 1, 0)
    literals = (I * 1)()
    steps = (Step * 1)(Step(0, 1, 1, 4, 0, 0))
    outputs = (Output * 1)(Output(None, None, 11, 1, -1, 0))
    nt, st = (I * 1)(1), (I * 2)(1, 11)
    request = Request(2, 0, 0, 2, 1, 4, 1, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nt), ptr(st))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for at, invalid in ((0, 0), (1, 1), (2, 2), (3, 5)):
        original = code[at]
        code[at] = invalid
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Metal rank" in C.string_at(err)
        lib.dfm_free(err)
        code[at] = original


check_rank_preflight_without_device()

def check_decimal128_preflight_without_device():
    def reduction_metadata(min_count=0, precision=38, scale=2, workers=8):
        return (
            min_count
            | (precision << 32)
            | (scale << 38)
            | (workers << 44)
        )

    inputs = (Input * 1)(Input(None, None, 3, 0, 0))
    outputs = (Output * 2)(Output(None, None, 12, 0, 208, 0),
                           Output(None, None, 3, 0, 11, 0))
    slots = (I * 1)(3)
    request = Request(2, (1 << 31) - 1, 0, 1, 1, 0, 0, 0, 0, 2,
                      -1, 1, 0, -1, ptr(inputs), None, None, None,
                      None, ptr(outputs), None, ptr(slots))
    # Typed requests require a node-types pointer even when there is no code.
    nodes = (I * 1)(3)
    request.node_types = ptr(nodes)
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result == 26 and memory.upload == 8 * request.rows
    inputs[0].dtype = slots[0] = 1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"unsupported accumulator" in C.string_at(err)
    lib.dfm_free(err)

    # Decimal128 SUM and MEAN carry the logical decimal contract and CPU
    # worker count in one signed-positive metadata word.
    inputs = (Input * 1)(Input(None, None, 12, 0, 0))
    metadata = reduction_metadata(min_count=1)
    outputs = (Output * 2)(
        Output(None, None, 12, 0, 215, metadata),
        Output(None, None, 11, 0, 216, metadata),
    )
    nodes, slots = (I * 1)(12), (I * 1)(12)
    request = Request(
        2, 131072, 0, 1, 1, 0, 0, 0, 0, 2, -1, 1, 0, -1,
        ptr(inputs), None, None, None, None, ptr(outputs), ptr(nodes), ptr(slots)
    )
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result == 26
    for invalid in (
        reduction_metadata(precision=0),
        reduction_metadata(precision=39),
        reduction_metadata(precision=2, scale=3),
        reduction_metadata(workers=0),
        reduction_metadata() | (1 << 60),
    ):
        outputs[0].min_count = invalid
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Decimal128 reduction metadata" in C.string_at(err)
        lib.dfm_free(err)
    outputs[0].min_count = -1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"Invalid Metal output descriptor" in C.string_at(err)
    lib.dfm_free(err)
    outputs[0].min_count = metadata
    outputs[0].dtype = 11
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"Invalid Metal reduction dtype" in C.string_at(err)
    lib.dfm_free(err)

    # Wide literals require a valid typed high-word payload. Estimates do not
    # read caller row pointers, including when the physical payload is 16 bytes.
    inputs[0].dtype = slots[0] = 12
    code = (I * 8)(1, -1, -1, 1, 1, -1, -1, -1)
    literals = (I * 2)(0, -2)
    nodes = (I * 2)(12, 3)
    slots = (I * 2)(12, 12)
    steps = (Step * 1)(Step(0, 1, 1, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 12, 1, -1, 0))
    request = Request(2, 513, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result == 513 * 16 + 65
    for payload in [-1, 0, 2]:
        code[3] = payload
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"literal high-word" in C.string_at(err)
        lib.dfm_free(err)


check_decimal128_preflight_without_device()
def check_extended_shapes_without_device():
    # Descriptor-only probes must accept the new boundaries without reading
    # caller input/output addresses, and reject the next shape before indexing.
    inputs = (Input * 256)(*(Input(None, None, 3, 5, 1) for _ in range(256)))
    outputs = (Output * 256)(*(Output(None, None, 3, i, -1, 0) for i in range(256)))
    slots = (I * 256)(*([3] * 256))
    request = Request(2, 17, 0, 256, 256, 0, 0, 0, 0, 256, -1, 0, 0, -1,
                      ptr(inputs), None, None, None, None, ptr(outputs), None, ptr(slots))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result >= 256 * (17 * 8 + 3)
    # Physical two-word storage is independent of the logical slot bound.
    for i in range(256):
        inputs[i].dtype = outputs[i].dtype = slots[i] = 12
    err = P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result == 256 * (17 * 16 + 3)
    for i in range(256):
        inputs[i].dtype = outputs[i].dtype = slots[i] = 3
    for field, invalid in (("slots", 257), ("input_count", 257), ("output_count", 257), ("step_count", 129)):
        original = getattr(request, field)
        setattr(request, field, invalid)
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Invalid Metal request shape" in C.string_at(err)
        lib.dfm_free(err)
        setattr(request, field, original)

    # Sorting reuses the source slot, exercising 128 device steps separately
    # from the slot limit and the 128-by-256 boundary-liveness storage.
    inputs = (Input * 1)(Input(None, None, 2, 0, 0))
    code = (I * 512)(*([200, 0, 0, 1] * 128))
    literals = (I * 128)()
    steps = (Step * 128)(*(Step(i, 1, 0, 2, i, 1) for i in range(128)))
    gathers = (I * 128)()
    outputs = (Output * 1)(Output(None, None, 2, 0, -1, 0))
    nodes, slots = (I * 128)(*([2] * 128)), (I * 1)(2)
    request = Request(2, 17, 0, 1, 1, 512, 128, 128, 128, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), ptr(gathers),
                      ptr(outputs), ptr(nodes), ptr(slots))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.launches >= 128 and memory.peak >= memory.result


check_extended_shapes_without_device()
def check_decimal_float_cast_preflight_without_device():
    def word(value):
        return C.cast(C.pointer(C.c_double(value)), C.POINTER(C.c_uint64)).contents.value

    for dtype, maximum in [(2, 9), (3, 18), (12, 38)]:
        inputs = (Input * 1)(Input(None, None, dtype, 5, 1))
        code = (I * 8)(0, -1, -1, 0, 213, 0, -1, 0)
        literals = (C.c_uint64 * 2)(0, word(1.0))
        nodes, slots = (I * 2)(dtype, 11), (I * 2)(dtype, 11)
        steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
        outputs = (Output * 1)(Output(None, None, 11, 1, -1, 0))
        request = Request(2, 513, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                          ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                          ptr(outputs), ptr(nodes), ptr(slots))
        for scale in range(maximum + 1):
            code[7], literals[1] = scale, word(float(10**scale))
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
            assert memory.result == 513 * 8 + 65
        for scale in [-1, maximum + 1, 2**63 - 1]:
            code[7] = scale
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"decimal Float64 cast descriptor" in C.string_at(err)
            lib.dfm_free(err)
        code[7] = maximum
        for bad in [0, word(-1.0), word(float('inf')), word(float('nan')), literals[1] ^ 1]:
            original = literals[1]
            literals[1] = bad
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"decimal Float64 cast divisor" in C.string_at(err)
            lib.dfm_free(err)
            literals[1] = original
        code[6] = 0
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal Float64 cast descriptor" in C.string_at(err)
        lib.dfm_free(err)
        code[6] = -1
        for bad_source in [1, 4, 5, 6, 7, 8, 9, 10, 11]:
            nodes[0] = slots[0] = inputs[0].dtype = bad_source
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"decimal Float64 cast descriptor" in C.string_at(err)
            lib.dfm_free(err)
        nodes[0] = slots[0] = inputs[0].dtype = dtype
        for bad_target in [1, 2, 3, 4, 12]:
            nodes[1] = slots[1] = outputs[0].dtype = bad_target
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"decimal Float64 cast descriptor" in C.string_at(err)
            lib.dfm_free(err)


check_decimal_float_cast_preflight_without_device()


def check_decimal_cast_preflight_without_device():
    def metadata(source=2, target=1, precision=9, strict=True,
                 source_decimal=True, target_decimal=True):
        return (source | target << 6 | precision << 12 | int(strict) << 18
                | int(source_decimal) << 19 | int(target_decimal) << 20)

    inputs = (Input * 1)(Input(None, None, 12, 5, 1))
    code = (I * 8)(0, -1, -1, 0, 217, 0, -1, metadata())
    literals = (I * 2)(0, 0)
    nodes, slots = (I * 2)(12, 2), (I * 2)(12, 2)
    steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 2, 1, -1, 0))
    request = Request(2, 513, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    memory, err = Memory(), P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    assert memory.result == 513 * 4 + 65
    for invalid in (-1, metadata(source=39), metadata(target=10),
                    metadata(precision=0), metadata(precision=10),
                    metadata(target_decimal=False),
                    metadata(source_decimal=False), metadata() | 1 << 21):
        code[7] = invalid
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal cast descriptor" in C.string_at(err)
        lib.dfm_free(err)
    code[7] = metadata()
    code[6] = 0
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"decimal cast descriptor" in C.string_at(err)
    lib.dfm_free(err)
    code[6] = -1
    nodes[1] = slots[1] = outputs[0].dtype = 1
    code[7] = metadata(target=0, precision=0, target_decimal=False)
    literals[1] = C.cast(C.pointer(C.c_double(100.0)), C.POINTER(I)).contents.value
    err = P()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    literals[1] ^= 1
    err = P()
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"decimal Float32 cast divisor" in C.string_at(err)
    lib.dfm_free(err)


check_decimal_cast_preflight_without_device()
def check_decimal_comparison_preflight_without_device():
    inputs = (Input * 2)(Input(None, None, 2, 5, 1), Input(None, None, 12, 0, 0))
    code = (I * 12)(0, -1, -1, 0, 0, -1, -1, 1, 218, 0, 1, 1 | 38 << 6 | 20 << 12)
    literals = (I * 3)(0, 0, 0)
    nodes, slots = (I * 3)(2, 12, 4), (I * 3)(2, 12, 4)
    steps = (Step * 1)(Step(0, 3, 2, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 4, 2, -1, 0))
    request = Request(2, 513, 0, 3, 2, 12, 3, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    for op in (8, 9, 20, 21, 22, 23):
        code[11] = 1 | 38 << 6 | op << 12
        memory, err = Memory(), P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        assert memory.result == 2 * 65
    for invalid in (-1, 10 | 20 << 12, 39 << 6 | 20 << 12,
                    7 << 12, 24 << 12, 20 << 12 | 1 << 18):
        code[11] = invalid
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal comparison descriptor" in C.string_at(err)
        lib.dfm_free(err)
    code[11] = 20 << 12
    for bad_type in (1, 4, 5, 6, 7, 8, 9, 10, 11):
        nodes[0] = slots[0] = inputs[0].dtype = bad_type
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal comparison descriptor" in C.string_at(err)
        lib.dfm_free(err)


check_decimal_comparison_preflight_without_device()
def check_decimal_arithmetic_preflight_without_device():
    inputs = (Input * 2)(Input(None, None, 2, 5, 1), Input(None, None, 12, 0, 0))
    code = (I * 12)(0, -1, -1, 0, 0, -1, -1, 1, 219, 0, 1, 1 | 38 << 6 | 5 << 12)
    literals = (I * 3)(0, 0, 0)
    nodes, slots = (I * 3)(2, 12, 12), (I * 3)(2, 12, 12)
    steps = (Step * 1)(Step(0, 3, 2, 0, 0, 0))
    outputs = (Output * 1)(Output(None, None, 12, 2, -1, 0))
    request = Request(2, 513, 0, 3, 2, 12, 3, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    for op in (5, 6, 7, 24):
        code[11] = 1 | 38 << 6 | op << 12
        memory, err = Memory(), P()
        check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        assert memory.result == 513 * 16 + 65
    for invalid in (-1, 10 | 5 << 12, 39 << 6 | 5 << 12, 8 << 12, 5 << 12 | 1 << 18):
        code[11] = invalid
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal arithmetic descriptor" in C.string_at(err)
        lib.dfm_free(err)
    for dtype in (2, 3, 12):
        nodes[0] = slots[0] = inputs[0].dtype = dtype
        nodes[2] = slots[2] = outputs[0].dtype = dtype
        code[8], code[10] = 220, -1
        for op in (50, 51):
            code[11] = op
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        code[11] = 52
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"decimal unary descriptor" in C.string_at(err)
        lib.dfm_free(err)


check_decimal_arithmetic_preflight_without_device()
def check_shift_fill_preflight_without_device():
    for dtype in range(1, 13):
        inputs = (Input * 1)(Input(None, None, dtype, 5, 1))
        code = (I * 4)(214, 0, 0, 4)
        literals = (I * 1)(-3)
        nodes, slots = (I * 1)(dtype), (I * 2)(dtype, dtype)
        steps = (Step * 1)(Step(0, 1, 1, 5, 0, 0))
        outputs = (Output * 1)(Output(None, None, dtype, 1, -1, 0))
        request = Request(2, 513, 0, 2, 1, 4, 1, 1, 0, 1, -1, 0, 0, -1,
                          ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                          ptr(outputs), ptr(nodes), ptr(slots))
        for method in (4, 5, 6):
            code[2], code[3] = int(method == 6), method
            for parameter in (-(1 << 63), -3, -1, 0, 1, (1 << 63)-1):
                literals[0] = parameter
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
                expected = (2 * 65 if dtype == 4 else 513 * 16 + 65 if dtype == 12
                            else 513 * {1:4,2:4,3:8,5:1,6:2,7:1,8:2,9:4,10:8,11:8}[dtype] + 65)
                assert memory.result == expected
        for method, reverse in ((-1, 0), (13, 0), (4, 1), (5, 1), (6, 0)):
            code[2], code[3] = reverse, method
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"cumulative descriptor" in C.string_at(err)
            lib.dfm_free(err)


def check_rolling_preflight_without_device():
    for dtype in range(1, 13):
        for method in range(7, 11):
            result = 11 if method == 8 else (3 if method == 7 and dtype in (5,6,7,8) else dtype)
            inputs = (Input * 1)(Input(None, None, dtype, 0, 0))
            code = (I * 4)(214, 0, 2, method)
            literals = (I * 1)(3)
            nodes, slots = (I * 1)(dtype), (I * 2)(dtype, result)
            steps = (Step * 1)(Step(0, 1, 1, 5, 0, 0))
            outputs = (Output * 1)(Output(None, None, result, 1, -1, 0))
            request = Request(2, 513, 0, 2, 1, 4, 1, 1, 0, 1, -1, 0, 0, -1,
                              ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                              ptr(outputs), ptr(nodes), ptr(slots))
            unsupported = dtype in (4,12) and method in (7,8)
            memory, err = Memory(), P()
            status = lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            if unsupported:
                assert status
                lib.dfm_free(err)
                continue
            check(status, err)
            for parameter in (0, -1, -(1 << 63)):
                literals[0] = parameter
                err = P()
                assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
                assert b"rolling window size" in C.string_at(err)
                lib.dfm_free(err)
            literals[0] = (1 << 63)-1
            for minimum in (0, 1, 513, (1 << 63)-1):
                code[2] = minimum
                err = P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
            code[2] = -1
            err = P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            lib.dfm_free(err)


def check_rolling_moments_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 11, 0, 0))
    code = (I * 4)(214, 0, 1, 11)
    literals = (C.c_uint64 * 1)(3 | (1 << 32))
    nodes, slots = (I * 1)(11), (I * 2)(11, 11)
    steps = (Step * 1)(Step(0, 1, 1, 5, 0, 0))
    outputs = (Output * 1)(Output(None, None, 11, 1, -1, 0))
    request = Request(2, 513, 0, 2, 1, 4, 1, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    for method in (11, 12):
        code[3] = method
        for window in (1, 3, (1 << 32)-1):
            for ddof in (0, 1, (1 << 32)-1):
                literals[0] = window | (ddof << 32)
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        for ddof in (0, 1, (1 << 32)-1):
            literals[0] = ddof << 32
            err = P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"rolling window size" in C.string_at(err)
            lib.dfm_free(err)
        literals[0] = 3
        for dtype in (1, 3, 4, 10, 12):
            inputs[0].dtype = nodes[0] = slots[0] = dtype
            err = P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"cumulative result type" in C.string_at(err)
            lib.dfm_free(err)
        inputs[0].dtype = nodes[0] = slots[0] = 11
        slots[1] = outputs[0].dtype = 1
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)
        slots[1] = outputs[0].dtype = 11


def check_moment_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 11, 0, 0))
    slots = (I * 1)(11)
    outputs = (Output * 1)(Output(None, None, 11, 0, 87, 1 << 47))
    request = Request(2, 513, 0, 1, 1, 0, 0, 0, 0, 1, -1, 1, 0, -1,
                      ptr(inputs), None, None, None, None, ptr(outputs), None, ptr(slots))
    for operation in (86,87):
        outputs[0].reduction = operation
        for workers in (1,2,8,65535):
            for ddof in (0,1,513,(1 << 31)-1):
                outputs[0].min_count = (workers << 47) | ddof
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
                assert memory.result == 9
        for metadata in (0,1,1 << 31,(1 << 47) | (1 << 31)):
            outputs[0].min_count = metadata
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"moment metadata" in C.string_at(err)
            lib.dfm_free(err)
        outputs[0].min_count = 1 << 47
        slots[0] = inputs[0].dtype = 3
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"accumulator" in C.string_at(err)
        lib.dfm_free(err)
        slots[0] = inputs[0].dtype = 11


def check_higher_moment_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 11, 0, 0))
    slots = (I * 1)(11)
    outputs = (Output * 1)(Output(None, None, 11, 0, 98, 1 << 47))
    request = Request(2, 513, 0, 1, 1, 0, 0, 0, 0, 1, -1, 1, 0, -1,
                      ptr(inputs), None, None, None, None, ptr(outputs), None, ptr(slots))
    for operation, maximum in ((98, 1), (99, 3)):
        outputs[0].reduction = operation
        for workers in (1, 2, 8, 65535):
            for flags in range(maximum + 1):
                outputs[0].min_count = (workers << 47) | flags
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        for metadata in (0, 1, (1 << 47) | (maximum + 1), (1 << 47) | (1 << 31)):
            outputs[0].min_count = metadata
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"moment metadata" in C.string_at(err)
            lib.dfm_free(err)
        outputs[0].min_count = 1 << 47
        slots[0] = inputs[0].dtype = 3
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)
        slots[0] = inputs[0].dtype = 11


def check_pair_moment_preflight_without_device():
    inputs = (Input * 2)(Input(None, None, 11, 0, 0), Input(None, None, 11, 0, 0))
    slots = (I * 3)(11, 11, 11)
    outputs = (Output * 1)(Output(None, None, 11, 0, 160, (1 << 47) | (1 << 31)))
    request = Request(2, 513, 0, 3, 2, 0, 0, 0, 0, 1, -1, 1, 0, -1,
                      ptr(inputs), None, None, None, None, ptr(outputs), None, ptr(slots))
    for operation in (160, 161):
        outputs[0].reduction = operation
        for workers in (1, 2, 8, 65535):
            for right in (0, 1):
                for ddof in (0, 1, (1 << 31)-1):
                    outputs[0].min_count = (workers << 47) | (right << 31) | ddof
                    memory, err = Memory(), P()
                    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
                    assert memory.result == 9
        for right in (2, 3, 65535):
            outputs[0].min_count = (1 << 47) | (right << 31)
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"paired moment right input" in C.string_at(err)
            lib.dfm_free(err)
        outputs[0].min_count = (1 << 47) | (1 << 31)
        inputs[1].dtype = slots[1] = 3
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"paired moment right input" in C.string_at(err)
        lib.dfm_free(err)
        inputs[1].dtype = slots[1] = 11
    # Group output literals carry the same logical right operand slot.
    code = (I * 12)(201, 1, 3, 4, 202, 0, 0, 0, 203, 0, 5, 160)
    literals = (I * 3)(0, 0, (1 << 47) | (1 << 31))
    nt, st = (I * 3)(3, 11, 11), (I * 6)(11, 11, 3, 3, 3, 11)
    steps = (Step * 1)(Step(0, 3, 2, 3, 0, 0))
    outputs[0].slot, outputs[0].reduction, outputs[0].min_count = 5, -1, 0
    request = Request(2, 513, 0, 6, 2, 12, 3, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nt), ptr(st))
    for operation in (160, 161):
        code[11] = operation
        for right in (0, 1):
            literals[2] = (8 << 47) | (right << 31) | 1
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
        for right in (2, 5, 6, 65535):
            literals[2] = (8 << 47) | (right << 31)
            memory, err = Memory(), P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"paired moment right input" in C.string_at(err)
            lib.dfm_free(err)


check_higher_moment_preflight_without_device()
check_pair_moment_preflight_without_device()

check_moment_preflight_without_device()
check_rolling_preflight_without_device()
check_rolling_moments_preflight_without_device()
check_shift_fill_preflight_without_device()
def empty_request(rows, reduction=0):
    values = (C.c_float * max(1, rows))(*([1.0] * max(1, rows)))
    inputs = (Input * 1)(Input(ptr(values), None, 1, 0, 0))
    request = Request(2, rows, 1, 1, 1, 0, 0, 0, 0, 0, -1, reduction, 0, -1,
                      ptr(inputs), None, None, None, None, None, None, None)
    return request, (values, inputs)


def check_empty_outputs_preflight_without_device():
    for rows in (0, 1, 513):
        for reduction in (0, 1):
            request, owners = empty_request(rows, reduction)
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
            assert memory.result == 0 and memory.launches == 1
            request.output_count = -1
            err = P()
            assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert b"Invalid Metal request shape" in C.string_at(err)
            lib.dfm_free(err)


def empty_input_request(rows, reduction=0, typed=False):
    return Request(2, rows, 0 if typed else 1, 0, 0, 0, 0, 0, 0, 0,
                   -1, reduction, 0, -1,
                   None, None, None, None, None, None, None, None)


def check_empty_inputs_preflight_without_device():
    for rows in (0, 1, 513):
        for typed in (False, True):
            for reduction in (0, 1):
                request = empty_input_request(rows, reduction, typed)
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
                assert memory.result == memory.upload == memory.launches == 0
                assert memory.peak > 0
                for field in ("slots", "input_count"):
                    setattr(request, field, -1)
                    err = P()
                    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
                    lib.dfm_free(err)
                    setattr(request, field, 0)


check_empty_inputs_preflight_without_device()
def check_duration_totals_preflight_without_device():
    inputs = (Input * 1)(Input(None, None, 3, 0, 0))
    code = (I * 8)(0, -1, -1, 0, 143, 0, 1, 1)
    literals = (I * 2)(0, 0)
    steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
    nodes, slots = (I * 2)(3, 3), (I * 2)(3, 3)
    outputs = (Output * 1)(Output(None, None, 3, 1, -1, 0))
    request = Request(2, 513, 0, 2, 1, 8, 2, 1, 0, 1, -1, 0, 0, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    for mode in (0, 1):
        code[7] = mode
        for factor in (1, 1000, 1000000, 86400000000000):
            code[6] = factor
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    for factor, mode in ((0, 1), (-1, 1), (86400000000001, 1), (1, -1), (1, 2)):
        code[6], code[7] = factor, mode
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"duration total descriptor" in C.string_at(err)
        lib.dfm_free(err)


check_duration_totals_preflight_without_device()
check_empty_outputs_preflight_without_device()
print("Native Metal ABI and pure request validation: PASS")
if lib.dfm_device_count() == 0:
    print("Native Metal execution: SKIP (no GPU exposed on this host)")
    sys.exit(0)
lib.dfm_probe.argtypes = [I, C.POINTER(Request), C.POINTER(ProbeInfo), C.POINTER(P)]
lib.dfm_probe.restype = C.c_int32
metadata = ProbeInfo()
err = P()
check(lib.dfm_probe(0, None, C.byref(metadata), C.byref(err)), err)
assert C.sizeof(ProbeInfo) == 64 and metadata.recommended > 0
assert metadata.max_buffer > 0 and metadata.unified == 1
assert metadata.cached == 0 and metadata.reclaimable == 0
assert b"model=" in metadata.host and b"os=" in metadata.host
before = metadata.allocated
for _ in range(10):
    check(lib.dfm_probe(0, None, C.byref(metadata), C.byref(err)), err)
    assert metadata.allocated == before and metadata.cached == 0
assert lib.dfm_probe(-1, None, C.byref(metadata), C.byref(err)) != 0
assert b"device index unavailable" in C.string_at(err)
lib.dfm_free(err)
probe_ctx = lib.dfm_context_create(0, 1, C.byref(err))
if not probe_ctx:
    check(1, err)
probe_values = (C.c_float * 1)(7)
probe_result = (C.c_float * 1)()
probe_bits = (C.c_uint8 * 1)()
probe_inputs = (Input * 1)(Input(ptr(probe_values), None, 1, 0, 0))
probe_code = (I * 4)(0, -1, -1, 0)
probe_literals = (I * 1)(0)
probe_steps = (Step * 1)(Step(0, 1, 1, 0, 0, 0))
probe_outputs = (Output * 1)(Output(ptr(probe_result), ptr(probe_bits), 1, 1, -1, 0))
probe_request = Request(2, 1, 1, 2, 1, 4, 1, 1, 0, 1, -1, 0, 0, -1,
                        ptr(probe_inputs), ptr(probe_code), ptr(probe_literals),
                        ptr(probe_steps), None, ptr(probe_outputs), None, None)
check(lib.dfm_probe(0, C.byref(probe_request), C.byref(metadata), C.byref(err)), err)
assert metadata.cached == 0
probe_stats = Stats()
check(lib.dfm_execute(probe_ctx, C.byref(probe_request), C.byref(probe_stats), C.byref(err)), err)
check(lib.dfm_probe(0, C.byref(probe_request), C.byref(metadata), C.byref(err)), err)
assert probe_result[0] == 7 and metadata.cached == 1 and metadata.reclaimable > 0
lib.dfm_context_release(probe_ctx)
print("Native Metal metadata-only probe and exact plan cache state: PASS")
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
    repeats=1,
    typed=False,
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
        11 if reduction in (211, 212) else dtype if reduction == 210 else
        (dtype if dtype in (1, 2, 9, 10) else 3) if reduction == 10 else 3
    ) if reduction >= 0 else (4 if output_bool else dtype)
    out_scalar = C.c_uint64 if out_type == 11 else SCALARS[out_type]
    result = (out_scalar * max(1, n))()
    bits = (C.c_uint8 * max(1, (n + 7) // 8))()
    if expect_error:
        result[0], bits[0] = 19, 0xA5
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
    if typed:
        assert not filter_nodes and not output_bool
        node_types = (I * len(programs))(*([dtype] * len(programs)))
        slot_types = (I * (slot + 1))(*([dtype] * (slot + 1)))
        request.dtype = 0
        request.node_types = ptr(node_types)
        request.slot_types = ptr(slot_types)
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
            assert result[0] == 19 and bits[0] == 0xA5, "failed request mutated caller outputs"
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
result, bits, _ = execute([1.0], 1, [col], [0.0], reduction=10)
assert result == [1.0] and bits == [True]


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
]:
    a, b = f32(a), f32(b)
    result, bits, _ = execute(
        [a],
        1,
        [col, (2, -1, -1, -1), (op, 0, 1, -1)],
        [0.0, b, 0.0],
        output_bool=op == 8,
    )
    expected = f32(a + b if op == 5 else a - b if op == 6 else a * b)
    assert result == [expected], (op, a, b, result, expected)


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
    result, bits, _ = execute([1, 2, 3], dtype, [col], [0], reduction=10)
    assert result == [6] and bits == [True]
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
                    C.cast(C.pointer(C.c_double(1.0)), C.POINTER(C.c_uint64))[
                        0
                    ] if t
                    == 1 else 1
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
    check(
        lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err
    )
    check(
        lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
        err,
    )
    selected = [i for i in range(n) if i % 13 != 0 and i % 3 != 0]
    assert stats.rows == len(selected)
    assert stats.memory.peak == memory.peak
    for j, t in enumerate(types):
        result = (
            unpack(results[j], stats.rows) if t
            == 4 else list(results[j])[: stats.rows]
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
            outputs = (Output * 1)(
                Output(ptr(result), ptr(valid), target, 1, -1, 0)
            )
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
                lib.dfm_execute(
                    ctx, C.byref(request), C.byref(stats), C.byref(err)
                ),
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
                    lo, hi = (0, 2**width - 1) if unsigned else (
                        -(2 ** (width - 1)),
                        2 ** (width - 1) - 1,
                    )
                    fits = not (source == 1 and not math.isfinite(x))
                    expected = math.trunc(x) if fits else 0
                    fits = fits and lo <= expected <= hi
                expected_valid.append(fits)
                if fits:
                    assert got[i] == expected or (
                        target == 1
                        and math.isnan(got[i])
                        and math.isnan(expected)
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
    # that point. Software binary64 must reproduce the CPU's two conversions.
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
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
    assert result[0] == C.c_float(float(data[0])).value
    assert unpack(valid, 1) == [True]


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
                Output(
                    C.addressof(results[i]) + 8,
                    ptr(bits[i]),
                    dtype,
                    0,
                    80 + i,
                    0,
                )
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
        rc = lib.dfm_execute(
            ctx, C.byref(request), C.byref(stats), C.byref(err)
        )
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
    float_bits = lambda u: C.cast(
        C.pointer(C.c_uint32(u)), C.POINTER(C.c_float)
    )[0]
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
        for denominator in [3, -3] if signed else [3]:
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
                    x // denominator if op == 25 else x % denominator
                    for x in values
                ]
                assert bits == valid
                assert all(
                    not yes or x == y
                    for x, y, yes in zip(result, expected, valid)
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
                [minimum],
                dtype,
                [col, (1, -1, -1, -1), (26, 0, 1, -1)],
                [0, -1, 0],
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
                    math.floor(x) if op
                    == 55 else (
                        math.ceil(x) if op
                        == 56 else math.floor(x + 0.5) if x
                        >= 0 else math.ceil(x - 0.5)
                    )
                )
                if expected == 0:
                    expected = math.copysign(0.0, x) if op != 57 or x != 0 else 0.0
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
                math.isnan(x) if op
                == 63 else (
                    not math.isnan(x) if op
                    == 64 else math.isfinite(x) if op
                    == 65 else math.isinf(x)
                )
            )
            for x in values
        ]
        assert result == expected, op
    result, _, _ = execute(
        values, 1, [col, (2, -1, -1, -1), (34, 0, 1, -1)], [0, 3.5, 0]
    )
    assert all(
        y == 3.5 if math.isnan(x) else y == x for x, y in zip(values, result)
    )
    _, bits, _ = execute(
        values, 1, [col, (12, -1, -1, -1), (34, 0, 1, -1)], [0, 0, 0]
    )
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

    def run(dtype, values, valid, descending, nulls_last, fallback=False):
        n = len(values)
        offset = 5
        scalar = SCALARS[dtype]
        data = packed(values, offset) if dtype == 4 else (scalar * max(1, n))(
            *values
        )
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
        check(
            lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)),
            err,
        )
        expected_radix = n >= 65536 and dtype != 1
        assert bool(lib.dfm_plan_features(C.byref(request), C.byref(err)) & 4) == expected_radix
        if fallback:
            assert expected_radix
            # Remove exactly the optional radix workspace from the peak budget.
            request.budget = memory.peak - (12 * n + 64 * ((n + 255) // 256) + 96)
            assert lib.dfm_plan_features(C.byref(request), C.byref(err)) & 4 == 0
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
            assert memory.peak == request.budget
        check(
            lib.dfm_execute(
                ctx, C.byref(request), C.byref(stats), C.byref(err)
            ),
            err,
        )
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
                c = (
                    0 if math.isnan(x)
                    and math.isnan(y) else 1 if math.isnan(x) else -1
                )
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
                assert C.string_at(
                    C.addressof(key) + i * size, size
                ) == C.string_at(C.addressof(data) + source * size, size), (
                    dtype,
                    i,
                    source,
                )

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
        # Algorithm threshold, partial blocks, and digit/sign boundaries.
        for n in (65535, 65536, 65537):
            large = (values * ((n + len(values) - 1) // len(values)))[:n]
            run(dtype, large, [i % 7 != 3 for i in range(n)], False, True)
        if dtype != 1:
            n = 65537
            large = (values * ((n + len(values) - 1) // len(values)))[:n]
            for descending, nulls_last in ((True, False), (False, False), (True, True)):
                run(dtype, large, [i % 7 != 3 for i in range(n)], descending, nulls_last)
            run(dtype, large, [False] * n, True, True)
            run(dtype, large, [True] * n, True, True, fallback=True)



test_stable_sort()


def test_radix_composite_after_filter():
    from functools import cmp_to_key

    n = 65537
    first = [((i * 103) % 31) - 15 for i in range(n)]
    second = [((i * 47) % 13) - 6 for i in range(n)]
    valid = [[i % 11 != 0 for i in range(n)], [i % 7 != 0 for i in range(n)]]
    arrays = [(C.c_int32 * n)(*first), (C.c_int32 * n)(*second), (I * n)(*range(n))]
    bitmaps = [packed(v, 3) for v in valid]
    inputs = (Input * 3)(Input(ptr(arrays[0]), ptr(bitmaps[0]), 2, 3, 1),
        Input(ptr(arrays[1]), ptr(bitmaps[1]), 2, 3, 1), Input(ptr(arrays[2]), None, 3, 0, 0))
    # Filter on an independent column, then sort mixed directions/null policies.
    words = [0, -1, -1, 2, 1, -1, -1, -1, 8, 0, 1, -1,
        200, 0, 0, 1, 200, 1, 1, 0]
    code = (I * len(words))(*words)
    literals = (I * 5)(0, n // 3, 0, 0, 0)
    steps = (Step * 2)(Step(0, 3, 3, 1, 0, 3), Step(3, 2, 0, 2, 3, 3))
    gathers = (I * 6)(0, 1, 2, 0, 1, 2)
    result = (I * n)()
    resultbits = packed([False] * n)
    outputs = (Output * 1)(Output(ptr(result), ptr(resultbits), 3, 2, -1, 0))
    nt, st = (I * 5)(3, 3, 4, 2, 2), (I * 4)(2, 2, 3, 4)
    request = Request(2, n, 0, 4, 3, len(words), 5, 2, 6, 1, -1, 0, 1, -1,
        ptr(inputs), ptr(code), ptr(literals), ptr(steps), ptr(gathers), ptr(outputs), ptr(nt), ptr(st))
    memory, stats, err = Memory(), Stats(), P()
    assert lib.dfm_plan_features(C.byref(request), C.byref(err)) & 4
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)

    def compare(a, b):
        for values, validity, descending, last in ((first, valid[0], False, True),
                (second, valid[1], True, False)):
            if validity[a] != validity[b]:
                return (-1 if validity[a] else 1) * (1 if last else -1)
            if validity[a]:
                c = (values[a] > values[b]) - (values[a] < values[b])
                if c:
                    return -c if descending else c
        return 0

    expected = sorted(range(n // 3 + 1, n), key=cmp_to_key(compare))
    assert stats.rows == len(expected)
    assert list(result)[:stats.rows] == expected
    assert all(unpack(resultbits, stats.rows))
    assert stats.memory.peak == memory.peak
    assert stats.memory.launches == memory.launches
    # An empty filtered table uses indirect one-block dispatches safely.
    literals[1] = n
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
    assert stats.rows == 0


test_radix_composite_after_filter()


def test_grouped_reductions():
    def run(dtype, n, key_count=2, unique=False, fallback=False, last=False):
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
        nodes = [(205 if last else 201, key_count, 4, 5)] + (
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
            Step(0, stepnodes, 3, 3, 0, 0),
            Step(stepnodes, 1, 0, 2, 0, len(outs)),
        )
        code = (I * (4 * len(nodes)))(*[x for node in nodes for x in node])
        literals = (I * len(words))(*words)
        types = (I * len(nt))(*nt)
        st = (I * len(slots))(*slots)
        arrays = [
            packed([False] * max(1, n)) if t
            == 4 else (SCALARS[t] * max(1, n))()
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
            canon = (
                "nan" if dtype == 1
                and raw is not None
                and math.isnan(raw) else raw
            )
            k = (canon, second[i] if bv[i] else None) if key_count else ()
            groups.setdefault(k, []).append(i)
        if not key_count and not n:
            groups[()] = []
        expected = []
        ordered = list(groups.values())
        if last:
            ordered.sort(key=lambda rows: rows[-1] if rows else 0)
        for rows in ordered:
            row = rows[-1 if last else 0] if rows else 0
            good = [i for i in rows if vv[i]]
            rank_value = lambda i: (
                (math.isnan(value[i]), value[i]) if dtype == 1 else value[i]
            )
            smallest = min(good, key=rank_value) if good else 0
            largest = max(good, key=rank_value) if good else 0
            record = [
                (key[row], kv[row]),
                (second[row], bv[row]),
            ] if key_count else []
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
        check(
            lib.dfm_estimate(C.byref(req), C.byref(memory), C.byref(err)), err
        )
        assert bool(lib.dfm_plan_features(C.byref(req), C.byref(err)) & 2) == (n >= 4096)
        if fallback:
            h = 1 << (2 * n - 1).bit_length()
            req.budget = memory.peak - (12 * n + 64 * ((n + 255) // 256) + 96 + 4 * h)
            assert lib.dfm_plan_features(C.byref(req), C.byref(err)) & 2 == 0
            check(lib.dfm_estimate(C.byref(req), C.byref(memory), C.byref(err)), err)
            assert memory.peak == req.budget
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
        status = lib.dfm_execute(
            ctx, C.byref(req), C.byref(stats), C.byref(err)
        )
        if overflow:
            assert status and b"overflow" in C.string_at(err)
            lib.dfm_free(err)
            return
        check(status, err)
        assert stats.rows == len(expected), (
            dtype,
            n,
            stats.rows,
            len(expected),
        )
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
                assert validity[row] == valid, (
                    dtype,
                    n,
                    c,
                    row,
                    validity[row],
                    valid,
                )
                if valid:
                    assert (
                        t == 1
                        and math.isnan(value0)
                        and math.isnan(actual[row])
                    ) or actual[row] == value0, (
                        dtype,
                        n,
                        c,
                        row,
                        actual[row],
                        value0,
                    )
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
        for n in (0, 1, 7, 257, 1025, 4095, 4096, 4097):
            run(t, n)
    run(5, 0, 0)
    run(5, 262401, 0)
    run(3, 1025, 2, True)
    run(3, 4097, 2, True)
    run(3, 4097, fallback=True)
    for t in range(1, 11):
        for n in (0, 7, 257, 4097):
            run(t, n, last=True)
    run(3, 4097, fallback=True, last=True)


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


def test_additional_reductions():
    def run(t, n, grouped, all_null=False, all_nan=False, all_valid=False):
        scalar = SCALARS[t]
        if t == 1:
            pool = [
                float("nan"),
                0.0,
                -0.0,
                1.0,
                -1.0,
                1.401298464324817e-45,
                float("inf"),
                -float("inf"),
            ]
            if all_nan:
                pool = [float("nan")]
        elif t == 4:
            pool = [False, True]
        elif t in (7, 8, 9, 10):
            pool = [0, (1 << (C.sizeof(scalar) * 8)) - 1, 1, 2]
        else:
            high = (1 << (C.sizeof(scalar) * 8 - 1)) - 1
            pool = [-high - 1, high, 0, -1, 1]
        values = [pool[(i * 7) % len(pool)] for i in range(n)]
        keys = [(i * 7) % 11 for i in range(n)]
        bools = [i % 5 == 0 for i in range(n)]
        vv = [not all_null and (all_valid or i % 13 != 0) for i in range(n)]
        kv = [i % 17 != 3 for i in range(n)]
        bv = [not all_null and (all_valid or i % 7 != 0) for i in range(n)]
        va = packed(values, 5) if t == 4 else (scalar * max(1, n))(*values)
        ka = (C.c_int32 * max(1, n))(*keys)
        ba = packed(bools, 5)
        vb, kb, bb = packed(vv, 5), packed(kv, 5), packed(bv, 5)
        inputs = (Input * 3)(
            Input(ptr(ka), ptr(kb), 2, 5, 1),
            Input(ptr(va), ptr(vb), t, 5, 1),
            Input(ptr(ba), ptr(bb), 4, 5, 1),
        )
        specs = [
            (1, 83, t, 0),
            (1, 84, t, 0),
            (1, 93, 3, 0),
            (1, 95, 9, 0),
            (1, 96, 9, 0),
            (2, 91, 4, 1),
            (2, 92, 4, 1),
            (2, 91, 4, 0),
            (2, 92, 4, 0),
        ]
        slots = [2, t, 4]
        nodes = []
        nt = []
        words = []
        step_list = []
        gather_list = []
        outs = []
        if grouped:
            slots += [3, 3, 3]
            nodes = [(201, 1, 4, 5), (202, 0, 0, 0)]
            nt = [3, 2]
            words = [0, 0]
            outs = [(0, 2, -1, 0)]
            for source, kind, outtype, param in specs:
                target = len(slots)
                slots.append(outtype)
                nodes.append((203, source, target, kind))
                nt.append(outtype)
                words.append(param)
                outs.append((target, outtype, -1, 0))
            size = len(nodes)
            nodes.append((200, 5, 0, 0))
            nt.append(3)
            words.append(0)
            gather_list = [slot for slot, t, kind, param in outs]
            step_list = [
                Step(0, size, 3, 3, 0, 0),
                Step(size, 1, 0, 2, 0, len(gather_list)),
            ]
        else:
            outs = [
                (source, outtype, kind, param)
                for source, kind, outtype, param in specs
            ]
        code = (I * len([x for node in nodes for x in node]))(
            *[x for node in nodes for x in node]
        )
        literals = (I * len(words))(*words)
        types = (I * len(nt))(*nt)
        st = (I * len(slots))(*slots)
        steps = (Step * len(step_list))(*step_list)
        gathers = (I * len(gather_list))(*gather_list)
        cap = max(1, n) if grouped else 1
        arrays = [
            packed([False] * cap) if outtype
            == 4 else (SCALARS[outtype] * cap)()
            for slot, outtype, kind, param in outs
        ]
        bits = [packed([False] * cap) for _ in outs]
        output = (Output * len(outs))(
            *[
                Output(ptr(a), ptr(b), outtype, slot, kind, param)
                for (slot, outtype, kind, param), a, b in zip(
                    outs, arrays, bits
                )
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
            len(steps),
            len(gathers),
            len(outs),
            -1,
            0 if grouped else 1,
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
        if grouped:
            for i in range(n):
                groups.setdefault(keys[i] if kv[i] else None, []).append(i)
        else:
            groups[()] = list(range(n))
        expected = []
        for key, rows in groups.items():
            record = [(key, key is not None)] if grouped else []
            for source, kind, outtype, param in specs:
                good = [
                    (pos, row)
                    for pos, row in enumerate(rows)
                    if (vv if source == 1 else bv)[row]
                ]
                if kind in (83, 84):
                    at = rows[0 if kind == 83 else -1] if rows else 0
                    record.append(
                        (values[at] if rows else 0, bool(rows) and vv[at])
                    )
                elif kind == 93:
                    record.append((len(rows) - len(good), True))
                elif kind in (95, 96):
                    numeric = [
                        (pos, row)
                        for pos, row in good
                        if not (t == 1 and math.isnan(values[row]))
                    ]
                    choices = numeric or good
                    chosen = (max if kind == 96 else min)(
                        choices, key=lambda pair: values[pair[1]]
                    ) if choices else (0, 0)
                    record.append((chosen[0], bool(choices)))
                else:
                    yes = any(bools[row] for pos, row in good)
                    no = any(not bools[row] for pos, row in good)
                    unknown = len(good) != len(rows)
                    record.append(
                        (
                            yes if kind == 91 else not no,
                            bool(param)
                            or not unknown
                            or (yes if kind == 91 else no),
                        )
                    )
            expected.append(record)
        memory, stats, err = Memory(), Stats(), P()
        check(
            lib.dfm_estimate(C.byref(req), C.byref(memory), C.byref(err)), err
        )
        check(
            lib.dfm_execute(ctx, C.byref(req), C.byref(stats), C.byref(err)),
            err,
        )
        assert stats.rows == len(expected), (
            t,
            n,
            grouped,
            stats.rows,
            len(expected),
        )
        assert stats.waits == 1 and memory.shared == stats.memory.shared
        assert memory.launches == stats.memory.launches
        for c, ((slot, outtype, kind, param), a, b) in enumerate(
            zip(outs, arrays, bits)
        ):
            actual = (
                unpack(a, stats.rows) if outtype == 4 else list(a)[: stats.rows]
            )
            valid = unpack(b, stats.rows)
            for row, record in enumerate(expected):
                value, ok = record[c]
                assert valid[row] == ok, (t, n, grouped, c, row, valid[row], ok)
                if ok:
                    assert (
                        outtype == 1
                        and math.isnan(value)
                        and math.isnan(actual[row])
                    ) or actual[row] == value, (
                        t,
                        n,
                        grouped,
                        c,
                        row,
                        actual[row],
                        value,
                    )
                    if outtype == 1 and value == 0:
                        assert C.string_at(
                            C.byref(C.c_float(actual[row])), 4
                        ) == C.string_at(C.byref(C.c_float(value)), 4)

    for t in range(1, 11):
        for grouped in (False, True):
            for n in (0, 1, 7, 521, 4097):
                run(t, n, grouped)
            run(t, 521, grouped, all_null=True)
            run(t, 521, grouped, all_valid=True)
    for grouped in (False, True):
        run(1, 521, grouped, all_nan=True)
    run(1, 4097, True, all_nan=True)
    run(1, 4097, True, all_null=True)
    run(4, 4097, True, all_valid=True)
    run(3, 262401, False)


test_additional_reductions()


def test_workspace_reuse():
    class Workspace(C.Structure):
        _fields_ = [
            (name, I)
            for name in ("bytes", "buffers", "hits", "allocations", "limit")
        ]

    lib.dfm_workspace_info.argtypes = [P, C.POINTER(Workspace), C.POINTER(P)]
    lib.dfm_workspace_trim.argtypes = [P, C.POINTER(P)]

    def info():
        result, error = Workspace(), P()
        check(
            lib.dfm_workspace_info(ctx, C.byref(result), C.byref(error)), error
        )
        assert 0 <= result.bytes <= result.limit
        assert 0 <= result.buffers <= 64
        return result

    error = P()
    check(lib.dfm_workspace_trim(ctx, C.byref(error)), error)
    assert info().bytes == info().buffers == 0
    n = 521
    values = list(range(n))
    first = execute(values, 2, [col], [0])
    cold = info()
    assert cold.bytes > 0 and cold.buffers > 0
    # Same allocations, different payload and nulls: stale data must not leak.
    valid = [i % 3 != 0 for i in range(n)]
    second = execute([-i for i in values], 2, [col], [0], valid=valid)
    # Validity storage changes shape; warm that shape before checking reuse.
    warm = info()
    third = execute(values, 2, [col], [0], valid=valid)
    hot = info()
    assert hot.allocations == warm.allocations
    assert hot.hits > warm.hits
    assert first[0] == values and all(first[1])
    assert second[1] == third[1] == valid
    for i in range(n):
        if valid[i]:
            assert second[0][i] == -i and third[0][i] == i
    execute(
        values,
        2,
        [col],
        [0],
        valid=valid,
        budget=0,
        expect_error="memory budget",
    )
    denied = info()
    assert denied.allocations == hot.allocations and denied.hits == hot.hits
    execute(
        [2147483647] * n,
        2,
        [col, (1, -1, -1, -1), (5, 0, 1, -1)],
        [0, 1, 0],
        expect_error="overflow",
    )
    recovered = execute(values, 2, [col], [0], valid=valid)
    assert recovered[1] == valid
    for i in range(n):
        if valid[i]:
            assert recovered[0][i] == i
    check(lib.dfm_workspace_trim(ctx, C.byref(error)), error)
    assert info().bytes == info().buffers == 0
    execute([], 2, [col], [0])
    info()


test_workspace_reuse()


def test_filtered_reduction_without_compaction():
    for n in (0, 1, 9, 257, 65537):
        values = [i % 11 - 5 for i in range(n)]
        valid = [i % 7 != 0 for i in range(n)]
        predicate = [col, (1, -1, -1, -1), (8, 0, 1, -1)]
        selected = [x for x, ok in zip(values, valid) if ok and x > 0]
        row = execute(
            values, 2, [col], [0], valid=valid, offset=3, filter_nodes=predicate
        )
        assert row[0] == selected
        for kind in (10, 11, 90):
            result, bitmap, stats = execute(
                values,
                2,
                [col],
                [0],
                valid=valid,
                offset=3,
                filter_nodes=predicate,
                reduction=kind,
            )
            expected = sum(selected) if kind == 10 else len(selected)
            assert result == [expected] and bitmap == [True]
            # Unpack, two expression segments, and two reduction stages. No
            # prefix ranks, block scan or column gather launches remain.
            assert stats.memory.launches == 5
            if n >= 257:
                assert stats.memory.shared < row[2].memory.shared
        for kind in (11, 90):
            # A null test retains null rows: count and length must differ.
            result, bitmap, _ = execute(
                values,
                2,
                [col],
                [0],
                valid=valid,
                offset=3,
                filter_nodes=[col, (61, 0, -1, -1)],
                reduction=kind,
            )
            assert result == [0 if kind == 11 else n - sum(valid)]
            assert bitmap == [True]
    result = execute(
        [2147483647, -1],
        2,
        [col, (1, -1, -1, -1), (5, 0, 1, -1)],
        [0, 1, 0],
        filter_nodes=[col, (1, -1, -1, -1), (20, 0, 1, -1)],
        reduction=10,
    )
    assert result[0] == [0] and result[1] == [True]
    execute(
        [2147483647, -1],
        2,
        [col, (1, -1, -1, -1), (5, 0, 1, -1)],
        [0, 1, 0],
        filter_nodes=[col, (1, -1, -1, -1), (8, 0, 1, -1)],
        reduction=10,
        expect_error="overflow",
    )
    result = execute(
        [1, 2, 3],
        2,
        [col],
        [0],
        filter_nodes=[col, (1, -1, -1, -1), (20, 0, 1, -1)],
        reduction=10,
        minimum=1,
    )
    assert result[1] == [False]

    # Two filters whose conjunction is empty. Inactive lanes of the second
    # predicate must be false even when a recycled matrix contains old values.
    n = 521
    data = (C.c_int32 * n)(*[i % 11 - 5 for i in range(n)])
    inputs = (Input * 1)(Input(ptr(data), None, 2, 0, 0))
    nodes = [
        col,
        (1, -1, -1, -1),
        (8, 0, 1, -1),
        col,
        (1, -1, -1, -1),
        (20, 0, 1, -1),
        col,
    ]
    code = (I * (4 * len(nodes)))(*[v for node in nodes for v in node])
    words = (I * len(nodes))()
    types = (I * len(nodes))(2, 2, 4, 2, 2, 4, 2)
    slots = (I * 4)(2, 4, 4, 2)
    steps = (Step * 3)(
        Step(0, 3, 1, 1, 0, 1), Step(3, 3, 2, 1, 1, 1), Step(6, 1, 3, 0, 2, 0)
    )
    gathers = (I * 2)(0, 0)
    value, bitmap = (C.c_int64 * 1)(), (C.c_uint8 * 1)()
    outputs = (Output * 1)(Output(ptr(value), ptr(bitmap), 3, 3, 90, 0))
    request = Request(
        2,
        n,
        0,
        4,
        1,
        len(code),
        len(nodes),
        3,
        2,
        1,
        -1,
        1,
        0,
        -1,
        ptr(inputs),
        ptr(code),
        ptr(words),
        ptr(steps),
        ptr(gathers),
        ptr(outputs),
        ptr(types),
        ptr(slots),
    )
    for _ in range(3):
        stats, err = Stats(), P()
        check(
            lib.dfm_execute(
                ctx, C.byref(request), C.byref(stats), C.byref(err)
            ),
            err,
        )
        assert value[0] == 0 and bitmap[0] == 1
        assert stats.memory.launches == 6


test_filtered_reduction_without_compaction()


def test_filter_gathers_only_columns_used_after_boundary():
    n = 521
    keys = [i % 7 - 3 for i in range(n)]
    key_valid = [i % 11 != 0 for i in range(n)]
    payload_valid = [i % 5 != 0 for i in range(n)]
    key_data = (C.c_int32 * n)(*keys)
    payload = (C.c_int32 * n)(*[1000 + i for i in range(n)])
    key_bits, payload_bits = packed(key_valid), packed(payload_valid)
    inputs = (Input * 2)(
        Input(ptr(key_data), ptr(key_bits), 2, 0, 1),
        Input(ptr(payload), ptr(payload_bits), 2, 0, 1),
    )
    nodes = [col, (1, -1, -1, -1), (8, 0, 1, -1), (0, -1, -1, 1)]
    code = (I * 16)(*[v for node in nodes for v in node])
    literals = (I * 4)()
    steps = (Step * 2)(Step(0, 3, 2, 1, 0, 2), Step(3, 1, 3, 0, 2, 0))
    gathers = (I * 2)(0, 1)
    result, bitmap = (C.c_int32 * n)(), packed([False] * n)
    outputs = (Output * 1)(Output(ptr(result), ptr(bitmap), 2, 3, -1, 0))
    request = Request(
        2,
        n,
        2,
        4,
        2,
        16,
        4,
        2,
        2,
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
    )
    stats, estimate, err = Stats(), Memory(), P()
    check(
        lib.dfm_estimate(C.byref(request), C.byref(estimate), C.byref(err)), err
    )
    check(
        lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)),
        err,
    )
    selected = [i for i in range(n) if key_valid[i] and keys[i] > 0]
    assert stats.rows == len(selected)
    actual_valid = unpack(bitmap, stats.rows)
    assert actual_valid == [payload_valid[i] for i in selected]
    for row, source in enumerate(selected):
        if actual_valid[row]:
            assert result[row] == payload[source]
    assert stats.memory.launches == estimate.launches == 8


test_filter_gathers_only_columns_used_after_boundary()


def test_integer_membership_exact_payload_and_descriptor_bounds():
    for dtype in [2, 3, 5, 6, 7, 8, 9, 10]:
        scalar = SCALARS[dtype]
        bits = C.sizeof(scalar) * 8
        unsigned = dtype in (7, 8, 9, 10)
        low = 0 if unsigned else -(1 << (bits - 1))
        high = (1 << (bits if unsigned else bits - 1)) - 1
        values = [low, 0, 1, 2, high, low, high, 7, 1]
        valid = [True, False, True, True, True, True, False, True, True]
        data = (scalar * len(values))(*values)
        bitmap = packed(valid, 3)
        inputs = (Input * 1)(Input(ptr(data), ptr(bitmap), dtype, 3, 1))
        # The executable program has two nodes. Three exact integer payloads
        # live outside that range; the duplicate target is intentional.
        code = (I * 20)(0, -1, -1, 0, 172, 0, 2, 3,
                        1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1)
        words = (C.c_uint64 * 5)(0, 0, low & (2**64 - 1), high, high)
        steps = (Step * 1)(Step(0, 2, 1, 0, 0, 0))
        result = (C.c_uint8 * 2)()
        output_valid = (C.c_uint8 * 2)()
        outputs = (Output * 1)(Output(ptr(result), ptr(output_valid), 4, 1, -1, 0))
        nt, st = (I * 5)(dtype, 4, dtype, dtype, dtype), (I * 2)(dtype, 4)
        request = Request(2, len(values), 0, 2, 1, 20, 5, 1, 0, 1, -1, 0, 1, -1,
                          ptr(inputs), ptr(code), ptr(words), ptr(steps), None,
                          ptr(outputs), ptr(nt), ptr(st))
        for repeat in range(2):
            stats, err = Stats(), P()
            check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
            assert unpack(output_valid, stats.rows) == valid
            assert all(not ok or got == (value in {low, high}) for got, value, ok in zip(unpack(result, stats.rows), values, valid))
            if repeat:
                assert stats.cache_hit
        for field, invalid in [(6, -1), (6, 5), (7, -1), (7, 4), (7, 2**63 - 1)]:
            before = code[field]
            code[field] = invalid
            memory, err = Memory(), P()
            status = lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
            assert status
            lib.dfm_free(err)
            code[field] = before
        nt[2] = 1
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)
        nt[2] = dtype
        # Runtime values, not generated shader text: reuse the same pipeline
        # with changed targets, then exercise empty sets and empty inputs.
        words[2], words[3], words[4] = 1, 1, 1
        stats, err = Stats(), P()
        check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
        assert stats.cache_hit
        assert all(not ok or got == (value == 1) for got, value, ok in zip(unpack(result, stats.rows), values, valid))
        code[7] = 0
        stats, err = Stats(), P()
        check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
        assert unpack(output_valid, stats.rows) == valid
        assert not any(unpack(result, stats.rows))
        request.rows = 0
        stats, err = Stats(), P()
        check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
        assert stats.rows == 0


test_integer_membership_exact_payload_and_descriptor_bounds()


def test_resident_slice_runtime_parameters():
    for n in (0, 1, 17, 257):
        values = (C.c_int32 * max(1, n))(*range(n))
        validity = [i % 5 != 2 for i in range(n)]
        valid = packed(validity)
        inputs = (Input * 1)(Input(ptr(values), ptr(valid), 2, 0, 1))
        code = (I * 12)(206, 1, 2, -1, 1, -1, -1, -1, 1, -1, -1, -1)
        words = (I * 3)(0, 0, -1)
        nt, st = (I * 3)(4, 3, 3), (I * 2)(2, 4)
        steps, gathers = (Step * 1)(Step(0, 1, 1, 1, 0, 1)), (I * 1)(0)
        output_values = (C.c_int32 * max(1, n))()
        output_valid = packed([False] * max(1, n))
        outputs = (Output * 1)(Output(ptr(output_values), ptr(output_valid), 2, 0, -1, 0))
        request = Request(2, n, 0, 2, 1, 12, 3, 1, 1, 1, -1, 0, 1, -1,
                          ptr(inputs), ptr(code), ptr(words), ptr(steps), ptr(gathers),
                          ptr(outputs), ptr(nt), ptr(st))
        for offset in (-(2**63), -300, -3, -1, 0, 1, 3, 300, 2**63-1):
            for length in (-1, 0, 1, 300, 2**63-1):
                words[1], words[2] = offset, length
                begin = min(n, max(0, n + offset) if offset < 0 else offset)
                count = n - begin if length == -1 else min(length, n - begin)
                stats, err = Stats(), P()
                check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
                assert stats.rows == count, (n, offset, length, stats.rows, count)
                assert unpack(output_valid, count) == validity[begin:begin+count]
                assert all(not validity[i] or output_values[i-begin] == i for i in range(begin, begin+count))
        words[2] = -2
        memory, err = Memory(), P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        assert b"Slice length" in C.string_at(err)
        lib.dfm_free(err)
        words[2] = -1
        code[1] = 3
        err = P()
        assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
        lib.dfm_free(err)


test_resident_slice_runtime_parameters()


def test_opaque_float64_storage_and_rejection():
    words = [0, 1 << 63, 0x3FF0000000000000, 0xBFF0000000000000,
             0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000042,
             0x7FF0000000000042, 1, (1 << 63) | 1, 0x7FEFFFFFFFFFFFFF]
    n = len(words)
    values = (C.c_uint64 * n)(*words)
    valid = [i % 4 != 1 for i in range(n)]
    bits = packed(valid, 5)
    inputs = (Input * 1)(Input(ptr(values), ptr(bits), 11, 5, 1))
    nodes = [(0, -1, -1, 0), (0, -1, -1, 0), (61, 0, -1, -1), (2, -1, -1, -1)]
    code = (I * 16)(*[x for node in nodes for x in node])
    literals = (C.c_uint64 * 4)(0, 0, 0, 0x8000000000000001)
    nt, st = (I * 4)(11, 11, 4, 11), (I * 4)(11, 11, 4, 11)
    steps = (Step * 3)(Step(0, 1, 1, 0, 0, 0), Step(1, 2, 2, 0, 0, 0), Step(3, 1, 3, 0, 0, 0))
    result, literal = (C.c_uint64 * n)(), (C.c_uint64 * n)()
    nulls = packed([False] * n)
    validity = [packed([False] * n) for _ in range(3)]
    outputs = (Output * 3)(Output(ptr(result), ptr(validity[0]), 11, 1, -1, 0),
                           Output(ptr(nulls), ptr(validity[1]), 4, 2, -1, 0),
                           Output(ptr(literal), ptr(validity[2]), 11, 3, -1, 0))
    request = Request(2, n, 0, 4, 1, 16, 4, 3, 0, 3, -1, 0, 1, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nt), ptr(st))
    stats, err = Stats(), P()
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
    assert unpack(validity[0], n) == valid
    assert all(not ok or got == want for ok, got, want in zip(valid, result, words))
    assert unpack(nulls, n) == [not v for v in valid]
    assert unpack(validity[1], n) == [True] * n
    assert list(literal) == [0x8000000000000001] * n
    assert unpack(validity[2], n) == [True] * n
    # Nonzero floating rounding carries the exact bound Float64 scale.
    # Malformed scale encodings reject before submission.
    code[8:12] = [57, 0, -1, 2]
    nt[2] = 11
    st[2] = 11
    outputs[1].dtype = 11
    memory, err = Memory(), P()
    literals[2] = C.cast(C.pointer(C.c_double(100)), C.POINTER(C.c_uint64))[0]
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    literals[2] = 0xBFF0000000000000
    assert lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err))
    assert b"Invalid Metal floating rounding scale" in C.string_at(err)
    lib.dfm_free(err)


test_opaque_float64_storage_and_rejection()


def test_bounded_wide_sums():
    for dtype in (3, 10):
        high = (1 << (63 if dtype == 3 else 64)) - 1
        for n in (0, 1, 257, 4097, 65537):
            bound = high // max(1, n)
            values = [bound if dtype == 10 or i % 3 else -bound for i in range(n)]
            valid = [i % 11 != 2 for i in range(n)]
            for minimum in (0, n + 1):
                result, bits, _ = execute(values, dtype, [col], [0], valid=valid,
                                          offset=5, reduction=10, minimum=minimum)
                assert bits == [sum(valid) >= minimum]
                if bits[0]:
                    assert result == [sum(v for v, ok in zip(values, valid) if ok)]
        result, bits, _ = execute([high], dtype, [col], [0], reduction=10)
        assert result == [high] and bits == [True]
        execute([high, 1], dtype, [col], [0], reduction=10,
                expect_error="exact wide accumulation")
        result, bits, _ = execute([high, high], dtype, [col], [0], valid=[False, False], reduction=10)
        assert result == [0] and bits == [True]
        # The predicate removes the value that cannot satisfy the bound.
        result, bits, stats = execute([high, 0], dtype, [col], [0], reduction=10,
                                      filter_nodes=[col, (1, -1, -1, -1), (20, 0, 1, -1)])
        assert result == [0] and bits == [True] and stats.memory.launches == 5
    result, bits, _ = execute([-(1 << 63)], 3, [col], [0], reduction=10)
    assert result == [-(1 << 63)] and bits == [True]
    execute([(1 << 63) - 1, -((1 << 63) - 1)], 3, [col], [0], reduction=10,
            expect_error="exact wide accumulation")
    result, bits, _ = execute([1, 2, 3], 3, [col], [0], reduction=10)
    assert result == [6] and bits == [True]


test_bounded_wide_sums()


def test_temporal_fields():
    import datetime as dt
    epoch = dt.date(1970, 1, 1)
    dates = [dt.date(1, 1, 1), dt.date(1600, 2, 29), dt.date(1900, 3, 1),
             dt.date(1969, 12, 31), epoch, dt.date(2000, 2, 29),
             dt.date(2000, 3, 1), dt.date(2024, 12, 31), dt.date(9999, 12, 31)]
    days = [(d - epoch).days for d in dates] * 29
    valid = [i % 7 != 1 for i in range(len(days))]
    for op, expected in [(130, [d.year for d in dates]),
                         (131, [d.month for d in dates]),
                         (132, [d.day for d in dates]),
                         (137, [d.isoweekday() for d in dates]),
                         (138, [d.timetuple().tm_yday for d in dates])]:
        result, bitmap, _ = execute(days, 3, [col, (op, 0, -1, 0)], [0, 0],
                                    valid=valid, offset=5, typed=True)
        assert bitmap == valid
        assert all(not ok or got == want for got, want, ok in zip(result, expected * 29, valid)), (op, result, expected)
    for kind, second in [(1, 1000), (2, 1000000), (3, 1000000000), (4, 1000000000)]:
        day = second * 86400
        values = [-(1 << 63), -day-1, -day, -second-1, -second, -1, 0, 1,
                  second-1, second, day-1, day, day+1, (1 << 63)-1] * 19
        for op in (133, 134, 135, 136, 137, 139, 140):
            if kind == 4 and op in (139, 140):
                continue
            expected = []
            for x in values:
                within = x % day
                seconds = within // second
                expected.append({133: seconds // 3600, 134: seconds // 60 % 60,
                                 135: seconds % 60, 136: within % second * (1000000000 // second),
                                 137: ((x // day + 3) % 7) + 1, 139: x // day,
                                 140: within * (1000000000 // second)}[op])
            result, bitmap, _ = execute(values, 3, [col, (op, 0, -1, kind)], [0, 0], typed=True)
            assert all(bitmap) and result == expected, (kind, op, result, expected)
    execute([0], 3, [col, (134, 0, -1, 9)], [0, 0], typed=True, expect_error="temporal field descriptor")
    execute([0], 2, [col, (134, 0, -1, 1)], [0, 0], typed=True, expect_error="temporal field descriptor")
    execute([0], 3, [col, (134, 0, -1, 1)], [0, 0], expect_error="temporal field descriptor")


test_temporal_fields()


def test_temporal_truncate():
    def i64(x):
        return C.c_int64(x).value
    for kind, second in [(0, 1), (1, 1000), (2, 1000000), (3, 1000000000), (4, 1000000000)]:
        day = 1 if kind == 0 else 86400 * second
        values = [-(1 << 63), -8 * day - 1, -7 * day, -4 * day, -1, 0, 1,
                  4 * day, 7 * day, 8 * day + 1, (1 << 63) - 1] * 25
        for step in [1, 3 * day, 7 * day, 14 * day]:
            origin = 4 * day if step % (7 * day) == 0 else 0
            expected = [i64((i64(x - origin) // step) * step + origin) for x in values]
            result, bitmap, _ = execute(values, 3, [col, (141, 0, step, kind)], [0, 0], typed=True)
            assert all(bitmap) and result == expected, (kind, step, result, expected)
    execute([0], 3, [col, (141, 0, 0, 1)], [0, 0], typed=True, expect_error="temporal field descriptor")


test_temporal_truncate()

def test_extended_expression_nodes():
    values, valid = [-7, 0, 9] * 9, [True, False, True] * 9
    for count in (65, 127, 128):
        nodes = [col] + [(50, i - 1, -1, -1) for i in range(1, count)]
        result, bitmap, _ = execute(values, 2, nodes, [0] * count,
                                     valid=valid, offset=5, typed=True)
        sign = -1 if (count - 1) % 2 else 1
        assert bitmap == valid
        assert all(not ok or got == sign * want for got, want, ok in zip(result, values, valid))
    nodes = [col] + [(50, i - 1, -1, -1) for i in range(1, 129)]
    execute(values, 2, nodes, [0] * 129, valid=valid, typed=True,
            expect_error="Invalid Metal expression step")


test_extended_expression_nodes()

def test_exact_wide_integer_reductions():
    import struct
    for dtype in (3, 10):
        high = (1 << (63 if dtype == 3 else 64)) - 1
        for n in (0, 1, 4, 257, 4097, 65537):
            values = [high if dtype == 10 or i % 4 < 2 else -high for i in range(n)]
            for valid in (None, [i % 7 != 2 for i in range(n)]):
                selected = values if valid is None else [v for v, ok in zip(values, valid) if ok]
                total = sum(selected)
                count = len(selected)
                for minimum in (0, n + 1):
                    expected_valid = count >= minimum
                    fits = -(1 << 63) <= total <= high if dtype == 3 else 0 <= total <= high
                    fault = "Integer overflow" if expected_valid and not fits else None
                    result = execute(values, dtype, [col], [0], valid=valid, offset=5,
                                     reduction=210, minimum=minimum, typed=True, expect_error=fault)
                    if fault is None:
                        output, bits, stats = result
                        assert bits == [expected_valid]
                        assert output == [total if expected_valid else 0]
                        assert stats.memory.peak > stats.memory.result
                output, bits, _ = execute(values, dtype, [col], [0], valid=valid,
                                         offset=5, reduction=211, typed=True)
                expected = struct.unpack("Q", struct.pack("d", float(total) / count))[0] if count else 0
                assert bits == [bool(count)] and output == [expected], (dtype, n, total, count, output, expected)
        execute([high, 1], dtype, [col], [0], reduction=210, typed=True,
                expect_error="Integer overflow")
        output, bits, _ = execute([high, 1], dtype, [col], [0], reduction=210,
                                 typed=True, limit=0)
        assert output == bits == []
    # Final fits after intermediate signed totals cross either 64-bit boundary.
    for values in ([-(1 << 63), -(1 << 63), (1 << 63)-1, (1 << 63)-1],
                   [(1 << 63)-1, (1 << 63)-1, -((1 << 63)-1), -((1 << 63)-1)]):
        output, bits, _ = execute(values, 3, [col], [0], reduction=210, typed=True)
        assert bits == [True] and output == [sum(values)]
    for dtype in (1, 2, 4, 5, 6, 7, 8, 9):
        for op in (210, 211):
            execute([1], dtype, [col], [0], reduction=op, typed=True,
                    expect_error="unsupported accumulator precision")
    print("PASS exact wide integer sum/mean native regressions")


test_exact_wide_integer_reductions()



def test_exact_decimal_means():
    import struct
    def expected(values, scale):
        if not values:
            return 0
        total, count = sum(values), len(values)
        per_row, left = divmod(abs(total), count)
        whole, digits = divmod(per_row, 10**scale)
        value = float(whole) + (float(digits) + float(left) / float(count)) / float(10**scale)
        if total < 0:
            value = -value
        return struct.unpack("Q", struct.pack("d", value))[0]
    for dtype, width, max_scale in ((2, 31, 9), (3, 63, 18)):
        high = (1 << width) - 1
        for scale in range(max_scale + 1):
            for values in ([high, high-1, -high, 7], [-high, -high+1, -3],
                           [high, high, high], [-(1 << width)], []):
                output, bits, _ = execute(values, dtype, [col], [0], reduction=212,
                                         minimum=scale, typed=True)
                assert bits == [bool(values)] and output == [expected(values, scale)], (dtype, scale, values, output)
        for n in (257, 4097, 65537):
            values = [high - i % 31 if i % 3 else -high for i in range(n)]
            valid = [i % 11 != 3 for i in range(n)]
            chosen = [v for v, ok in zip(values, valid) if ok]
            for scale in (0, max_scale):
                output, bits, _ = execute(values, dtype, [col], [0], valid=valid,
                                         offset=5, reduction=212, minimum=scale, typed=True)
                assert bits == [True] and output == [expected(chosen, scale)]
        output, bits, _ = execute([high, -high], dtype, [col], [0], valid=[False, False],
                                 reduction=212, minimum=max_scale, typed=True)
        assert output == [0] and bits == [False]
        for scale in (-1, max_scale + 1):
            execute([high], dtype, [col], [0], reduction=212, minimum=scale,
                    typed=True, expect_error="Invalid Metal output" if scale < 0 else "decimal mean scale")
        execute([high], dtype, [col], [0], reduction=212, minimum=max_scale,
                typed=True, budget=0, expect_error="budget")
    for dtype in (1, 4, 5, 6, 7, 8, 9, 10):
        execute([1], dtype, [col], [0], reduction=212, typed=True,
                expect_error="unsupported accumulator precision")
    print("PASS exact Decimal32/64 mean native regressions")


test_exact_decimal_means()

def test_empty_outputs_native():
    for rows in (0, 1, 513):
        for reduction in (0, 1):
            request, owners = empty_request(rows, reduction)
            memory, err = Memory(), P()
            check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
            for limit in (-1, 0, 3):
                request.limit = limit
                stats, err = Stats(), P()
                check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
                expected = 1 if reduction else rows
                assert stats.rows == (expected if limit < 0 else min(expected, limit))
                assert stats.memory.result == 0
                assert stats.memory.shared == memory.shared
                assert stats.memory.launches == 1
            request.budget = memory.peak - 1
            stats, err = Stats(), P()
            assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
            assert b"budget" in C.string_at(err)
            lib.dfm_free(err)
    print("PASS zero-column native cardinality, reduction, limits and budget regressions")


test_empty_outputs_native()
def test_empty_inputs_native():
    for rows in (0, 1, 513):
        for typed in (False, True):
            for reduction in (0, 1):
                request = empty_input_request(rows, reduction, typed)
                memory, err = Memory(), P()
                check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
                for limit in (-1, 0, 3):
                    request.limit = limit
                    stats, err = Stats(), P()
                    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
                    expected = 1 if reduction else rows
                    assert stats.rows == (expected if limit < 0 else min(expected, limit))
                    assert stats.memory.result == stats.memory.upload == stats.memory.launches == 0
                    assert stats.memory.shared == memory.shared and stats.waits == 1
                request.budget = memory.peak - 1
                stats, err = Stats(), P()
                assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
                assert b"budget" in C.string_at(err)
                lib.dfm_free(err)
    print("PASS zero-column source native cardinality, no-input/no-output accounting, limits and budget regressions")


test_empty_inputs_native()
def test_duration_totals_native():
    values = [-(1 << 63), -(1 << 63) + 1, -1000000001, -1001, -1, 0,
              1, 1001, 1000000001, (1 << 63)-1]
    for factor in (1, 1000, 1000000, 86400000000000):
        for mode in (0, 1):
            for validity in ([True] * len(values), [i % 3 != 1 for i in range(len(values))]):
                output, bits, _ = execute(values, 3, [col, (143, 0, factor, mode)], [0, 0],
                                          valid=validity, offset=5, typed=True)
                reference = [((-1 if x < 0 else 1)*(abs(x)//factor)) if mode
                             else ((x*factor+(1 << 63)) % (1 << 64))-(1 << 63) for x in values]
                assert bits == validity
                assert [x for x, valid in zip(output, bits) if valid] == [x for x, valid in zip(reference, validity) if valid]
    print("PASS signed duration totals, full-width storage, nulls and slice bitmaps")


test_duration_totals_native()
lib.dfm_context_release(ctx)
print(
    "Native Metal ABI, boundaries, filters, bitmaps, integer checks, reductions and cache: PASS"
)
