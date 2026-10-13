"""Independent fixed-duration window membership and arithmetic references."""
from pathlib import Path
import ctypes as C
import math
import random
import struct
import sys
from decimal import Decimal, localcontext

source = Path(__file__).with_name("test_native.py").read_text()
exec(compile(source.split("probe_ctx =", 1)[0], "test_native.py", "exec"))
err = P()
ctx = lib.dfm_context_create(0, 0, C.byref(err))
if not ctx:
    check(1, err)
rng = random.Random(0xD0A710)
formats = {1: "f", 2: "i", 3: "q", 5: "b", 6: "h", 7: "B", 8: "H", 9: "I", 10: "Q", 11: "d"}
widths = {dtype: struct.calcsize(fmt) for dtype, fmt in formats.items()}
fields = 0


def run(dtype, method, closed, minimum, ddof, period, n):
    global fields
    groups = [i % 3 for i in range(n)]
    gv = [i % 13 != 9 for i in range(n)]
    times = []
    previous = {}
    for i, group in enumerate(groups):
        key = group if gv[i] else None
        previous[key] = previous.get(key, -100) + rng.randrange(3)
        times.append(previous[key])
    valid = [i % 7 not in (1, 2) for i in range(n)]
    if dtype in (1, 11):
        values = [float(i % 29) - 10.25 for i in range(n)]
        if method >= 4:
            values = [1e12 + float(i % 29) / 8 for i in range(n)]
        for i in range(0, n, 17):
            values[i] = [math.nan, math.inf, -math.inf, -0.][(i // 17) % 4]
    else:
        values = [i % 29 for i in range(n)]
        if dtype in (2, 3, 5, 6):
            values = [value - 10 for value in values]
    offset = 3
    data = C.create_string_buffer(b"".join(struct.pack("=" + formats[dtype], x) for x in values) or b"\0")
    validity = packed(valid, offset)
    axis = (I * max(1, n))(*times)
    keys = (C.c_int32 * max(1, n))(*groups)
    key_valid = packed(gv, offset)
    output_type = 11 if method in (1, 4, 5) else 3 if method == 0 and dtype in (5, 6, 7, 8) else dtype
    result = (C.c_uint8 * max(1, n * widths[output_type]))()
    mask = (C.c_uint8 * max(1, (n + 7) // 8))()
    inputs = (Input * 3)(Input(ptr(data), ptr(validity), dtype, offset, 1),
                          Input(ptr(axis), None, 3, 0, 0),
                          Input(ptr(keys), ptr(key_valid), 2, offset, 1))
    code = (I * 12)(226, 0, 1, method, 227, minimum, closed, ddof, 200, 2, 0, 1)
    literals = (I * 3)(period, 0, 0)
    nodes, slots = (I * 3)(dtype, 3, 2), (I * 4)(dtype, 3, 2, output_type)
    steps = (Step * 1)(Step(0, 3, 3, 5, 0, 0))
    outputs = (Output * 1)(Output(ptr(result), ptr(mask), output_type, 3, -1, 0))
    request = Request(2, n, 0, 4, 3, 12, 3, 1, 0, 1, -1, 0, 1, -1,
                      ptr(inputs), ptr(code), ptr(literals), ptr(steps), None,
                      ptr(outputs), ptr(nodes), ptr(slots))
    memory = Memory()
    check(lib.dfm_estimate(C.byref(request), C.byref(memory), C.byref(err)), err)
    request.budget = memory.peak
    stats = Stats()
    check(lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err)), err)
    assert stats.memory.peak == memory.peak and stats.rows == n
    for row, time in enumerate(times):
        key = groups[row] if gv[row] else None
        candidates = [values[j] for j, other in enumerate(times)
                      if (groups[j] if gv[j] else None) == key and valid[j]
                      and (other >= time - period if closed & 1 else other > time - period)
                      and (other <= time if closed & 2 else other < time)]
        count = len(candidates)
        expected_valid = count >= minimum and (method == 0 or count > (ddof if method >= 4 else 0))
        actual_valid = bool(mask[row // 8] & (1 << (row % 8)))
        assert actual_valid == expected_valid, (dtype, method, closed, minimum, row, "validity")
        if not expected_valid:
            continue
        if method == 0:
            expected = sum(candidates)
            if output_type == 1:
                expected = struct.unpack("=f", struct.pack("=f", expected))[0]
        elif method == 1:
            expected = sum(candidates) / count
        elif method in (2, 3):
            nans = [value for value in candidates if math.isnan(value)]
            expected = nans[-1] if nans else (min(candidates) if method == 2 else max(candidates))
        elif any(not math.isfinite(value) for value in candidates):
            expected = math.nan
        else:
            with localcontext() as context:
                context.prec = 100
                numbers = [Decimal.from_float(value) for value in candidates]
                mean = sum(numbers) / count
                variance = sum((value - mean) ** 2 for value in numbers) / (count - ddof)
                expected = float(variance.sqrt() if method == 5 else variance)
        actual = struct.unpack("=" + formats[output_type], bytes(result[row * widths[output_type]:(row + 1) * widths[output_type]]))[0]
        if math.isnan(expected):
            assert math.isnan(actual)
        elif method >= 4:
            assert abs(actual - expected) <= 1e-12 * max(1, abs(expected)), (dtype, method, closed, minimum, n, row, groups[row] if gv[row] else None, time, candidates, actual, expected)
        else:
            assert actual == expected, (dtype, method, closed, minimum, row, actual, expected)
        fields += 1
    if n == 33 and closed == 2 and minimum == 0:
        request.budget = memory.peak - 1
        assert lib.dfm_execute(ctx, C.byref(request), C.byref(stats), C.byref(err))
        assert b"budget" in C.string_at(err)
        lib.dfm_free(err)


def exact_case(dtype, values, times, *, method=0, period=1, closed=2,
               minimum=0, valid=None, axis_valid=None, expected=None,
               expect_error=None, raw=None):
    n=len(times)
    output_type = 3 if method == 0 and dtype in (5,6,7,8) else dtype
    data=C.create_string_buffer(raw if raw is not None else b"".join(struct.pack("="+formats[dtype],x) for x in values))
    axis=(I*n)(*times)
    validity=packed(valid if valid is not None else [True]*n,3)
    av=packed(axis_valid if axis_valid is not None else [True]*n,3)
    result=(C.c_uint8*(n*widths[output_type]))(*([0x5a]*(n*widths[output_type])))
    mask=(C.c_uint8*((n+7)//8))(*([0xa5]*((n+7)//8)))
    inputs=(Input*2)(Input(ptr(data),ptr(validity),dtype,3,1),Input(ptr(axis),ptr(av),3,3,1))
    code=(I*8)(226,0,1,method,227,minimum,closed,1)
    literals=(I*2)(period,0);nodes=(I*2)(dtype,3);slots=(I*3)(dtype,3,output_type)
    steps=(Step*1)(Step(0,2,2,5,0,0));outputs=(Output*1)(Output(ptr(result),ptr(mask),output_type,2,-1,0))
    request=Request(2,n,0,3,2,8,2,1,0,1,-1,0,1,-1,ptr(inputs),ptr(code),ptr(literals),ptr(steps),None,ptr(outputs),ptr(nodes),ptr(slots))
    stats,err=Stats(),P()
    before,prior_mask=bytes(result),bytes(mask)
    status=lib.dfm_execute(ctx,C.byref(request),C.byref(stats),C.byref(err))
    if expect_error is not None:
        assert status
        assert expect_error in C.string_at(err),C.string_at(err)
        lib.dfm_free(err)
        assert bytes(result)==before and bytes(mask)==prior_mask
        return
    check(status,err)
    for row,value in enumerate(expected):
        assert bool(mask[row//8]&(1<<(row%8)))==(value is not None)
        if value is not None:
            word=bytes(result[row*widths[output_type]:(row+1)*widths[output_type]])
            assert word==(value if isinstance(value,bytes) else struct.pack("="+formats[output_type],value)),(row,word,value)


def protocol_cases():
    maximum=(1<<64)-1
    exact_case(10,[maximum,1,maximum],[0,3,6],expected=[maximum,1,maximum])
    exact_case(10,[maximum,1],[0,0],expect_error=b"overflow")
    exact_case(10,[maximum,1],[0,0],minimum=3,expected=[None,None])
    exact_case(3,[(1<<63)-1,1],[0,0],expect_error=b"overflow")
    exact_case(2,[-(1<<31),-1],[0,0],expect_error=b"overflow")
    exact_case(9,[(1<<32)-1,1],[0,0],expect_error=b"overflow")
    exact_case(3,[1,2],[2,1],expect_error=b"not sorted")
    exact_case(3,[1,2],[0,1],axis_valid=[True,False],expect_error=b"contains nulls")
    exact_case(3,[1],[-(1<<63)],valid=[False],minimum=3,expect_error=b"offset overflow")
    exact_case(3,[1],[-(1<<63)],axis_valid=[False],expect_error=b"contains nulls")
    exact_case(3,[1,2,3],[-(1<<53)-1,-(1<<53),(1<<53)+1],closed=3,expected=[1,3,3])
    for dtype in formats:
        exact_case(dtype,[0,0],[0,1],valid=[False,False],expected=[0,0])
    words=[0xfff8000000000042,0x3ff0000000000000,0x7ff0000000000123]
    payload=b"".join(struct.pack("=Q",word) for word in words)
    for method in (2,3):
        exact_case(11,[],[0,1,2],method=method,period=3,closed=3,raw=payload,
                   expected=[struct.pack("=Q",words[0]),struct.pack("=Q",words[0]),struct.pack("=Q",words[2])])
    print("PASS full-domain UInt64 sums, declared integer overflow, exact time words, zero-count sums, NaN payloads and atomic index/bounds failures")


try:
    protocol_cases()
    for n in (0, 1, 33, 257):
        for closed in range(4):
            for method in range(6):
                for minimum in (0, 2):
                    for dtype in ((11,) if method in (1, 4, 5) else formats.keys()):
                        run(dtype, method, closed, minimum, 1, 3, n)
    for ddof in (0, 2, (1 << 63) - 1):
        for method in (4, 5):
            run(11, method, 3, 0, ddof, 7, 33)
finally:
    lib.dfm_context_release(ctx)
print(f"PASS {fields} independent duration window fields: all primitive types, closed sides, duplicates, null partitions, Decimal variance/std references, exact budgets and empty inputs")
