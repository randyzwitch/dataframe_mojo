"""Exact Python-integer references for precision-bounded duration decimal sums."""
from pathlib import Path
import ctypes as C
import random
import sys

source=Path(__file__).with_name("test_native.py").read_text()
exec(compile(source.split("probe_ctx =",1)[0],"test_native.py","exec"))
err=P()
ctx=lib.dfm_context_create(0,0,C.byref(err))
if not ctx:check(1,err)
rng=random.Random(0xDEC1D0)
fields=errors=0


def run(precision,closed,minimum,n,safe=False):
    global fields,errors
    limit=10**precision
    keys=[i%3 for i in range(n)]
    kv=[i%11!=7 for i in range(n)]
    times=[];previous={}
    for row,key in enumerate(keys):
        group=key if kv[row] else None
        previous[group]=previous.get(group,0)+(1 if safe else rng.randrange(3))
        times.append(previous[group])
    valid=[i%7 not in (1,2) for i in range(n)]
    pool=[0,1,-1,limit-1,1-limit]
    values=[pool[i%len(pool)] if i%2 else rng.randrange(1-limit,limit) for i in range(n)]
    if safe:
        factor=(1<<100) if precision==38 else 1
        span=1 if precision==1 else 7
        values=[rng.randrange(-span,span+1)*factor for _ in range(n)]
    payload=b"".join((value%(1<<128)).to_bytes(16,"little") for value in values)
    data=C.create_string_buffer(payload or b"\0")
    axis=(I*max(1,n))(*times);groups=(C.c_int32*max(1,n))(*keys)
    validity=packed(valid,5);key_valid=packed(kv,5)
    result=(C.c_uint8*max(1,n*16))(*([0x5a]*max(1,n*16)))
    mask=(C.c_uint8*max(1,(n+7)//8))(*([0xa5]*max(1,(n+7)//8)))
    inputs=(Input*3)(Input(ptr(data),ptr(validity),12,5,1),Input(ptr(axis),None,3,0,0),Input(ptr(groups),ptr(key_valid),2,5,1))
    code=(I*12)(226,0,1,0,227,minimum,closed,precision,200,2,0,1)
    literals=(I*3)(2,0,0);nodes=(I*3)(12,3,2);slots=(I*4)(12,3,2,12)
    steps=(Step*1)(Step(0,3,3,5,0,0));outputs=(Output*1)(Output(ptr(result),ptr(mask),12,3,-1,0))
    request=Request(2,n,0,4,3,12,3,1,0,1,-1,0,1,-1,ptr(inputs),ptr(code),ptr(literals),ptr(steps),None,ptr(outputs),ptr(nodes),ptr(slots))
    memory,err=Memory(),P()
    check(lib.dfm_estimate(C.byref(request),C.byref(memory),C.byref(err)),err)
    request.budget=memory.peak
    expected=[];overflow=False
    for row,time in enumerate(times):
        key=keys[row] if kv[row] else None
        selected=[values[j] for j,other in enumerate(times) if (keys[j] if kv[j] else None)==key and valid[j]
                  and (other>=time-2 if closed&1 else other>time-2)
                  and (other<=time if closed&2 else other<time)]
        if len(selected)<minimum:
            expected.append(None);continue
        total=0
        for value in selected:
            total+=value
            if not -limit<value<limit or not -limit<total<limit:
                overflow=True;break
        expected.append(total)
    before,prior_mask=bytes(result),bytes(mask)
    stats=Stats()
    status=lib.dfm_execute(ctx,C.byref(request),C.byref(stats),C.byref(err))
    if overflow:
        assert status
        assert b"decimal precision" in C.string_at(err),C.string_at(err)
        lib.dfm_free(err)
        assert bytes(result)==before and bytes(mask)==prior_mask
        errors+=1
        return
    check(status,err)
    assert stats.memory.peak==memory.peak and stats.rows==n
    for row,value in enumerate(expected):
        assert bool(mask[row//8]&(1<<(row%8)))==(value is not None)
        if value is not None:
            actual=int.from_bytes(bytes(result[row*16:(row+1)*16]),"little",signed=True)
            assert actual==value,(precision,closed,minimum,n,row,actual,value)
            fields+=1
    if n==33 and (minimum>n or safe and minimum==0 and closed==2):
        request.budget=memory.peak-1
        before,prior_mask=bytes(result),bytes(mask)
        assert lib.dfm_execute(ctx,C.byref(request),C.byref(stats),C.byref(err))
        assert b"budget" in C.string_at(err)
        lib.dfm_free(err)
        assert bytes(result)==before and bytes(mask)==prior_mask


try:
    for precision in (1,3,9,18,38):
        for closed in range(4):
            for n in (0,1,33,257):
                for minimum in (0,1,n+1):
                    run(precision,closed,minimum,n)
            for n in (33,257):
                for minimum in (0,1):
                    run(precision,closed,minimum,n,True)
finally:
    lib.dfm_context_release(ctx)
print(f"PASS {fields} exact decimal duration fields and {errors} atomic prefix failures: precisions, signed extremes, closed sides, null partitions, masks, empty and budget cases")
