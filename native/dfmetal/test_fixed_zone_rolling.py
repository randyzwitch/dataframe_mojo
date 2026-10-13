"""Fixed-zone duration membership with independent local-calendar and UTC references."""
from pathlib import Path
import ctypes as C
import math
import random
import struct
import sys
from decimal import Decimal, localcontext

source = Path(__file__).with_name("test_native.py").read_text()
exec(compile(source.split("probe_ctx =", 1)[0], "test_native.py", "exec"))
err=P();ctx=lib.dfm_context_create(0,0,C.byref(err))
if not ctx: check(1,err)
rng=random.Random(0xCA1E0DA7)
lo,hi=-(1<<63),(1<<63)-1
fields=failures=0

def wrap(value): return (value-lo) % (1<<64)+lo

def checked(value):
    if not lo <= value <= hi: raise OverflowError
    return value

def year_start(year):
    previous=year-1
    return 365*previous+previous//4-previous//100+previous//400-719162

def lengths(year):
    return [31,29 if year%4==0 and (year%100!=0 or year%400==0) else 28,31,30,31,30,31,31,30,31,30,31]

def calendar_lower(value,unit,months,fixed):
    per_second=(0,1000,1000000,1000000000)[unit]
    days=value if not unit else value//(86400*per_second)
    within=0 if not unit else value%(86400*per_second)
    left,right=-30000000000000000,30000000000000000
    while left+1<right:
        middle=(left+right)//2
        if year_start(middle)<=days: left=middle
        else: right=middle
    year,day,month=left,days-year_start(left),0
    old=lengths(year)
    while day>=old[month]: day-=old[month];month+=1
    year,month=divmod(wrap(year*12+month-months),12)
    days=wrap(year_start(year)+sum(lengths(year)[:month])+min(day+1,lengths(year)[month])-1)
    value=days if not unit else checked(checked(checked(checked(days*86400)+within//per_second)*per_second)+within%per_second)
    return checked(value-fixed) if fixed else value

def lower(value,unit,months,fixed,days,offset):
    per_second=(0,1000,1000000,1000000000)[unit]
    def relocalize(local):
        seconds,fraction=divmod(local,per_second)
        return checked(checked((seconds-offset)*per_second)+fraction)
    if months:
        value=relocalize(calendar_lower(wrap(value+offset*per_second),unit,months,0))
    if days:
        local=wrap(value+offset*per_second)
        value=relocalize(checked(local+checked(-days*86400*per_second)))
    return checked(value-fixed)

def run(unit,months,fixed,method,closed,minimum,calendar_days=0,offset=0,extreme=False):
    global fields,failures
    per_day=(1,86400000,86400000000,86400000000000)[unit]
    # Includes leap-century changes, month ends, negative epochs and duplicate indices.
    days=sorted([year_start(y)+sum(lengths(y)[:m])+lengths(y)[m]-1 for y in (1900,1999,2000,2001) for m in range(12)])
    times=sorted([day*per_day+(0 if not unit else 12345) for day in days]+[0,0,-1,1])
    if extreme: times=sorted([lo,lo+1,hi-1,hi]+[rng.randrange(lo,hi+1) for _ in range(48)])
    n=len(times);keys=[i%3 for i in range(n)];kv=[i%13!=4 for i in range(n)]
    valid=[i%7 not in (1,2) for i in range(n)]
    values=[float(1000000+i%19)/8 if method else i%19-9 for i in range(n)]
    groups=[key if ok else None for key,ok in zip(keys,kv)]
    try: starts=[lower(time,unit,months,fixed,calendar_days,offset) for time in times]
    except OverflowError: starts=None
    dtype=11 if method else 3
    data=(C.c_double*n)(*values) if dtype==11 else (I*n)(*values)
    axis=(I*n)(*times);keydata=(C.c_int32*n)(*keys)
    masks=packed(valid,5);keymask=packed(kv,5)
    result=(C.c_uint8*(8*n))(*([0x5a]*(8*n)));bits=(C.c_uint8*((n+7)//8))(*([0xa5]*((n+7)//8)))
    inputs=(Input*3)(Input(ptr(data),ptr(masks),dtype,5,1),Input(ptr(axis),None,3,0,0),Input(ptr(keydata),ptr(keymask),2,5,1))
    words=[226,0,1,method,227,minimum,closed,1]
    node_types=[dtype,3]
    if months:
        words.extend([228,months,unit,0]);node_types.append(3)
    words.extend([229,calendar_days,unit,offset,200,2,0,1]);node_types.extend([3,2])
    size=len(node_types);code=(I*len(words))(*words)
    literals=(I*size)(fixed,*([0]*(size-1)));nodes=(I*size)(*node_types);slots=(I*4)(dtype,3,2,dtype)
    steps=(Step*1)(Step(0,size,3,5,0,0));outputs=(Output*1)(Output(ptr(result),ptr(bits),dtype,3,-1,0))
    request=Request(2,n,0,4,3,len(words),size,1,0,1,-1,0,1,-1,ptr(inputs),ptr(code),ptr(literals),ptr(steps),None,ptr(outputs),ptr(nodes),ptr(slots))
    memory=Memory();check(lib.dfm_estimate(C.byref(request),C.byref(memory),C.byref(err)),err)
    request.budget=memory.peak;stats=Stats();before,prior=bytes(result),bytes(bits)
    status=lib.dfm_execute(ctx,C.byref(request),C.byref(stats),C.byref(err))
    if starts is None:
        assert status and b"offset overflow" in C.string_at(err),C.string_at(err)
        lib.dfm_free(err);err.value=None
        assert bytes(result)==before and bytes(bits)==prior
        failures+=1;return
    check(status,err);assert stats.memory.peak==memory.peak
    for row,(time,start) in enumerate(zip(times,starts)):
        candidates=[values[j] for j,other in enumerate(times) if groups[j]==groups[row] and valid[j]
                    and (other>=start if closed&1 else other>start) and (other<=time if closed&2 else other<time)]
        count=len(candidates);ok=count>=minimum and (method==0 or count>(1 if method>=4 else 0))
        assert bool(bits[row//8]&(1<<(row%8)))==ok,(unit,months,method,closed,row,"validity")
        if ok:
            if method==0: expected=sum(candidates)
            elif method==1: expected=sum(candidates)/count
            elif method==2: expected=min(candidates)
            elif method==3: expected=max(candidates)
            else:
                with localcontext() as context:
                    context.prec=100;numbers=[Decimal.from_float(value) for value in candidates]
                    mean=sum(numbers)/count;variance=sum((value-mean)**2 for value in numbers)/(count-1)
                    expected=float(variance.sqrt() if method==5 else variance)
            actual=struct.unpack("=d" if dtype==11 else "=q",bytes(result[8*row:8*(row+1)]))[0]
            assert abs(actual-expected)<=abs(expected)*2e-12+1e-12,(unit,months,fixed,method,closed,row,actual,expected)
        fields+=1
    request.budget=memory.peak-1;before,prior=bytes(result),bytes(bits)
    assert lib.dfm_execute(ctx,C.byref(request),C.byref(stats),C.byref(err))
    assert b"budget" in C.string_at(err);lib.dfm_free(err);err.value=None
    assert bytes(result)==before and bytes(bits)==prior

try:
    for unit in (1,2,3):
        for offset in (-86340,-12600,20700,50400,86340):
            for method in range(6):
                for closed in range(4):
                    run(unit,1,5,method,closed,0,1,offset)
            for months in (0,1):
                for method in (0,4,5):
                    run(unit,months,0,method,3,0,200000,offset)
            for months,days,fixed in ((0,0,5),(0,1,0),(1,0,0),(1,1,5)):
                for method in (0,4,5):
                    run(unit,months,fixed,method,3,0,days,offset,True)
finally: lib.dfm_context_release(ctx)
print(f"PASS {fields} independent fixed-offset duration fields and {failures} atomic bound failures: signed minute offsets, local month/day stages, wrapping local ticks, checked UTC reconstruction, wide periods, closed sides and exact budgets")
