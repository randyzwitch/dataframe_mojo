"""Known requested-byte lifetimes for the optional compiler allocation probe."""
from std.benchmark.compiler import keep
from std.ffi import external_call
from std.memory import Pointer


def main():
    var a = external_call["KGEN_CompilerRT_AlignedAlloc", Int](16, 64)
    var b = external_call["KGEN_CompilerRT_AlignedAlloc", Int](16, 32)
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a)[] = 11
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=b)[] = 22
    keep(a)
    keep(b)
    external_call["KGEN_CompilerRT_AlignedFree", NoneType](a)
    var c = external_call["KGEN_CompilerRT_AlignedAlloc", Int](16, 128)
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=c)[] = 33
    keep(c)
    external_call["KGEN_CompilerRT_AlignedFree", NoneType](b)
    external_call["KGEN_CompilerRT_AlignedFree", NoneType](c)
