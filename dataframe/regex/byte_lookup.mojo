"""Sixteen-byte table lookup for nibble indices in the regex run scanner."""
from std.sys import CompilationTarget, llvm_intrinsic


@always_inline
def lookup16(
    table: SIMD[DType.uint8, 16], indices: SIMD[DType.uint8, 16]
) -> SIMD[DType.uint8, 16]:
    # Indices are masked to four bits, giving both target instructions the
    # same semantics. Other targets retain the scalar definition.
    comptime if CompilationTarget.has_neon():
        return llvm_intrinsic[
            "llvm.aarch64.neon.tbl1.v16i8",
            SIMD[DType.uint8, 16],
            has_side_effect=False,
        ](table, indices)
    elif CompilationTarget.is_x86() and CompilationTarget._has_feature[
        "ssse3"
    ]():
        return llvm_intrinsic[
            "llvm.x86.ssse3.pshuf.b.128",
            SIMD[DType.uint8, 16],
            has_side_effect=False,
        ](table, indices)
    else:
        var result = SIMD[DType.uint8, 16]()
        comptime for i in range(16):
            result[i] = table[Int(indices[i])]
        return result
