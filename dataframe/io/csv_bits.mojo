"""Polars CSV SIMD masks and target-selected inclusive quote parity.

Maps Rust Mask::to_bitmask and polars-utils/src/clmul.rs. The private target
feature query is pinned to this repository's Mojo compiler version.
"""
from std.memory import bitcast, pack_bits
from std.sys import CompilationTarget, llvm_intrinsic


@always_inline
def _mask64(matches: SIMD[DType.bool, 64]) -> UInt64:
    return pack_bits[DType.uint64](matches)


@always_inline
def _prefix_xor_inclusive(mask: UInt64) -> UInt64:
    comptime if CompilationTarget.is_x86() and CompilationTarget._has_feature[
        "pclmul"
    ]():
        var lhs = SIMD[DType.uint64, 2](mask, 0)
        var rhs = SIMD[DType.uint64, 2](UInt64(0xFFFFFFFFFFFFFFFF), 0)
        return llvm_intrinsic[
            "llvm.x86.pclmulqdq", SIMD[DType.uint64, 2], has_side_effect=False
        ](lhs, rhs, UInt8(0))[0]
    elif CompilationTarget.has_neon() and CompilationTarget._has_feature[
        "aes"
    ]():
        var product = llvm_intrinsic[
            "llvm.aarch64.neon.pmull64",
            SIMD[DType.uint8, 16],
            has_side_effect=False,
        ](mask, UInt64(0xFFFFFFFFFFFFFFFF))
        return bitcast[DType.uint64, 2](product)[0]
    else:
        var parity = mask
        comptime for i in range(6):
            parity ^= parity << UInt64(1 << i)
        return parity
