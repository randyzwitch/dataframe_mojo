"""Strict CSV integer conversion ported from atoi_simd 0.18.1.

Polars 1.44.2 pins atoi_simd 0.18.1.  This module implements the complete
`parse::<_, true, true>` contract for the CSV primitive builders: optional `+`,
signed `-` for signed destinations, unlimited leading-zero skipping, complete
input validation, and destination-range checks. It is CSV-only and does not
change dataframe.parse's cast semantics.

Source mapping:
* `short.rs` supplies the 1..3 byte scalar route.
* `fallback.rs::{load_8,check_len_8,process_8,process_16}` supplies the
  bounds-safe packed fallback.
* `simd/sse_avx.rs::parse_simd_sse` supplies the 16-byte vector validation and
  pair/group reduction shape. The actual intrinsic path is compile-time gated
  for x86 SSSE3/SSE4.1 (and AVX2 for 17..20-digit 64-bit values); other targets
  use the packed SWAR fallback. A dedicated Neon madd reduction is not exposed
  by this module yet, so it is intentionally not described as a SIMD backend.

atoi_simd is MIT OR Apache-2.0; this repository retains its MIT text at
`third_party/ATOI_SIMD_LICENSE`.
"""
from std.memory import bitcast, pack_bits
from std.bit import count_trailing_zeros
from std.sys import CompilationTarget, size_of, llvm_intrinsic
from std.sys.info import is_little_endian


@always_inline
def _load_8(bytes: Span[UInt8, _], start: Int, digits: Int) -> UInt64:
    """atoi_simd fallback::load_8, with ASCII-zero padding outside input."""
    if digits == 8:
        return bitcast[DType.uint64, 1](
            bytes.unsafe_ptr().unsafe_load[width=8](start)
        )
    var word = UInt64(0)
    for i in range(digits):
        word |= UInt64(bytes[start + i]) << UInt64(i * 8)
    var mask = (UInt64(1) << UInt64(digits * 8)) - 1
    return word | (UInt64(0x3030303030303030) & ~mask)


@always_inline
def _all_digits_8(word: UInt64) -> Bool:
    return (
        (
            (word + UInt64(0x4646464646464646))
            | (word - UInt64(0x3030303030303030))
        )
        & UInt64(0x8080808080808080)
    ) == 0


@always_inline
def _process_8(word: UInt64, digits: Int) raises -> UInt64:
    """atoi_simd fallback::process_8 after its full-byte digit check."""
    if not _all_digits_8(word):
        raise Error("invalid CSV integer byte")
    var value = word << UInt64((8 - digits) * 8)
    value = (value & UInt64(0x0F0F0F0F0F0F0F0F)) * UInt64(0xA01) >> 8
    value = (value & UInt64(0x00FF00FF00FF00FF)) * UInt64(0x640001) >> 16
    return (value & UInt64(0x0000FFFF0000FFFF)) * UInt64(0x271000000001) >> 32


@always_inline
def _parse_short(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt64:
    """atoi_simd short::{parse_short_pos,parse_short_neg}'s 1..3-byte route."""
    var value = UInt64(0)
    for i in range(digits):
        var byte = bytes[start + i]
        if byte < 48 or byte > 57:
            raise Error("invalid CSV integer byte")
        value = value * 10 + UInt64(byte - 48)
    return value


@always_inline
def _parse_swar_at_most_16(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt64:
    """fallback::parse_16_by_8 for a known strict, nonempty suffix."""
    if digits <= 8:
        return _process_8(_load_8(bytes, start, digits), digits)
    var low = _process_8(_load_8(bytes, start, 8), 8)
    var high_digits = digits - 8
    var high = _process_8(_load_8(bytes, start + 8, high_digits), high_digits)
    return low * _pow10(high_digits) + high


@always_inline
def _load_le_up_to_8(bytes: Span[UInt8, _], start: Int, digits: Int) -> UInt64:
    """The bounded 0..8-byte half of sse_avx.rs::load's sized switch."""
    if digits == 0:
        return 0
    if digits == 1:
        return UInt64(bytes[start])
    if digits == 2:
        return UInt64(
            bitcast[DType.uint16, 1](
                bytes.unsafe_ptr().unsafe_load[width=2](start)
            )
        )
    if digits == 3:
        return (
            UInt64(
                bitcast[DType.uint16, 1](
                    bytes.unsafe_ptr().unsafe_load[width=2](start)
                )
            )
            | UInt64(bytes[start + 2]) << 16
        )
    if digits == 4:
        return UInt64(
            bitcast[DType.uint32, 1](
                bytes.unsafe_ptr().unsafe_load[width=4](start)
            )
        )
    if digits == 5:
        return (
            UInt64(
                bitcast[DType.uint32, 1](
                    bytes.unsafe_ptr().unsafe_load[width=4](start)
                )
            )
            | UInt64(bytes[start + 4]) << 32
        )
    if digits == 6:
        return (
            UInt64(
                bitcast[DType.uint32, 1](
                    bytes.unsafe_ptr().unsafe_load[width=4](start)
                )
            )
            | UInt64(
                bitcast[DType.uint16, 1](
                    bytes.unsafe_ptr().unsafe_load[width=2](start + 4)
                )
            )
            << 32
        )
    if digits == 7:
        return (
            UInt64(
                bitcast[DType.uint32, 1](
                    bytes.unsafe_ptr().unsafe_load[width=4](start)
                )
            )
            | UInt64(
                bitcast[DType.uint16, 1](
                    bytes.unsafe_ptr().unsafe_load[width=2](start + 4)
                )
            )
            << 32
            | UInt64(bytes[start + 6]) << 48
        )
    return bitcast[DType.uint64, 1](
        bytes.unsafe_ptr().unsafe_load[width=8](start)
    )


@always_inline
def _load_sse_lanes(
    bytes: Span[UInt8, _], start: Int, digits: Int
) -> Tuple[UInt128, SIMD[DType.uint8, 16]]:
    """sse_avx.rs::load for a bounded 1..16 byte field, zero padded above."""
    if digits <= 8:
        var raw = UInt128(_load_le_up_to_8(bytes, start, digits))
        return (raw, bitcast[DType.uint8, 16](raw))
    var low = UInt128(_load_le_up_to_8(bytes, start, 8))
    var high = UInt128(_load_le_up_to_8(bytes, start + 8, digits - 8))
    var raw = low | (high << 64)
    return (raw, bitcast[DType.uint8, 16](raw))


@always_inline
def _parse_sse_madd(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt64:
    """Literal SSE `load_len` + maddubs/madd/pack/madd reduction from atoi_simd.

    The sized-load switch supplies a zero-padded vector. Its packed bad-byte
    mask yields the same prefix length as `_mm_movemask_epi8(...).trailing_zeros`.
    """
    var loaded = _load_sse_lanes(bytes, start, digits)
    var raw = loaded[0]
    var chars = loaded[1]
    var below = chars.lt(SIMD[DType.uint8, 16](48))
    var above = chars.gt(SIMD[DType.uint8, 16](57))
    var bad = pack_bits[DType.uint16](below | above)[0]
    var parsed = Int(count_trailing_zeros(bad))
    if parsed != digits:
        raise Error("invalid CSV integer byte")

    # _mm_bslli_si128(chunk, 16-len), then `to_numbers(chunk)`.
    var numbers = (
        raw & UInt128(0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F)
    ) << UInt128((16 - digits) * 8)
    var aligned = bitcast[DType.uint8, 16](numbers)
    var pairs = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128",
        SIMD[DType.uint16, 8],
        has_side_effect=False,
    ](
        aligned,
        SIMD[DType.uint8, 16](
            10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1
        ),
    )
    var quads = llvm_intrinsic[
        "llvm.x86.sse2.pmadd.wd",
        SIMD[DType.uint32, 4],
        has_side_effect=False,
    ](pairs, SIMD[DType.uint16, 8](100, 1, 100, 1, 100, 1, 100, 1))
    var packed = llvm_intrinsic[
        "llvm.x86.sse41.packusdw",
        SIMD[DType.uint16, 8],
        has_side_effect=False,
    ](quads, quads)
    var groups = llvm_intrinsic[
        "llvm.x86.sse2.pmadd.wd",
        SIMD[DType.uint32, 4],
        has_side_effect=False,
    ](packed, SIMD[DType.uint16, 8](10000, 1, 10000, 1, 0, 0, 0, 0))
    return UInt64(groups[0]) * UInt64(100000000) + UInt64(groups[1])


@always_inline
def _parse_avx_17_to_20(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt128:
    """atoi_simd sse_avx::{load_avx_len,process_avx} for its 17..20 route."""
    var first = _load_sse_lanes(bytes, start, 16)
    var first64 = bitcast[DType.uint64, 2](first[1])
    var tail = _load_le_up_to_8(bytes, start + 16, digits - 16)
    var raw64 = SIMD[DType.uint64, 4](first64[0], first64[1], tail, 0)
    var chars = bitcast[DType.uint8, 32](raw64)
    var below = chars.lt(SIMD[DType.uint8, 32](48))
    var above = chars.gt(SIMD[DType.uint8, 32](57))
    var bad = pack_bits[DType.uint32](below | above)[0]
    var parsed = Int(count_trailing_zeros(bad))
    if parsed != digits:
        raise Error("invalid CSV integer byte")

    # process_avx first aligns the 17..31 byte forms to its 32-byte reduction.
    # These 17..20 forms shift by 12..15 bytes, so the cross-lane expression is
    # the direct scalar spelling of its alignr/permute preparation.
    var shift = UInt64((32 - digits) * 8 - 64)
    var aligned64 = SIMD[DType.uint64, 4](
        0,
        first64[0] << shift,
        (first64[1] << shift) | (first64[0] >> UInt64(64 - shift)),
        (tail << shift) | (first64[1] >> UInt64(64 - shift)),
    )
    var aligned = bitcast[DType.uint8, 32](aligned64) & SIMD[DType.uint8, 32](
        15
    )
    var pairs = llvm_intrinsic[
        "llvm.x86.avx2.pmadd.ub.sw",
        SIMD[DType.uint16, 16],
        has_side_effect=False,
    ](
        aligned,
        SIMD[DType.uint8, 32](
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
            10,
            1,
        ),
    )
    var quads = llvm_intrinsic[
        "llvm.x86.avx2.pmadd.wd",
        SIMD[DType.uint32, 8],
        has_side_effect=False,
    ](
        pairs,
        SIMD[DType.uint16, 16](
            100,
            1,
            100,
            1,
            100,
            1,
            100,
            1,
            100,
            1,
            100,
            1,
            100,
            1,
            100,
            1,
        ),
    )
    var packed = llvm_intrinsic[
        "llvm.x86.avx2.packusdw",
        SIMD[DType.uint16, 16],
        has_side_effect=False,
    ](quads, quads)
    # _mm256_permute4x64_epi64(..., 8) selects packed q0..q3 then q4..q7.
    var packed64 = bitcast[DType.uint64, 4](packed)
    var selected = bitcast[DType.uint16, 8](
        SIMD[DType.uint64, 2](packed64[0], packed64[2])
    )
    var groups = llvm_intrinsic[
        "llvm.x86.sse2.pmadd.wd",
        SIMD[DType.uint32, 4],
        has_side_effect=False,
    ](selected, SIMD[DType.uint16, 8](10000, 1, 10000, 1, 10000, 1, 10000, 1))
    # process_avx subsequently combines these four 8-digit groups. Keeping
    # the UInt128 fold makes the 20-digit range check explicit and exact.
    var value = UInt128(0)
    for i in range(4):
        value = value * UInt128(100000000) + UInt128(groups[i])
    return value


@always_inline
def _parse_simd_16(bytes: Span[UInt8, _], start: Int) raises -> UInt64:
    return _parse_sse_madd(bytes, start, 16)


@always_inline
def _parse_simd_at_most_16(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt64:
    return _parse_sse_madd(bytes, start, digits)


@always_inline
def _has_x86_atoi_simd_backend() -> Bool:
    # linker/simd_32.rs' x86 cfg: SSE2/SSE3/SSE4.1/SSSE3.
    comptime if (
        CompilationTarget.is_x86()
        and CompilationTarget._has_feature["sse2"]()
        and CompilationTarget._has_feature["sse4.1"]()
        and CompilationTarget._has_feature["sse3"]()
        and CompilationTarget._has_feature["ssse3"]()
    ):
        return True
    return False


@always_inline
def _has_wide_atoi_simd_backend() -> Bool:
    # linker/simd_64.rs' x86 cfg adds AVX+AVX2.
    comptime if (
        _has_x86_atoi_simd_backend()
        and CompilationTarget.has_avx()
        and CompilationTarget.has_avx2()
    ):
        return True
    return False


@always_inline
def _pow10(digits: Int) -> UInt64:
    # `digits` is 0..8 at all call sites, directly matching process_16 tails.
    comptime powers: SIMD[DType.uint64, 16] = SIMD[DType.uint64, 16](
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    # A static SIMD table is the available fixed-table representation in this
    # pinned Mojo toolchain (InlineArray is not exported by this stdlib build).
    return powers[digits]


@always_inline
def _parse_magnitude_u64[
    wide: Bool
](bytes: Span[UInt8, _], start: Int, digits: Int) raises -> UInt64:
    """Monomorphized u64 normal route from atoi_simd's checked parsers."""
    comptime assert is_little_endian(), "atoi_simd packed path requires LE"
    comptime if _has_x86_atoi_simd_backend():
        comptime if wide:
            comptime if _has_wide_atoi_simd_backend():
                if digits < 4:
                    return _parse_short(bytes, start, digits)
                return _parse_simd_at_most_16(bytes, start, digits)
            else:
                if digits < 5:
                    return _parse_short(bytes, start, digits)
                return _parse_swar_at_most_16(bytes, start, digits)
        else:
            if digits < 4:
                return _parse_short(bytes, start, digits)
            return _parse_simd_at_most_16(bytes, start, digits)
    else:
        if digits < 5:
            return _parse_short(bytes, start, digits)
        return _parse_swar_at_most_16(bytes, start, digits)


@always_inline
def _parse_wide_fallback_17_to_20(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt128:
    """fallback::parse_fb_64_pos's 16-byte block plus bounded tail."""
    var low = _parse_swar_at_most_16(bytes, start, 16)
    var tail_digits = digits - 16
    var tail = _parse_swar_at_most_16(bytes, start + 16, tail_digits)
    return UInt128(low) * UInt128(_pow10(tail_digits)) + UInt128(tail)


@always_inline
def _parse_wide_17_to_20(
    bytes: Span[UInt8, _], start: Int, digits: Int
) raises -> UInt128:
    """Compile-time AVX2/fallback selection for atoi_simd's wide route."""
    comptime if _has_wide_atoi_simd_backend():
        return _parse_avx_17_to_20(bytes, start, digits)
    return _parse_wide_fallback_17_to_20(bytes, start, digits)


@always_inline
def _skip_zeroes(bytes: Span[UInt8, _], mut index: Int) -> Int:
    # SKIP_ZEROES recovery after the source route reaches a size boundary.
    while index < len(bytes) and bytes[index] == 48:
        index += 1
    return index


@always_inline
def _parse_csv_unsigned[
    wide: Bool
](text: StringSlice, maximum: UInt64) raises -> UInt64:
    """Typed atoi_simd unsigned front end; the normal route returns a u64."""
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty CSV integer")
    var index = 0
    var negative = False
    if bytes[0] == 43 or bytes[0] == 45:
        negative = bytes[0] == 45
        index = 1
    if index == len(bytes):
        raise Error("CSV integer sign without digits")

    var digits = len(bytes) - index
    var normal_limit = 16
    comptime if wide:
        normal_limit = 20
        # AVX's <17 path does not defer a 16-byte leading-zero field.
        comptime if not _has_wide_atoi_simd_backend():
            if digits == 16 and bytes[index] == 48:
                index = _skip_zeroes(bytes, index)
                if index == len(bytes):
                    if negative:
                        raise Error("CSV unsigned integer overflow")
                    return 0
                digits = len(bytes) - index
    else:
        # parse_simd_16::<true> and fallback::parse_16_by_8 only examine
        # leading zeroes after a full 16-byte block.
        if digits == 16 and bytes[index] == 48:
            index = _skip_zeroes(bytes, index)
            if index == len(bytes):
                if negative:
                    raise Error("CSV unsigned integer overflow")
                return 0
            digits = len(bytes) - index

    # SKIP_ZEROES is a recovery path after a size-boundary failure, never a
    # prepass for ordinary values.
    if digits > normal_limit:
        index = _skip_zeroes(bytes, index)
        if index == len(bytes):
            if negative:
                raise Error("CSV unsigned integer overflow")
            return 0
        digits = len(bytes) - index
    if digits > normal_limit:
        for i in range(index, len(bytes)):
            if bytes[i] < 48 or bytes[i] > 57:
                raise Error("invalid CSV integer byte")
        raise Error("CSV integer overflow")

    if digits <= 16:
        var magnitude = _parse_magnitude_u64[wide](bytes, index, digits)
        if negative or magnitude > maximum:
            raise Error("CSV unsigned integer overflow")
        return magnitude

    comptime if wide:
        var long_magnitude = _parse_wide_17_to_20(bytes, index, digits)
        if negative or long_magnitude > UInt128(maximum):
            raise Error("CSV unsigned integer overflow")
        return UInt64(long_magnitude)
    raise Error("CSV integer overflow")


@always_inline
def _parse_csv_signed[
    wide: Bool
](text: StringSlice, maximum: UInt64) raises -> Int64:
    """Typed atoi_simd signed front end; the normal route returns an i64."""
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty CSV integer")
    var index = 0
    var negative = False
    if bytes[0] == 43 or bytes[0] == 45:
        negative = bytes[0] == 45
        index = 1
    if index == len(bytes):
        raise Error("CSV integer sign without digits")

    var digits = len(bytes) - index
    var normal_limit = 16
    comptime if wide:
        normal_limit = 20
        comptime if not _has_wide_atoi_simd_backend():
            if digits == 16 and bytes[index] == 48:
                index = _skip_zeroes(bytes, index)
                if index == len(bytes):
                    return 0
                digits = len(bytes) - index
    else:
        if digits == 16 and bytes[index] == 48:
            index = _skip_zeroes(bytes, index)
            if index == len(bytes):
                return 0
            digits = len(bytes) - index

    if digits > normal_limit:
        index = _skip_zeroes(bytes, index)
        if index == len(bytes):
            return 0
        digits = len(bytes) - index
    if digits > normal_limit:
        for i in range(index, len(bytes)):
            if bytes[i] < 48 or bytes[i] > 57:
                raise Error("invalid CSV integer byte")
        raise Error("CSV integer overflow")

    if digits <= 16:
        var magnitude = _parse_magnitude_u64[wide](bytes, index, digits)
        var limit = maximum + UInt64(1) if negative else maximum
        if magnitude > limit:
            raise Error("CSV signed integer overflow")
        if not negative:
            return Int64(magnitude)
        if magnitude == UInt64(9223372036854775808):
            return Int64(-9223372036854775807) - 1
        return -Int64(magnitude)

    comptime if wide:
        var long_magnitude = _parse_wide_17_to_20(bytes, index, digits)
        var limit = UInt128(maximum + UInt64(1)) if negative else UInt128(
            maximum
        )
        if long_magnitude > limit:
            raise Error("CSV signed integer overflow")
        if not negative:
            return Int64(UInt64(long_magnitude))
        if long_magnitude == UInt128(9223372036854775808):
            return Int64(-9223372036854775807) - 1
        return -Int64(UInt64(long_magnitude))
    raise Error("CSV integer overflow")


def parse_csv_uint64(text: StringSlice) raises -> UInt64:
    """atoi_simd::parse::<u64, true, true> with strict full consumption."""
    return _parse_csv_unsigned[True](text, UInt64.MAX)


def parse_csv_int64(text: StringSlice) raises -> Int64:
    """atoi_simd::parse::<i64, true, true> with strict full consumption."""
    return _parse_csv_signed[True](text, UInt64(9223372036854775807))


def parse_csv_integer[D: DType](text: StringSlice) raises -> Scalar[D]:
    """CSV-only atoi_simd parse for the native 8/16/32/64 integer dtypes."""
    comptime assert D.is_integral(), "parse_csv_integer needs an integer dtype"
    comptime if D == DType.int64:
        return rebind[Scalar[D]](parse_csv_int64(text))
    comptime if D == DType.uint64:
        return rebind[Scalar[D]](parse_csv_uint64(text))
    comptime if D.is_signed():
        var maximum = Scalar[D].MAX.cast[DType.uint64]()
        comptime if size_of[Scalar[D]]() >= 8:
            return _parse_csv_signed[True](text, maximum).cast[D]()
        else:
            return _parse_csv_signed[False](text, maximum).cast[D]()
    else:
        var maximum = Scalar[D].MAX.cast[DType.uint64]()
        comptime if size_of[Scalar[D]]() >= 8:
            return _parse_csv_unsigned[True](text, maximum).cast[D]()
        else:
            return _parse_csv_unsigned[False](text, maximum).cast[D]()
