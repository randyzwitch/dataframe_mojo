"""Packed optional moment tail for the existing composed scan ABI.

Each partial value reserves20 Int64 words. Float32 uses18 wide square bins
(or7 bins at offset6 in the narrow variant). Integer fields use six signed
radix word sums, count, and a raw UInt64 maximum magnitude in the first8.
Float32 maximum-magnitude raw bits occupy word18; word19 is reserved.
"""
comptime MOMENT_PARTIAL_WORDS = 20
comptime MOMENT_FLAG_SHIFT = 24
