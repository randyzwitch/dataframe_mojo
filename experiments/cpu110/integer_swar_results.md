# Generic integer packed-parser result

The retained path is limited to generic integer parsing. `parse_int64` and
8-bit integer types remain scalar. Signed and nonnegative 16-bit-or-wider
fields use a scalar no-overflow path for one to three digits, then bounded
packed 4- and 8-byte reductions for lengths that are provably in range.
Unsigned negative fields retain the reference range/error path.

Paired 20M-field measurements: two digits improved from about 80 ms to 74 ms
for Int16, 69 ms for Int32, and 67 ms for UInt64. Long safe-width fields were
about 131 to 82 ms (Int16), 249 to 121 ms (Int32), and 532 to 215 ms (UInt64).
The focused scalar-reference differential suite and existing integer parser
tests passed after integration.
