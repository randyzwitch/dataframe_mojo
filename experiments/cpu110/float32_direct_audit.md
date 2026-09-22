# Float32 decimal conversion audit

The pinned Mojo standard library has no `Float32(StringSlice)` constructor.
Compiling that expression reports no matching initializer. Its private `_atof`
function also returns `Float64`; assigning it to `Float32` fails with
`cannot implicitly convert 'Float64' value to 'Float32'`.

The current Float64-then-cast behavior is observably different from a fully
correct direct Float32 decimal conversion. For `1.0000000596046448`, the exact
decimal is `2.4609375e-17` above the Float32 midpoint
`1 + 2^-24`, but less than half an IEEE Float64 ULP from that midpoint. The
current conversion first produces the exact Float64 midpoint and then
ties-to-even to Float32 bits `0x3f800000`. A correctly rounded direct Float32
conversion must produce `0x3f800001`. The analogous below-midpoint input
`1.0000001788139343` currently gives `0x3f800002`; a direct Float32 parser
must give `0x3f800001`.

`float32_direct_subset.mojo` is an isolated direct parser for a safe subset:
plain, non-exponent decimals with a mantissa at most `2^24` and at most seven
fractional digits. Both the mantissa and `10^k` are exact Float32 integers, so
one Float32 division rounds the decimal rational directly. It falls back to
the established Float64 contract otherwise. It matched the current result
bit-for-bit for 10,000 deterministic mantissas, every decimal placement and
both signs.

This subset cannot make a general direct-Float32 proposal safe: general
correctly rounded parsing changes current output at the midpoint examples.
Replicating current output for all inputs requires the Float64 conversion
before narrowing, which is not a direct Float32 improvement.

The 20M-field paired benchmark measured a 2--7% subset-path improvement, but
the intentionally falling-back inputs were about 35% slower because the
candidate scans them before the established Float64 parser scans them again.
The default path is therefore rejected.
