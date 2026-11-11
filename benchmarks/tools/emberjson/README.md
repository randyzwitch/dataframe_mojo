# EmberJSON integration reproducers

These tools retain the inputs, exact-rounding oracle, and measurement procedure
used while removing dataframe_mojo's Float32 decimal parser. They are development
mechanism probes, not claims about general dataframe performance. No benchmark
results or generated fixtures belong in the source tree.

The standalone `upstream_float32.mojo` imports only EmberJSON and the Mojo standard
library. It can be copied into EmberJSON without dataframe_mojo. Python here only
generates independent test expectations and orchestrates measurements; the
library and the upstream reproducer use Mojo APIs.

## Versions and baseline

- EmberJSON 0.5.0: `e7feca8ebbdf2a338f586fed737bf8d01e6cad05`.
- EmberSerde 0.2.0: `8a5cde9bb7e838cf1e8e6b753c5594ed612f2aa9`.
- Mojo/MAX nightly: `1.3.0.dev2026111005` / `26.8.0.dev2026111005`.
- The compiler compatibility patches are in `packages/{emberjson,emberserde}`.
  They change API syntax and ownership declarations, not numeric algorithms.
- The dataframe baseline before Float32 replacement is
  `207a2e4f1d02f0c456be59598528fc2e07b26225`. Its Float32 parser and tables are
  available through that commit; these tools do not retain another code copy.

Run inside a matching Pixi environment (`pixi install --locked`, then `pixi shell`),
or prefix each Mojo command with `pixi run --locked`.

## Upstream reproducer

```sh
mojo build benchmarks/tools/emberjson/upstream_float32.mojo -o /tmp/emberjson-f32
/tmp/emberjson-f32 > /tmp/emberjson-f32.log
```

The rounding assertions must pass before timings are meaningful. In particular,
`1.000000059604644775390625000001` rounds to Float32 bits `0x3f800001`.
Converting through Float64 without a rounding guard produces `0x3f800000`.
The reproducer also checks the half-minimum-subnormal boundary, minimum normal,
maximum finite rounding, signed zero, and underflow. Timing output gives input
family, type, conversion count, nanoseconds, and a checked checksum. It alternates
Float32 and Float64 order, with one warmup and seven timed rounds.

Float64 is only a diagnostic for the extra work in the narrower-target path. It
is not a correctness-equivalent replacement for Float32 conversion.

### Candidate upstream improvements

1. **Float32 subnormal conversion.** `Parser.expect_float` currently parses to
   Float64 and calls `from_chars_slow[type]` for every nonzero value below the
   target's minimum normal, as well as values on target rounding midpoints.
   Explore a correctly rounded direct target-precision path or a cheaper safe
   guard. Preserve the exact midpoint and overflow-boundary tests; removing the
   guard and casting Float64 is incorrect. Use the `subnormal` family to reproduce
   the extra cost, then check normal/scientific inputs for regressions.
2. **Typed numeric range errors.** Both `"Infinite float"` (Float64 range) and
   `"float overflow"` (narrower target range) currently use
   `DerErrorKind.InvalidValue`. A distinct range kind would let CSV adapters accept
   overflow without matching diagnostic strings. Full input validation remains
   necessary: an overflowing prefix such as `1e400x` must still fail.
3. **Reusable numeric parsing policy.** CSV accepts leading `+`, leading zeroes,
   `.5`, `1.`, NaN/infinity, and overflow to infinity. A separate bounded numeric
   API could expose these choices while retaining strict JSON defaults. The
   dataframe adapter currently normalizes syntax; this is not a request to loosen
   JSON parsing itself.
4. **Public source-span cursor.** The JSON integration uses the private
   `EmberJsonCursor` to retain raw Int128/decimal number spellings and reusable
   flat subtree boundaries. A public cursor that exposes bounded source spans,
   validates skipped subtrees, and documents index size would remove this private
   API dependency. Existing dataframe reader contracts cover malformed ignored
   values, duplicate keys, escapes, and source lifetime.

These are retained integration findings and proposed work, not submitted upstream
issues or claims that fixes have already been implemented.

## Independent exact-rounding corpus

```sh
python3 benchmarks/tools/emberjson/make_cases.py /tmp/float32-cases.tsv
mojo build -I . benchmarks/tools/emberjson/check_cases.mojo -o /tmp/check-f32
/tmp/check-f32 /tmp/float32-cases.tsv
```

The generator uses integer arithmetic and `fractions.Fraction`, with round-to-nearest
and ties-to-even. It does not use a host float parser as its oracle. Cases include
values below, on, and above midpoints across all Float32 exponent fields, the
finite/infinity boundary, seeded long decimal inputs, huge exponents, special
values, and byte mutations. The Mojo checker tries each field at four offsets
with numeric-looking bytes immediately after the borrowed end. Non-numeric bytes
must be rejected rather than silently truncated.

Run the same checker source against baseline and candidate checkouts with `-I`.
The permanent regression subset lives in
`tests/kernel_equivalence/test_csv_float32.mojo` in the replacement change.

## CSV measurements

Use the same benchmark source and compiler for both library revisions. For example,
with baseline and candidate worktrees already checked out:

```sh
mojo build -I /path/to/baseline/providers/cpu -I /path/to/baseline \
  benchmarks/tools/emberjson/csv_float32.mojo -o /tmp/csv-base
mojo build -I /path/to/candidate/providers/cpu -I /path/to/candidate \
  benchmarks/tools/emberjson/csv_float32.mojo -o /tmp/csv-new
python3 benchmarks/tools/emberjson/compare_csv.py \
  --baseline /tmp/csv-base --candidate /tmp/csv-new \
  --baseline-revision BASELINE_SHA --candidate-revision CANDIDATE_SHA \
  --output /tmp/emberjson-csv-comparison
```

Use actual commit IDs and disclose dirty source changes in the revision label.
Wait for compilers and unrelated CPU-heavy work to finish before timing. Record
`mojo --version`, the CPU model, and the compatibility patch revision with any
shared report. Test native macOS separately; Linux measurements do not establish
macOS performance.

The driver constructs 100,000-row fixtures outside the timed region. It compares
ordinary decimals, scientific notation, CSV-specific spellings, subnormal values,
and a four-column mix. Each family includes the original order and a seeded
shuffled variant with nulls and quoted fields. Reads are warm-cache end-to-end CSV
operations. Each run checks row count and samples output bits/nulls outside the
timed region; the independent corpus supplies broader conversion correctness.

Four alternating-order process pairs, each with seven timed rounds after warmup,
produce 28 samples per binary/family/variant at one and eight requested workers.
The driver checks matching output checksums, retains raw samples and logs, and
writes medians plus binary/tool/fixture hashes to `results.json` under the output
directory. Report base and perturbed results, including slow cases; do not hide
subnormal or CSV-syntax regressions behind an aggregate.
