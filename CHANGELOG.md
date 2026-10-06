# Changelog

All notable changes are recorded here. The API is pre-1.0 and provisional:
breaking changes can happen in any release and are listed under **Breaking**.

## Unreleased

### Breaking

- Parquet timestamps with a time zone now read as zone-aware datetimes
  instead of naive UTC ones, and Arrow import accepts them instead of
  raising (#222).

### Added

- Categorical columns (dictionary-encoded strings): `cast("categorical")`
  from String and back, stored as UInt32 codes with the dictionary of
  distinct values carried by the dtype, as Arrow's
  `dictionary<uint32, large_utf8>`. Grouping, joins, `unique` and value counts
  work on the codes; sorting is by value; joins and `concat` across different
  dictionaries move both sides onto the union; expressions, display, `get`
  and CSV writing read the values. Arrow export and import keep the
  dictionary (pyarrow round-trips unchanged). On 1M rows with 100 distinct
  20-byte values, grouping takes 4.9 ms against 8.6 ms for String (Int64: 3.2
  ms), and the column takes 4 MB against 28 MB (#106).
- An expression over a categorical that reads no other column and works row
  by row (comparisons, `is_in`, `.str()` operations, `when`, casts) runs
  once per dictionary value and is spread over the rows by code, and
  `n_unique` counts codes. On 1M rows with 100 distinct values, `==` takes
  0.5 ms against 12.7 ms for String and 46 ms before, `to_uppercase` 9 ms
  against 85 ms, and `n_unique` 1.9 ms against 54 ms before (#106).
- `LazyFrame.join(other, left_on=..., right_on=..., how, suffix, coalesce)`
  joins on differently named key pairs, as the eager join does; projection
  pushdown passes each input only its own keys and used columns, predicate
  pushdown moves filters to the side that owns them, and `explain()` shows
  the key pairs. `LazyFrame.sort` takes a direction and null placement per
  key. Multi-table queries such as TPC-H's can now be written lazily
  (#329).
- Zone-aware datetimes: `DataType.datetime(unit, time_zone)` holds UTC
  instants shown and split into fields at local time, like an Arrow timestamp
  with its zone set. `dt().replace_time_zone(zone, ambiguous, non_existent)`
  and `dt().convert_time_zone(zone)`; local-time fields, `truncate`,
  `offset_by`, `strftime` (`%z`, `%:z`, `%Z`) and casts with Polars' DST
  rules; `strptime` with `%z` gives UTC-aware values; `datetime_range(...,
  time_zone)`; CSV inference and `CsvField.datetime(time_zone)`; Arrow
  `tsu:<zone>` import and export; and Parquet timestamps keep their zone.
  Zones come from the system tzdata (#222).
- `DataType.BINARY` columns of arbitrary bytes (Arrow `large_binary`):
  `Series.binary(...)` and `AnyValue.bytes()`, byte-wise sorting, grouping,
  joins and reductions, casts to and from string with UTF-8 checks, Polars'
  `b"..."` display, Arrow `Z`/`z` import and export, and Parquet binary
  columns read and written (#120).
- Expressions `interpolate`, `interpolate_by`, `cut`, and `qcut`, including
  per-partition interpolation, Polars-compatible bin boundaries and labels,
  and optional breakpoint structs (#227).
- `DataType.decimal(precision, scale)` decimal128 fixed-point columns, including
  arithmetic, comparisons, sorting, grouping and join keys, reductions, casts,
  explicit-schema CSV support, and Arrow `d:p,s` interoperability (#229).
  Decimal `mean` returns Float64, as Polars does, rather than a decimal
  truncated to the input scale (#341). Products, quotients and casts that
  drop decimal digits round half to even, as Polars does (#342).
- Reductions `arg_min`, `arg_max`, `mode`, `skew(bias)` and
  `kurtosis(fisher, bias)`, and the two-column `corr(a, b, method)` (Pearson
  or Spearman) and `cov(a, b, ddof)`, globally, in `group_by(...).agg(...)`
  and in `over(...)` (except `mode`), with Polars' null and NaN rules (#226).
- `DataFrame.describe(percentiles, interpolation)` in Polars' layout,
  `sample(n, fraction, with_replacement, shuffle, seed)` on DataFrame and
  Series with a seeded, platform-independent generator, and the reduction
  `value_counts(sort, name, normalize)` (#220).
- `write_parquet` writes supported logical and nested column types with zstd,
  snappy or uncompressed output and configurable row-group sizes.
- `read_csv` accepts ordered `(name, dtype)` pairs as a concise explicit schema.

- `read_parquet(path, columns=, row_groups=)` and the lazy `scan_parquet(path)`,
  which pushes projection into the reader and decodes only the row groups
  whose footer statistics can hold a match for a filter with constant
  bounds. `parquet_row_group_statistics(path)` returns those bounds as a
  frame. The reader is Arrow C++ built with only its Parquet parts behind a
  15 MB shared library that is loaded on first use; see the README.

### Changed

- Hash aggregation samples a mixed position within each input stratum,
  avoiding cardinality underestimates from periodic keys. The sample
  budget and 85% cutoff stay the same; exact bounds stop sampling early
  when the full-sample decision is already determined.
- Hash buckets with multiple non-null offsets-backed string keys cache
  buffer access for exact row comparisons, avoiding per-key storage
  dispatch on each hash match.
- Native sums/means and dense integer/categorical key encoding cache
  validity bitmap pointers and slice offsets outside their row loops,
  reducing repeated metadata reads for nullable inputs.
- Distinct counts by group (`n_unique` in a group-by) use one hash set of
  (group, value) pairs per partition instead of a set per group, which
  allocated once for every group holding a second value. PDS-H q21, which
  counts suppliers for 1.5M orders, 624 → 251 ms at scale factor 1.

- Numeric casts run as one typed loop when no value can fail to fit (Bool
  to any number, an integer to a wider type or to a float, Float32 to
  Float64) or when none does (an integer to a narrower or differently
  signed integer type). The general path, which reads each row into a
  128-bit intermediate inside a try block, remains for floats to integers
  and for columns holding a value that does not fit. Over 10M rows: Bool
  to Int64 196 → 8 ms, Int32 to Int64 196 → 4.5 ms, Int64 to Float64 137
  → 5 ms, Int32 to Int16 192 → 3 ms.

- A lazy chain of inner joins starts from the input that saves the most
  hashing. The chain streams its first input through hash tables built on
  the others, so `small.join(large)` hashed the large table. When the
  first input supplies only its join keys, the result is aggregated, and
  another input is larger, the chain is re-rooted there and the remaining
  joins follow outward along the same key pairs. Input sizes are
  estimated by applying each filter to 64 evenly spaced runs of 1,024
  rows; a right input keyed by an arithmetic progression counts as free,
  since it is looked up by position. PDS-H q3 80 → 40 ms, q11 25 → 10 ms
  and q8 46 → 39 ms at scale factor 1.

- Composite categorical keys reuse the bounded dictionary lookup for
  first-occurrence codes, avoiding numeric hashing for each key column.
  The new lookup checks valid codes before indexing the dictionary domain.

- Grouped sums and means of a plain numeric column of any width feed the
  reducer kernels by source interval, avoiding per-batch expression setup.
  The kernels read each width at its own type and add in 128 bits or as
  Float64, so Int8 to Int32, the unsigned types and Float32 no longer copy
  every batch to Int64 or Float64 first: three such aggregates over 10M
  rows and 100 groups take 13 ms instead of 56 ms, level with Int64.

- Hash joins with one integer probe key of any width hash it as each
  worker reads it, removing the probe hash/histogram pass and an
  eight-byte-per-row buffer, and compare keys as stored words instead of
  through the general row comparison. This was Int64 only; Int8 to Int32,
  the unsigned types, narrow decimals and categorical codes now probe the
  same way: 1M Int32 probes against 1M build rows take 11 ms instead of
  20 ms, level with Int64.

- String window ranking orders borrowed source rows by partition and value
  with the packed sort, then assigns ranks without dense-rank preprocessing.

- Group-by reuses its bounded cardinality preference across strategy
  selection, avoiding repeated sample gathers and hashes within one call.

- Cardinality sampling hashes the same bounded row sample in one pass per
  key, reducing repeated storage dispatch for contiguous and chunked keys.
- Partitioned grouping selects indexed or gathered evaluation per aggregate.
  Numeric first/last, standard deviation, variance and distinct counts can
  read source rows directly; decimal and computed reductions use bounded
  batches. A median/quantile or categorical expression gathers only its
  own referenced columns, while all expressions share group IDs. When no
  aggregate needs gathered columns, the row order is no longer copied for
  an empty gather: sums with a computed sum and a `first` over 1M rows and
  100,000 groups take 16.5 ms instead of 19.8 ms.
- General sorting compares normalized row keys and resolves long-string
  prefix ties before later keys, avoiding a dense-rank sort for each input
  column. Packed sorts retain their existing path; rank-based consumers
  now encode supported keys even when another key needs dense ranking.

- A frame whose columns are all chunked at the same rows (Parquet row
  groups, a partitioned group-by's output) filters chunk by chunk in
  parallel for any row-local predicate, each chunk taking whichever filter
  path suits it, with no merge first. This replaces a path that served only
  a comparison of a Float64 column with a float literal on 50,000 rows or
  more; it now runs for 25 ClickBench and 3 PDS-H queries instead of one,
  at the same speed. No row-count threshold gates either path.
- `read_parquet` keeps the dictionary codes of string columns the file
  stores dictionary-encoded in every selected row group, beside the strings.
  The column is still a String column with the same values. Grouping by
  it groups the codes, then returns string keys: H2O q2 (two string keys,
  10M rows) 102 → 55 ms, q3 108 → 76 ms (219 → 134 ms with 5% nulls),
  q10 658 → 527 ms. Reading such a file costs 8–21% more time and up to
  11% more memory (the codes). libdfparquet gains
  `dfq_read_parquet_stream_dict`; an older library reads as before.
- Those codes now survive a filter, take, sort or join. The gathered column
  keeps the source's codes and the gather's row list rather than copying
  codes, so a filter costs nothing extra, and the codes are gathered only
  when something groups by the column. Before, any gather dropped them
  and the group-by hashed strings again. On the 10M-row H2O file, a filter
  keeping 60% of rows followed by a group-by on `id3` (100,000 strings)
  takes 70 ms instead of 185 ms for the group-by; two keys after a 33%
  filter take 19 ms instead of 27 ms.
- Grouping by one categorical key numbers its rows by direct lookup on the
  codes instead of the general encoder: H2O q1 (one coded string key) 30 →
  17 ms, and 23 → 16 ms on sorted data.
- A filter on `a & b & ...` over 4,096 rows or more evaluates its parts
  one at a time, each only on the rows the earlier ones kept, with parts
  reading strings last. It narrows while a part keeps at most an eighth
  of its rows (half when a string part is still to come); otherwise the
  remaining parts run together and the masks combine. ClickBench q22
  97 → 45 ms, q37 63 → 49 ms, q40 27 → 16 ms, q41 22 → 14 ms, q38 26 →
  16 ms. Turning a mask into row numbers now reads 64 rows at a time.
- Lazy plans order chains of inner joins by how much each input narrows
  its table, measured by running the input when the plan executes: a
  selective join moves into the input that holds its keys (customers
  joined to the nations a filter allows, then orders to those customers),
  and the most selective joins run first. A filter on an OR of ANDs also
  filters each input by what it implies, and a filter on `a & b` moves as
  two. PDS-H q7 101 → 50 ms, q21 344 → 279 ms, q19 37 → 34 ms.
- The partitioned group-by no longer gathers aggregated columns into bucket
  order when every aggregation is a sum, mean, min, max, count or len of an
  Int64 or Float64 column: each bucket reads values at their source rows
  and updates its groups' states in place. ClickBench q15 124 → 105 ms, q16
  206 → 175 ms, q35 108 → 84 ms; H2O q3 (10M rows) 99 → 78 ms, and 137 →
  114 ms with 5% nulls.
- A filter on `rank("ordinal").over(keys) <= k` (or `< k + 1`), the "top
  N per group" idiom, keeps each partition's first k rows by scanning the
  rows once with a small sorted list per partition, instead of ranking
  every row. Partitions are numbered as `over()` numbers them and split
  among workers; the order is `rank`'s own, so ties keep the earlier row
  and NaN ranks as it does there. H2O q8 (10M rows) 414 → 237 ms at k=100,
  500 → 275 ms at k=10 and 604 → 431 ms at k=2.
- Decimals keep the width their source declares: Arrow `decimal32` and
  `decimal64` (`d:p,s,32` and `d:p,s,64`) import and export at that width,
  stored in Int32 and Int64, alongside `decimal128`. Parquet decimals
  stored as INT32 or INT64 read as `decimal32` or `decimal64` (Arrow's
  smallest-decimal reader option). `DataType.decimal(precision, scale,
  width=128)` names them `decimal32[p,s]` and `decimal64[p,s]`. Filters,
  sorts, grouping and joins keep the width; arithmetic computes at 128 bits
  and returns `decimal128`, and a narrow decimal's sum widens to
  `decimal(38, s)`, so a result never overflows its storage. Comparisons
  and no-null arithmetic read 64-bit decimals at their width. PDS-H decimal
  q6 36 → 11 ms, q11 33 → 27 ms and q3 93 → 79 ms at scale 1.
- Grouping by one Int64 key without nulls into a moderate number of groups
  (about 200K or fewer, from a 65,536-row sample) aggregates in place: each
  worker numbers its row range with its own hash table and reduces its
  values in row order, and the ranges' groups merge by hash part on every
  worker, so no aggregated column is gathered into bucket order (#383).
  Serves sum, mean, min, max, count and len of Int64 and Float64 columns,
  and keys with nulls (a null key is one group). H2O q5 (10M rows, 100K
  groups) 133 → 64 ms, and 233 → 73 ms with 5% null keys.
- The partitioned group-by confirms a string key's hash match eight bytes
  at a time instead of one. ClickBench q33 and q34 (group by URL, 10M rows)
  413 → 376 ms.
- A numeric comparison of columns without validity bitmaps returns its
  values with an empty (all-valid) bitmap, instead of filling an all-ones
  bitmap, applying it to every output byte, and handing it to every later
  AND. ClickBench filter queries q1, q7 and q36-q42 4-13% faster.
- A semi or anti join whose right side is at least 512K rows and four
  times the left hashes the left keys and scans the right, marking the left
  rows it matches, instead of hashing the right. A lazy plan decides this
  from both real sizes once its right side has run, and runs neither side
  twice. PDS-H q4 59 → 48 ms and q22 45 → 35 ms at scale 1.
- Numbering an Int64 key whose values span fewer than 4,096 integers reads
  values and writes ids through pointers, and reads no validity when the key
  has no nulls. H2O q4 (10M rows) 34 → 23 ms at k=100 and 34 → 25 ms at k=2.
- A chunked series (a Parquet table's row groups) keeps the contiguous
  array its first `rechunk` makes, shared by every copy, instead of merging
  the same chunks again for every join, kernel and gather that needs it.
  Merging string-view chunks copies each chunk's descriptors in one block
  when its buffer indexes do not change. H2O join (10M rows) q1 127 → 80 ms,
  q2 142 → 96 ms, q4 242 → 176 ms and q5 530 → 419 ms; H2O group-by q3 199
  → 153 ms and q10 802 → 681 ms; PDS-H q14 17 → 12 ms and q19 46 → 37 ms.
- A lazy plan over a large in-memory input that streams through joins
  uses batches up to four times the default, as long as every worker still
  gets one: each batch probes every join, and fewer, larger batches cost
  less per row. PDS-H q14 43 → 18 ms, q19 102 → 45 ms and q21 477 → 386 ms
  at scale 1.
- Grouping or partitioning by one numeric key without nulls numbers each
  hash bucket by hash alone: the key's hash is a bijection on its 64 bits,
  so equal hashes are equal keys and rows need no comparison, which read
  both rows' keys at random. H2O q5 (10M rows) 358 → 264 ms at k=2 and 200
  → 159 ms at k=100; ClickBench q8, q9, q15 and q35 11-14% faster; PDS-H
  q18 137 → 118 ms and q21 483 → 442 ms at scale 1.
- A filter whose mask keeps every row (`is_not_null()` on a column
  without nulls) returns the frame's columns instead of copying them, and a
  rank over many small partitions sorts every row once by (partition, key)
  sooner: when partitions times workers exceed the row count rather than
  four times it, which also skips a serial prefix over partitions times
  workers. A rank of a column without nulls fills its validity in one call.
  H2O q8 (top two per group, 10M rows) 898 → 611 ms at k=2, 554 → 511 ms at
  k=10 and 470 → 444 ms at k=100; ClickBench q27 137 → 106 ms.
- Decimal addition, subtraction and multiplication of columns without
  nulls read values through pointers and build no validity, and a product's
  rounding division by 10 to 10,000 is specialized when compiling, so a
  quotient that fits 64 bits needs no division instruction. PDS-H decimal q1
  180 → 143 ms at scale 1.
- Boolean `fill_null`, and the step that keeps nulls in numeric `is_in`,
  work on bitmaps a byte (eight rows) at a time instead of building two
  lists of Booleans per row and packing them. Numeric `is_in` runs both
  once per listed value. ClickBench q40 (10M rows) 63 → 29 ms.
- A lazy join whose left input is made by other joins, so its size is
  unknown until it runs, against a right input of 512K rows or more, also
  leaves the stream and runs eagerly on the smaller side; the joins above
  still stream. PDS-H q5 123 → 68 ms, q18 242 → 140 ms, q11 44 → 27 ms, q7
  158 → 114 ms and q2 35 → 28 ms at scale 1. q9, whose left input there is
  large, is 14% slower than with the bounded rule alone.
- Decimal addition, subtraction, multiplication and sums read the
  precision limit once per column instead of computing `10^precision` for
  every row, and a decimal sum without nulls reads values and group ids
  through pointers. PDS-H decimal q1 322 → 188 ms at scale 1.
- In a lazy chain of joins, a join whose left input a filter bounds to at
  most an eighth of a large right input now runs eagerly, building its hash
  table on the smaller side, and the joins above stream from its result.
  Before, only a plan's single join did (#377); in a chain the streaming
  join built on the right. PDS-H q8 158 → 62 ms and q9 224 → 128 ms at
  scale 1.
- A group-by on several keys with few values each (at most 256 in a
  sample of each key, at most 65,536 combinations) aggregates by worker row
  ranges instead of hash partitions. A sample of whole key rows looked
  mostly distinct for two 100-value keys, so 10,000 groups took the
  partitioned path built for millions. H2O (10M rows, k=100) q2 322 → 103
  ms and q9 294 → 108 ms.
- `DATAFRAME_THREADS` is now a limit: at most that many threads run jobs
  at once, across nested and concurrent parallel steps. Steps that cut work
  into more jobs than threads used to give every job its own thread, and a
  grouped reduce inside each hash bucket, or a median, started rounds of
  their own; H2O q6 ran up to 118 threads at `DATAFRAME_THREADS=8`. Extra
  jobs are now claimed by the allowed threads, and a job that starts jobs
  while every thread is busy runs them itself. Benchmarks at 8 threads are
  now measured on 8; that makes some slower than reported before (#390).
- Lazy group-bys over an in-memory frame run eagerly when that is faster:
  after a filter, on keys made by an earlier step, and on keys with few or
  mostly distinct values. Unfiltered keys in between still stream. The
  parallel string-key encoder merges its ranges with the string encoder on
  every worker instead of a `Dict[String, Int]` on one thread. ClickBench
  (10M rows) q18 1,647 → 377 ms, q14 470 → 69 ms, q12 355 → 66 ms, q39 364
  → 97 ms and q27 171 → 145 ms; an eager group-by on SearchPhrase 910 → 375
  ms (#381).
- `over()` numbers its partition keys on every worker instead of one, and
  the partitioned key encoder writes ids back to row order in parallel.
  H2O q8 (top two per group) 2,090 → 850 ms at k=2 and 570 → 443 ms at
  k=100 on 10M rows (#387).
- The hash join's Int64 probe reads hashes, keys and bucket slots through
  pointers, and the build and probe hashes are moved into the index instead
  of copied. H2O join q5 571 → 490 ms; PDS-H q3 85 → 74 ms, q5 117 → 102
  ms, q9 211 → 192 ms and q21 441 → 416 ms at scale 1 (#378).
- A lazy plan whose single join has a left input bounded to a quarter of
  a large right input (a filtered scan, for example) runs eagerly, so the
  join builds its hash table on the smaller side instead of the right one.
  PDS-H q17 97 → 42 ms at scale 1 (#377).
- Aggregation loops read values and group ids through pointers, skip
  validity checks for columns without nulls, and keep ungrouped totals in a
  local: H2O group-by q4 (`mean` of three columns by `id4`) 53 → 35 ms at
  10M rows (#382).
- Group keys are numbered without standard-library dictionaries: number
  columns through a typed open-addressing table, string columns through the
  single-key path that packs short strings into 128-bit keys, and several
  columns combined through a direct array or the same table. H2O group-by
  with 500,000 groups by two string keys (k2 q2) 144 → 95 ms (#380).
- `+`, `-` and `*` on signed 8-, 16- and 32-bit columns without nulls run
  16 rows at a time with one overflow check per block, and an ungrouped
  `sum`, `mean`, `min` or `max` of a computed column reads it with SIMD
  loops instead of copying it to Int64 first. ClickBench q29 (90 sums of
  `ResolutionWidth + i`) 3.5 s → 0.45 s at 10M rows (#384).
- Decimal arithmetic, comparisons and sums no longer divide 128-bit
  integers on every row: `pow10` (read by every precision check) is two
  multiplications instead of a loop of up to 38, same-scale operands skip
  rescaling, and products of 64-bit values skip the overflow division.
  PDS-H decimal q1 1,022 → 427 ms and q6 216 → 38 ms at scale 1 (#385).
- `str.contains`, `starts_with` and `ends_with` run on every worker over
  raw bytes, `contains` with a SIMD first- and last-byte filter, and
  `str.slice` returns views into the source's bytes. New `str.like(pattern)`
  evaluates SQL `LIKE` ('%' and '_') without regex. ClickBench q20 89 → 47
  ms and PDS-H q22 105 → 42 ms at full scale (#374).
- String view columns list each byte buffer once when chunks are merged,
  gathers share a view column's buffer list instead of copying it, views
  are built in registers, and the mask filter copies runs of kept rows.
  This undoes the slowdowns #375 and #376 caused: ClickBench q18 2,137 →
  1,640 ms and PDS-H q1 201 → 167 ms at full scale (#395).
- Gathering string rows (`take`, and the gathers behind filters, joins and
  sorts) returns 16-byte views into the source column's bytes instead of
  copying every byte: values of up to 12 bytes sit in the view, longer ones
  point into the shared buffer. A gathered column keeps its source's byte
  buffer alive, as Polars' does (#375).
- `filter` copies fixed-width and Boolean columns straight from the mask's
  64-bit words, copying all-true words in bulk and skipping empty ones,
  instead of building a list of kept row numbers and gathering every column
  from it. String and nested columns still use the row list (#376).
- Comparing a string column with a literal (`==`, `!=`, `<`, `<=`, `>`,
  `>=`, either side) runs on every worker over the raw bytes and writes the
  result bitmap directly, instead of building a string slice per row on one
  thread. `is_in` over string literals is one operation that compares each
  row once (a hash table for more than eight values) instead of a chain of
  `==`, `fill_null` and `|` per value; other `is_in` values work as before
  (#373).
- Joins concatenate their workers' matched rows on every worker instead of
  appending them one at a time on one thread, probe each chunk of a Parquet
  key in place instead of copying it into one buffer, and hand their row
  lists to the gathers without copying them. On H2O's joins at 8 threads,
  10M rows by 10,000 on `id2` goes from 222 ms to 153 ms (Polars: 114 ms),
  the same join with one value per side from 117 ms to 45 ms (Polars: 35
  ms), and the left join from 129 ms to 66 ms (#335).
- Casting strings to integers and floats parses each row straight into the
  output type, in row ranges on every worker, instead of dispatching on the
  source and target types per row through a 128-bit intermediate; accepted
  text, results and error messages are unchanged, and a strict cast still
  names the first failing row. On 5M rows at 8 threads, short integers to
  Int64 go from 82 ms to 13 ms (Polars: 10 ms) and decimals to Float64 from
  82 ms to 18 ms (Polars: 14 ms) (#149).
- Grouped and global `median` and `quantile` find the one or two positions
  they read by selection instead of sorting every value, and groups finish
  on every worker, several per worker in ranges of equal size. Results are
  unchanged to the bit. On 10M H2O rows at 8 threads, `median` by `id4, id5`
  over 4 groups goes from 1,280 ms to 201 ms (Polars: 140 ms) and over
  10,000 groups from 252 ms to 227 ms (#337).
- A lazy group-by over an in-memory frame streams only when its keys repeat
  (at most half of 4,096 sampled rows distinct), and otherwise runs the eager
  group-by, whose hash-partitioned encoding beats merging every batch's
  groups again; keys that repeat stream, which is faster than eager for
  them. Each group's first distinct number for `n_unique` is kept inline, and
  its set is created only for a second. On 10M ClickBench rows at 8 threads,
  lazy grouping by URL goes from 904 ms to 351 ms and by WatchID and ClientIP
  from 1,406 ms to 322 ms, and a streamed Parquet scan grouped by UserID
  with `n_unique` from 1,510 ms to 383 ms (#326).
- `n_unique` counts distinct values by hash partition instead of keeping a
  Dict per group: rows are keyed (group, value), scattered by hash into
  cache-sized partitions on every worker, and each partition's distinct keys
  are counted on its own, so no merge runs on one thread however skewed the
  groups. Each worker first drops keys it has just seen, which keeps a
  dominant value from filling one partition. Grouping by several
  low-cardinality keys also encodes them on every worker. A lazy plan over an
  in-memory frame counts distinct values this way rather than in streamed
  batch sets; file scans still stream. On 10M ClickBench
  rows at 8 threads: `UserID.n_unique()` from 276 ms to 33 ms,
  `SearchPhrase.n_unique()` from 414 ms to 46 ms, and
  `group_by("MobilePhoneModel").agg(col("UserID").n_unique())` from 288 ms to
  104 ms (#336).
- Ungrouped `len()` returns the row count without reading its column, and
  integer `sum`, `mean`, `min` and `max` scan eight values (64 bytes when
  there are no nulls) per step instead of one row at a time. Sums stay
  exact in 128 bits: 64-bit values add as 32-bit halves in separate lanes.
  `cast(Int64)` of a narrower integer column before one of these reads the
  column itself instead of materializing the cast. Counting validity bits
  (`count()`, `null_count()`) goes a 64-bit word at a time. On 10M ClickBench
  rows at 8 threads: `len()` from 10 ms to 0.01 ms, Int16 `sum` from 17 ms to
  0.3 ms, Int16 `max` from 5.8 ms to 0.3 ms, `ClientIP.cast(Int64).sum()` from
  133 ms to 1 ms, and `count()` from 0.7 ms to 0.2 ms (#333).
- A lazy sort followed by `head(k)` or `slice(offset, k)` selects its first
  `offset + k` rows instead of sorting every row; `explain()` shows it as
  `TOP_K`, and streaming plans keep each batch's first rows. Eager `top_k`
  and `bottom_k` use the same selection, one 65,536-row chunk per worker. At
  8 threads on 10M rows, `sort("v3").head(10)` goes from 494 ms to 15 ms and
  `top_k(10, "v3")` from 133 ms to 12 ms (#332).
- Sorting packs each row's keys and its row index into one 64- or 128-bit
  integer and sorts those directly, bucketed by their top bits and then
  sorted per bucket in parallel, instead of ranking string keys with a sort
  of their own and merging row indices through per-key lookups. Keys are
  stored relative to their range and strings past their shared prefix, so
  most sorts by one to three keys pack; long strings settle ties by
  comparing the strings, and anything wider keeps the general path. At 8
  threads on 10M rows `arg_sort` by a Float64 goes from 1,110 ms to 142 ms,
  by a short string from 2,009 ms to 217 ms, and by a string and a Float64
  from 3,205 ms to 286 ms. Gathers no longer zero their output first, and
  string gathers read each source offset once and copy short values inline;
  a 9-column `sort` goes from 1,556 ms to 489 ms (#331).
- Numeric comparisons and Boolean `&`, `|`, `^` and `~` write packed bitmaps
  directly, eight rows per step, instead of looping per row; `select_exprs`,
  `select` and `filter` default to 8192-row batches, as `with_columns`
  already did. PDS-H q6's predicate on `lineitem` goes from 113 ms to 7 ms at
  8 threads, and the whole filter, including the row gather, from 127 ms to
  33 ms (#327).
- Gathers copy runs of consecutive rows with `memcpy` instead of appending
  row by row: fixed-width columns, and strings through their offsets, which
  used to cost about 85 ns per row whatever their length. Chunked (Parquet)
  columns gather each chunk's share of the filter indices in place. Filters
  and joins get faster; PDS-H runs in 0.70 of the time (#328).
- `rank` of numeric and temporal columns, globally and with `over`, sorts
  each partition once: every row packs into one 128-bit integer (an
  order-preserving key above its row), partitions are bucketed and ranked on
  all workers, a single large partition is sorted in parallel runs and
  merged, and very many small partitions share one parallel sort. H2O q8
  at 10M rows goes from 3.7 s to 0.62 s (10.8x to 1.66x Polars), and every
  data variant runs 4.5-6x faster. Descending ranks now put NaN
  first, as Polars does; the old path put it last (#330).
- String group keys cost much less. The row-range encoder reads offsets and
  bytes in place, builds a short key with one copy and reuses the previous
  row's code; hash-partitioned grouping encodes each bucket from the key
  hashes the partitioner already computed, comparing rows in place on a
  match (with prefetching), so key columns are no longer gathered or
  re-hashed; and taking a few rows of a chunked column no longer rechunks
  it. At 10M rows and 8 threads, `group_by(...).agg(len())` on id1 goes from
  90 to 30 ms, id3 from 299 to 171 ms, and (id1, id2) from 351 to 209 ms
  (1.8x, 1.16x and 1.21x Polars) (#334).
- Streaming lazy group-by merges batch states in groups, sized so total merge
  work stays linear, instead of re-encoding every group seen so far on each
  batch. Large merges use the partitioned key encoder, and per-group state
  grows in place rather than being rebuilt. The 20 affected ClickBench
  queries run 22x faster (geometric mean; q32 434 s to 5 s) (#326).
- Streaming group-by with many groups splits its state into hash parts that
  merge on separate workers through a persistent key index, so no merge
  re-encodes the groups already seen; groups are interleaved back into
  first-occurrence order at the end. ClickBench q32 goes from 5.0 s to 1.9 s
  and the affected queries run 1.23x faster than after #338 (#326).
- Retired the in-repo benchmarks the external suites supersede or whose
  target code is gone: `bench_vs_polars` (`bench-polars`), the upstream join
  ports, `bench_progression`, `bench_aggregate_fusion` and `bench_agg_shapes`.
  docs/benchmarks.md lists what remains and why.
- PDS-H is a development benchmark suite from 2026-10-05, and the quick
  tier runs it: no development suite covered multi-join plans, the one area
  behind Polars. TPC-DS replaces it as the held-out suite, from DuckDB's
  `dsdgen` and `tpcds_queries()`. DuckDB runs all 99 queries; 48 are
  translated for this library and Polars (the 23 single-block queries and
  25 with derived tables, subqueries or EXISTS but no CTE), and the rest are reported
  as not translated. The suite runner now counts a worker
  crash as one failed query and runs the remaining queries in a new worker.
- Lazy projection pushdown reaches through joins: each join input reads only
  its keys and the columns the plan above uses, for frame, CSV and Parquet
  scans and through filters. The inner-join count shortcut, which answered
  `join(...).select(len())` without running the join, is removed (#304).
- Group-by reduces worker row ranges into mergeable state for any list of
  supported reductions on low-cardinality keys of any type, replacing two
  paths that accepted only Float64 sum/count/mean on one Int64 key. Other
  aggregation lists run 3.3 to 6.5x faster there; the former two-aggregation
  special case is slower. See [the measurements](docs/group-aggregation.md) (#306).
- Lazy collect executes supported pipelines in bounded ordered batches,
  merges aggregate state, streams supported join probes, and annotates
  materialization boundaries in explain. `streaming=False` keeps the
  materializing executor available for comparison.

- Large Arrow record batches import columns in parallel on Linux; Parquet
  reuses the pool across row groups. Measured Mac imports remain serial.
- Worker pools park when they exceed physical/performance cores. Gather
  partitions and bounded-index builders respect core budgets, with 4/8/16
  worker calibration on Threadripper and Apple M1.
- Progression joins accept matching Date, Datetime, Duration and Time keys,
  including temporal units. Build keys with repeated values (equal runs) no
  longer take the progression path; they use the join index (#308).
- Bounded Int64 joins no longer compute a GCD stride for wide shuffled keys;
  those keys use the hash index (#307).

- Join and sorted-chunk-gather dispatch uses measured working-set and job
  costs instead of the remaining fixed 2,000,000-row switches. Bounded join
  tables share an overflow-safe byte budget, with measured density and
  membership-cache limits; ordered CSR IDs and chunked progression probes
  avoid parallel paths that lose on those inputs. See [the measurements](docs/join-cutoffs.md).
- Parquet reads decode one selected row group at a time through an Arrow C
  stream and preserve imported chunks, avoiding whole-file batch assembly.
- Sort bucket dispatch uses measured domain, rank-buffer and worker-balance
  limits, with calibration on Threadripper and Apple M1.

- Semi and anti joins whose keys do not fit a direct-address range probe
  the right-row hash index for membership instead of encoding both inputs
  as dictionary ids. On 10M rows with wide Int64 keys, semi and anti went
  from about 1.6 s to 0.2 s.
- Two row-count cutoffs were re-measured by sweeping sizes: fused
  expressions keep source chunks from 200,000 rows (was 2,000,000), and
  Float64 comparison filters run on aligned chunks from 50,000 rows (was
  2,000,000). At 1M rows this makes expression chains about 3x and filters
  about 1.7x faster.
- List and struct column types: `DataType.list(inner)` and
  `DataType.struct(names, dtypes)`, backed by `ListColumn` and `StructColumn`
  in the Arrow `large_list` and `struct` layouts, with take, slice, concat,
  equality, display, and Arrow import and export (so `read_parquet` now
  returns nested columns). `str.split`, `DataFrame.explode` and
  `LazyFrame.explode`, the `.list()` expression namespace (`len`, `get`,
  `first`, `last`, `contains`, `join`, `sum`, `min`, `max`, `mean`),
  `pack_struct`, `unnest` (eager and lazy) and `field(name)`. Operations
  that do not support nested columns yet raise a clear error.
- `col(x).implode()` (a group's values as a list inside `agg`, or a whole
  column as one row), `as_struct([...])` to pack expressions into a struct
  column, and struct columns as `group_by`, `unique` and inner/left/semi/anti
  join keys, compared field by field with a null struct distinct from a
  struct of nulls.

### Breaking

- Public read_csv now uses the single source-derived Polars CSV pipeline; the
  former scalar reader is no longer available as a fallback. Observable
  contract changes include positional explicit schemas, nullable short records,
  100-row default inference, opt-in temporal inference, Polars duplicate
  header suffixes, and Polars numeric parsing. See
  [the CSV integration notes](https://github.com/randyzwitch/dataframe_mojo/wiki/csv-polars-port).

- Unsigned CSV integer fields now reject `-0`, matching Polars' `atoi_simd`
  parser. Generic casts retain their existing behavior.

### Changed

- CSV integer fields use a source port of Polars' `atoi_simd` parser, with
  destination-width overflow checks, x86 SIMD reductions and a packed
  fallback. The public reader remains the only CSV implementation. See
  [measurements and limitations](https://github.com/randyzwitch/dataframe_mojo/wiki/csv-integer-experiments).

- Float64 parsing passes validated wide decimal mantissas directly to Mojo's
  Lemire converter and uses its borrowed-span converter for exponent and
  long inputs. This avoids reparsing wide decimals and constructing an owned
  string for successful conversion, while retaining existing result bits and
  strict grammar. The private converter dependency is covered by reference
  comparisons; see [numeric parsing measurements](https://github.com/randyzwitch/dataframe_mojo/wiki/numeric-parsing-149).

- Simple quoted CSV fields decode borrowed input spans with SIMD structural
  scanning. Quoted empty strings and quoted null tokens keep their existing
  meaning; escaped quotes and embedded line breaks retain the state machine.

- CSV boundary scanning computes quote parity in blocks instead of repeatedly
  loading overlapping bytes around quotes. Workers construct and reserve
  their builders using scanned record counts. Validity packing and shifted
  bitmap assembly write groups of bits at once; string concatenation splits
  bytes, offsets, and validity into independent jobs within the thread cap.
  Fully validated wide plain decimals skip a duplicate grammar scan while
  retaining the standard Float64 converter.

- Plain CSV records decode borrowed input spans without copying fields into
  a record buffer. Finished column builders transfer their buffers, and
  Int64 fields of at most 18 digits avoid per-digit overflow checks. Quoted
  and irregular records retain the strict state machine.

- CSV reads publish small record-aligned chunks while scanning the file,
  and workers claim them dynamically. Tokenization reuses structural masks
  across fields; numeric grammar checks consume borrowed text before
  handing it to the existing numeric conversion. See [the comparison notes](https://github.com/randyzwitch/dataframe_mojo/wiki/csv-pipeline-152)
  for measurements and the remaining work in #152.

- Concatenation sizes each output column once instead of growing into it.
  Every parallel stage reassembles its result this way -- a CSV read
  produces one frame per range -- and letting the column double copied it
  again at every step. The final height and the text size are both known
  from the inputs. Reassembling 1M rows of 8 columns from 32 ranges drops
  from 21 ms to 7 ms, and the whole parallel CSV read from a median of
  86 ms to 67 ms across 32 threads (#107).

- Signed decimals take the one-pass Float64 parser. Only unsigned text did,
  so every negative field went to the strict parser, which allocates a
  String to check the grammar -- half the fields of a column centred on
  zero. Reading 1M rows of 8 columns drops 14% single-threaded (720 ms to
  622 ms) and about 10% at 4 and 32 threads. The accepted grammar is
  unchanged: the fast path takes the one leading sign the strict grammar
  already allowed, and anything else it cannot finish still falls through
  (#107).

- A parallel CSV read maps the file instead of reading it into a buffer.
  Mapping the 50 MB benchmark file costs 2.4 ms against 58 ms to read it,
  and the whole file being addressable at once removes the block loop and
  the partial record carried between blocks. A range also reads straight
  out of the mapping rather than copying itself out first, which was a
  second pass over every byte. Reading 1M rows of 8 columns drops from a
  median of 103 ms to 88 ms across 32 threads. A path with no length to map
  -- a pipe, a character device -- still reads, through the block reader
  (#107).

### Fixed

- The ungrouped mean of a decimal column stored at 32 or 64 bits no longer
  raises once its running total passes the column's precision; the total
  is kept at precision 38, as a sum's is.
- `n_unique` of a decimal column no longer crashes the process. Decimals
  are counted by their scaled integers; a value needing more than 64 bits
  is reported as unsupported.
- `when().then()` on a decimal column no longer crashes the process when
  the column is stored at 128 bits or the conditional has no `otherwise`.
  Branches of different decimal types now widen to decimal(38) at their
  shared scale instead of being rejected; branches of different scales
  are an error (#464).
- `fill_null` and `coalesce` accept decimal operands, with the same type
  rule (#465).
- Composite key numbering reuses dense integer lookups and owned code
  arrays, avoiding redundant first-key renumbering while retaining null
  insertion, Boolean ordering, and exact key equality (#380).
- Join planning reuses inputs executed to measure selectivity even when
  the original join order is kept or a candidate is rejected. Unreachable
  input and materialization slots are released after planning (#436).
- Grouped ordinal top-k filters bound scratch space by input rows and
  group counts, including huge k and singleton groups. Bounded heaps
  replace O(Nk) insertion lists, workers visit disjoint rows or groups,
  and `< Int64.MIN` returns no rows instead of overflowing (#435).
- Grouped top-k filters with many small groups no longer route rows to
  their owning worker through a serial pass over every row; each worker
  scans the group ids and keeps its own groups' rows. That pass had made
  H2O q8 20–27% slower on the two high-cardinality variants: at 10M rows
  the filter now takes 262 ms instead of 341 ms (k=10) and 362 ms instead
  of 424 ms (k=2), and uses 8 bytes a row less scratch.
- Fused Float64 expressions that reference a column repeatedly now work on
  chunked inputs, including nonzero batch offsets and misaligned chunks.
  Each source window is sliced once instead of once per expression leaf
  (#434).
- A filter with decimal comparisons inside an AND no longer runs them row by
  row under the AND's mask: a comparison is safe on rows the mask skips, so
  it takes its whole-column path. The selective AND filter had made PDS-H
  decimal q6 13 -> 58 ms and q19 38 -> 86 ms; now 15 and 32 ms. Masks that
  start on a byte boundary also combine and count whole bytes at a time.
- Lazy filters holding a date literal (`str().to_date()`) or another
  operator with a text argument again move below joins: the argument was
  read as a column name, which no input has.
- `write_csv` writes a float NaN as `NaN`, as Polars does, instead of
  `nan`, which both `read_csv` and Polars read back as a string, so a
  Float64 column with NaN now round-trips as Float64 (#368).

## 0.2.0 - 2026-09-21

### Breaking

- Mojo 1.1 is no longer supported; this release requires the 1.2 series.
  A compiled Mojo package loads only in the version that produced it, so a
  consumer on 1.1 must stay on 0.1.3. The package was built and its suite
  run against 1.2.0.dev2026092105, which is what the bound names: 1.2 is a
  nightly series today.

- Development tracks the same nightly. The nightly channel is now a
  workspace channel and there is one environment again, so plain
  `pixi run test` is the supported toolchain; the separate `nightly`
  environment added in 0.1.3's cycle has nothing left to do.

### Added

- `Pool`, worker threads reused across the rounds of one operation instead
  of created and joined per round. A pool is **scoped**: it joins its
  workers when released, and never outlives the operation that made it. A
  process-wide pool is not possible here -- its workers would still be
  parked in JIT-compiled code when the process exits, which crashes about
  one run in three under `mojo run`, and no Mojo code can run at exit to
  join them first. Sorting uses one for its run pass and merge rounds: a
  two-key sort of 1M rows drops from 156 ms to 150 ms, and at 100k rows
  from 38 ms to 33 ms (#103).

### Changed

- The CSV reader no longer builds a `String` for every field. Fields of a
  record are written end to end into one buffer and passed on as slices
  over it, which needs no allocation; a 1M-row file of 8 columns was
  allocating 8 million Strings. Reading that file single-threaded drops
  from 1,336 ms to 992 ms, and across 32 threads from 155 ms to 118 ms.
  Strings are still built where they are kept rather than parsed: schema
  inference's sample, a header's names, and lossy decoding, which
  substitutes U+FFFD and so changes the bytes (#107).

- The CSV reader copies runs of ordinary field bytes in bulk, finding the
  next separator, newline or quote a block at a time rather than
  dispatching on every byte. Reading 1M rows of 8 columns on one thread
  drops from 990 ms to 925 ms. Across many threads the read is bound by
  something else and the difference is within run-to-run noise, so this is
  for machines with few cores (#107).

- `concat` builds its output columns on worker threads. Each output column
  is assembled from that column of every input frame and touches nothing
  else, so there is nothing to coordinate. A parallel CSV read concatenates
  one frame per range per block -- 64 of them for a 50 MB file -- and doing
  that one column after another was a third of the read: 1M rows of 8
  columns drops from 158 ms to 133 ms (#107).

- A join assembles its output across worker threads. It gathered one
  column at a time on the calling thread, which was about half of a join:
  98 ms of 198 ms for 1M x 500k inner. `take_parallel` already writes
  disjoint output ranges, and now carries the -1 that a join uses for
  "no row on this side", so every column of both sides is gathered at
  once. The join drops from 174 ms to 119 ms at 1M rows, and from 19 ms
  to 15 ms at 100k (#105).

- Encoding a single key column no longer builds a hash map to combine it
  with the others, there being no others: the distinct values are
  renumbered in row order through an array indexed by code. This is on the
  path every single-key join and group_by takes. A 1M-row join drops from
  119 ms to 116 ms and a 100k-row one from 15 ms to 13 ms; grouping 100k
  rows on a high-cardinality key from 4.2 ms to 3.6 ms (#105).

- Sorting by fixed-width keys no longer ranks its key columns. Each value
  maps to an Int whose signed order is the value's order, in one linear
  pass, which is what the sort compares anyway; ranking existed to give
  strings an order, and string keys still take that path. An n-column sort
  did n sorts before the one that orders the rows, and now does none: for
  1M rows and two keys, building the keys drops from 77 ms to 11 ms and the
  whole sort from 147 ms to 85 ms, which is 3.0x to 1.5x of Polars. Row
  order is unchanged, including -0.0 equal to 0.0, NaN after the numbers in
  either direction, and null placement independent of direction (#108).

- String sort keys are encoded the same way, as their first 24 bytes plus
  their length, so a sort by a string key no longer ranks it either.
  Padding with zeros and comparing the length last is exact for any two
  values that fit, including one that is a prefix of the other and
  including embedded NUL bytes. A value longer than 24 bytes cannot be
  compared from its prefix alone, so such a column falls back to ranking.
  Sorting 1M rows by a string key drops from 331 ms to 75 ms (#108).

- Sorting ranks its numeric key columns by sorting `(value, row)` pairs and
  walking them, instead of reducing the values to the distinct ones and
  binary-searching every row back in. That search was the largest single
  cost of a sort -- 114 ms of the 182 ms spent ranking two key columns of
  1M rows -- and the string path already ranked by walking a sorted order.
  A two-key sort of 1M rows drops from 341 ms to 261 ms; ranking a
  high-cardinality Int64 column drops from 195 ms to 72 ms. Row order is
  unchanged for every dtype, direction and null placement (#108).

- A sort's merge rounds are split across threads. Each round halves the
  number of merges, so the last round was one thread merging the whole
  array; every merge is now cut into output slices, located by binary
  search so that a slice starts at the same place in both runs. Runs are
  also formed one per thread rather than one per 65,536 rows, which that
  minimum -- sized for a linear scan -- had capped at 15 on a 32-core
  machine, and which left sorts below 131,072 rows entirely serial.
  Merging and run-sorting 1M rows drops from 114 ms to 31 ms; a whole
  two-key sort from 261 ms to 191 ms, and at 100k rows from 43 ms to
  38 ms. Row order is unchanged: ranks break ties by row index, so the
  comparison is a total order and a slice boundary falls in exactly one
  place (#108).

- A sort with several key columns ranks them at once rather than one after
  another, which is worth doing because ranking a column is itself serial
  and is the largest part of a sort. A two-key sort of 1M rows spends
  107 ms ranking before and 76 ms after, for 191 ms to 155 ms overall
  (#108).

- CSV reads decode records on worker threads. A block is split at record
  boundaries -- decided by quote parity, so a newline inside a quoted field
  is never mistaken for one -- each range is decoded by its own reader, and
  the partial frames are concatenated in order, which keeps the output
  identical to a serial read. Blocks are read sequentially with the
  trailing partial record carried forward, so memory stays bounded by the
  block size rather than the file size. 1M rows of 8 columns drops from
  1,081 ms to 145 ms, and a 100k-row file from 66 ms to 18 ms. Reads using
  `n_rows`, `skip_rows`, `comment_prefix`, `ignore_errors` or
  `truncate_ragged_lines` stay serial, since those count records from the
  start of the file (#107).

- Appending one column to another copies the window in bulk instead of one
  bounds-checked element at a time. That is how `concat`, `vstack`, batch
  reassembly and every parallel stage reassemble their output, so it is not
  specific to CSV; concatenating the 64 partial frames of a 1M-row CSV read
  drops from 69 ms to 51 ms (#107).

## 0.1.3 - 2026-09-20

### Added

- `pixi run -e oracle bench-polars`: a head-to-head benchmark against Polars
  on identical CSV inputs with thread counts pinned equal, covering CSV
  read, elementwise arithmetic and comparison, filter, global sum, grouping
  at low, high and skewed cardinality on Int64 and String keys, inner join,
  and multi-column sort. Each workload's row count and a column total must
  agree between the engines before their times are compared (#102).

### Changed

- `sort`, `arg_sort`, `top_k` and `bottom_k` sort one row range per worker
  and merge the runs, instead of one serial mergesort. The result is the
  same stable order for any worker count: ranges are cut in row order and
  merges prefer the earlier run on ties. Sorting 1M rows by two keys drops
  from 594 ms to 346 ms (#108).

- Joins encode their key columns one hash bucket at a time when keys are
  many, instead of building one dictionary over both sides. A join's row
  order comes from iterating rows rather than from the id numbering, so the
  ids may be assigned in any consistent order; the documented match order
  is unchanged. The rows per key id also moved from one list per key to a
  flat index. A 1M-row inner join on 500k distinct keys drops from 283 ms
  to 188 ms (#105).

- Grouping is parallel at every cardinality. Rows are hashed by key and
  partitioned into buckets, and each bucket is encoded and reduced on its
  own: equal keys always share a bucket, so no dictionaries or reduction
  states are ever merged, which is what the earlier merge-based attempts
  (#8) lost at high cardinality. Output order is bucket order unless
  `maintain_order=True`, which sorts groups by their first input row in
  O(groups). Frames below the parallel threshold keep the serial path,
  and `GroupBy.len` and `group_indices` are unchanged (#104).

- CSV reads plain decimal Float64 fields in one pass that validates and
  computes together, instead of checking the grammar with one scan and
  converting with another. Values are unchanged and exact: the fast path
  only runs when the mantissa and the power of ten are both exact, so the
  single division is correctly rounded, and anything else (signs,
  exponents, `nan`, infinities, long mantissas) uses the original parser.
  Float parsing was about 35% of a CSV read; a 4-column file drops from
  80.5 ms to 66 ms and a 1M-row 8-column file from 1,260 ms to 1,081 ms
  (#107).

### Fixed

- CSV and temporal parsing accept an ISO 8601 zone designator on a
  datetime: a trailing `Z` means UTC, and `+HH:MM`, `-HH:MM`, `+HHMM` or
  `+HH` are converted to UTC, since datetimes here are naive and hold UTC.
  Both forms previously raised "unexpected trailing text", which rejected
  the most common datetime spelling in real data. Dates and times still
  reject designators (#107).

- Format directives with no separator between them take their exact width,
  so `%Y%m%d` reads `20240228` instead of letting `%Y` consume six digits
  and then failing. A directive followed by a literal still accepts
  one-digit months and days as before (#107).

## 0.1.2 - 2026-09-20

### Added

- Public access to a column's Arrow buffers, so a consumer reads values in
  place instead of one tagged `AnyValue` per element: `unsafe_values()`,
  `unsafe_validity()` and `validity_offset()` on `Column`, `BoolColumn` and
  `StringColumn` (plus `unsafe_bytes()` / `unsafe_offsets()` for the
  `large_utf8` layout), a non-raising `is_valid(i)`, and `to_list()`
  promoted from `_to_list()`. Summing a million-row Float64 column drops
  from about 3,500 us through `Series.get` to about 700 us through the
  buffers. Null slots hold whatever the buffer holds, as in Arrow, where
  they are undefined (#91).

- `DataFrame.group_indices(keys)` returns which rows belong to which group
  without aggregating them, for callers that want each group's rows rather
  than one summary row: `count`, `ids`, `rows(g)`, `all_rows`, `sizes`,
  and `representative(g)` for where to read a group's key values. Groups are
  numbered in first-occurrence order and null keys form their own group,
  both as in `group_by`. Building sub-frames from the indices is left to the
  caller, who may prefer to read the shared buffers directly (#92).

## 0.1.1 - 2026-09-20

Everything below is the initial feature set. v0.1.0 was tagged the same day and
is identical except that it was missing bit-packed Boolean columns, whose merge
had been stranded on a branch; prefer 0.1.1.

### Breaking

- `Series.bool()` returns a `BoolColumn` (bit-packed values) instead of
  `Column[Bool]`, with the same `value`, `is_null`, `null_count`, `take`,
  and `slice` methods. `Series(name, Column[Bool](...))` and
  `DataFrame.filter(Column[Bool])` still work and pack on construction (#80).
- Arrow import keeps narrow integer and float32 types instead of widening
  them to Int64 / Float64, and accepts UInt64.
- `Series.string()` returns a `StringColumn` (Arrow `large_utf8`: one UTF-8
  buffer plus Int64 offsets) instead of `Column[String]`. It has the same
  `value`, `is_null`, `null_count`, `take`, and `slice` methods.
  `Series(name, Column[String](...))` still works and converts (#33).
- Dtypes are structured `DataType` values instead of strings:
  `Series.dtype()`, `DataFrame.dtypes()`, `Field.dtype`, `AnyValue.dtype()`,
  and `CsvField.dtype` return `DataType`. Compare with constants
  (`s.dtype() == DataType.INT64`) or use `String(dtype)` / `dtype.name()`
  for the old string. Name-taking APIs still accept strings.

- Removed the legacy column kernels `sum_int64`, `sum_float64`,
  `greater_than`, and `multiply`, and `DataFrame.group_by_sum`. Use
  expressions instead: `col("x").sum(min_count=1)` for a null-for-empty sum,
  `col("x") > lit(...)`, `col("x") * lit(...)`, and
  `group_by(key).agg(col("v").sum(min_count=1))`. Expression sums check
  overflow on the exact final total rather than on each prefix, so inputs such
  as `[MAX, 1, -1]` now succeed.
- `group_by` and `join` accept keys of any dtype, and `join` supports `how="full"`;
  code that relied on these raising must be updated.

### Changed

- Columns are windows (offset, length) onto reference-counted, immutable
  buffers, as in Arrow arrays. `select`, `rename`, `drop`, `head`, `slice`,
  `column()`, typed extraction, GroupBy snapshots, and expression batch slices
  share storage in O(1) instead of copying. Global reductions are 1.6-2x and
  low-cardinality grouping 1.5x faster at 1M rows (#34).

### Fixed

- Float64 text parsing (CSV, casts, inference) no longer accepts malformed
  numbers that Mojo's parser tolerates, such as `2024-02-28` (read as
  2024002028.0), `1-2`, or `1.5.5`.

### Added

- Installable as a Mojo package straight from GitHub (`[package]` with the
  `pixi-build-mojo` backend), so another Pixi workspace can depend on
  `dataframe_mojo` by git tag or path and `from dataframe import ...`. Adds
  a LICENSE file (MIT).

- Boolean columns store one bit per value (Arrow layout): 8x less memory,
  zero-copy Arrow export, and faster Boolean results (nullable compare
  4.7 -> 2.7 ms, filter 10.7 -> 8.7 ms at 1M rows) (#80).

- Row-wise expressions and filters run on worker threads for large inputs:
  arithmetic and comparisons ~4x faster and filter ~3.5x faster at 1M rows,
  with identical row order and results (#5, #7).

- Parallel reductions: global and grouped reductions over large inputs run
  on worker threads (POSIX threads through the C FFI; no new dependency) with
  worker-private states merged in row order. `DATAFRAME_THREADS` caps the
  thread count (1 disables). Global sums are ~3.7x faster at 1M rows
  (#6, #8).
- Unfused float kernels read contiguous SIMD vectors directly from shared
  column buffers (about 1.45x faster for Float32 arithmetic, `%`, and math
  functions) (#3).
- `DataFrame.fill_null(0)` / `fill_null(0.5)`: a bare number fills every
  column it can adopt (integers: all numeric columns; floats: float
  columns), each in its own dtype; `fill_null("x")` fills string columns.

- Expression sugar: bare numbers and Bools work wherever an `Expr` is
  expected (`col("x") > 0`, `col("x") * 2.5`, `1 + col("x")`,
  `.fill_null(0)`, `when(...).then(1)`), `==` / `!=` build expressions, and
  strings are accepted by comparisons, `is_in`, `fill_null`, and
  `then`/`otherwise`. Bare numbers are untyped and adopt the other operand's
  dtype at bind time (range-checked), keeping the no-implicit-promotion
  rule (#75).

- Numeric dtypes Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, and
  Float32 alongside Int64 and Float64, through every operation: checked
  arithmetic at each width, Float32 SIMD kernels, exact 128-bit sums (8/16-bit
  sums produce Int64, as in Polars), sorting, grouping and join keys, casts
  with exact range checks, CSV fields, display, Arrow (zero-copy, native
  formats), and typed literals (`lit(Int32(1))`, generic `lit[D](Scalar[D])`).
  `Series.numeric[D]()` / `.int8()` ... `.float32()` and matching `AnyValue`
  accessors; `Expr.cast` also takes a `DataType` (#30).

- Arrow C Data Interface: `export_arrow` / `import_arrow` (frames as struct
  arrays) and `export_arrow_series` / `import_arrow_series`, with
  `ArrowArray` / `ArrowSchema` structs. Int64, Float64, String, Datetime,
  Duration, and Time export zero-copy; Bool and Date convert. Import copies
  and accepts narrower integer, float32, utf8, date64, and time32 inputs.
  See docs/arrow.md (#35).

- Date, Datetime, Duration, and Time types with a `.dt()` namespace, temporal
  arithmetic and casts, strptime/strftime-style formats, CSV support,
  inference, and `date_range`/`datetime_range`.

- Frame utilities, display, concatenation, multi-column sort, multi-key
  grouping and joins, reshaping, deduplication, and a Series API.
- Expression operators, Kleene logic, conditionals, reductions, casts, string
  and window operations, selectors, and `over()` partitions.
- `write_csv`, `to_csv_string`, and `CsvSchema.of`.
