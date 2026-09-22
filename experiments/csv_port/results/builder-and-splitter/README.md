# Builder state alignment

Same one-million-row fixture, machine, compiler and seven-sample warm-cache
method as ../initial-source-port. `projection` is 64c78e7; `splitter` adds
f568d61. `builder` adds builder-owned quote/encoding/scratch state and moves
null-token matching to the decoder as in Polars builder.rs/parser.rs.
The benchmark used the original integer frontend and checked span access.

| Read | Threads | Splitter ms | Builder ms |
|---|---:|---:|---:|
| Full | 1 | 355.24 | 275.09 |
| Projected | 1 | 257.77 | 172.17 |
| Full | 32 | 19.54 | 16.97 |
| Projected | 32 | 17.30 | 22.37 |

The projected 32-thread regression reproduced in reverse-order runs:
22.33 vs 18.40 ms and 22.44 vs 17.90 ms. This is a source-alignment change
with substantial single-thread gains, **not an across-the-board improvement**.
No benchmark ran concurrently with compilation or tests. Decoder 9/9,
buffers 5/5, lazy-validity 4/4, and actual Polars differential 24/24 passed.
