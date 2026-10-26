# IPC buffer measurements

The same `benchmarks/bench_ipc_buffers.mojo` program runs against separate
before/after source worktrees and their own native libraries. Build both with
the pinned Mojo compiler, `-O3 -I providers/cpu -I .`, from inside each source
worktree so implicit imports cannot select another revision. Build each native
library from that revision's `native/dfparquet/build.sh`; the Arrow build cache
can be reused. Keep baseline and candidate binaries and libraries separate.

`pixi run -e oracle python3 benchmarks/tools/generate_ipc_buffers.py` produces
independent PyArrow files and a manifest under `build/ipc-buffers/data/`
(or a directory selected with `--output`).
It generates one million rows of eight Int64 columns, 500,000 rows of mixed
nullable numeric/string/binary/Boolean/dictionary/list/struct columns, and
500,000 rows of Arrow string/binary views. Files use 65,536-row batches and
uncompressed/LZ4/Zstd codecs. The manifest records file digests, shape, and a
checksum covering every non-null payload byte/value and nested child.

Build the allocation observer separately:

```sh
# Linux
cc -shared -fPIC -std=c11 -O2 benchmarks/tools/ipc_allocations.c -ldl -o build/ipc_allocations.so
# Apple
cc -dynamiclib -std=c11 -O2 -undefined dynamic_lookup benchmarks/tools/ipc_allocations.c -o build/ipc_allocations.dylib
```

Run the driver with the two built programs, libraries, and exact source revisions:

```sh
python3 benchmarks/tools/bench_ipc_buffers.py \
  --baseline /absolute/path/baseline-bench \
  --candidate /absolute/path/candidate-bench \
  --baseline-revision BEFORE_COMMIT --candidate-revision AFTER_COMMIT \
  --baseline-library /absolute/path/baseline/libdfparquet.so \
  --candidate-library /absolute/path/candidate/libdfparquet.so \
  --observer /absolute/path/ipc_allocations.so \
  --data build/ipc-buffers/data --output build/ipc-buffer-measurements.json
```

Use `.dylib` library paths on Apple. The driver verifies input digests, alternates
variant order, and checks each result against the independent manifest. Five
fresh-process repetitions measure read completion and full checksum traversal,
plus process peak RSS. Inputs are warmed by digest verification; destruction
is outside the reported traversal endpoint. Mojo uses four workers; Arrow
retains its host defaults. The owned-import case checks ordinary copying
imports against the same million-row, eight-column input shape.

Timing and RSS runs do not load the allocation observer. Separate instrumented
runs count actual Mojo allocation requests at the pinned compiler runtime's
`KGEN_CompilerRT_AlignedAlloc(alignment, bytes)` boundary. Native allocators,
including Arrow's private pool, are excluded; process RSS includes the entire
process. This runtime boundary is version-specific. The driver rejects an
observer that cannot see the ordinary 64 MB numeric payload allocation, so
an incompatible compiler cannot silently produce incomplete counters.
Apple observer runs launch the benchmark directly so the system `time` binary
cannot strip its injection environment. Instrumented timings are not used as
latency samples.

The driver waits for host compilers and rejects samples overlapping one. If
the existing dataframe timing lock is present, it also coordinates with that
gate. Store generated inputs and measurements outside tracked paths; reports
belong with the relevant PR evidence, following `docs/design/benchmarks.md`.
