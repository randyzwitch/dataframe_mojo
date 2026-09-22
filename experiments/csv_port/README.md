# CSV port benchmark

This harness measures the public Mojo `read_csv` API and pinned Polars 1.44.2
on one caller-supplied file. There is no internal-reader selection: the public
API is the sole implementation in this checkout. Each run uses this fixed
explicit schema:

    id:Int64,value:Float64,active:Bool,label:String

The file must have that header. The full scenario reads all four columns. The
projected scenario reads `id,label`, so Float64 and Bool parsing is excluded in
every engine. Each process warms once, retains raw read samples as CSV, and
checks full frame equality (Mojo) or Polars frame equality plus shape/null
counts. The timed region contains only the read call.

Create the deterministic fixture outside the timed process (numeric nulls,
quoted separators, escaped quotes, Unicode, and occasional embedded newlines):

    python3 experiments/csv_port/prepare.py build/csv_port/mixed.csv --rows 1000000

Build the public Mojo harness without timing it:

    pixi run mojo build -I . experiments/csv_port/bench_csv_port.mojo -o /tmp/bench_csv_port_current

With an existing `build/csv_port/mixed.csv`, collect raw samples in separate
processes. `DATAFRAME_THREADS` must be set before launching Mojo and
`POLARS_MAX_THREADS` before importing Polars.

    mkdir -p experiments/csv_port/results
    DATAFRAME_THREADS=1 /tmp/bench_csv_port_current build/csv_port/mixed.csv 8 full > experiments/csv_port/results/mojo_public_current_full_t1.csv
    POLARS_MAX_THREADS=1 pixi run -e oracle python experiments/csv_port/bench_csv_port_polars.py build/csv_port/mixed.csv 8 full > experiments/csv_port/results/polars_full_t1.csv

    DATAFRAME_THREADS=32 /tmp/bench_csv_port_current build/csv_port/mixed.csv 8 projected > experiments/csv_port/results/mojo_public_current_projected_t32.csv
    POLARS_MAX_THREADS=32 pixi run -e oracle python experiments/csv_port/bench_csv_port_polars.py build/csv_port/mixed.csv 8 projected > experiments/csv_port/results/polars_projected_t32.csv

Repeat both commands for both scenarios at each desired thread count. Keep every
output file; calculate medians only afterward:

    python experiments/csv_port/summarize.py experiments/csv_port/results/*.csv

For a before/after Mojo claim, build this same public-only harness in an
unchanged `main` checkout (or use a recorded baseline binary), then use the
same fixture, environment, and invocation:

    git worktree add --detach /tmp/dataframe-mojo-main-bench origin/main
    cp experiments/csv_port/bench_csv_port.mojo /tmp/dataframe-mojo-main-bench/experiments/csv_port/
    cd /tmp/dataframe-mojo-main-bench
    pixi run mojo build -I . experiments/csv_port/bench_csv_port.mojo -o /tmp/bench_csv_port_main
    DATAFRAME_THREADS=32 /tmp/bench_csv_port_main /path/to/mixed.csv 8 full > /path/to/mojo_public_main_full_t32.csv

Record both commits and the binary build commands with new samples. The retained
result directories are historical evidence from earlier internal-port stages;
their `legacy` and `explicit` labels are deliberately unchanged and must not be
mixed with a public-API comparison without recording the build provenance.

Untimed cross-engine validation:

    pixi run mojo build -I . experiments/csv_port/verify_reader.mojo -o /tmp/csv-verify-reader
    pixi run -e oracle python experiments/csv_port/verify_reader_oracle.py /tmp/csv-verify-reader

The initial source-port readout and raw samples are in
[results/initial-source-port](results/initial-source-port/README.md).

Expanded deterministic differential matrix (24 comparisons):

    pixi run mojo build -I . experiments/csv_port/verify_reader_differential.mojo -o /tmp/csv-verify-reader-differential
    pixi run -e oracle python experiments/csv_port/verify_reader_differential.py /tmp/csv-verify-reader-differential
