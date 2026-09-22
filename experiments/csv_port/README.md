# CSV port benchmark

This harness compares the legacy Mojo CSV reader, the internal explicit reader
port, and pinned Polars 1.44.2 on one caller-supplied file. Each run uses this
fixed explicit schema:

    id:Int64,value:Float64,active:Bool,label:String

The file must have that header. The full scenario reads all four columns. The
projected scenario reads id,label, so Float64 and Bool parsing is excluded in
every engine. Each process warms once, retains raw read samples as CSV, and
checks full frame equality (Mojo) or Polars frame equality plus shape/null
counts. The timed region contains only the read call.

Create the deterministic fixture outside the timed process (numeric nulls,
quoted separators, escaped quotes, Unicode, and occasional embedded newlines):

    python3 experiments/csv_port/prepare.py build/csv_port/mixed.csv --rows 1000000

Build the Mojo binary without timing it:

    pixi run mojo build -I . experiments/csv_port/bench_csv_port.mojo -o /tmp/bench_csv_port

With an existing build/csv_port/mixed.csv, collect raw samples for each thread
count in separate processes. DATAFRAME_THREADS must be set before launching
Mojo and POLARS_MAX_THREADS before importing Polars.

    mkdir -p experiments/csv_port/results
    DATAFRAME_THREADS=1 /tmp/bench_csv_port legacy build/csv_port/mixed.csv 8 full > experiments/csv_port/results/mojo_legacy_full_t1.csv
    DATAFRAME_THREADS=1 /tmp/bench_csv_port explicit build/csv_port/mixed.csv 8 full > experiments/csv_port/results/mojo_explicit_full_t1.csv
    POLARS_MAX_THREADS=1 pixi run -e oracle python experiments/csv_port/bench_csv_port_polars.py build/csv_port/mixed.csv 8 full > experiments/csv_port/results/polars_full_t1.csv

    DATAFRAME_THREADS=32 /tmp/bench_csv_port legacy build/csv_port/mixed.csv 8 projected > experiments/csv_port/results/mojo_legacy_projected_t32.csv
    DATAFRAME_THREADS=32 /tmp/bench_csv_port explicit build/csv_port/mixed.csv 8 projected > experiments/csv_port/results/mojo_explicit_projected_t32.csv
    POLARS_MAX_THREADS=32 pixi run -e oracle python experiments/csv_port/bench_csv_port_polars.py build/csv_port/mixed.csv 8 projected > experiments/csv_port/results/polars_projected_t32.csv

Repeat the three commands for both scenarios at each desired thread count.
Keep every output file; calculate medians only afterward:

    python experiments/csv_port/summarize.py experiments/csv_port/results/*.csv

`legacy` means the public reader in the checkout being compiled. On the port
branch it also uses the shared chunk-storage changes, so it is not an untouched
`main` baseline. Record the checkout commit with results and build an unchanged
main reader separately before making a before/after performance claim.

Untimed cross-engine validation:

    pixi run mojo build -I . experiments/csv_port/verify_reader.mojo -o /tmp/csv-verify-reader
    pixi run -e oracle python experiments/csv_port/verify_reader_oracle.py /tmp/csv-verify-reader

The initial source-port readout and raw samples are in
[results/initial-source-port](results/initial-source-port/README.md).

Expanded deterministic differential matrix (24 comparisons):

    pixi run mojo build -I . experiments/csv_port/verify_reader_differential.mojo -o /tmp/csv-verify-reader-differential
    pixi run -e oracle python experiments/csv_port/verify_reader_differential.py /tmp/csv-verify-reader-differential
