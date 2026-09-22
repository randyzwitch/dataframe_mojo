#!/usr/bin/env bash
set -euo pipefail
for threads in 1 32; do
  for rows in 1000000 10000000; do
    for spec in int64:repeat:100 int64:repeat:$((rows / 10)) int64:sparse:$((rows / 10)) int64:skew:100 string:repeat:$((rows / 10)) string:skew:100; do
      IFS=: read -r kind shape cardinality <<< "$spec"
      suffix="${kind}_${shape}_${rows}_c${cardinality}_t${threads}_full110.log"
      DATAFRAME_THREADS=$threads /tmp/raw_partitioned_join_matrix_bench "$kind" "$rows" 3 "$threads" "$cardinality" "$shape" > "experiments/cpu110/results/raw_join_${suffix}"
      POLARS_MAX_THREADS=$threads pixi run -e oracle python experiments/cpu110/raw_join_polars.py "$kind" "$rows" 3 "$cardinality" "$shape" > "experiments/cpu110/results/polars_join_${suffix}"
    done
  done
done
