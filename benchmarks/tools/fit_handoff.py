"""Fit the hand-off placement calibration from a `bench_handoff` sweep.

    build/bench_handoff 5 > sweep.tsv
    python3 benchmarks/tools/fit_handoff.py sweep.tsv > handoff.json
    DATAFRAME_HANDOFF_CALIBRATION=handoff.json ...

For each operator key (`filter_sum:<dtype>:<case>`) and engine, fit
`fixed_ns + row_ns * rows` to the measured best times by least squares on
relative error, so small inputs set the fixed cost and large inputs the
per-row cost. Costs are clamped at zero. The output is the JSON that
`dataframe/placement.mojo` reads: `cpu` for the executor, `accel` for the
provider. Rows the sweep did not hand off are skipped, so CPU time is
never recorded as the accelerator's.
"""

import csv
import json
import sys


def fit(points):
    """Weighted least squares of y = a + b x with weights 1 / y**2."""
    sw = sx = sy = sxx = sxy = 0.0
    for x, y in points:
        w = 1.0 / (y * y)
        sw += w
        sx += w * x
        sy += w * y
        sxx += w * x * x
        sxy += w * x * y
    det = sw * sxx - sx * sx
    if det == 0:
        return max(0.0, sy / sw), 0.0
    b = (sw * sxy - sx * sy) / det
    a = (sy - b * sx) / sw
    return max(0.0, a), max(0.0, b)


def main(path):
    series = {}
    with open(path) as source:
        rows = [r for r in csv.DictReader(source, delimiter="\t") if r.get("rows")]
    for row in rows:
        if row["handed"] != "True":
            continue
        key = f"filter_sum:{row['dtype']}:{row['case']}"
        n = float(row["rows"])
        for engine, column in (("cpu", "cpu_ms"), ("accel", "metal_ms")):
            series.setdefault(key, {}).setdefault(engine, []).append(
                (n, float(row[column]) * 1e6)
            )
    operators = {}
    for key, engines in sorted(series.items()):
        operators[key] = {}
        for engine, points in engines.items():
            fixed, per_row = fit(points)
            operators[key][engine] = {
                "fixed_ns": round(fixed, 1),
                "row_ns": round(per_row, 6),
            }
    json.dump({"operators": operators}, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fit_handoff.py SWEEP.tsv")
    main(sys.argv[1])
