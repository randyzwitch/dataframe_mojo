#!/usr/bin/env python3
"""Read-only keyed diagnostics for every wrong-answer H2O group-by export."""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
SCRIPT_SHA = sys.argv[3]
LOGROOT = ROOT.parent / "logs"
KEYS = {
    "q3": ["id3"], "q4": ["id4"], "q5": ["id6"],
    "q6": ["id4", "id5"], "q9": ["id2", "id4"],
}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def schema_file(path):
    doc = json.loads(path.read_text())
    return {"rows": doc["rows"], "columns": [
        {"name": c["name"], "dtype": c["dtype"]["name"], "encoded": c["dtype"].get("encoded")}
        for c in doc["columns"]]}

def exports(log):
    out = {}
    for line in log.read_text().splitlines():
        if line.startswith("answer_export\t"):
            item = json.loads(line.split("\t", 1)[1])
            out[item["engine"]] = (Path(item["path"]), Path(item["schema_path"]))
    return out

def ordered_bits(bits):
    sign = 1 << 63
    return (~bits) & ((1 << 64) - 1) if bits & sign else bits | sign

def key_rows(frame, keys):
    return [tuple(row) for row in frame.select(keys).rows()]

def compare(left, right, keys):
    left_keys, right_keys = key_rows(left, keys), key_rows(right, keys)
    left_positions = {key: i for i, key in enumerate(left_keys)}
    right_positions = {key: i for i, key in enumerate(right_keys)}
    result = {
        "rows": {"left": left.height, "right": right.height},
        "key_columns": keys,
        "schema_equal": [(n, str(d)) for n, d in zip(left.columns, left.dtypes)] == [(n, str(d)) for n, d in zip(right.columns, right.dtypes)],
        "key_sets_equal": set(left_positions) == set(right_positions),
        "duplicate_keys": len(left_positions) != len(left_keys) or len(right_positions) != len(right_keys),
        "key_order_equal": left_keys == right_keys,
        "key_positions_differ": sum(a != b for a, b in zip(left_keys, right_keys)),
        "columns": {},
    }
    if not result["key_sets_equal"] or result["duplicate_keys"] or left.height != right.height:
        result["value_alignment"] = "not_computed: keys are not a unique equal set"
        return result
    for name, dtype in zip(left.columns, left.dtypes):
        if name in keys:
            continue
        pairs = [(left[name][left_positions[k]], right[name][right_positions[k]]) for k in left_positions]
        if dtype in (pl.Float32, pl.Float64):
            bit_diff = numeric_diff = 0
            ulps, absdiff, reldiff = [], [], []
            for a, b in pairs:
                if a is None or b is None:
                    if a != b:
                        bit_diff += 1; numeric_diff += 1
                    continue
                typ = np.float32 if dtype == pl.Float32 else np.float64
                rawtype = np.uint32 if dtype == pl.Float32 else np.uint64
                ra = int(np.asarray([a], dtype=typ).view(rawtype)[0])
                rb = int(np.asarray([b], dtype=typ).view(rawtype)[0])
                if ra != rb: bit_diff += 1
                if a != b: numeric_diff += 1
                if np.isfinite(a) and np.isfinite(b):
                    ulps.append(abs(ordered_bits(ra) - ordered_bits(rb)))
                    delta = abs(float(a) - float(b)); absdiff.append(delta)
                    reldiff.append(delta / max(abs(float(a)), np.finfo(np.float64).tiny))
            result["columns"][name] = {
                "dtype": str(dtype), "exact_bit_differences": bit_diff,
                "numeric_differences": numeric_diff, "rows": len(pairs),
                "nulls": {"left": left[name].null_count(), "right": right[name].null_count()},
                "ulp_max_finite": max(ulps, default=0),
                "ulp_over_1_finite": sum(x > 1 for x in ulps),
                "absolute_error_max": max(absdiff, default=0.0),
                "relative_error_max": max(reldiff, default=0.0),
                "exact": bit_diff == 0,
            }
        else:
            mismatches = sum(a != b for a, b in pairs)
            result["columns"][name] = {"dtype": str(dtype), "value_differences": mismatches,
                "rows": len(pairs), "nulls": {"left": left[name].null_count(), "right": right[name].null_count()},
                "exact": mismatches == 0}
    result["value_alignment"] = "unique declared key columns"
    return result

report = {"answer_root": str(ROOT), "log_root": str(LOGROOT),
          "coverage_run": "metal-868-dev-coverage-a52334e7",
          "analysis": {"script_path": "/tmp/h2o_all_a523_compare.py", "script_sha256": SCRIPT_SHA,
                       "python_environment": "pixi --locked, -e oracle, manifest ~/GitHub/dataframe_mojo-metal-868/pixi.toml",
                       "scope": "H2O group-by wrong_answer logs only; read-only answer exports/logs; exact float differences remain failures",
                       "key_source": "benchmarks/tools/metal_case_metadata.json alignment_keys / suite query definitions",
                       "duplicate_handling": "keyed comparison only when declared keys form a unique equal set; otherwise report alignment unavailable without forcing row pairing"},
          "cases": {}}
for log in sorted(LOGROOT.glob("h2o_groupby-*.txt")):
    body = log.read_text()
    classes = [line for line in body.splitlines() if line.startswith("coverage\t")]
    if not any("\twrong_answer\t" in line for line in classes):
        continue
    m = re.fullmatch(r"h2o_groupby-(.+)-(q\d+)", log.stem)
    if not m: continue
    variant, query = m.group(1), m.group(2)
    case = f"{variant}/{query}"
    key_cols = KEYS.get(query)
    exported = exports(log)
    entry = {"log_path": str(log), "log_sha256": sha(log), "coverage_classification": "wrong_answer",
             "variant": variant, "query": query, "key_columns": key_cols,
             "exports": {}, "comparisons": {}}
    for engine, (parquet, schema) in exported.items():
        entry["exports"][engine] = {"parquet_path": str(parquet), "parquet_sha256": sha(parquet),
            "parquet_bytes": parquet.stat().st_size, "schema_path": str(schema),
            "schema_sha256": sha(schema), "schema_bytes": schema.stat().st_size,
            "schema": schema_file(schema)}
    if not key_cols:
        entry["status"] = "no declared key mapping for query"
    elif "cpu" not in exported:
        entry["status"] = "CPU answer export missing"
    else:
        cpu = pl.read_parquet(exported["cpu"][0])
        for engine in sorted(exported):
            if engine == "cpu": continue
            other = pl.read_parquet(exported[engine][0])
            result = compare(cpu, other, key_cols)
            result["encoded_schema_equal"] = schema_file(exported["cpu"][1]) == schema_file(exported[engine][1])
            entry["comparisons"][f"cpu0_vs_{engine}"] = result
        entry["status"] = "compared exact available exports; exact mismatches remain failures"
    report["cases"][case] = entry
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
print(OUTPUT)
