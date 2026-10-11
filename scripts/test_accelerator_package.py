#!/usr/bin/env python3
"""Check identified provider packages with synthetic policies, never calibration."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from bench_accelerator_cost import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="shared-provider-consumer-") as directory:
        work = Path(directory)
        source = work / "app.mojo"
        shutil.copyfile(ROOT / "tests/accelerator_package_consumer.mojo", source)
        binary = args.binary.resolve() if args.binary else work / "app"
        if not args.binary:
            subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", str(package), str(source), "-o", str(binary)], cwd=work, check=True)
        env = dict(os.environ, DATAFRAME_THREADS="8")
        for name in ("DATAFRAME_ACCEL_COST_TABLE", "DATAFRAME_ACCEL_DEVICE", "DATAFRAME_ACCEL_MEMORY_LIMIT"):
            env.pop(name, None)
        key = json.loads(subprocess.check_output([str(binary), "fixture"], cwd=work, env=env, text=True))
        assert key["cpu_build_id"] != "development-source-no-calibration"
        assert key["runtime_build_id"] != "development-source-no-calibration"

        def model(fixed):
            return dict(coefficients={k: fixed if k == "fixed" else 0 for k in FEATURES}, minimum=dict.fromkeys(FEATURES, 0), maximum=dict.fromkeys(FEATURES, 1e30), residual_factor=1)

        record = dict(key, validated=True, validation_failures=0,
                      training_queries=8, validation_queries=8,
                      expires_at=int(time.time()) + 3600,
                      distribution_coverage="conservative-null-selectivity-envelope",
                      safety_factor=1.5, cpu=model(1e15), accelerator=model(1))
        table = dict(format="dataframe-analytical-cost-v1", memory_fraction=.8, records=[record])
        path = work / "synthetic-do-not-deploy.json"

        def run(mode, costs=None, **settings):
            selected = dict(env, **settings)
            if costs is not None:
                path.write_text(json.dumps(costs))
                selected["DATAFRAME_ACCEL_COST_TABLE"] = str(path)
            subprocess.run([str(binary), mode], cwd=work, env=selected, check=True)

        run("cpu")
        run("gpu", table)
        run("cpu", table, DATAFRAME_ACCEL_MEMORY_LIMIT="0")
        run("cpu", table, DATAFRAME_ACCEL_DEVICE="invalid")
        run("cpu", table, DATAFRAME_THREADS="1")
        for field, value in (("provider", "other"), ("device", "other"), ("runtime_build_id", "stale"), ("cpu_build_id", "stale"), ("host", "other"), ("expires_at", 0), ("validated", False), ("validation_failures", 1)):
            invalid = copy.deepcopy(table)
            invalid["records"][0][field] = value
            run("cpu", invalid)
        invalid = copy.deepcopy(table)
        invalid["records"][0]["accelerator"] = model(1e18)
        run("cpu", invalid)
        run("cpu", dict(table, records=[]))
        for fraction in (0, -1, 1.1):
            run("cpu", dict(table, memory_fraction=fraction))
        path.write_text("invalid JSON")
        subprocess.run([str(binary), "cpu"], cwd=work, env=dict(env, DATAFRAME_ACCEL_COST_TABLE=str(path)), check=True)
        if key["provider"] == "metal":
            run("cpu", table, DATAFRAME_METAL_LIBRARY="/nonexistent/shared-provider-native.dylib")
        print("Synthetic fixtures were temporary; no calibration was installed.")


if __name__ == "__main__":
    main()
