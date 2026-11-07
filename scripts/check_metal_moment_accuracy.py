"""Check Metal moment finishing against independent exact/high-precision references.

Run inside the project environment: pixi run python scripts/check_metal_moment_accuracy.py
This uses the CPU provider and requires no Apple device. Artifacts stay in build/.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess


def _generate(p: Path) -> tuple[int, int]:
    rng = random.Random(20261107)
    datasets = []

    def coeff(raw):
        e = (raw >> 23) & 255
        assert e != 255
        m = (raw & 0x7FFFFF) | (0x800000 if e else 0)
        return (-1 if raw >> 31 else 1) * (m << max(0, e - 1))

    for e in range(255):
        raw = (e << 23) | rng.randrange(1, 0x7FFFFE)
        for weights in [(1, 1), (123456789, 123456788)]:
            datasets.append(
                (-149, [(coeff(raw), weights[0]), (coeff(raw + 1), weights[1])])
            )
            datasets.append(
                (-149, [(coeff(raw), weights[0]), (-coeff(raw), weights[1])])
            )
    for _ in range(180):
        datasets.append(
            (
                -149,
                [
                    (
                        coeff(rng.randrange(255 << 23) | (rng.randrange(2) << 31)),
                        rng.randrange(1, 10000),
                    )
                    for i in range(rng.randrange(2, 8))
                ],
            )
        )
    for vals in [
        [-(1 << 63), (1 << 63) - 1],
        [(1 << 60), (1 << 60) + 256],
        [(1 << 60) + 1, (1 << 60) + 2],
        [-(1 << 31), (1 << 31) - 1],
        [0, 1],
        [-1, 0, 1],
        [17, 17],
    ]:
        for weights in [1, 123456789]:
            datasets.append((0, [(int(float(x)), weights) for x in vals]))
    datasets.extend(
        [
            (-149, [(1, 1), (-1, (1 << 32) - 2)]),
            (0, [(0, (1 << 32) - 2), (1, 1)]),
            (-149, []),
            (0, [(7, 1)]),
        ]
    )
    fixtures = []
    lines = []
    for unit, values in datasets:
        n = sum(w for x, w in values)
        s = sum(x * w for x, w in values)
        q = sum(x * x * w for x, w in values)
        sums = [
            sum(
                (-1 if x < 0 else 1) * ((abs(x) >> (32 * i)) & 0xFFFFFFFF) * w
                for x, w in values
            )
            for i in range(9)
        ]
        squares = [
            sum((((x * x) >> (32 * i)) & 0xFFFFFFFF) * w for x, w in values)
            for i in range(18)
        ]
        for ddof in sorted({0, 1, max(0, n - 1), n}):
            for std in [0, 1]:
                expected = None if n <= ddof else Fraction(
                    n * q - s * s, n * (n - ddof)
                ) * Fraction(2) ** (2 * unit)
                case = {
                    "unit": unit,
                    "count": n,
                    "ddof": ddof,
                    "std": std,
                    "numerator": str(expected.numerator) if expected
                    is not None else None,
                    "denominator": str(expected.denominator) if expected
                    is not None else None,
                }
                fixtures.append(case)
                parts = [unit, n, ddof, std]
                for x in sums + squares:
                    assert abs(x) <= (1 << 64) - 1
                    parts.extend([int(x < 0), abs(x) & 0xFFFFFFFF, abs(x) >> 32])
                lines.append(",".join(map(str, parts)))
    (p / "fixtures.csv").write_text("\n".join(lines) + "\n")
    (p / "fixtures.json").write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
    return len(fixtures), len(datasets)


def _verify(p: Path) -> dict:
    cases = json.loads((p / "fixtures.json").read_text())
    lines = (p / "actual.log").read_text().splitlines()
    assert len(lines) == len(cases), (len(lines), len(cases))
    max_relative = Decimal(0)
    max_ulps = 0
    numeric = 0
    nulls = 0
    with localcontext() as context:
        context.prec = 500
        limit = Decimal(2) ** -51
        for i, (case, line) in enumerate(zip(cases, lines)):
            index, value = line.split()
            assert int(index) == i
            if case["numerator"] is None:
                assert value == "NULL", (i, value)
                nulls += 1
                continue
            assert value != "NULL", i
            bits = int(value)
            actual = struct.unpack(">d", struct.pack(">Q", bits))[0]
            expected = Fraction(int(case["numerator"]), int(case["denominator"]))
            exact_decimal = Decimal(expected.numerator) / Decimal(expected.denominator)
            if case["std"]:
                exact_decimal = exact_decimal.sqrt()
                error = abs(Decimal.from_float(actual) - exact_decimal)
                assert error <= limit * abs(exact_decimal), (
                    i,
                    case,
                    actual,
                    error,
                    exact_decimal,
                )
            else:
                error = abs(Fraction(actual) - expected)
                assert error <= Fraction(1, 1 << 51) * abs(expected), (
                    i,
                    case,
                    actual,
                    error,
                )
            if exact_decimal:
                relative = abs(Decimal.from_float(actual) - exact_decimal) / abs(
                    exact_decimal
                )
                max_relative = max(max_relative, relative)
            else:
                assert actual == 0.0
            reference = float(exact_decimal)
            reference_bits = struct.unpack(">Q", struct.pack(">d", reference))[0]
            ulps = abs(bits - reference_bits)
            max_ulps = max(max_ulps, ulps)
            assert ulps <= 4, (i, ulps, case)
            numeric += 1
    output = {
        "status": "PASS independent exact Fraction variance and 500-digit Decimal std oracle",
        "cases": len(cases),
        "numeric": numeric,
        "nulls": nulls,
        "relative_error_limit": "4u = 2**-51",
        "maximum_observed_relative_error": format(max_relative, ".8E"),
        "maximum_observed_ULP_distance": max_ulps,
        "files_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in p.iterdir()
            if f.is_file()
        },
    }
    (p / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=repo / "build/metal-moment-accuracy"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("mojo")
    if compiler is None:
        parser.error(
            "mojo is not on PATH; run this script inside the project pixi environment"
        )
    count, distributions = _generate(output)
    print(f"Generated {count} fixtures from {distributions} distributions", flush=True)
    driver = Path(__file__).with_suffix(".mojo")
    shutil.copyfile(driver, output / "host.mojo")
    task_tmpdir = output / "compiler-temp"
    task_tmpdir.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(task_tmpdir))
    command = [
        compiler,
        "build",
        "-O3",
        "--target-accelerator=unknown",
        "-I",
        str(repo / "providers/cpu"),
        "-I",
        str(repo),
        str(output / "host.mojo"),
    ]
    with (output / "build.log").open("w") as log:
        subprocess.run(
            command + ["-o", str(output / "host")],
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (output / "actual.log").open("w") as log:
        subprocess.run(
            [str(output / "host"), str(output / "fixtures.csv")],
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (output / "ir-build.log").open("w") as log:
        subprocess.run(
            command + ["--emit", "llvm", "-o", str(output / "strict.ll")],
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    llvm = (output / "strict.ll").read_text()
    assert "fdiv double" in llvm and "@llvm.sqrt.f64" in llvm
    assert not re.search(r"\bfdiv\s+(fast|arcp|reassoc|afn)\b", llvm)
    report = _verify(output)
    report["finisher_sha256"] = hashlib.sha256(
        (repo / "dataframe_metal/moment_finish.mojo").read_bytes()
    ).hexdigest()
    report["oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["driver_sha256"] = hashlib.sha256(driver.read_bytes()).hexdigest()
    report[
        "strict_IR"
    ] = "PASS Float64 division/sqrt without approximate division flags"
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
