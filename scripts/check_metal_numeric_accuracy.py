"""Check binary32 division against an independent exact Fraction oracle.

Run inside the pinned development environment. This checks host execution of
the portable scalar primitive; Apple contracts separately prove device use.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess


def magnitude(raw):
    exponent = (raw >> 23) & 255
    significand = raw & 0x7FFFFF
    if exponent:
        significand |= 1 << 23
    power = (exponent or 1) - 150
    return (Fraction(significand << power) if power >= 0
            else Fraction(significand, 1 << -power))


def nearest(value):
    """Search adjacent binary32 values, then compare exact rational distances.

    This deliberately does not use the primitive's significand division or
    exponent/shift rounding algorithm, nor Float64 as a rounding intermediary.
    """
    if value >= Fraction((1 << 128) - (1 << 103)):
        return 0x7F800000
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high + 1) // 2
        if magnitude(middle) <= value:
            low = middle
        else:
            high = middle - 1
    if low == 0x7F7FFFFF:
        return low
    below, above = value - magnitude(low), magnitude(low + 1) - value
    return low if below < above or below == above and low % 2 == 0 else low + 1


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/metal-numeric-accuracy"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(70031)
    edges = [0, 1, 2, 0x3FFFFF, 0x7FFFFF, 0x800000, 0x800001,
             0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
             0x40000000, 0x7F7FFFFE, 0x7F7FFFFF]
    pairs = [(a | sa << 31, b | sb << 31)
             for a in edges for b in edges if b
             for sa in (0, 1) for sb in (0, 1)]
    pairs += [(rng.randrange(0x7F800000) | rng.randrange(2) << 31,
               rng.randrange(1, 0x7F800000) | rng.randrange(2) << 31)
              for _ in range(12000)]
    expected = []
    for a, b in pairs:
        raw = nearest(magnitude(a & 0x7FFFFFFF) / magnitude(b & 0x7FFFFFFF))
        expected.append(f"{a},{b},{raw | ((a ^ b) & 0x80000000)}")
    fixture = output / "fixtures.csv"
    fixture.write_text("\n".join(expected) + "\n")
    driver = repo / "scripts/check_metal_numeric_accuracy.mojo"
    shutil.copy2(driver, output / "driver.mojo")
    shutil.copy2(Path(__file__), output / "oracle.py")
    binary = output / "check"
    command = ["mojo", "build", "-O1", "--target-accelerator=unknown",
               "-I", str(repo / "providers/cpu"), "-I", str(repo),
               str(driver), "-o", str(binary)]
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = output / "actual.csv"
    with (output / "results.log").open("w") as log:
        subprocess.run([str(binary), str(fixture), str(actual)], cwd=repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    assert actual.read_text().splitlines() == expected
    files = ["fixtures.csv", "actual.csv", "driver.mojo", "oracle.py", "check", "build.log", "results.log"]
    report = {"status": "PASS exact Fraction binary-search oracle", "cases": len(pairs),
              "scope": "Host scalar primitive only; Apple device and public query checks required separately.",
              "primitive_sha256": sha256(repo / "dataframe_metal/accelerator/gpu/native_numeric.mojo"),
              "command": command, "files_sha256": {name: sha256(output / name) for name in files}}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
