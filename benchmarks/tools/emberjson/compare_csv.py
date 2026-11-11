"""Generate CSV fixtures and compare prebuilt binaries with balanced run order.

Results stay outside the repository. These development fixtures isolate numeric
input families; they do not establish general dataframe performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

VALUES = {
    "plain": [
        "0",
        "-0",
        "1.25",
        "-37.125",
        "1348.123",
        "0.1",
        "123456.75",
        "1030",
        "0.00012345",
        "98765.43",
    ],
    "scientific": [
        "1e-30",
        "2.7182818e5",
        "-1.2e12",
        "3.4e38",
        "9.8765E-20",
        "5e-7",
        "1e20",
        "-1e-2",
        "123e12",
        "-17e3",
    ],
    "csv-forms": [
        "+1.25",
        ".5",
        "-.125",
        "1.",
        "1.e-2",
        "00001.5",
        "+0",
        "-0000",
        "+.2",
        "001E3",
    ],
    "subnormal": [
        "1e-45",
        "1e-46",
        "1.1754942e-38",
        "1.401298464324817e-45",
        "7.006492321624085e-46",
        "-1e-40",
    ],
}
ROWS = 100_000


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in [*VALUES, "mixed"]:
        values = VALUES["plain" if name == "mixed" else name]
        for variant in ["base", "shuffled_nulls"]:
            order = list(range(ROWS))
            if variant != "base":
                random.Random(32459).shuffle(order)
            lines = ["id,value,active,label" if name == "mixed" else "value"]
            for row, source in enumerate(order):
                value = values[source % len(values)]
                if variant != "base":
                    if row % 11 == 0:
                        value = ""
                    elif row % 5 == 0:
                        value = '"' + value + '"'
                if name == "mixed":
                    value = f'{source},{value},{"true" if source % 2 else "false"},"group,{source % 97}"'
                lines.append(value)
            path = root / f"{name}-{variant}.csv"
            path.write_text("\n".join(lines) + "\n")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.output / "fixtures"
    hashes = fixtures(root)
    samples = []
    checksums = {}
    for threads in [1, 8]:
        for repeat in range(4):
            order = ["baseline", "candidate"] if repeat % 2 == 0 else [
                "candidate",
                "baseline",
            ]
            for version in order:
                binary = getattr(args, version).resolve()
                result = subprocess.run(
                    [str(binary), str(root.resolve())],
                    env=dict(os.environ, DATAFRAME_THREADS=str(threads)),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                (
                    args.output / f"{version}-t{threads}-r{repeat}.log"
                ).write_text(result.stdout + result.stderr)
                lines = result.stdout.splitlines()
                if len(lines) != 70:
                    raise RuntimeError(
                        "Expected seven samples for each of ten workload/variant pairs"
                    )
                for line in lines:
                    name, variant, elapsed, checksum = line.split()
                    key = (threads, name, variant)
                    if key in checksums and checksums[key] != checksum:
                        raise RuntimeError(f"Output bits differ for {key}")
                    checksums[key] = checksum
                    samples.append(
                        dict(
                            version=version,
                            threads=threads,
                            workload=name,
                            variant=variant,
                            elapsed_ns=int(elapsed),
                            checksum=int(checksum),
                            repeat=repeat,
                        )
                    )
                print("completed", version, threads, repeat, flush=True)
    summary = []
    for threads, name, variant in checksums:
        times = {}
        for version in ["baseline", "candidate"]:
            values = [
                s["elapsed_ns"]
                for s in samples
                if (s["threads"], s["workload"], s["variant"], s["version"])
                == (threads, name, variant, version)
            ]
            if len(values) != 28:
                raise RuntimeError("Missing workload samples")
            times[version] = statistics.median(values)
        summary.append(
            dict(
                threads=threads,
                workload=name,
                variant=variant,
                **times,
                ratio=times["candidate"] / times["baseline"],
            )
        )
    tools = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.iterdir()
        if p.suffix in {".py", ".mojo"}
    }
    binaries = {
        v: hashlib.sha256(getattr(args, v).read_bytes()).hexdigest()
        for v in ["baseline", "candidate"]
    }
    report = dict(
        platform=platform.platform(),
        cpu=platform.processor(),
        baseline_revision=args.baseline_revision,
        candidate_revision=args.candidate_revision,
        binary_sha256=binaries,
        tool_sha256=tools,
        fixture_sha256=hashes,
        samples=samples,
        summary=summary,
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for row in summary:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
