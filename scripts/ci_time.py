"""Time a command without hiding its output or exit status; summarize CI costs."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--label", default="command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = Path(os.environ.get("CI_TIMINGS_DIR", "build/ci-timings"))
    if args.summary:
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
        lines = [
            "### CI command timings", "",
            "Nested commands overlap: do not sum these rows. RSS is the largest child process, not aggregate job memory.",
            "", "| Command | Seconds | Child CPU seconds | Peak child RSS MiB | Exit |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(rows, key=lambda r: r["started"]):
            lines.append(
                f'| {row["label"]} | {row["seconds"]:.1f} | {row["cpu_seconds"]:.1f}'
                f' | {row["peak_rss_mib"]:.1f} | {row["exit_code"]} |'
            )
        report = "\n".join(lines) + "\n"
        print(report)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
                output.write(report)
        return 0
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    started = time.time()
    start = time.monotonic()
    try:
        result = subprocess.run(command)
        code = result.returncode if result.returncode >= 0 else 128 - result.returncode
    except OSError as error:
        print(error, file=sys.stderr)
        code = 127
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = dict(
        label=args.label, started=started, seconds=time.monotonic() - start,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        peak_rss_mib=usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        exit_code=code,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{uuid.uuid4()}.json").write_text(json.dumps(row) + "\n")
    print(f'CI timing: {args.label}: {row["seconds"]:.1f}s, exit {code}', flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
