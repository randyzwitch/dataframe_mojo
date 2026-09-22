#!/usr/bin/env python3
"""Create deterministic mixed CSV outside all timed benchmark regions."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument(
        "--profile", choices=("mixed", "short-ascii", "long-ascii"), default="mixed"
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    args.path.parent.mkdir(parents=True, exist_ok=True)
    with args.path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "value", "active", "label"))
        for i in range(args.rows):
            value = "" if i % 11 == 0 else f"{i / 7:.12g}"
            if args.profile == "mixed":
                label = f'group,{i % 97}: "café"'
                if i % 101 == 0:
                    label += "\nsecond line"
            else:
                label = f"group{i % 97}"
                if args.profile == "long-ascii":
                    label += "_abcdefghijklmnopqrstuvwxyz0123456789"
            writer.writerow((i, value, "true" if i % 2 == 0 else "false", label))
    print(f"{args.rows} rows, {args.path.stat().st_size} bytes: {args.path}")


if __name__ == "__main__":
    main()
