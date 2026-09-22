"""Generate CSV integer benchmark fixtures outside timed runs."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=("mixed", "wide"))
parser.add_argument("path", type=Path)
parser.add_argument("--rows", type=int, default=1_000_000)
args = parser.parse_args()
if args.rows < 1:
    parser.error("--rows must be positive")
args.path.parent.mkdir(parents=True, exist_ok=True)
with args.path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output, lineterminator="\n")
    if args.kind == "wide":
        writer.writerow(("signed", "unsigned"))
        for i in range(args.rows):
            signed = 9223372036854775807 - i * 104729
            if i % 2:
                signed = -signed
            writer.writerow((signed, 18446744073709551615 - i * 130363))
    else:
        writer.writerow(("id", "value", "active", "label"))
        for i in range(args.rows):
            value = "" if i % 11 == 0 else f"{i / 7:.12g}"
            label = f'group,{i % 97}: "café"'
            if i % 101 == 0:
                label += "\nsecond line"
            writer.writerow((i, value, "true" if i % 2 == 0 else "false", label))
print(args.path, args.path.stat().st_size)
