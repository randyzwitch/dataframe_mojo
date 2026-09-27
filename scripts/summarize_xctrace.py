"""Summarize an xctrace time-profile XML export (self and inclusive weights).

Export with xctrace's /trace-toc/run/data/table[@schema="time-profile"] XPath.
Weights are sampled CPU time, not elapsed time or hardware cycles. Inclusive
weights overlap. Repeated/inlined frame names are counted once per stack.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(path):
    root = ET.parse(path).getroot()
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}

    def resolve(element):
        while element is not None and element.get("ref"):
            element = ids[element.get("ref")]
        return element

    own, inclusive = Counter(), Counter()
    total = samples = 0
    for row in root.iter("row"):
        weight = resolve(row.find("weight"))
        stack = resolve(row.find("tagged-backtrace"))
        if weight is None or stack is None:
            continue
        trace = resolve(stack.find("backtrace"))
        if trace is None:
            continue
        names = [resolve(frame).get("name", "unknown") for frame in trace]
        if not names:
            continue
        value = int(weight.text)
        total += value
        samples += 1
        own[names[0]] += value
        for name in set(names):
            inclusive[name] += value
    if not total:
        raise ValueError("no weighted time-profile stacks found")

    def table(counter):
        return [
            {
                "symbol": name,
                "weight_ns": value,
                "percent": round(100 * value / total, 3),
            }
            for name, value in counter.most_common(30)
        ]

    return {
        "samples": samples,
        "total_weight_ns": total,
        "self": table(own),
        "inclusive": table(inclusive),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.input), indent=2) + "\n")


if __name__ == "__main__":
    main()
