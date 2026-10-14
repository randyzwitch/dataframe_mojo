"""Generate independent arbitrary-precision oracles for Mojo GPU integers."""
import argparse
from pathlib import Path
import random
import struct


def vectors():
    rng = random.Random(7285)
    for bits in (8, 16, 32, 64):
        for unsigned in (False, True):
            low = 0 if unsigned else -(1 << (bits - 1))
            high = (1 << (bits - int(not unsigned))) - 1
            edges = sorted({low, low + 1, 0, 1, 2, high - 1, high})
            if not unsigned:
                edges += [-2, -1]
            pairs = [(a, b) for a in edges for b in edges]
            if bits == 8:
                pairs += [(a, b) for a in range(low, high + 1)
                          for b in range(low, high + 1)]
            pairs += [(rng.randint(low, high), rng.randint(low, high))
                      for _ in range(2048)]
            for a, b in pairs:
                for op in (25, 26):
                    yield op, a, b, bits, unsigned
            for a in edges + [rng.randint(low, high) for _ in range(256)]:
                for exponent in ([0, 1, 2, 3, 7, 8, 15, 31, 63, high]
                                 + ([] if unsigned else [-1, low])):
                    yield 27, a, exponent, bits, unsigned


def expected(op, a, b, bits, unsigned):
    low = 0 if unsigned else -(1 << (bits - 1))
    high = (1 << (bits - int(not unsigned))) - 1
    if op == 27:
        if b < 0:
            return 0, 1, 1
        if b > 64 and abs(a) > 1:
            return 0, 1, 1
        value = pow(a, b)
    elif not b:
        return 0, 0, 0
    else:
        value = a // b if op == 25 else a % b
    if not low <= value <= high:
        return 0, 1, 1
    return value & ((1 << 64) - 1), 1, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = 0
    mask = (1 << 64) - 1
    with args.output.open("wb") as file:
        for case in vectors():
            op, a, b, bits, unsigned = case
            file.write(struct.pack("<8Q", op, a & mask, b & mask, bits,
                                   unsigned, *expected(*case)))
            count += 1
    print(f"Generated {count} independent integer cases")


if __name__ == "__main__":
    main()
