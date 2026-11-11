"""Generate independent exact Float32 expectations; no floating-point oracle."""
import argparse
import re, random
from fractions import Fraction
from pathlib import Path

rng = random.Random(32459)
num = re.compile(
    rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)


def value(b):
    if b == 0:
        return Fraction(0)
    e = b >> 23
    m = b & 0x7FFFFF
    return Fraction(m if e == 0 else m + (1 << 23)) * Fraction(2) ** (
        e - 150 if e else -149
    )


def expected(raw):
    sign = 0x80000000 if raw.startswith(b"-") else 0
    plain = raw.lstrip(b"+-").lower()
    if len(raw) - len(raw.lstrip(b"+-")) <= 1:
        if plain == b"nan":
            return sign | 0x7FC00000
        if plain in (b"inf", b"infinity"):
            return sign | 0x7F800000
    if not num.fullmatch(raw):
        return -1
    text = raw.decode().lstrip("+-").lower()
    c, _, e = text.partition("e")
    exp = int(e or 0)
    whole, _, frac = c.partition(".")
    coefficient = int(whole + frac)
    exp -= len(frac)
    if coefficient == 0:
        return sign
    digits = len(str(coefficient))
    if digits + exp > 42:
        return sign | 0x7F800000
    if digits + exp < -48:
        return sign
    x = Fraction(coefficient) * Fraction(10) ** exp
    # Search monotone positive Float32 bit patterns with exact rational distances.
    if x >= value(0x7F7FFFFF) + Fraction(2) ** 103:
        return sign | 0x7F800000
    lo, hi = 0, 0x7F7FFFFF
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if value(mid) <= x:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0x7F7FFFFF:
        return sign | lo
    delta = x - value(lo)
    other = value(lo + 1) - x
    return sign | (lo + int(delta > other or delta == other and lo & 1))


def decimal(f):
    n, d = f.numerator, f.denominator
    assert d & (d - 1) == 0
    k = d.bit_length() - 1
    s = str(n * 5**k)
    return (s.zfill(k + 1)[:-k] + "." + s.zfill(k + 1)[-k:]) if k else s


assert expected(b"0.1") == 0x3DCCCCCD
assert expected(b"1.000000059604644775390625000001") == 0x3F800001
assert expected(b"340282356779733661637539395458142568448") == 0x7F800000
assert expected(b"-0e99999") == 0x80000000
assert expected(b"1e400x") == -1
cases = set()


def add(s):
    cases.add(s.encode() if isinstance(s, str) else s)


for sign in ("", "+", "-"):
    for s in (
        "0",
        "0.0",
        "0000",
        ".0",
        "0.",
        "1",
        "01.5",
        ".5",
        "1.e2",
        "1e99999",
        "1e-99999",
        "0e99999",
        "0e-99999",
        "NaN",
        "NAN",
        "nan",
        "inf",
        "INF",
        "Infinity",
        "iNfInItY",
    ):
        add(sign + s)
# Exact midpoint + tiny decimal residual on either side; all Float32 exponents.
for exp in range(255):
    for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF):
        bits = (exp << 23) | mantissa
        mid = (value(bits) + value(bits + 1)) / 2
        s = decimal(mid)
        if "." not in s:
            s += "."
        digits = len(s.split(".")[1])
        # Fraction decimal constructor for denominator containing 5 requires exact formatting.
        scale = 10 ** (digits + 10)
        scaled = int(mid * scale)
        for offset in (-1, 0, 1):
            v = str(scaled + offset).zfill(digits + 11)
            text = v[: -(digits + 10)] + "." + v[-(digits + 10) :]
            add(text)
            add("-" + text)
for _ in range(10000):
    n = rng.choice([1, 2, 6, 9, 17, 19, 20, 30, 60, 120, 800])
    digits = "".join(str(rng.randrange(10)) for _ in range(n))
    position = rng.randrange(n + 1)
    s = digits[:position] + "." + digits[position:]
    if rng.randrange(2):
        s += "e" + str(rng.randrange(-n - 60, 61))
    add(rng.choice(["", "+", "-"]) + s)
for seed in (b"1.25", b"-1.2e-3", b"+001.50E+02"):
    for pos in range(len(seed)):
        for byte in range(256):
            add(seed[:pos] + bytes([byte]) + seed[pos + 1 :])
for s in (
    b"",
    b"+",
    b"-",
    b".",
    b"+-1",
    b"--1",
    b"1e",
    b"1e+",
    b"1e-",
    b"1e2.",
    b"1.2.",
    b"1e400.",
    b"1_0",
    b"0x1p2",
    b"nan(1)",
    b"InfinitYx",
    b"1,2",
    b"1\x00",
    b"1\n",
    b"\t1",
    b"1 ",
    b" 1",
    b"1\r\n",
):
    add(s)
for s in (
    "1e" + "9" * 1000,
    "0e" + "9" * 1000,
    "-0e-" + "9" * 1000,
    "1e-" + "9" * 1000,
    "0." + "0" * 1500 + "1e1500",
    "1" + "0" * 1500 + "e-1500",
):
    add(s)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for raw in sorted(cases):
        f.write(str(expected(raw)) + "\t" + raw.hex() + "\n")
print(len(cases), "cases; exact integer/rational oracle")
