"""Independent Python integer oracle for paired-word Mojo GPU decimals.

Python arbitrary-precision integers supply arithmetic and comparisons. Explicit
128-bit wrapping models the public CPU decimal contract, including fixed-width
rescaling. Python floats supply each separately rounded mean conversion.
"""
from pathlib import Path
import argparse
import random
import struct
import subprocess
import tempfile

WORD_MASK = (1 << 64) - 1
MASK = (1 << 128) - 1
MINIMUM = -(1 << 127)
MAXIMUM = (1 << 127) - 1
PRECISION = 10**38


def signed(value):
    value &= MASK
    return value - (1 << 128) if value >> 127 else value


def words(value):
    return value & WORD_MASK, (value >> 64) & WORD_MASK


def nearest(value, unit):
    quotient, remainder = divmod(abs(value), unit)
    if 2 * remainder > unit or (2 * remainder == unit and quotient % 2):
        quotient += 1
    return -quotient if value < 0 else quotient


def scaled(value, scale):
    if scale and abs(value) > MAXIMUM // 10**scale:
        return 0, False
    return signed(value * 10**scale), True


def division(a, b):
    value = abs(a) // abs(b)
    return (-value if (a < 0) != (b < 0) else value), a % b


def expected(op, a, b, ls, rs, option, bits, flags):
    """Return two value words and two remainder/status words."""
    raw_a, raw_b = a & MASK, b & MASK
    a, b = signed(a), signed(b)
    value = extra = 0
    if op == 0:
        value = raw_a + raw_b
    elif op == 1:
        value = raw_a - raw_b
    elif op == 2:
        value = raw_a * raw_b
    elif op == 3:
        value, extra = divmod(raw_a, raw_b)
    elif op == 4:
        value, extra = division(a, b)
    elif op == 5:
        value = nearest(a, 10**ls)
    elif op == 6:
        value, extra = scaled(a, ls)
    elif op in (7, 8, 9, 10, 12):
        common = max(ls, rs)
        left, left_ok = scaled(a, common - ls)
        right, right_ok = scaled(b, common - rs)
        if not left_ok or not right_ok:
            extra = 0 if op == 12 else 1
        elif op in (7, 8):
            value = signed(left + (-right if op == 8 else right))
            extra = 2 if abs(value) >= PRECISION else 0
        elif op == 9:
            if a == MINIMUM:
                extra = 1
            elif a == 0:
                value = 0
            elif abs(b) > ((-MINIMUM if b < 0 else MAXIMUM) // abs(a)):
                extra = 1
            else:
                value = signed(a * b)
                scale = min(ls, rs)
                if scale:
                    if value == MINIMUM and scale <= 18:
                        value = 0
                    elif value == MINIMUM:
                        value = -nearest(value, 10**scale)
                    else:
                        value = nearest(value, 10**scale)
                extra = 2 if abs(value) >= PRECISION else 0
        elif op == 10:
            if not b:
                extra = 4
            else:
                divisor = signed(abs(b))
                value, remainder = division(signed(abs(a)), divisor)
                if abs(value) >= PRECISION:
                    value, extra = 0, 2
                else:
                    for _ in range(common + rs - ls):
                        value = signed(value * 10)
                        if abs(value) >= PRECISION:
                            value, extra = 0, 2
                            break
                        q, remainder = division(signed(remainder * 10), divisor)
                        value = signed(value + q)
                    if not extra:
                        rest = signed(divisor - remainder)
                        if rest < remainder or (
                            rest == remainder and value % 2
                        ):
                            value = signed(value + 1)
                        if abs(value) >= PRECISION:
                            value, extra = 0, 2
                        elif (a < 0) != (b < 0):
                            value = -value
        else:
            extra = 1
            value = {
                9: left == right,
                23: left != right,
                20: left < right,
                22: left <= right,
                8: left > right,
                21: left >= right,
            }[option]
    elif op == 11:
        if a == -(1 << (bits - 1)):
            extra = 3
        else:
            value = abs(a) if flags else -a
    elif op == 13:
        value = nearest(a, 10 ** (ls - rs)) if ls > rs else signed(
            a * 10 ** (rs - ls)
        )
        magnitude = abs(value)
        if option:
            extra = magnitude < 10**option
        elif flags & 2:
            value, extra = int(magnitude != 0), True
        else:
            maximum = (1 << bits) - 1 if flags & 1 else (1 << (bits - 1)) - int(
                value >= 0
            )
            extra = magnitude <= maximum and not ((flags & 1) and value < 0)
    elif op in (14, 15):
        count = raw_b & WORD_MASK
        if count:
            quotient, remainder = divmod(abs(a), count)
            whole, digits = divmod(quotient, 10**ls)
            mean = float(whole) + (
                float(digits) + float(remainder) / float(count)
            ) / float(10**ls)
            if a < 0:
                mean = -mean
            value = struct.unpack("<Q", struct.pack("<d", mean))[0]
    elif op == 16:
        value = 10**ls
    elif op == 17:
        value = raw_a >= raw_b
    elif op == 18:
        value = a < b
    elif op == 19:
        value = -a
    elif op == 20:
        value = raw_a + 1
    elif op == 21:
        value, remainder = divmod(raw_a, raw_b)
        return value & WORD_MASK, 0, remainder, 0
    else:
        raise ValueError(op)
    if op in (3, 4):
        return *words(value), *words(extra)
    return *words(value), int(extra), 0


def generate(path):
    rng = random.Random(0x4D4F4A4F444543)
    count = 0
    with path.open("wb") as out:

        def emit(op, a=0, b=0, ls=0, rs=0, option=0, bits=64, flags=0):
            nonlocal count
            record = (op, *words(a), *words(b), ls, rs, option, bits, flags)
            out.write(
                struct.pack(
                    "<14Q",
                    *record,
                    *expected(op, a, b, ls, rs, option, bits, flags),
                )
            )
            count += 1

        edge = [
            0,
            1,
            -1,
            2,
            -2,
            5,
            -5,
            15,
            -15,
            25,
            -25,
            35,
            -35,
            (1 << 31) - 1,
            -(1 << 31),
            (1 << 63) - 1,
            -(1 << 63),
            1 << 64,
            -(1 << 64),
            PRECISION - 1,
            1 - PRECISION,
            PRECISION,
            -PRECISION,
            MAXIMUM,
            MINIMUM,
        ]
        for a in edge:
            for b in edge:
                for op in (0, 1, 2, 17, 18):
                    emit(op, a, b)
                if b:
                    emit(3, a, abs(b))
                    emit(4, a, b)
                for ls, rs in ((0, 0), (1, 0), (0, 1), (18, 19), (38, 38)):
                    for op in (7, 8, 9, 10):
                        emit(op, a, b, ls, rs)
                    for comparison in (9, 23, 20, 22, 8, 21):
                        emit(12, a, b, ls, rs, comparison)
        for a in edge:
            for op in (19, 20):
                emit(op, a)
            for scale in range(39):
                emit(5, a, ls=scale)
                emit(6, a, ls=scale)
                emit(13, a, ls=scale, rs=38 - scale, option=38)
                for bits in (8, 16, 32, 64):
                    for flags in (0, 1, 2):
                        emit(13, a, ls=scale, bits=bits, flags=flags)
            for bits in (32, 64, 128):
                for flags in (0, 1):
                    emit(11, a, bits=bits, flags=flags)
        for scale in range(39):
            emit(16, ls=scale)
            for count_value in (0, 1, 2, 3, 257, (1 << 31) - 1):
                for total in (
                    0,
                    1,
                    -1,
                    2**53 + 1,
                    -(2**53 + 1),
                    PRECISION - 1,
                    1 - PRECISION,
                ):
                    emit(15, total, count_value, scale)
                    if scale <= 18 and abs(total) < count_value * (1 << 63):
                        emit(14, total, count_value, scale)
        for _ in range(4000):
            a, b = rng.randrange(1 << 128), rng.randrange(1, 1 << 127)
            for op in (0, 1, 2, 3, 4, 17, 18):
                emit(op, a, b)
            divisor = rng.randrange(1, 10**18 + 1)
            numerator = (rng.randrange(divisor) << 64) | rng.getrandbits(64)
            emit(21, numerator, divisor)
            ls, rs = rng.randrange(39), rng.randrange(39)
            a = rng.randrange(-PRECISION + 1, PRECISION)
            b = rng.randrange(-PRECISION + 1, PRECISION)
            for op in (7, 8, 9, 10):
                emit(op, a, b, ls, rs)
        # Halfway values cover ties and adjacent rounding at each valid scale.
        for scale in range(1, 39):
            unit = 10**scale
            for quotient in (0, 1, 2, 3, 7):
                for delta in (-1, 0, 1):
                    a = quotient * unit + unit // 2 + delta
                    if a <= MAXIMUM:
                        emit(5, a, ls=scale)
                        emit(5, -a, ls=scale)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate", type=Path, help="Write corpus without launching Mojo"
    )
    args = parser.parse_args()
    if args.generate:
        print("Generated decimal cases:", generate(args.generate))
        return
    with tempfile.TemporaryDirectory(prefix="mojo-gpu-decimal-") as directory:
        corpus = Path(directory) / "expected.bin"
        print("Generated decimal cases:", generate(corpus), flush=True)
        subprocess.run(
            [
                "mojo",
                "run",
                "-I",
                ".",
                "tests/tooling/mojo_gpu_decimal_probe.mojo",
                str(corpus),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
