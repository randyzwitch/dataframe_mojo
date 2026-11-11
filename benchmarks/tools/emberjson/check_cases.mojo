"""Check generated exact Float32 expectations through bounded CSV fields."""
from dataframe.io.csv_numeric import parse_csv_float32
from std.sys import argv
from std.memory import bitcast


def _hex(c: UInt8) -> UInt8:
    return c - 48 if c <= 57 else c - 87


def main() raises:
    var count = 0
    var errors = 0
    with open(argv()[1], "r") as file:
        var lines = file.read().splitlines()
        for line in lines:
            var fields = line.split("\t")
            var expected = Int64(Int(fields[0]))
            var hex = fields[1].as_bytes()
            var bytes = List[UInt8]()
            for i in range(0, len(hex), 2):
                bytes.append((_hex(hex[i]) << 4) | _hex(hex[i + 1]))
            var text = String(unsafe_from_utf8=bytes)
            for offset in [0, 1, 7, 15]:
                var prefix = String()
                for _ in range(offset):
                    prefix += "9"
                var owner = prefix + text + "987e42"
                var field = StringSlice(
                    unsafe_from_utf8=owner.as_bytes()[
                        offset : offset + text.byte_length()
                    ]
                )
                var right = Int64(-1)
                try:
                    right = Int64(
                        bitcast[DType.uint32](parse_csv_float32(field))
                    )
                except:
                    pass
                if right != expected:
                    errors += 1
                    if errors <= 10:
                        print("mismatch", expected, right, fields[1])
                count += 1
                _ = owner^
    print("checks", count, "errors", errors)
    if errors:
        raise Error("Float32 corpus mismatch")
