"""Crash-only and split-invariance fuzzing of the CSV tokenizer.

Random byte strings drawn from CSV-significant bytes (separators, quotes,
CR, LF, digits, multi-byte and invalid UTF-8) must either parse or raise a
clean Error, and the outcome, including the error text, must not depend on
the input buffer size.
"""
from std.testing import TestSuite, assert_equal
from dataframe import CsvField, CsvSchema, DataFrame, read_csv

comptime PATH = "/tmp/dataframe_mojo_csv_fuzz.csv"
comptime CASES = 60


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def outcome(
    size: Int, schema: CsvSchema, lossy: Bool, permissive: Bool
) -> String:
    try:
        var frame = read_csv(
            PATH,
            schema,
            buffer_size=size,
            n_rows=1 << 30,  # Force streaming so buffer_size=1 is scalar.
            encoding="utf8-lossy" if lossy else "utf8",
            ignore_errors=permissive,
        )
        var out = String("ok ", frame.height(), ":")
        for r in range(frame.height()):
            for c in range(frame.width()):
                out += String(frame._columns[c].get(r)) + "|"
        return out^
    except e:
        return "error: " + String(e)


def test_random_bytes_parse_or_fail_cleanly_at_every_split() raises:
    var alphabet: List[UInt8] = [
        97,
        49,
        50,
        45,
        46,
        44,
        44,
        34,
        34,
        10,
        10,
        13,
        32,
        195,
        169,
        255,
        116,
        114,
        117,
        101,
    ]
    var schema = CsvSchema([CsvField.int64("a"), CsvField.string("b")])
    var text_schema = CsvSchema([CsvField.string("a"), CsvField.string("b")])
    var rng = Lcg(424242)
    for index in range(CASES):
        var bytes: List[UInt8] = [97, 44, 98, 10]
        for _ in range(rng.next(320)):
            bytes.append(alphabet[rng.next(len(alphabet))])
        with open(PATH, "w") as file:
            file.write_bytes(bytes)
        for mode in range(4):
            var chosen = schema.copy() if mode % 2 == 0 else text_schema.copy()
            var reference = outcome(
                len(bytes) + 1, chosen, mode >= 2, mode == 3
            )
            for size in [1, 2, 3, 7, 63, 64, 65]:
                assert_equal(
                    outcome(size, chosen, mode >= 2, mode == 3),
                    reference,
                    msg="case " + String(index) + " mode " + String(mode),
                )


def test_valid_quoted_records_at_every_simd_alignment() raises:
    var schema = CsvSchema([CsvField.string("a"), CsvField.string("b")])
    for padding in range(128):
        var prefix = String()
        for _ in range(padding):
            prefix += "x"
        var text = String("a,b\r\n")
        for _ in range(5):
            text += prefix + ',"a,b\r\nc""d"\r\n'
            text += '"",plain\n'
        with open(PATH, "w") as file:
            file.write(text)
        var reference = outcome(1, schema, False, False)
        for size in [63, 64, 65, 127, 128, 129, 4096]:
            assert_equal(outcome(size, schema, False, False), reference)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
