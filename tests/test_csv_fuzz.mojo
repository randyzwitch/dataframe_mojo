"""Seeded grammar fuzzing for the public Polars-derived CSV pipeline.

The former fuzz target compared a scalar tokenizer against its SIMD shortcut.
This replacement generates valid quoted CSV records at every alignment and
requires identical typed data at one and 32 workers. A compact malformed
corpus separately verifies that scheduling cannot change success into failure.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import CsvField, CsvSchema, DataFrame, read_csv

comptime PATH = "/tmp/dataframe_mojo_csv_pipeline_fuzz.csv"
comptime CASES = 60


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def schema() raises -> CsvSchema:
    return CsvSchema([CsvField.int64("id"), CsvField.string("text")])


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def read_at_workers(workers: Int) raises -> DataFrame:
    set_threads(workers)
    return read_csv(PATH, schema())


def test_seeded_valid_csv_matches_at_one_and_32_workers() raises:
    var rng = Lcg(424242)
    for case_index in range(CASES):
        var alignment = rng.next(193)
        var prefix = String()
        for _ in range(alignment):
            prefix += "x"
        var text = String("id,text\n")
        var rows = 24 + rng.next(48)
        for row in range(rows):
            text += String(row) + ","
            # The first six rows guarantee plain, separator, embedded LF,
            # escaped quote, Unicode, and quoted-empty coverage per seed.
            var kind = row if row < 6 else rng.next(6)
            if kind == 0:
                text += prefix + "plain"
            elif kind == 1:
                text += '"' + prefix + ',comma"'
            elif kind == 2:
                text += '"' + prefix + '\nnewline"'
            elif kind == 3:
                text += '"' + prefix + '""quote"""'
            elif kind == 4:
                text += '"' + prefix + '🔥"'
            else:
                text += '""'
            # Every case uses CRLF and LF; alternating cases end at EOF with
            # no final terminator, as allowed by the Polars parser.
            if row + 1 < rows or case_index % 2 == 0:
                text += "\r\n" if row % 3 == 0 else "\n"
        write(text)
        var one = read_at_workers(1)
        var many = read_at_workers(32)
        assert_true(one.equals(many), "seeded case " + String(case_index))


def succeeds_at_workers(workers: Int) -> Bool:
    try:
        _ = read_at_workers(workers)
        return True
    except:
        return False


def test_malformed_records_have_worker_invariant_outcome() raises:
    for text in [
        "id,text\n1,ok\n2,extra,field\n",
        'id,text\n1,"unterminated\n',
        "id,text\nbad,plain\n",
        'id,text\n1,"x"tail\n',
        "id,text\n1,ok\r2,next\n",
    ]:
        write(text)
        assert_equal(succeeds_at_workers(1), succeeds_at_workers(32))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
