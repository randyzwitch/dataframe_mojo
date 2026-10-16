"""Hash joins on several keys, whose candidates are matched a key column at
a time over batches of probe rows, against a nested-loop model: probes
longer than a batch, strings stored as offsets or as Arrow views (inline
lengths 0 to 12, and long strings that share their first bytes), floats
with NaN and signed zero, Bool and Int128 keys, and a null in each key."""
# contract-group: kernel.order_keys
from std.testing import assert_equal

from dataframe import BoolColumn, Column, Series, StringColumn
from dataframe.aggregate import float_key
from dataframe.join_hash import (
    direct_hash_join_rows,
    direct_hash_semi_anti_rows,
)
from dataframe.string_view import StringViewBuilder

# Longer than a batch of probe rows (1024), and not a multiple of one.
comptime PROBE = 2_500
comptime BUILD = 300


def word_of(k: Int) -> String:
    """Inline view lengths 0 to 12 and long strings that agree on their
    first twelve and four bytes."""
    var kind = k % 6
    if kind == 0:
        # Each a prefix of the longer ones.
        var digits = String()
        for d in range(k % 13):
            digits += String(d % 10)
        return digits^
    if kind == 1:
        return "shared-long-prefix-" + String(k % 7)
    if kind == 2:
        return "abcd" + String(k % 5) + "-long-tail-string"
    if kind == 3:
        return "é" + String(k % 4)
    if kind == 4:
        return "a\x00b" + String(k % 3)
    return "abcdefghijk" + String(k % 2)


def strings(
    count: Int, seed: Int, native: Bool, nulls: Bool
) raises -> Tuple[Series, List[String], List[Bool]]:
    var values = List[String]()
    var valid = List[Bool]()
    for row in range(count):
        values.append(word_of((row * seed) % 41))
        valid.append(not nulls or (row * seed) % 29 != 3)
    if native:
        var builder = StringViewBuilder(count)
        for row in range(count):
            if valid[row]:
                builder.append(StringSlice(values[row]))
            else:
                builder.append_null()
        return (Series("s", StringColumn(builder^.finish())), values^, valid^)
    return (
        Series("s", StringColumn(values.copy(), valid.copy())),
        values^,
        valid^,
    )


struct Side(Movable):
    var keys: List[Series]
    var text: List[String]
    var text_valid: List[Bool]
    var number: List[Int64]
    var number_valid: List[Bool]
    var real: List[Float64]
    var flag: List[Bool]
    var wide: List[Int128]

    def __init__(
        out self, count: Int, seed: Int, native: Bool, nulls: Bool
    ) raises:
        var text = strings(count, seed, native, nulls)
        self.text = text[1].copy()
        self.text_valid = text[2].copy()
        self.number = List[Int64]()
        self.number_valid = List[Bool]()
        self.real = List[Float64]()
        self.flag = List[Bool]()
        self.wide = List[Int128]()
        var nan = Float64(0) / Float64(0)
        var specials: List[Float64] = [0.0, -0.0, nan, 1.5, -2.25]
        for row in range(count):
            self.number.append(Int64((row * seed) % 3))
            self.number_valid.append(not nulls or (row * seed) % 31 != 5)
            self.real.append(specials[(row * seed) % 5])
            self.flag.append((row * seed) % 7 < 4)
            self.wide.append(Int128((row * seed) % 2) << 100)
        self.keys = [
            text[0].copy(),
            Series(
                "n", Column[Int64](self.number.copy(), self.number_valid.copy())
            ),
            Series("f", Column[Float64](self.real.copy())),
            Series("b", BoolColumn(self.flag.copy())),
            Series("w", Column[Int128](self.wide.copy())),
        ]

    def equal(self, other: Side, i: Int, j: Int, keys: Int) -> Bool:
        """The model: every one of the first `keys` keys is valid on both
        sides and equal, floats compared as `float_key` compares them."""
        if not (self.text_valid[i] and other.text_valid[j]):
            return False
        if self.text[i] != other.text[j]:
            return False
        if keys > 1 and not (
            self.number_valid[i]
            and other.number_valid[j]
            and self.number[i] == other.number[j]
        ):
            return False
        if keys > 2 and float_key(self.real[i]) != float_key(other.real[j]):
            return False
        if keys > 3 and self.flag[i] != other.flag[j]:
            return False
        if keys > 4 and self.wide[i] != other.wide[j]:
            return False
        return True


def check(left: Side, right: Side, keys: Int) raises:
    var probe = List[Series]()
    var build = List[Series]()
    for k in range(keys):
        probe.append(left.keys[k].copy())
        build.append(right.keys[k].copy())
    var matches = List[List[Int]]()
    for i in range(PROBE):
        var found = List[Int]()
        for j in range(BUILD):
            if left.equal(right, i, j, keys):
                found.append(j)
        matches.append(found^)
    for outer in [False, True]:
        var expected_left = List[Int]()
        var expected_right = List[Int]()
        for i in range(PROBE):
            for j in matches[i]:
                expected_left.append(i)
                expected_right.append(j)
            if outer and len(matches[i]) == 0:
                expected_left.append(i)
                expected_right.append(-1)
        var actual = direct_hash_join_rows(probe, build, outer)
        assert_equal(actual[0], expected_left)
        assert_equal(actual[1], expected_right)
    var semi = List[Int]()
    var anti = List[Int]()
    for i in range(PROBE):
        if len(matches[i]) > 0:
            semi.append(i)
        else:
            anti.append(i)
    assert_equal(direct_hash_semi_anti_rows(probe, build, True), semi)
    assert_equal(direct_hash_semi_anti_rows(probe, build, False), anti)


def test_string_and_integer_keys_match_over_batches_in_every_storage() raises:
    for left_native in [False, True]:
        for right_native in [False, True]:
            for nulls in [False, True]:
                var left = Side(PROBE, 7, left_native, nulls)
                var right = Side(BUILD, 11, right_native, nulls)
                check(left, right, 2)


def test_float_bool_and_int128_keys_narrow_matches_column_by_column() raises:
    for native in [False, True]:
        var left = Side(PROBE, 7, native, True)
        var right = Side(BUILD, 11, native, True)
        for keys in [3, 4, 5]:
            check(left, right, keys)
