"""Join matrix: every mode against a nested-loop reference, plus schemas."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import AnyValue, DataType, Column, DataFrame, Series


def nan() -> Float64:
    return Float64(0) / Float64(0)


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def random_side(rows: Int, seed: UInt64, value: String) raises -> DataFrame:
    var rng = Lcg(seed)
    var k1 = List[Int64]()
    var v1 = List[Bool]()
    var k2 = List[String]()
    var v2 = List[Bool]()
    var f = List[Float64]()
    var vf = List[Bool]()
    var payload = List[Int64]()
    var words: List[String] = ["", "a", "b"]
    for i in range(rows):
        k1.append(Int64(rng.next(4)))
        v1.append(rng.next(6) != 0)
        k2.append(words[rng.next(3)])
        v2.append(rng.next(6) != 0)
        var pick = rng.next(4)
        f.append(
            nan() if pick
            == 0 else (-0.0 if pick == 1 else (0.0 if pick == 2 else 2.5))
        )
        vf.append(rng.next(6) != 0)
        payload.append(Int64(i))
    return DataFrame(
        [
            Series("k1", Column[Int64](k1^, v1^)),
            Series("k2", Column[String](k2^, v2^)),
            Series("f", Column[Float64](f^, vf^)),
            Series(value, Column[Int64](payload^)),
        ]
    )


def matches(
    left: DataFrame,
    right: DataFrame,
    i: Int,
    j: Int,
    left_on: List[String],
    right_on: List[String],
) raises -> Bool:
    for k in range(len(left_on)):
        var a = left.item(i, left_on[k])
        var b = right.item(j, right_on[k])
        if a.is_null() or b.is_null() or not (a == b):
            return False
    return True


def reference_rows(
    left: DataFrame,
    right: DataFrame,
    left_on: List[String],
    right_on: List[String],
    how: String,
) raises -> Tuple[List[Int], List[Int]]:
    var lr = List[Int]()
    var rr = List[Int]()
    if how == "right":
        for j in range(right.height()):
            var any = False
            for i in range(left.height()):
                if matches(left, right, i, j, left_on, right_on):
                    lr.append(i)
                    rr.append(j)
                    any = True
            if not any:
                lr.append(-1)
                rr.append(j)
        return (lr^, rr^)
    var right_used = List[Bool](length=right.height(), fill=False)
    for i in range(left.height()):
        var any = False
        for j in range(right.height()):
            if matches(left, right, i, j, left_on, right_on):
                any = True
                right_used[j] = True
                if how != "semi" and how != "anti":
                    lr.append(i)
                    rr.append(j)
        if how == "semi" and any:
            lr.append(i)
        elif how == "anti" and not any:
            lr.append(i)
        elif not any and (how == "left" or how == "full"):
            lr.append(i)
            rr.append(-1)
    if how == "full":
        for j in range(right.height()):
            if not right_used[j]:
                lr.append(-1)
                rr.append(j)
    return (lr^, rr^)


def cell(
    frame: DataFrame, row: Int, name: String, dtype: DataType
) raises -> AnyValue:
    if row < 0:
        return AnyValue.null(dtype)
    return frame.item(row, name)


def check(
    left: DataFrame,
    right: DataFrame,
    left_on: List[String],
    right_on: List[String],
    how: String,
    coalesce: Bool,
) raises:
    var expected = reference_rows(left, right, left_on, right_on, how)
    var actual = left.join(
        right, left_on=left_on, right_on=right_on, how=how, coalesce=coalesce
    )
    ref lr = expected[0]
    ref rr = expected[1]
    assert_equal(actual.height(), len(lr), msg=how)
    for r in range(actual.height()):
        for c in range(left.width()):
            var name = left.columns()[c]
            var dtype = left.dtypes()[c]
            var want = cell(left, lr[r], name, dtype)
            var key = -1
            for k in range(len(left_on)):
                if left_on[k] == name:
                    key = k
            var mixed = how == "right" or (how == "full" and coalesce)
            if key >= 0 and mixed and lr[r] < 0:
                want = cell(right, rr[r], right_on[key], dtype)
            assert_true(actual.item(r, name) == want, msg=how + " " + name)
        if how == "semi" or how == "anti":
            assert_equal(actual.width(), left.width())
            continue
        var position = left.width()
        for c in range(right.width()):
            var name = right.columns()[c]
            var is_key = False
            for k in right_on:
                is_key = is_key or k == name
            if is_key and not (how == "full" and not coalesce):
                continue
            var want = cell(right, rr[r], name, right.dtypes()[c])
            assert_true(
                actual._columns[position].get(r) == want,
                msg=how + " right " + name,
            )
            position += 1


def test_every_mode_matches_nested_loop_reference() raises:
    var hows: List[String] = ["inner", "left", "right", "full", "semi", "anti"]
    var keys: List[List[String]] = [
        ["k1"],
        ["k1", "k2"],
        ["f"],
        ["k2", "f", "k1"],
    ]
    for seed in range(1, 3):
        var left = random_side(14, UInt64(seed), "lv")
        var right = random_side(11, UInt64(seed * 31), "rv")
        for key in keys:
            for how in hows:
                check(left, right, key, key, how, True)
                check(left, right, key, key, how, False)
                check(left, right.clear(), key, key, how, True)
                check(left.clear(), right, key, key, how, True)
                check(left, left, key, key, how, True)


def test_left_on_right_on_and_schemas() raises:
    var orders = DataFrame(
        [
            Series("customer", Column[Int64]([1, 2, 1, 4])),
            Series("amount", Column[Float64]([10, 20, 30, 40])),
        ]
    )
    var customers = DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3])),
            Series("name", Column[String](["ann", "bo", "cy"])),
            Series("amount", Column[Float64]([0, 0, 0])),
        ]
    )
    var inner = orders.join(customers, left_on=["customer"], right_on=["id"])
    assert_equal(
        inner.columns(), [String("customer"), "amount", "name", "amount_right"]
    )
    assert_equal(inner.height(), 3)
    var right = orders.join(
        customers, left_on=["customer"], right_on=["id"], how="right"
    )
    assert_equal(right.height(), 4)
    # Right joins follow right rows; the key comes from whichever side exists.
    assert_equal(right.item(3, "customer").int64(), Int64(3))
    assert_true(right.item(3, "amount").is_null())
    assert_equal(right.item(3, "name").string(), "cy")
    var full = orders.join(
        customers, left_on=["customer"], right_on=["id"], how="full"
    )
    assert_equal(full.height(), 5)
    assert_equal(full.item(4, "customer").int64(), Int64(3))
    var separate = orders.join(
        customers,
        left_on=["customer"],
        right_on=["id"],
        how="full",
        coalesce=False,
    )
    assert_equal(
        separate.columns(),
        [String("customer"), "amount", "id", "name", "amount_right"],
    )
    assert_true(separate.item(4, "customer").is_null())
    assert_equal(separate.item(4, "id").int64(), Int64(3))
    assert_true(separate.item(3, "id").is_null())
    var semi = orders.join(
        customers, left_on=["customer"], right_on=["id"], how="semi"
    )
    assert_equal(semi.columns(), orders.columns())
    assert_equal(semi.height(), 3)
    var anti = orders.join(
        customers, left_on=["customer"], right_on=["id"], how="anti"
    )
    assert_equal(anti.item(0, "customer").int64(), Int64(4))


def test_cross_join() raises:
    var a = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var b = DataFrame(
        [
            Series("x", Column[String](["p", "q", "r"])),
            Series("y", Column[Bool]([True, False, True])),
        ]
    )
    var cross = a.join(b, how="cross")
    assert_equal(cross.columns(), [String("x"), "x_right", "y"])
    assert_equal(cross.height(), 6)
    assert_equal(cross.item(2, "x").int64(), Int64(1))
    assert_equal(cross.item(3, "x").int64(), Int64(2))
    assert_equal(cross.item(3, "x_right").string(), "p")
    assert_equal(a.join(b.clear(), how="cross").height(), 0)
    assert_equal(a.join(b, how="cross", suffix="_b").columns()[1], "x_b")
    with assert_raises(contains="requires key columns"):
        _ = a.join(b, how="inner")
    with assert_raises(contains="takes no keys"):
        _ = a.join(a, "x", "cross")


def test_validation() raises:
    var left = random_side(3, 1, "lv")
    var right = random_side(3, 2, "rv")
    with assert_raises(
        contains="Join key dtypes differ: k1 is int64 but k2 is string"
    ):
        _ = left.join(right, left_on=["k1"], right_on=["k2"])
    with assert_raises(contains="same nonzero number"):
        _ = left.join(right, left_on=["k1", "k2"], right_on=["k1"])
    with assert_raises(contains="same nonzero number"):
        _ = left.join(right, List[String]())
    with assert_raises(contains="Duplicate join key"):
        _ = left.join(right, ["k1", "k1"])
    with assert_raises(contains="Unknown column"):
        _ = left.join(right, ["k1", "zzz"])
    with assert_raises(contains="Join how must be"):
        _ = left.join(right, "k1", "outer")
    # Right "lv" becomes "lv_right", which the right side already uses.
    var clash = right.rename({"rv": "lv"}).with_row_index("lv_right")
    with assert_raises(contains="Join output name collision: lv_right"):
        _ = left.join(clash, "k1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
