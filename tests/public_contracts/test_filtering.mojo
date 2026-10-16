"""Each predicate selects scalar true rows; unknown and false rows are dropped."""
# contract-group: table.filter
from std.testing import assert_equal, assert_raises
from dataframe import Column, Expr, col
from tests.support.reference_model import generated_model
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model


def scalar_and(left: Int, right: Int) -> Int:
    if left == 0 or right == 0:
        return 0
    return -1 if left < 0 or right < 0 else 1


def scalar_or(left: Int, right: Int) -> Int:
    if left == 1 or right == 1:
        return 1
    return -1 if left < 0 or right < 0 else 0


def test_predicates_across_types_layouts_and_null_logic() raises:
    var predicates: List[Expr] = [
        col("i") > 0,
        col("f") > 0,
        col("s") == "é",
        col("b"),
        (col("i") > 0) & col("b"),
        ((col("i") > 0) & col("b")) | (col("s") == "é"),
    ]
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for predicate in range(len(predicates)):
                    var kept = List[Int]()
                    for index in range(n):
                        ref row = model.rows[index]
                        var i = Int(row.i > 0) if row.iv else -1
                        var f = Int(row.f > 0) if row.fv else -1
                        var s = Int(row.s == "é") if row.sv else -1
                        var b = Int(row.b) if row.bv else -1
                        var expected: List[Int] = [
                            i,
                            f,
                            s,
                            b,
                            scalar_and(i, b),
                            scalar_or(scalar_and(i, b), s),
                        ]
                        if expected[predicate] == 1:
                            kept.append(index)
                    assert_model(
                        frame.filter(predicates[predicate]), model.take(kept)
                    )
                for all_true in [False, True]:
                    var mask = Column[Bool](List[Bool](length=n, fill=all_true))
                    assert_model(
                        frame.filter(mask),
                        model.copy() if all_true else model.slice(0, 0),
                    )
                with assert_raises():
                    _ = frame.filter(col("i"))
                with assert_raises():
                    _ = frame.filter(
                        Column[Bool](List[Bool](length=n + 1, fill=True))
                    )


def test_false_or_null_at_any_bitmap_position_disables_all_rows_shortcut() raises:
    for n in [1, 7, 8, 9, 63, 64, 65]:
        var model = generated_model(n, 3)
        var frame = frame_from_model(model, 1)
        for excluded in range(n):
            var kept = List[Int]()
            for row in range(n):
                if row != excluded:
                    kept.append(row)
            for by_null in [False, True]:
                var payloads = List[Bool](length=n, fill=True)
                var valid = List[Bool](length=n, fill=True)
                if by_null:
                    valid[excluded] = False
                else:
                    payloads[excluded] = False
                assert_model(
                    frame.filter(Column[Bool](payloads^, valid^)),
                    model.take(kept),
                )
