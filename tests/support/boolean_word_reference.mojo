"""Scalar Boolean state generator shared by bitmap operation contracts."""
from dataframe.bool_column import BoolColumn


def truth_state(row: Int, mode: Int, side: Int) -> Int:
    if mode == 2 or (mode == 1 and (row + side * 3) % 5 == 2):
        return -1
    return Int((row * 7 + side * 5) % 3 == 1)


def truth_column(
    n: Int, offset: Int, mode: Int, side: Int
) raises -> BoolColumn:
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(n + offset):
        var state = truth_state(row, mode, side)
        values.append(state != 0)  # Poison true under null.
        valid.append(state >= 0)
    return BoolColumn(values, valid).slice(offset, n)
