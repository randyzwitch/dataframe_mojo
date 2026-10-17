"""Typed scalar equality and stable order, independent of encoded key words."""
from std.math import isnan


struct KeyCell(Copyable, Movable):
    var kind: Int
    var valid: Bool
    var integer: Int128
    var floating: Float64
    var text: String

    def __init__(
        out self,
        kind: Int,
        valid: Bool = True,
        integer: Int128 = 0,
        floating: Float64 = 0,
        var text: String = "",
    ):
        self.kind = kind
        self.valid = valid
        self.integer = integer
        self.floating = floating
        self.text = text^


def cell_equal(left: KeyCell, right: KeyCell) -> Bool:
    if left.valid != right.valid or left.kind != right.kind:
        return False
    if not left.valid:
        return True
    if left.kind == 1:
        return (
            isnan(left.floating) and isnan(right.floating)
        ) or left.floating == right.floating
    if left.kind == 2:
        return left.text == right.text
    return left.integer == right.integer


def cell_less(
    left: KeyCell, right: KeyCell, descending: Bool, nulls_last: Bool
) -> Bool:
    if not left.valid or not right.valid:
        return left.valid if nulls_last else right.valid
    if left.kind == 1:
        var ln = isnan(left.floating)
        var rn = isnan(right.floating)
        if ln or rn:
            # NaN is the largest value: last ascending, first descending.
            return (ln and not rn) if descending else (not ln and rn)
        return (
            left.floating
            > right.floating if descending else left.floating
            < right.floating
        )
    if left.kind == 2:
        return left.text > right.text if descending else left.text < right.text
    return (
        left.integer
        > right.integer if descending else left.integer
        < right.integer
    )


def row_equal(rows: List[List[KeyCell]], a: Int, b: Int) -> Bool:
    for column in range(len(rows[a])):
        if not cell_equal(rows[a][column], rows[b][column]):
            return False
    return True


def expected_key_ids(
    rows: List[List[KeyCell]], nulls_equal: Bool
) -> Tuple[List[Int], List[Int]]:
    var ids = List[Int]()
    var representatives = List[Int]()
    for row in range(len(rows)):
        var missing = False
        for cell in rows[row]:
            missing = missing or not cell.valid
        if missing and not nulls_equal:
            ids.append(-1)
            continue
        var id = -1
        for group in range(len(representatives)):
            if row_equal(rows, row, representatives[group]):
                id = group
                break
        if id < 0:
            id = len(representatives)
            representatives.append(row)
        ids.append(id)
    return (ids^, representatives^)


def row_less(
    rows: List[List[KeyCell]],
    a: Int,
    b: Int,
    descending: List[Bool],
    nulls_last: List[Bool],
) -> Bool:
    for key in range(len(rows[a])):
        if cell_equal(rows[a][key], rows[b][key]):
            continue
        return cell_less(
            rows[a][key], rows[b][key], descending[key], nulls_last[key]
        )
    return False


def expected_order(
    rows: List[List[KeyCell]], descending: List[Bool], nulls_last: List[Bool]
) -> List[Int]:
    # Stable insertion sort: equal keys retain original row positions.
    var order = List[Int]()
    for row in range(len(rows)):
        order.append(row)
        var at = len(order) - 1
        while at > 0 and row_less(
            rows, order[at], order[at - 1], descending, nulls_last
        ):
            var previous = order[at - 1]
            order[at - 1] = order[at]
            order[at] = previous
            at -= 1
    return order^
