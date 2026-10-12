"""Scalar row equality, survivor positions and reshape cardinality models."""
from tests.support.key_kernel_reference import cell_equal
from tests.support.selection_types_reference import (
    selection_key,
    selection_valid,
)


def set_cell_equal(kind: Int, a: Int, b: Int, mode: Int) raises -> Bool:
    if kind != 27:
        return cell_equal(
            selection_key(kind, a, mode), selection_key(kind, b, mode)
        )
    var av = selection_valid(a, mode)
    var bv = selection_valid(b, mode)
    if av != bv:
        return False
    if not av:
        return True
    # Both struct fields share a child mask. Visible number is the exact
    # global row literal, making non-null records equal only to themselves.
    var ac = a % 3 != 1
    var bc = b % 3 != 1
    return ac == bc and (not ac or a == b)


def set_key_equal(
    kind: Int, a: Int, b: Int, mode: Int, compound: Bool
) raises -> Bool:
    if not set_cell_equal(kind, a, b, mode):
        return False
    if not compound:
        return True
    var av = a % 5 != 2
    var bv = b % 5 != 2
    return av == bv and (not av or (a % 3 == 1) == (b % 3 == 1))


def set_counts(
    kind: Int, rows: List[Int], mode: Int, compound: Bool
) raises -> List[Int]:
    var result = List[Int](length=len(rows), fill=0)
    for i in range(len(rows)):
        for other in rows:
            result[i] += Int(
                set_key_equal(kind, rows[i], other, mode, compound)
            )
    return result^


def set_survivors(
    kind: Int, rows: List[Int], mode: Int, compound: Bool, keep: String
) raises -> List[Int]:
    var result = List[Int]()
    for i in range(len(rows)):
        var first = True
        var last = True
        var count = 0
        for j in range(len(rows)):
            if set_key_equal(kind, rows[i], rows[j], mode, compound):
                count += 1
                first = first and j >= i
                last = last and j <= i
        if (
            (keep == "none" and count == 1)
            or (keep == "last" and last)
            or ((keep == "first" or keep == "any") and first)
        ):
            result.append(rows[i])
    return result^


def explode_parent_valid(row: Int, mode: Int) -> Bool:
    return mode == 0 or mode == 1 and row % 5 != 2


def explode_child_count(row: Int, mode: Int) -> Int:
    # Null parents own no hidden children; the separate core explode witness
    # deliberately tests hidden null-parent spans as a known-failure contract.
    return row % 3 if explode_parent_valid(row, mode) else 0


def exploded_positions(
    n: Int, parent_offset: Int, parent_mode: Int, child_offset: Int = 7
) -> Tuple[List[Int], List[Int]]:
    var parents = List[Int]()
    var children = List[Int]()
    var child = child_offset
    for row in range(parent_offset + n):
        var count = explode_child_count(row, parent_mode)
        if row >= parent_offset:
            for i in range(max(1, count)):
                parents.append(row)
                children.append(child + i if count > 0 else -1)
        child += count
    return (parents^, children^)
