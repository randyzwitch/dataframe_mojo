"""Scalar relational predicates; no planner, expression or dataframe helpers."""
from tests.support.reference_model import (
    ReferenceTable,
    RowPair,
    reference_join,
)


def filtered_join_pairs(
    left: ReferenceTable,
    right: ReferenceTable,
    keys: List[String],
    how: String,
    predicate: Int,
) raises -> List[RowPair]:
    var pairs = reference_join(left, right, keys, how)
    var retained = List[RowPair]()
    for pair in pairs:
        var left_positive = (
            pair.left >= 0
            and left.rows[pair.left].iv
            and left.rows[pair.left].i > 0
        )
        var right_positive = (
            pair.right >= 0
            and right.rows[pair.right].fv
            and right.rows[pair.right].f > 0
        )
        var right_null = pair.right < 0 or not right.rows[pair.right].fv
        var keep = (
            left_positive if predicate
            == 0 else right_positive if predicate
            == 1 else right_null if predicate
            == 2 else left_positive
            and right_positive
        )
        if predicate == 4:
            # The public coalesced key is supplied by the retained side.
            var use_right = how == "right" or pair.left < 0
            keep = (
                pair.right >= 0
                and right.rows[pair.right].iv
                and right.rows[pair.right].i > 0
            ) if use_right else left_positive
        if keep:
            retained.append(pair.copy())
    return retained^
