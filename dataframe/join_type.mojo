"""Join types as integer codes, parsed once from the public `how` name."""

from dataframe.dtype import DataType
from std.collections import Optional

comptime JOIN_INNER = 0
comptime JOIN_LEFT = 1
comptime JOIN_RIGHT = 2
comptime JOIN_FULL = 3
comptime JOIN_SEMI = 4
comptime JOIN_ANTI = 5
comptime JOIN_CROSS = 6
# Physical join types with no public `how` name: a semi or anti join built on its
# left input flags the build rows a probe matched, then emits the flagged
# (RIGHT_SEMI) or unflagged (RIGHT_ANTI) ones; a MARK join emits every
# probe row with whether it matched.
comptime JOIN_RIGHT_SEMI = 7
comptime JOIN_RIGHT_ANTI = 8
comptime JOIN_MARK = 9


def join_emits_build_rows(how: Int) -> Bool:
    """Whether a probe flags the build rows it matches for a scan after
    the probe."""
    return (
        how == JOIN_RIGHT
        or how == JOIN_FULL
        or how == JOIN_RIGHT_SEMI
        or how == JOIN_RIGHT_ANTI
    )


def join_code(how: String) -> Int:
    """The code for a join type name, or -1 when the name is unknown."""
    if how == "inner":
        return JOIN_INNER
    if how == "left":
        return JOIN_LEFT
    if how == "right":
        return JOIN_RIGHT
    if how == "full":
        return JOIN_FULL
    if how == "semi":
        return JOIN_SEMI
    if how == "anti":
        return JOIN_ANTI
    if how == "cross":
        return JOIN_CROSS
    return -1


def join_type(how: String) raises -> Int:
    """The code for a join type name; raises for an unknown name."""
    var code = join_code(how)
    if code < 0:
        raise Error(
            "Join how must be inner, left, right, full, semi, anti, or cross"
        )
    return code


def join_name(code: Int) -> String:
    """The public name of a join code, for messages and plans."""
    if code == JOIN_INNER:
        return "inner"
    if code == JOIN_LEFT:
        return "left"
    if code == JOIN_RIGHT:
        return "right"
    if code == JOIN_FULL:
        return "full"
    if code == JOIN_SEMI:
        return "semi"
    if code == JOIN_ANTI:
        return "anti"
    if code == JOIN_CROSS:
        return "cross"
    if code == JOIN_RIGHT_SEMI:
        return "right_semi"
    if code == JOIN_RIGHT_ANTI:
        return "right_anti"
    if code == JOIN_MARK:
        return "mark"
    return "unknown"


def shared_join_key_type(
    left: DataType, right: DataType, how: Int
) raises -> Optional[DataType]:
    """The one type both keys of a pair take before the join, when they
    hold the same decimal at two storage widths; None for every other pair.

    A decimal's precision and scale fix its values, and its storage width
    (32, 64 or 128 bits) only says how they are held, so such a pair joins
    on value once both sides share a representation. Every join path (hash
    table, probe, coalesced key and unmatched rows) then sees one type. The
    shared type is the wider one, which holds every value of the narrower
    exactly. A semi, anti or mark join outputs only the probe side's rows
    as they are, so its other key takes the probe key's type instead; the
    precision fits either width, so that cast loses nothing."""
    if not (left.is_decimal() and right.is_decimal()) or left == right:
        return None
    if left.precision() != right.precision() or left.scale() != right.scale():
        return None
    if how == JOIN_SEMI or how == JOIN_ANTI or how == JOIN_MARK:
        return left
    if how == JOIN_RIGHT_SEMI or how == JOIN_RIGHT_ANTI:
        return right
    return DataType.decimal(
        left.precision(),
        left.scale(),
        max(left.decimal_width(), right.decimal_width()),
    )
