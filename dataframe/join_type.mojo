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


# A join's `validate` option as bits: which inputs must have unique keys.
# A plan that swaps a join's inputs swaps the two side bits and toggles
# JOIN_VALIDATE_SWAPPED, so messages still name the sides as written.
comptime JOIN_VALIDATE_LEFT = 1
comptime JOIN_VALIDATE_RIGHT = 2
comptime JOIN_VALIDATE_SWAPPED = 4
# An unknown `validate` name, reported when the plan executes.
comptime JOIN_VALIDATE_UNKNOWN = -1


def join_validate_code(validate: String) -> Int:
    """The bits for a `validate` name ("m:m", "1:m", "m:1" or "1:1"), or
    JOIN_VALIDATE_UNKNOWN."""
    if validate == "m:m":
        return 0
    if validate == "1:m":
        return JOIN_VALIDATE_LEFT
    if validate == "m:1":
        return JOIN_VALIDATE_RIGHT
    if validate == "1:1":
        return JOIN_VALIDATE_LEFT | JOIN_VALIDATE_RIGHT
    return JOIN_VALIDATE_UNKNOWN


def join_validate_swapped(code: Int) -> Int:
    """The bits of a join whose inputs trade places."""
    if code <= 0:
        return code
    var swapped = code ^ JOIN_VALIDATE_SWAPPED
    var left = code & JOIN_VALIDATE_LEFT
    var right = code & JOIN_VALIDATE_RIGHT
    swapped &= ~(JOIN_VALIDATE_LEFT | JOIN_VALIDATE_RIGHT)
    if left != 0:
        swapped |= JOIN_VALIDATE_RIGHT
    if right != 0:
        swapped |= JOIN_VALIDATE_LEFT
    return swapped


def join_validate_name(code: Int) -> String:
    """The `validate` name of the bits, as the join was written."""
    var written = code
    if code > 0 and code & JOIN_VALIDATE_SWAPPED != 0:
        written = join_validate_swapped(code)
    var left = written > 0 and written & JOIN_VALIDATE_LEFT != 0
    var right = written > 0 and written & JOIN_VALIDATE_RIGHT != 0
    if left and right:
        return "1:1"
    if left:
        return "1:m"
    if right:
        return "m:1"
    return "m:m"


def check_join_validate(code: Int, how: Int) raises:
    """Raise when the bits are unknown, or ask for unique keys on a join
    whose output cannot repeat rows of an input: a semi or anti join emits
    each kept row once, and a cross join has no keys."""
    if code == JOIN_VALIDATE_UNKNOWN:
        raise Error("Join validate must be 'm:m', '1:m', 'm:1' or '1:1'")
    if code == 0 or code == JOIN_VALIDATE_SWAPPED:
        return
    if how in [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
    ]:
        return
    var name = (
        "semi" if how
        in [JOIN_SEMI, JOIN_RIGHT_SEMI] else "anti" if how
        in [JOIN_ANTI, JOIN_RIGHT_ANTI] else join_name(how)
    )
    raise Error(
        "Join validate='"
        + join_validate_name(code)
        + "' does not apply to a "
        + name
        + " join"
    )


def join_validate_checks(code: Int, right_side: Bool) -> Bool:
    """Whether the bits ask for unique keys on the plan's right input
    (`right_side`) or its left."""
    if code <= 0:
        return False
    return (
        code & (JOIN_VALIDATE_RIGHT if right_side else JOIN_VALIDATE_LEFT) != 0
    )


def join_validate_side_name(code: Int, right_side: Bool) -> String:
    """The side, as the join was written, of the plan's right input
    (`right_side`) or its left."""
    var swapped = code > 0 and code & JOIN_VALIDATE_SWAPPED != 0
    return "right" if right_side != swapped else "left"
