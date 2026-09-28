"""Join types as integer codes, parsed once from the public `how` name."""

comptime JOIN_INNER = 0
comptime JOIN_LEFT = 1
comptime JOIN_RIGHT = 2
comptime JOIN_FULL = 3
comptime JOIN_SEMI = 4
comptime JOIN_ANTI = 5
comptime JOIN_CROSS = 6


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
    return "unknown"
