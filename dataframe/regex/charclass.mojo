"""POSIX-style ASCII character classes for regex syntax."""


def ascii_class(name: String) -> List[UInt32]:
    """`[[:name:]]` ASCII classes; empty for an unknown name."""
    var r: List[UInt32]
    if name == "alnum":
        r = [48, 57, 65, 90, 97, 122]
    elif name == "alpha":
        r = [65, 90, 97, 122]
    elif name == "ascii":
        r = [0, 127]
    elif name == "blank":
        r = [9, 9, 32, 32]
    elif name == "cntrl":
        r = [0, 31, 127, 127]
    elif name == "digit":
        r = [48, 57]
    elif name == "graph":
        r = [33, 126]
    elif name == "lower":
        r = [97, 122]
    elif name == "print":
        r = [32, 126]
    elif name == "punct":
        r = [33, 47, 58, 64, 91, 96, 123, 126]
    elif name == "space":
        r = [9, 13, 32, 32]
    elif name == "upper":
        r = [65, 90]
    elif name == "word":
        r = [48, 57, 65, 90, 95, 95, 97, 122]
    elif name == "xdigit":
        r = [48, 57, 65, 70, 97, 102]
    else:
        r = List[UInt32]()
    return r^
