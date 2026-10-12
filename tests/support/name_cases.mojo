"""Names are opaque strings, including characters used by internal encodings."""


def opaque_names() -> List[String]:
    return [
        "",
        "x\x1fy",
        "x\x1ey",
        "\x1e0\x1e1",
        "a, b",
        "é雪",
        '"|\\',
        "line\nname",
    ]
