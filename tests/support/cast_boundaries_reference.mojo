"""Literal conversion examples and first-observed dictionary reference."""


def first_observed_dictionary(
    text: List[String], valid: List[Bool]
) -> Tuple[List[String], List[UInt32]]:
    var entries = List[String]()
    var codes = List[UInt32]()
    for row in range(len(text)):
        var found = -1
        if valid[row]:
            for code in range(len(entries)):
                if entries[code] == text[row]:
                    found = code
            if found < 0:
                found = len(entries)
                entries.append(text[row].copy())
        codes.append(UInt32(max(0, found)))
    return (entries^, codes^)


def float_lexical_examples() -> Tuple[List[String], List[Float64], List[Bool]]:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var text: List[String] = [
        "0",
        "-0",
        "+0",
        "1.25",
        "-1.25",
        "+1e2",
        ".5",
        "5.",
        "1E-2",
        "nan",
        "NaN",
        "inf",
        "+inf",
        "-inf",
        "Infinity",
        "+Infinity",
        "-Infinity",
        "",
        " ",
        " 1",
        "1 ",
        "1x",
        "1_000",
        "0x1",
        "+nan",
        "-nan",
        "NAN",
        "INF",
        "infinity",
        "+",
        ".",
        "1e",
        "1e+",
        "1e9999",
        "-1e9999",
    ]
    var values: List[Float64] = [
        0.0,
        -0.0,
        0.0,
        1.25,
        -1.25,
        100.0,
        0.5,
        5.0,
        0.01,
        nan,
        nan,
        inf,
        inf,
        -inf,
        inf,
        inf,
        -inf,
    ]
    var valid = List[Bool](length=len(text), fill=False)
    for i in range(len(values)):
        valid[i] = True
    while len(values) < len(text):
        values.append(0.0)
    return (text^, values^, valid^)
