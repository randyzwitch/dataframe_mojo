from std.sys import argv
from std.time import monotonic
from dataframe.regex import Matcher, Regex


def _span(text: String) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


def _captures(mut matcher: Matcher, text: Span[UInt8, ImmutAnyOrigin]) -> Int:
    var at = 0
    var count = 0
    while matcher.captures(text, at):
        for k in range(0, len(matcher.slots), 2):
            if matcher.slots[k] >= 0:
                count += 1
        if matcher.slots[1] <= at:
            break
        at = matcher.slots[1]
    return count


def _run(
    mut matcher: Matcher, text: Span[UInt8, ImmutAnyOrigin], mode: Int
) -> Int:
    if mode == 2:
        return _captures(matcher, text)
    if mode == 3:
        var total = 0
        var start = 0
        for at in range(len(text) + 1):
            if at == len(text) or text[at] == 10:
                var end = at
                if end > start and text[end - 1] == 13:
                    end -= 1
                total += _captures(matcher, text[start:end])
                start = at + 1
        return total
    if mode == 0:
        return matcher.count(text)
    matcher.reset()
    var total = 0
    while True:
        var found = matcher.next(text)
        if found[0] < 0:
            break
        total += found[1] - found[0]
    return total


def main() raises:
    var args = argv()
    var manifest = open(String(args[1]), "r").read()
    for line in manifest.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        var name = String(fields[0])
        try:
            var pattern = open(String(fields[1]), "r").read()
            var text = open(String(fields[2]), "r").read()
            var mode = (
                1 if fields[3]
                == "count-spans" else 2 if fields[3]
                == "count-captures" else 3 if fields[3]
                == "grep-captures" else 0
            )
            var expected = Int(String(fields[4]))
            var regex = Regex(pattern)
            var matcher = Matcher(regex)
            var warm = _run(matcher, _span(text), mode)
            if warm != expected:
                print(name, "WRONG", warm, expected, sep="\t")
                continue
            comptime ITERATIONS = 20
            for rep in range(3):
                var start = monotonic()
                var total = 0
                for _ in range(ITERATIONS):
                    total += _run(matcher, _span(text), mode)
                var ns = monotonic() - start
                print(
                    name,
                    "OK" if total == expected * ITERATIONS else "WRONG",
                    rep,
                    ns // ITERATIONS,
                    total // ITERATIONS,
                    sep="\t",
                )
            _ = text
        except error:
            print(name, "ERROR", error, sep="\t")
