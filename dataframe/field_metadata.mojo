"""Internal comparison of owned Arrow C field metadata (arbitrary bytes)."""


def _length(data: List[UInt8], mut at: Int) -> Int:
    var result = 0
    for i in range(4):
        result |= Int(data[at + i]) << (8 * i)
    at += 4
    return result


def _entries(data: List[UInt8]) -> List[List[UInt8]]:
    var at = 0
    var count = _length(data, at)
    var result = List[List[UInt8]](capacity=count)
    for _ in range(count):
        var start = at
        var size = _length(data, at)
        at += size
        size = _length(data, at)
        at += size
        var entry = List[UInt8](capacity=at - start)
        for i in range(start, at):
            entry.append(data[i])
        result.append(entry^)
    return result^


def _equal_metadata(left: List[UInt8], right: List[UInt8]) -> Bool:
    # Pair order is not significant; duplicate pairs retain their multiplicity.
    if len(left) != len(right):
        return False
    var a = _entries(left)
    var b = _entries(right)
    if len(a) != len(b):
        return False
    var used = List[Bool](length=len(b), fill=False)
    for entry in a:
        var found = False
        for j in range(len(b)):
            if used[j] or len(entry) != len(b[j]):
                continue
            var equal = True
            for k in range(len(entry)):
                if entry[k] != b[j][k]:
                    equal = False
                    break
            if equal:
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True
