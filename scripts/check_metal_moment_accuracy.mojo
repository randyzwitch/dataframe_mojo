from std.sys import argv
from std.memory import bitcast
from dataframe_metal.moment_finish import finish_variance


def main() raises:
    var text = String()
    with open(String(argv()[1]), "r") as source:
        text = source.read()
    var index = 0
    for line in text.split("\n"):
        if not line:
            continue
        var parts = line.split(",")
        var sums = List[Int128]()
        var squares = List[Int128]()
        for word in range(27):
            var at = 4 + 3 * word
            var value = Int128(Int(parts[at + 1])) + (
                Int128(Int(parts[at + 2])) << 32
            )
            if Int(parts[at]):
                value = -value
            if word < 9:
                sums.append(value)
            else:
                squares.append(value)
        var result = finish_variance(
            sums,
            squares,
            Int(parts[1]),
            Int(parts[2]),
            Int(parts[0]),
            Bool(Int(parts[3])),
        )
        if result:
            print(index, bitcast[DType.uint64](result.value()))
        else:
            print(index, "NULL")
        index += 1
