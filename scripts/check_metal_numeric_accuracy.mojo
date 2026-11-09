"""Host execution of the device-compatible binary32 division primitive."""
from std.sys import argv
from std.testing import assert_equal, assert_false
from dataframe_metal.accelerator.gpu.native_numeric import float32_divide_bits


def main() raises:
    var args = argv()
    assert_equal(len(args), 3)
    var count = 0
    with open(args[1], "r") as source:
        with open(args[2], "w") as target:
            for line in source.read().split("\n"):
                if not line:
                    continue
                var parts = line.split(",")
                var a = UInt32(Int(parts[0]))
                var b = UInt32(Int(parts[1]))
                var expected = UInt32(Int(parts[2]))
                var result = float32_divide_bits(a, b)
                assert_false(result[1], String(line))
                assert_equal(result[0], expected, String(line))
                target.write(
                    String(a) + "," + String(b) + "," + String(result[0]) + "\n"
                )
                count += 1
    print("PASS independent binary32 division", count)
