"""Actual GPU packed-word counts: independent offsets, tails and null masks."""
from std.memory import Pointer
from std.testing import assert_equal
from max.gpu.host import DeviceContext
from dataframe_metal.accelerator.gpu.logical_words import logical_word_partials


def main() raises:
    var ctx = DeviceContext(0, api="metal")
    comptime BYTES = 8192
    var values = List[UInt8]()
    var bits = List[UInt8]()
    for pattern in range(5):
        for byte in range(BYTES):
            values.append(
                UInt8(
                    0 if pattern
                    == 1 else 255 if pattern
                    == 2 else (byte * 37 + 7) % 256
                )
            )
            bits.append(
                UInt8(
                    0 if pattern
                    == 3 else 255 if pattern
                    == 4 else (byte * 53 + 11) % 256
                )
            )
    var expected_valid = List[UInt32]()
    var expected_true = List[UInt32]()
    var descriptors = List[List[Int]]()
    for pattern in range(5):
        for vo in [0, 1, 3, 7]:
            for bo in [0, 1, 3, 7, 4095]:
                for rows in [
                    0,
                    1,
                    7,
                    8,
                    31,
                    32,
                    33,
                    63,
                    64,
                    65,
                    255,
                    256,
                    257,
                    8193,
                    32769,
                ]:
                    for flags in range(4):
                        var n = UInt32(0)
                        var t = UInt32(0)
                        for row in range(rows):
                            var valid = flags & 2 == 0 or Bool(
                                (
                                    bits[pattern * BYTES + (bo + row) // 8]
                                    >> UInt8((bo + row) % 8)
                                )
                                & 1
                            )
                            var value = Bool(
                                (
                                    values[pattern * BYTES + (vo + row) // 8]
                                    >> UInt8((vo + row) % 8)
                                )
                                & 1
                            )
                            n += UInt32(valid)
                            t += UInt32(valid and value and flags & 1 != 0)
                        expected_valid.append(n)
                        expected_true.append(t)
                        descriptors.append([pattern, vo, bo, rows, flags])
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(values))
    var db = ctx.enqueue_create_buffer[DType.uint8](len(bits))
    var output = ctx.enqueue_create_buffer[DType.uint32](len(descriptors) * 4)
    var actual = List[UInt32](length=len(descriptors) * 4, fill=UInt32.MAX)
    ctx.enqueue_copy(dv, values.unsafe_ptr())
    ctx.enqueue_copy(db, bits.unsafe_ptr())
    var kernel = ctx.compile_function[logical_word_partials]()
    for index in range(len(descriptors)):
        ref d = descriptors[index]
        var v = dv.create_sub_buffer[DType.uint8](d[0] * BYTES, BYTES)
        var b = db.create_sub_buffer[DType.uint8](d[0] * BYTES, BYTES)
        ctx.enqueue_function(
            kernel,
            v,
            b,
            UInt32(d[1]),
            UInt32(d[2]),
            UInt32(d[3]),
            UInt32(d[4]),
            UInt32(512),
            output,
            UInt32(index * 4),
            grid_dim=2,
            block_dim=256,
        )
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    ctx.synchronize()
    for index in range(len(descriptors)):
        assert_equal(
            actual[index * 4] + actual[index * 4 + 2], expected_valid[index]
        )
        assert_equal(
            actual[index * 4 + 1] + actual[index * 4 + 3], expected_true[index]
        )
    print("PASS packed-word GPU cases:", len(descriptors))
    _ = (dv^, db^, output^, actual^, values^, bits^)
