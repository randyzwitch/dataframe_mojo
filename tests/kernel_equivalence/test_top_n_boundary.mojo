"""Shared variable-key boundaries own bytes across growth and reader copies."""
# contract-group: table.sort
from std.testing import assert_equal, assert_true
from dataframe.sorting.algorithms import Rec
from dataframe.sorting.top_n import TopNBoundary, _BoundaryView


def test_variable_top_n_boundary_keeps_bytes_after_growth_and_source_reuse() raises:
    var boundary = TopNBoundary.new()
    var released = False
    try:
        var oldest = _BoundaryView()
        var current = _BoundaryView()
        var sizes: List[Int] = [9, 65, 513, 33, 2049]
        for step in range(len(sizes)):
            var value = UInt8(100 - step)
            var bytes = List[UInt8](length=sizes[step], fill=value)
            var record = Rec(0)
            record[0] = UInt64(value) * 0x0101010101010101
            record[1] = UInt64(len(bytes))
            record[2] = UInt64(Int(bytes.unsafe_ptr()))
            record[3] = UInt64(step)
            assert_true(boundary.publish[0, True](record))
            if step == 0:
                oldest.refresh[0, True](boundary)
            for i in range(len(bytes)):
                bytes[i] = 255
            current.refresh[0, True](boundary)
            assert_equal(current.version, Int64(step + 1))
            assert_equal(Int(current.rec[1]), sizes[step])
            assert_equal(Int(current.rec[3]), step)
            for i in range(sizes[step]):
                assert_equal(current.blob[i], value)
            for i in range(9):
                assert_equal(oldest.blob[i], UInt8(100))
        boundary.free()
        released = True
        # Reader copies remain valid after the shared cell has been released.
        assert_equal(current.blob[2048], UInt8(96))
        assert_equal(oldest.blob[8], UInt8(100))
    except error:
        if not released:
            boundary.free()
        raise error^
