"""GPU probe for exact row-observed text pair discovery and ID mapping."""
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.accelerator.gpu import observed_text_pairs


def _run_case(
    left: List[UInt32],
    right: List[UInt32],
    left_valid: List[UInt8],
    right_valid: List[UInt8],
    left_errors: List[UInt32],
    right_errors: List[UInt32],
    expected_unique: Int,
) raises:
    var rows = len(left)
    assert_equal(len(right), rows)
    assert_equal(len(left_valid), rows)
    assert_equal(len(right_valid), rows)
    assert_equal(len(left_errors), rows)
    assert_equal(len(right_errors), rows)
    var ctx = DeviceContext()
    var dl = ctx.enqueue_create_buffer[DType.uint32](rows)
    var dr = ctx.enqueue_create_buffer[DType.uint32](rows)
    var dlv = ctx.enqueue_create_buffer[DType.uint8](rows)
    var drv = ctx.enqueue_create_buffer[DType.uint8](rows)
    var dle = ctx.enqueue_create_buffer[DType.uint32](rows)
    var dre = ctx.enqueue_create_buffer[DType.uint32](rows)
    if rows:
        ctx.enqueue_copy(dl, left.unsafe_ptr())
        ctx.enqueue_copy(dr, right.unsafe_ptr())
        ctx.enqueue_copy(dlv, left_valid.unsafe_ptr())
        ctx.enqueue_copy(drv, right_valid.unsafe_ptr())
        ctx.enqueue_copy(dle, left_errors.unsafe_ptr())
        ctx.enqueue_copy(dre, right_errors.unsafe_ptr())
    var result = observed_text_pairs.run(ctx, dl, dlv, dle, dr, drv, dre, rows)
    var unique_left = List[UInt32](length=max(1, rows), fill=UInt32.MAX)
    var unique_right = List[UInt32](length=max(1, rows), fill=UInt32.MAX)
    var pair_ids = List[UInt32](length=rows, fill=UInt32.MAX)
    var pair_errors = List[UInt32](length=rows, fill=0)
    var metadata = List[UInt32](length=2, fill=0)
    ctx.enqueue_copy(unique_left.unsafe_ptr(), result.pair_left)
    ctx.enqueue_copy(unique_right.unsafe_ptr(), result.pair_right)
    if rows:
        ctx.enqueue_copy(pair_ids.unsafe_ptr(), result.row_ids)
        ctx.enqueue_copy(pair_errors.unsafe_ptr(), result.row_errors)
    ctx.enqueue_copy(metadata.unsafe_ptr(), result.metadata)
    ctx.synchronize()
    assert_equal(metadata[0], UInt32(expected_unique))
    assert_equal(metadata[1], UInt32(0))
    assert_equal(observed_text_pairs.table_capacity(rows), result.capacity)

    for row in range(rows):
        var error = left_errors[row] | right_errors[row]
        assert_equal(pair_errors[row], error)
        if error or not left_valid[row] or not right_valid[row]:
            assert_equal(pair_ids[row], UInt32.MAX)
            continue
        var pair_id = Int(pair_ids[row])
        assert_true(pair_id >= 0 and pair_id < expected_unique)
        assert_equal(unique_left[pair_id], left[row])
        assert_equal(unique_right[pair_id], right[row])
        for previous in range(row):
            var previous_error = left_errors[previous] | right_errors[previous]
            if (
                previous_error
                or not left_valid[previous]
                or not right_valid[previous]
            ):
                continue
            var same_pair = (
                left[row] == left[previous] and right[row] == right[previous]
            )
            assert_equal(pair_ids[row] == pair_ids[previous], same_pair)

    # Every compact dictionary control is referenced by at least one row.
    for pair_id in range(expected_unique):
        var found = False
        for row in range(rows):
            found |= pair_ids[row] == UInt32(pair_id)
        assert_true(found)

    # Publish a post-host-label-map code per row, checking null/error behavior
    # and a nonzero slot base used by resident output matrices.
    var code_map = List[UInt32](length=max(1, rows), fill=0)
    for pair_id in range(expected_unique):
        code_map[pair_id] = UInt32(100 + pair_id)
    var output_cells = rows + 2
    var words = List[UInt64](length=output_cells * 2, fill=0xBAD)
    var valid = List[UInt8](length=rows + 2, fill=7)
    var errors = List[UInt32](length=rows + 2, fill=0xBAD)
    var device_codes = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var device_words = ctx.enqueue_create_buffer[DType.uint64](output_cells * 2)
    var device_valid = ctx.enqueue_create_buffer[DType.uint8](rows + 2)
    var device_errors = ctx.enqueue_create_buffer[DType.uint32](rows + 2)
    ctx.enqueue_copy(device_codes, code_map.unsafe_ptr())
    ctx.enqueue_copy(device_words, words.unsafe_ptr())
    ctx.enqueue_copy(device_valid, valid.unsafe_ptr())
    ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
    if rows:
        ctx.enqueue_function[observed_text_pairs.write_output](
            result.row_ids.unsafe_ptr(),
            device_codes.unsafe_ptr(),
            result.row_errors.unsafe_ptr(),
            device_words.unsafe_ptr(),
            device_valid.unsafe_ptr(),
            device_errors.unsafe_ptr(),
            UInt32(rows),
            Int64(2),
            Int64(output_cells),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
    ctx.enqueue_copy(words.unsafe_ptr(), device_words)
    ctx.enqueue_copy(valid.unsafe_ptr(), device_valid)
    ctx.enqueue_copy(errors.unsafe_ptr(), device_errors)
    ctx.synchronize()
    for row in range(rows):
        var error = left_errors[row] | right_errors[row]
        assert_equal(errors[row + 2], error)
        if error or not left_valid[row] or not right_valid[row]:
            assert_equal(words[row + 2], UInt64(0))
            assert_equal(valid[row + 2], UInt8(0))
        else:
            assert_equal(words[row + 2], UInt64(100) + UInt64(pair_ids[row]))
            assert_equal(valid[row + 2], UInt8(1))
        assert_equal(words[output_cells + row + 2], UInt64(0))
    assert_equal(words[0], UInt64(0xBAD))
    assert_equal(words[1], UInt64(0xBAD))
    assert_equal(words[output_cells], UInt64(0xBAD))
    assert_equal(words[output_cells + 1], UInt64(0xBAD))
    _ = result^
    _ = dl^
    _ = dr^
    _ = dlv^
    _ = drv^
    _ = dle^
    _ = dre^
    _ = unique_left^
    _ = unique_right^
    _ = pair_ids^
    _ = pair_errors^
    _ = metadata^
    _ = code_map^
    _ = words^
    _ = valid^
    _ = errors^
    _ = device_codes^
    _ = device_words^
    _ = device_valid^
    _ = device_errors^


def _test_pack_codes() raises:
    # The requested slot begins after a prefix of other slots and uses a
    # stride larger than its row count, matching a resident matrix slice.
    var words = List[UInt64](length=18, fill=0xBAD)
    words[10] = 7
    words[11] = UInt64(0x0000000100000002)
    words[12] = 12
    var packed = List[UInt32](length=3, fill=0)
    var ctx = DeviceContext()
    var device_words = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var device_packed = ctx.enqueue_create_buffer[DType.uint32](len(packed))
    ctx.enqueue_copy(device_words, words.unsafe_ptr())
    ctx.enqueue_function[observed_text_pairs.pack_codes](
        device_words.unsafe_ptr(),
        device_packed.unsafe_ptr(),
        UInt32(3),
        UInt32(2),
        UInt32(5),
        grid_dim=1,
        block_dim=32,
    )
    ctx.enqueue_copy(packed.unsafe_ptr(), device_packed)
    ctx.synchronize()
    assert_equal(packed[0], UInt32(7))
    assert_equal(packed[1], UInt32.MAX)
    assert_equal(packed[2], UInt32(12))
    _ = words^
    _ = packed^
    _ = device_words^
    _ = device_packed^


def main() raises:
    _test_pack_codes()
    var shape_ctx = DeviceContext()
    var one_codes = shape_ctx.enqueue_create_buffer[DType.uint32](1)
    var no_validity = shape_ctx.enqueue_create_buffer[DType.uint8](0)
    var one_errors = shape_ctx.enqueue_create_buffer[DType.uint32](1)
    with assert_raises(contains="shorter than row count"):
        _ = observed_text_pairs.run(
            shape_ctx,
            one_codes,
            no_validity,
            one_errors,
            one_codes,
            no_validity,
            one_errors,
            1,
        )
    # More than one million possible dictionary pairs, but only 1,100 pairs
    # actually occur in the rows. Repetition also forces hash collisions.
    var left = List[UInt32]()
    var right = List[UInt32]()
    var left_valid = List[UInt8]()
    var right_valid = List[UInt8]()
    var left_errors = List[UInt32]()
    var right_errors = List[UInt32]()
    var initial_buckets = List[UInt32]()
    var collision = False
    var hash_capacity = observed_text_pairs.table_capacity(2200)
    for value in range(1100):
        var bucket = observed_text_pairs._hash_pair(
            UInt32(value), UInt32((value * 37) % 1100)
        ) & UInt32(hash_capacity - 1)
        for earlier in initial_buckets:
            collision |= earlier == bucket
        initial_buckets.append(bucket)
    assert_true(collision)
    for repeat in range(2):
        for value in range(1100):
            left.append(UInt32(value))
            right.append(UInt32((value * 37) % 1100))
            left_valid.append(1)
            right_valid.append(1)
            left_errors.append(0)
            right_errors.append(0)
    assert_true(1100 * 1100 > 1_048_576)
    _run_case(
        left^,
        right^,
        left_valid^,
        right_valid^,
        left_errors^,
        right_errors^,
        1100,
    )
    assert_equal(
        observed_text_pairs.estimate_bytes(2200),
        observed_text_pairs.table_capacity(2200) * 8 + 2200 * 16 + 8,
    )

    # Null sides are excluded, and input failures are carried unchanged as
    # their bitwise union. A failure row must never become a dictionary pair.
    _run_case(
        [1, 2, 3, 1, 4, 5],
        [7, 8, 9, 7, 10, 11],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 4],
        1,
    )
    _run_case([], [], [], [], [], [], 0)
    _run_case(
        [1, 2, 3],
        [4, 5, 6],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        0,
    )
    print(
        "PASS observed GPU text pairs: sparse domains, collisions, nulls, errors, empty"
    )
