"""Packed CPU Bool buffers mapped/uploaded and sliced on actual Metal."""
from std.memory import Pointer
from std.testing import assert_equal
from max.gpu import global_idx
from dataframe import DataFrame, Series
from dataframe.storage.boolean import BoolColumn
from dataframe.expr import COL
from dataframe_metal.accelerator.gpu.query_scope import Scope
from dataframe_metal.accelerator.gpu.typed_storage import NativeColumnAny
from dataframe_metal.accelerator.gpu.erased_input import (
    numeric_views,
    single_native_span,
)
from dataframe_metal.accelerator.gpu.compound_predicate import predicate_matches


def _storage_probe(
    values: Pointer[UInt8, MutUntrackedOrigin],
    bits: Pointer[UInt8, MutUntrackedOrigin],
    program: Pointer[UInt64, MutUntrackedOrigin],
    output: Pointer[UInt32, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row < Int(rows):
        output[unsafe_offset=row] = UInt32(
            predicate_matches(
                program,
                0,
                row,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                bits,
                bits,
                bits,
                bits,
                bits,
                bits,
                bits,
                bits,
                bits,
            )
        )


def main() raises:
    from tests.support.metal_outcomes import require_metal

    _ = require_metal()
    var checks = 0
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(1024):
        values.append(row % 5 < 2)
        valid.append(row % 11 != 0)
    var original = Series("b", BoolColumn(values^, valid^))
    var scope = Scope()
    var ctx = scope.context()
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var kernel = ctx.compile_function[_storage_probe]()
    for offset in range(8):
        var sliced = original.slice(offset, 513)
        var shared = Series._from_chunks(
            [sliced.slice(0, 129), sliced.slice(129, 384)]
        )
        # Independently owned chunks exercise packed bitmap concatenation.
        var left_values = List[Bool]()
        var right_values = List[Bool]()
        var left_valid = List[Bool]()
        var right_valid = List[Bool]()
        for i in range(513):
            if i < 129:
                left_values.append((i + offset) % 5 < 2)
                left_valid.append((i + offset) % 11 != 0)
            else:
                right_values.append((i + offset) % 5 < 2)
                right_valid.append((i + offset) % 11 != 0)
        var independent = Series._from_chunks(
            [
                Series("b", BoolColumn(left_values^, left_valid^)),
                Series("b", BoolColumn(right_values^, right_valid^)),
            ]
        )
        assert_equal(single_native_span(shared), True)
        assert_equal(single_native_span(independent), False)
        for layout in range(3):
            var source = (
                sliced.copy() if layout
                == 0 else shared.copy() if layout
                == 1 else independent.copy()
            )
            var frame = DataFrame([source])
            for zero_copy in [False, True]:
                for coalesce in [False, True]:
                    var owners = List[NativeColumnAny]()
                    var views = numeric_views(
                        frame,
                        "b",
                        scope,
                        zero_copy,
                        empty,
                        owners,
                        coalesce=coalesce,
                    )
                    var base = offset
                    for original_view in views:
                        for skip in [0, 3]:
                            var view = original_view.view(
                                skip, original_view.rows - skip
                            )
                            var program = List[UInt64](length=15, fill=0)
                            program[0] = 1
                            program[1] = 2
                            program[10] = (
                                UInt64(view.bit_offset)
                                | (UInt64(view.nullable) << 32)
                                | (UInt64(3) << 33)
                                | (UInt64(view.value_bit_offset) << 35)
                            )
                            program[14] = UInt64(3 | (1 << 2) | (COL << 5))
                            var metadata = ctx.enqueue_create_buffer[
                                DType.uint64
                            ](15)
                            var output = ctx.enqueue_create_buffer[
                                DType.uint32
                            ](view.rows)
                            ctx.enqueue_copy(metadata, program.unsafe_ptr())
                            ctx.enqueue_function(
                                kernel,
                                view.values,
                                view.bits,
                                metadata,
                                output,
                                UInt32(view.rows),
                                grid_dim=(view.rows + 255) // 256,
                                block_dim=256,
                            )
                            var actual = List[UInt32](length=view.rows, fill=0)
                            ctx.enqueue_copy(actual.unsafe_ptr(), output)
                            scope.synchronize()
                            for i in range(view.rows):
                                var row = base + skip + i
                                assert_equal(
                                    actual[i],
                                    UInt32(row % 5 < 2 and row % 11 != 0),
                                )
                                checks += 1
                        base += original_view.rows
    print("PASS", checks, "packed CPU Bool storage checks on Metal")
