"""Requested allocation sizes for the bounded resident row executor."""
from .accel_rows import RowPlan
from .accel_memory import AccelMemory, ACCEL_THREADS, ACCEL_MAX_BLOCKS
from .dtype import DataType


def checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or a > Int.MAX - b:
        raise Error("Accelerator memory estimate overflows Int")
    return a + b


def checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b != 0 and a > Int.MAX // b):
        raise Error("Accelerator memory estimate overflows Int")
    return a * b


def bitmap_bytes(rows: Int, offset: Int) raises -> Int:
    if rows < 0 or offset < 0 or offset > 7:
        raise Error("Invalid accelerator bitmap shape")
    return rows // 8 + (rows % 8 + offset + 7) // 8 if rows else 0


@fieldwise_init
struct RowMemory(Copyable):
    var summary: AccelMemory
    var capacity: Int
    var matrix: Int
    var alternate: Int
    var ranks: Int
    var blocks: Int
    var chunk: Int
    var packed: Int
    var validity_bytes: List[Int]
    var bool_bytes: List[Int]
    var offsets: List[Int]
    var has_validity: List[Bool]
    var upload_bytes: Int


def _source_shape[
    D: DType
](plan: RowPlan, index: Int) raises -> Tuple[Int, Bool]:
    var column = plan.source._columns[index].numeric[D]()
    return (column.validity_offset(), len(column._bits[]) != 0)


def row_memory(plan: RowPlan) raises -> RowMemory:
    var rows = plan.source.height()
    var capacity = max(1, rows)
    var matrix = checked_mul(capacity, plan.slots)
    var filtered = False
    var launches = plan.source.width()
    for step in plan.steps:
        filtered |= step.filter
        launches = checked_add(launches, 4 if step.filter else 1)
    var alternate = matrix if filtered else 1
    var ranks = capacity if filtered else 1
    var blocks = max(
        1,
        min(
            ACCEL_MAX_BLOCKS,
            rows // ACCEL_THREADS + Int(rows % ACCEL_THREADS != 0),
        ),
    )
    var chunk = checked_mul(
        capacity // (blocks * ACCEL_THREADS)
        + Int(capacity % (blocks * ACCEL_THREADS) != 0),
        ACCEL_THREADS,
    )
    var packed = max(1, bitmap_bytes(rows, 0))
    var item_bytes = 4 if plan.dtype == DataType.FLOAT32 else 8
    var workspace = checked_mul(checked_add(matrix, alternate), item_bytes + 1)
    workspace = checked_add(workspace, checked_mul(ranks, 8))
    workspace = checked_add(workspace, checked_mul(blocks, 16))
    workspace = checked_add(
        workspace, 32
    )  # row counts + scalar reduction outputs
    workspace = checked_add(workspace, checked_mul(packed, 2))
    var descriptors = checked_add(
        checked_mul(max(1, len(plan.code)), 8),
        checked_mul(max(1, len(plan.literals)), 8),
    )
    descriptors = checked_add(
        descriptors, checked_mul(max(1, len(plan.gathers)), 8)
    )
    workspace = checked_add(workspace, descriptors)
    var upload = checked_mul(
        checked_add(
            checked_add(len(plan.code), len(plan.literals)), len(plan.gathers)
        ),
        8,
    )
    upload = checked_add(upload, 16)
    var validity = List[Int]()
    var bools = List[Int]()
    var offsets = List[Int]()
    var has = List[Bool]()
    for i in range(plan.source.width()):
        var shape: Tuple[Int, Bool]
        var boolean = plan.source._columns[i].dtype() == DataType.BOOL
        if boolean:
            var column = plan.source._columns[i].bool()
            shape = (column.validity_offset(), len(column._bits[]) != 0)
        elif plan.dtype == DataType.FLOAT32:
            shape = _source_shape[DType.float32](plan, i)
        else:
            shape = _source_shape[DType.float64](plan, i)
        var bytes = bitmap_bytes(rows, shape[0] % 8)
        var valid_bytes = bytes if shape[1] else 0
        var bool_bytes = bytes if boolean else 0
        validity.append(valid_bytes)
        bools.append(bool_bytes)
        offsets.append(shape[0])
        has.append(shape[1])
        workspace = checked_add(
            workspace, checked_add(max(1, valid_bytes), max(1, bool_bytes))
        )
        upload = checked_add(
            upload,
            checked_add(
                valid_bytes,
                bool_bytes if boolean else checked_mul(rows, item_bytes),
            ),
        )
    launches = checked_add(
        launches, checked_mul(len(plan.outputs), 2 if plan.reductions else 1)
    )
    # Matrix storage includes input, intermediates and output slots. Count it
    # once as workspace; upload traffic is reported separately by the executor.
    var summary = AccelMemory(0, workspace, 0, workspace, 0, blocks, launches)
    return RowMemory(
        summary^,
        capacity,
        matrix,
        alternate,
        ranks,
        blocks,
        chunk,
        packed,
        validity^,
        bools^,
        offsets^,
        has^,
        upload,
    )
