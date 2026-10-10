"""Requested allocation sizes for the bounded resident row executor."""
from .rows import RowPlan
from dataframe.dtype import DataType
from dataframe.expr import LEN

comptime ACCEL_THREADS = 256
comptime ACCEL_MAX_BLOCKS = 1024


@fieldwise_init
struct MemoryEstimate(Copyable):
    var input_bytes: Int
    var workspace_bytes: Int
    var output_bytes: Int
    var peak_bytes: Int
    var download_bytes: Int
    var blocks: Int
    var launches: Int

    def require_budget(self, budget: Int) raises:
        if budget < 0:
            raise Error("Accelerator memory budget must be nonnegative")
        if self.peak_bytes > budget:
            raise Error(
                "Accelerator memory preflight rejected: estimated payload "
                + String(self.peak_bytes)
                + " bytes exceeds budget "
                + String(budget)
                + " bytes (input="
                + String(self.input_bytes)
                + ", workspace="
                + String(self.workspace_bytes)
                + ", output="
                + String(self.output_bytes)
                + ")"
            )

    def describe(self) -> String:
        return (
            "input="
            + String(self.input_bytes)
            + " B; workspace="
            + String(self.workspace_bytes)
            + " B; output="
            + String(self.output_bytes)
            + " B; peak_requested="
            + String(self.peak_bytes)
            + " B; download="
            + String(self.download_bytes)
            + " B; kernels="
            + String(self.launches)
        )


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
    var summary: MemoryEstimate
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


def row_memory(
    plan: RowPlan,
    *,
    threads: Int = ACCEL_THREADS,
    max_blocks: Int = ACCEL_MAX_BLOCKS,
    wide_integer: Bool = True,
) raises -> RowMemory:
    if threads < 1 or max_blocks < 1:
        raise Error("Accelerator launch dimensions must be positive")
    var rows = plan.source.height()
    var integer = plan.dtype.is_integer()
    var accumulator_bytes = 16 if integer and wide_integer else 8
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
            max_blocks,
            rows // threads + Int(rows % threads != 0),
        ),
    )
    var chunk = checked_mul(
        capacity // (blocks * threads)
        + Int(capacity % (blocks * threads) != 0),
        threads,
    )
    var packed = max(1, bitmap_bytes(rows, 0))
    var item_bytes = (
        4 if plan.dtype == DataType.FLOAT32
        or plan.dtype == DataType.INT32 else 8
    )
    var workspace = checked_mul(checked_add(matrix, alternate), item_bytes + 1)
    workspace = checked_add(workspace, checked_mul(ranks, 8))
    workspace = checked_add(
        workspace, checked_mul(blocks, accumulator_bytes + 8)
    )
    # Row metadata (including the integer error position) and scalar outputs.
    workspace = checked_add(
        workspace, (24 if integer else 16) + accumulator_bytes + 8
    )
    if integer:
        workspace = checked_add(workspace, checked_mul(capacity, 8))
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
    upload = checked_add(upload, 24 if integer else 16)
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
        elif plan.dtype == DataType.INT32:
            shape = _source_shape[DType.int32](plan, i)
        elif plan.dtype == DataType.INT64:
            shape = _source_shape[DType.int64](plan, i)
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
    for output in plan.outputs:
        if not plan.reductions or output.reduction != LEN:
            launches = checked_add(launches, 2 if plan.reductions else 1)
    if integer:
        launches = checked_add(launches, 1)  # checked-error reduction
    # Matrix storage includes input, intermediates and output slots. Count it
    # once as workspace; upload traffic is reported separately by the executor.
    var summary = MemoryEstimate(
        0, workspace, 0, workspace, 0, blocks, launches
    )
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
