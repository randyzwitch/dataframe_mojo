"""Private occurrence-ranked row matrices with caller-controlled observation.

There is no Table conversion and no automatic output-error observer. A caller
must certify original AST/provenance and its exact CPU phase order, then consume
these tokens with that phase's observer before publishing separate results.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import Pointer
from dataframe_metal.accelerator.rows import RowPlan
from dataframe.expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    IS_NULL,
    IS_NOT_NULL,
    WHEN,
    AND,
    OR,
    XOR,
    NOT,
)
from .program import Program
from .residency import Scope, Table, CELL_BYTES, checked_add, checked_mul
from .rows import row_program_ordered


@fieldwise_init
struct Resources(Copyable):
    var matrix_bytes: Int
    var program_bytes: Int
    var control_bytes: Int
    var descriptor_bytes: Int
    var peak_bytes: Int
    var upload_bytes: Int
    var download_bytes: Int
    var kernel_launches: Int
    var synchronizations: Int
    var partial_count: Int


def _validate(plan: RowPlan, rows: Int) raises:
    if (
        rows < 0
        or rows > Int(Int32.MAX)
        or len(plan.steps) != 1
        or len(plan.outputs) != 1
        or plan.reductions
        or plan.limit >= 0
        or plan.steps[0].filter
        or plan.steps[0].order
        or plan.steps[0].grouped
        or plan.steps[0].start != 0
        or plan.steps[0].nodes <= 0
        or plan.steps[0].nodes > 128
        or plan.source.width() <= 0
        or plan.slots < plan.source.width()
        or plan.slots > 256
        or plan.outputs[0].slot < 0
        or plan.outputs[0].slot >= plan.slots
        or len(plan.ordered_ranks[]) != len(plan.literals)
        or len(plan.node_dtypes) != len(plan.literals)
        or len(plan.code) != 4 * len(plan.literals)
        or len(plan.key_values)
        or len(plan.zone_words)
    ):
        raise Error(
            "Private ranked rows require a certified one-step ROWS program"
        )
    # Canonical occurrence programs increase positive ranks in code order;
    # generated infallible wrappers and non-executable auxiliary words are0.
    var previous = UInt32(0)
    for index in range(plan.steps[0].nodes):
        var rank = plan.ordered_ranks[][index]
        if not rank and Int(plan.code[4 * index]) not in List[Int](
            [
                COL,
                LIT_INT,
                LIT_FLOAT,
                LIT_BOOL,
                LIT_NULL,
                IS_NULL,
                IS_NOT_NULL,
                WHEN,
                AND,
                OR,
                XOR,
                NOT,
            ]
        ):
            raise Error(
                "Private ranked row unranked executable helper is not certified infallible"
            )
        if rank and (rank <= previous or rank > 128):
            raise Error("Private ranked row occurrence order changed")
        if rank:
            previous = rank
    if not previous:
        raise Error("Private ranked row occurrence certificate is empty")
    for index in range(plan.steps[0].nodes, len(plan.ordered_ranks[])):
        if plan.ordered_ranks[][index]:
            raise Error(
                "Private ranked row auxiliary words have occurrence ranks"
            )


def estimate(
    plan: RowPlan, rows: Int, host_input: Bool = False
) raises -> Resources:
    """Full additional peak above existing Scope owners, including copy scratch.

    Host/device copy descriptors are48bytes/input column; only24bytes/column
    are uploaded. Program counts include both live host and device owners.
    The source-zero proof has32bytes on device and32bytes on the host. Rows0
    is structural and callers skip evaluation without initializing a context.
    """
    _validate(plan, rows)
    if not rows:
        return Resources(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var program = Program(plan)
    var upload = checked_add(
        checked_add(
            checked_mul(len(program.code), 8),
            checked_mul(len(program.literals), 8),
        ),
        checked_add(
            checked_mul(len(program.node_types), 8),
            checked_mul(len(program.ordered_ranks[]), 4),
        ),
    )
    var cells = checked_mul(rows, plan.slots)
    var inputs = checked_mul(rows, plan.source.width())
    var partials = min(1024, (inputs + 255) // 256)
    var matrix = checked_mul(cells, CELL_BYTES)
    var program_bytes = checked_mul(upload, 2)
    var controls = checked_add(64, checked_mul(partials, 4))
    var descriptors = checked_mul(plan.source.width(), 48)
    var uploaded = checked_add(upload, checked_mul(plan.source.width(), 24))
    if host_input:
        uploaded = checked_add(uploaded, checked_mul(inputs, CELL_BYTES))
    return Resources(
        matrix,
        program_bytes,
        controls,
        descriptors,
        checked_add(
            checked_add(matrix, program_bytes),
            checked_add(controls, descriptors),
        ),
        uploaded,
        32,
        4,
        3,
        partials,
    )


struct _Reservation(Movable):
    var scope: Scope
    var bytes: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.bytes = bytes

    def release(mut self, bytes: Int):
        self.scope._release(bytes)
        self.bytes -= bytes

    def __deinit__(deinit self):
        self.scope._release(self.bytes)


struct RankedRows(Movable):
    var scope: Scope
    var rows: Int
    var slots: Int
    var target: Int
    var cells: Int
    var words: DeviceBuffer[DType.uint64]
    var validity: DeviceBuffer[DType.uint8]
    var tokens: DeviceBuffer[DType.uint32]
    var retained: Int

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        slots: Int,
        target: Int,
        var words: DeviceBuffer[DType.uint64],
        var validity: DeviceBuffer[DType.uint8],
        var tokens: DeviceBuffer[DType.uint32],
    ) raises:
        var cells = checked_mul(rows, slots)
        if (
            rows <= 0
            or slots <= 0
            or target < 0
            or target >= slots
            or (
                len(words) != checked_mul(cells, 2)
                or len(validity) != cells
                or len(tokens) != cells
            )
        ):
            raise Error("Private ranked row matrix geometry is invalid")
        var bytes = checked_mul(cells, CELL_BYTES)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.slots = slots
        self.target = target
        self.cells = cells
        self.words = words^
        self.validity = validity^
        self.tokens = tokens^
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def _zero_partials(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    cells: UInt64,
    groups: UInt32,
):
    var group = global_idx.x
    if group >= Int(groups):
        return
    var width = (cells + UInt64(groups) - 1) // UInt64(groups)
    var begin = UInt64(group) * width
    var end = min(cells, begin + width)
    var failure = UInt32(0)
    for cell in range(Int(begin), Int(end)):
        failure |= errors[unsafe_offset=cell]
    partials[unsafe_offset=group] = failure


def _zero_finish(
    partials: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
):
    if global_idx.x != 0:
        return
    for at in range(8):
        control[unsafe_offset=at] = 0
    for group in range(Int(groups)):
        control[unsafe_offset=0] |= partials[unsafe_offset=group]


def evaluate(plan: RowPlan, source: Table, scope: Scope) raises -> RankedRows:
    """Execute a caller-certified operand without observing output tokens.

    This is an internal device primitive, not an admission/provenance grant.
    The caller validates its original expression certificate after ALL rewrites
    and must independently specify the CPU batch/run/state/control schedule.
    Empty operands are handled structurally by the caller before this function.
    """
    _validate(plan, source.rows)
    if (
        not source.rows
        or not source.storage
        or not source.matrix()[].scope.same_query(scope)
    ):
        raise Error(
            "Private ranked rows require a nonempty same-query device source"
        )
    var resources = estimate(plan, source.rows)
    var mapping = List[Int64](capacity=plan.source.width())
    for field in plan.source._columns:
        var found = -1
        for index in range(source.schema.width()):
            ref actual = source.schema._columns[index]
            if (
                actual.name() == field.name()
                and actual.dtype() == field.dtype()
            ):
                found = index
                break
        if found < 0:
            raise Error("Private ranked row source schema changed")
        mapping.append(Int64(found))
    _ = scope.check(resources.peak_bytes)
    # copy_to checks/owns descriptor scratch separately while these live owners
    # are charged, so exclude descriptors from this persistent reservation.
    var reservation = _Reservation(
        scope, resources.peak_bytes - resources.descriptor_bytes
    )
    var program = Program(plan)
    var rows = source.rows
    var cells = checked_mul(rows, plan.slots)
    var input_cells = checked_mul(rows, plan.source.width())
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var tokens = ctx.enqueue_create_buffer[DType.uint32](cells)
    var code = ctx.enqueue_create_buffer[DType.int64](len(program.code))
    var literals = ctx.enqueue_create_buffer[DType.uint64](
        len(program.literals)
    )
    var types = ctx.enqueue_create_buffer[DType.int64](len(program.node_types))
    var ranks = ctx.enqueue_create_buffer[DType.uint32](
        len(program.ordered_ranks[])
    )
    var partials = ctx.enqueue_create_buffer[DType.uint32](
        resources.partial_count
    )
    var control = ctx.enqueue_create_buffer[DType.uint32](8)
    var host = List[UInt32](length=8, fill=0)
    var held_source = source.copy()
    try:
        _ = source.copy_to(
            scope, words, validity, tokens, rows, plan.slots, mapping
        )
        ctx.enqueue_function[_zero_partials](
            tokens.unsafe_ptr(),
            partials.unsafe_ptr(),
            UInt64(input_cells),
            UInt32(resources.partial_count),
            grid_dim=(resources.partial_count + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[_zero_finish](
            partials.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(resources.partial_count),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(2)
        scope.read_control_into(control, host)
        scope.record_download(32)
        if host[0]:
            raise Error(
                "Private ranked rows require already observed successful inputs"
            )
        ctx.enqueue_copy(code, program.code.unsafe_ptr())
        ctx.enqueue_copy(literals, program.literals.unsafe_ptr())
        ctx.enqueue_copy(types, program.node_types.unsafe_ptr())
        ctx.enqueue_copy(ranks, program.ordered_ranks[].unsafe_ptr())
        scope.record_upload(resources.program_bytes // 2)
        ctx.enqueue_function[row_program_ordered](
            words.unsafe_ptr(),
            validity.unsafe_ptr(),
            tokens.unsafe_ptr(),
            code.unsafe_ptr(),
            literals.unsafe_ptr(),
            types.unsafe_ptr(),
            ranks.unsafe_ptr(),
            UInt32(plan.steps[0].start),
            UInt32(plan.steps[0].nodes),
            UInt32(plan.steps[0].slot),
            UInt32(rows),
            UInt32(rows),
            UInt32(plan.slots),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(1)
        scope.synchronize()
    except error:
        scope.synchronize()
        _ = program^
        _ = words^
        _ = validity^
        _ = tokens^
        _ = code^
        _ = literals^
        _ = types^
        _ = ranks^
        _ = partials^
        _ = control^
        _ = host^
        _ = held_source^
        raise error^
    _ = program^
    _ = code^
    _ = literals^
    _ = types^
    _ = ranks^
    _ = partials^
    _ = control^
    _ = host^
    _ = held_source^
    reservation.release(resources.matrix_bytes)
    return RankedRows(
        scope,
        rows,
        plan.slots,
        plan.outputs[0].slot,
        words^,
        validity^,
        tokens^,
    )
