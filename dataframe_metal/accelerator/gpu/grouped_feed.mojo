"""CPU-order checked reduction feeds over one scheduler-owned row phase.

Input programs contain occurrence-ranked tokens, never ordinary error flags.
A complete operand batch is observed before any row in that batch updates a
state. States and output columns are private until control proves success.
Phase/job order and paired source-error-zero proof belong to the caller.
"""
from max.gpu import global_idx
from std.memory import Pointer
from dataframe.expr import (
    SUM,
    MEAN,
    MIN,
    MAX,
    COUNT,
    NULL_COUNT,
    LEN,
    FIRST,
    LAST,
    ARG_MIN,
    ARG_MAX,
    ANY,
    ALL,
)
from .reductions import State, STATE_WORDS
from .decimal_descriptors import DECIMAL_SUM, DECIMAL_MEAN
from . import ordered_faults
from dataframe_metal.accelerator.memory import checked_add, checked_mul
from .residency import CELL_BYTES

comptime CONTROL_WORDS = 8
comptime INPUT = UInt32(1)
comptime UPDATE = UInt32(2)
comptime FINISH = UInt32(3)
comptime PROTOCOL = UInt32(4)
comptime MERGE = UInt32(5)


@fieldwise_init
struct Resources(Copyable):
    var state_bytes: Int
    var result_bytes: Int
    var control_bytes: Int
    var peak_bytes: Int
    var kernel_launches: Int
    var download_bytes: Int


def resources(
    groups: Int, jobs: Int, batches: Int, result_slots: Int = 1
) raises -> Resources:
    """Additional feed owners, excluding caller-owned inputs/maps/program.

    One control per job stays on device; only the final job0 control is read.
    Result matrices may become the final shared owner after scratch release.
    """
    if (
        groups < 0
        or groups > Int(Int32.MAX)
        or jobs < 1
        or jobs > 65535
        or batches < 0
        or result_slots < 1
        or result_slots > 256
    ):
        raise Error("Mojo grouped feed metadata exceeds supported geometry")
    if groups == 0:
        return Resources(0, 0, 0, 0, 0, 0)
    var states = checked_mul(checked_mul(groups, jobs), STATE_WORDS * 8)
    var result = checked_mul(checked_mul(groups, result_slots), CELL_BYTES)
    var controls = checked_mul(checked_add(jobs, 1), CONTROL_WORDS * 4)
    return Resources(
        states,
        result,
        controls,
        checked_add(checked_add(states, result), controls),
        checked_add(checked_add(jobs, batches), 1 + Int(jobs > 1)),
        CONTROL_WORDS * 4,
    )


def admitted(op: Int64, dtype: Int64, target: Int64) -> Bool:
    """Basic states only; floating sums need their own association protocol."""
    if dtype < 1 or dtype > 12 or target < 1 or target > 12:
        return False
    if op == SUM or op == MEAN:
        return dtype != 1 and dtype != 11 and dtype != 12
    if op == DECIMAL_SUM or op == DECIMAL_MEAN:
        return dtype == 2 or dtype == 3 or dtype == 12
    return (
        op == MIN
        or op == MAX
        or op == COUNT
        or op == NULL_COUNT
        or op == LEN
        or op == FIRST
        or op == LAST
        or op == ARG_MIN
        or op == ARG_MAX
        or op == ANY
        or op == ALL
    )


def _fail(
    control: Pointer[UInt32, MutAnyOrigin],
    category: UInt32,
    row: UInt32,
    rank: UInt32,
    stage: UInt32,
    group: UInt32,
    batch: UInt32,
):
    # [category,row,rank,stage,globalGroup,batch,0,0]. Category is a public
    # raw fault only when stage!=PROTOCOL; it is never a token or OR union.
    control[unsafe_offset=0] = category
    control[unsafe_offset=1] = row
    control[unsafe_offset=2] = rank
    control[unsafe_offset=3] = stage
    control[unsafe_offset=4] = group
    control[unsafe_offset=5] = batch


def initialize(
    states: Pointer[UInt64, MutAnyOrigin],
    result_words: Pointer[UInt64, MutAnyOrigin],
    result_validity: Pointer[UInt8, MutAnyOrigin],
    result_errors: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    active_order: Pointer[UInt32, ImmutAnyOrigin],
    active_count: UInt32,
    total_groups: UInt32,
    result_slots: UInt32,
    target_slot: UInt32,
):
    """Initialize just the groups owned by this phase; retain earlier results."""
    if global_idx.x != 0:
        return
    for word in range(CONTROL_WORDS):
        control[unsafe_offset=word] = 0
    control[unsafe_offset=1] = UInt32.MAX
    control[unsafe_offset=4] = UInt32.MAX
    if target_slot >= result_slots:
        _fail(control, UInt32(1), UInt32.MAX, 0, PROTOCOL, UInt32.MAX, 0)
        return
    var base = Int(target_slot) * Int(total_groups)
    var high_base = Int(result_slots) * Int(total_groups)
    for at in range(Int(active_count)):
        var group = active_order[unsafe_offset=at]
        if group >= total_groups:
            _fail(control, UInt32(1), UInt32.MAX, 0, PROTOCOL, group, 0)
            return
        State().store(states, Int(group) * STATE_WORDS)
        result_words[unsafe_offset=base + Int(group)] = 0
        result_words[unsafe_offset=high_base + base + Int(group)] = 0
        result_validity[unsafe_offset=base + Int(group)] = 0
        result_errors[unsafe_offset=base + Int(group)] = 0


def feed_batch(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    tokens: Pointer[UInt32, ImmutAnyOrigin],
    row_to_group: Pointer[UInt32, ImmutAnyOrigin],
    group_ordinal: Pointer[UInt32, ImmutAnyOrigin],
    states: Pointer[UInt64, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    source_slot: UInt32,
    begin: UInt32,
    end: UInt32,
    stride: UInt32,
    slots: UInt32,
    total_groups: UInt32,
    batch_index: UInt32,
    op: Int64,
    dtype: Int64,
    parameter: Int64,
):
    """One launch per exact CPU _feed batch, in phase-local source order.

    The bounded serial update preserves cross-group first-update fault order.
    Later launches on the same queue do no work after the first failure.
    """
    if global_idx.x != 0 or control[unsafe_offset=0]:
        return
    if (
        begin > end
        or end > stride
        or source_slot >= slots
        or not admitted(op, dtype, dtype)
    ):
        _fail(control, UInt32(1), begin, 0, PROTOCOL, UInt32.MAX, batch_index)
        return
    var base = Int(source_slot) * Int(stride)
    var winner = UInt32(0)
    var winner_row = UInt32.MAX
    # A batch is an independent full AST evaluation. Earlier input faults
    # preempt every update in this same batch, even an update on an earlier row.
    for row in range(Int(begin), Int(end)):
        var token = tokens[unsafe_offset=base + row]
        if token and (token >> 7 == 0 or ordered_faults.category(token) == 0):
            _fail(
                control,
                UInt32(1),
                UInt32(row),
                0,
                PROTOCOL,
                UInt32.MAX,
                batch_index,
            )
            return
        if token and (
            not winner
            or (token >> 7) < (winner >> 7)
            or ((token >> 7) == (winner >> 7) and UInt32(row) < winner_row)
        ):
            winner = token
            winner_row = UInt32(row)
    if winner:
        _fail(
            control,
            ordered_faults.category(winner),
            winner_row,
            ordered_faults.rank(winner),
            INPUT,
            UInt32.MAX,
            batch_index,
        )
        return
    for row in range(Int(begin), Int(end)):
        var group = row_to_group[unsafe_offset=row]
        if group >= total_groups:
            _fail(
                control, UInt32(1), UInt32(row), 0, PROTOCOL, group, batch_index
            )
            return
        var state = State.load(
            states.unsafe_mut_cast[False]().unsafe_origin_cast[
                ImmutAnyOrigin
            ](),
            Int(group) * STATE_WORDS,
        )
        var at = base + row
        state.add(
            words[unsafe_offset=at],
            validity[unsafe_offset=at] != 0,
            UInt32(0),
            UInt64(group_ordinal[unsafe_offset=row]),
            op,
            dtype,
            words[unsafe_offset=Int(stride) * Int(slots) + at] if dtype
            == 12 else UInt64(0),
            parameter,
        )
        state.store(states, Int(group) * STATE_WORDS)
        if state.error:
            _fail(
                control,
                UInt32(state.error),
                UInt32(row),
                0,
                UPDATE,
                group,
                batch_index,
            )
            return


def merge_jobs(
    states: Pointer[UInt64, MutAnyOrigin],
    controls: Pointer[UInt32, MutAnyOrigin],
    merge_order: Pointer[UInt32, ImmutAnyOrigin],
    merge_counts: Pointer[UInt32, ImmutAnyOrigin],
    total_groups: UInt32,
    jobs: UInt32,
    op: Int64,
    dtype: Int64,
    parameter: Int64,
):
    """Match run_jobs error order, then Reducer.merge(worker, group) order.

    Worker states are [jobs,total_groups,STATE_WORDS]; controls are
    [jobs,CONTROL_WORDS]. Merge in place into job0 and return control in job0.
    merge_order is [jobs,total_groups] of canonical global IDs, in each
    worker's own CPU merge order; consume merge_counts[job] entries only.
    Shared-mapping _ReduceJob callers repeat the common order for each job.
    Finish must run only after this pass, not separately for each worker.
    """
    if global_idx.x != 0:
        return
    if jobs == 0 or not admitted(op, dtype, dtype):
        _fail(controls, UInt32(1), UInt32.MAX, 0, PROTOCOL, UInt32.MAX, 0)
        return
    # CPU waits for every job, then scans errors in job order. Even a later
    # worker input fault precedes overflow that would occur only during merge.
    for job in range(Int(jobs)):
        var at = job * CONTROL_WORDS
        if controls[unsafe_offset=at]:
            for word in range(CONTROL_WORDS):
                controls[unsafe_offset=word] = controls[unsafe_offset=at + word]
            controls[unsafe_offset=6] = UInt32(job)
            return
    for job in range(1, Int(jobs)):
        var count = merge_counts[unsafe_offset=job]
        if count > total_groups:
            _fail(controls, UInt32(1), UInt32.MAX, 0, PROTOCOL, UInt32.MAX, 0)
            return
        for at in range(Int(count)):
            var group = merge_order[unsafe_offset=job * Int(total_groups) + at]
            if group >= total_groups:
                _fail(controls, UInt32(1), UInt32.MAX, 0, PROTOCOL, group, 0)
                return
            var target = Int(group) * STATE_WORDS
            var source = (job * Int(total_groups) + Int(group)) * STATE_WORDS
            var state = State.load(
                states.unsafe_mut_cast[False]().unsafe_origin_cast[
                    ImmutAnyOrigin
                ](),
                target,
            )
            state.merge(
                State.load(
                    states.unsafe_mut_cast[False]().unsafe_origin_cast[
                        ImmutAnyOrigin
                    ](),
                    source,
                ),
                op,
                dtype,
                parameter,
            )
            state.store(states, target)
            if state.error:
                _fail(
                    controls,
                    UInt32(state.error),
                    UInt32.MAX,
                    0,
                    MERGE,
                    group,
                    0,
                )
                controls[unsafe_offset=6] = UInt32(job)
                return


def finish(
    states: Pointer[UInt64, ImmutAnyOrigin],
    group_sizes: Pointer[UInt32, ImmutAnyOrigin],
    active_order: Pointer[UInt32, ImmutAnyOrigin],
    result_words: Pointer[UInt64, MutAnyOrigin],
    result_validity: Pointer[UInt8, MutAnyOrigin],
    result_errors: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    active_count: UInt32,
    total_groups: UInt32,
    batches: UInt32,
    op: Int64,
    dtype: Int64,
    target: Int64,
    parameter: Int64,
    result_slots: UInt32,
    target_slot: UInt32,
):
    """Finish after all feeds, in CPU group-ID order, before the next phase."""
    if global_idx.x != 0 or control[unsafe_offset=0]:
        return
    if target_slot >= result_slots or not admitted(op, dtype, target):
        _fail(control, UInt32(1), UInt32.MAX, 0, PROTOCOL, UInt32.MAX, batches)
        return
    var base = Int(target_slot) * Int(total_groups)
    var high_base = Int(result_slots) * Int(total_groups)
    for at in range(Int(active_count)):
        var group = active_order[unsafe_offset=at]
        if group >= total_groups:
            _fail(control, UInt32(1), UInt32.MAX, 0, PROTOCOL, group, batches)
            return
        var state = State.load(states, Int(group) * STATE_WORDS)
        var result = state.finish(
            op,
            dtype,
            target,
            parameter,
            UInt64(group_sizes[unsafe_offset=Int(group)]),
        )
        if result[2]:
            _fail(control, result[2], UInt32.MAX, 0, FINISH, group, batches)
            return
        result_words[unsafe_offset=base + Int(group)] = result[0]
        result_words[
            unsafe_offset=high_base + base + Int(group)
        ] = state.result_high(op, target, result[1])
        result_validity[unsafe_offset=base + Int(group)] = result[1]
        result_errors[unsafe_offset=base + Int(group)] = 0
