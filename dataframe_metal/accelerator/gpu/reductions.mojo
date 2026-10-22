"""Mergeable Mojo GPU reduction states over dataframe words and validity.

Integer sums use paired UInt64 words, exact for every supported row count.
Only final typed sums can overflow. Floating arithmetic uses the shared Mojo
IEEE primitives; extrema retain source words and choose the earliest tie.
"""
from max.gpu import global_idx, block_idx, thread_idx, barrier
from std.memory import Pointer, unsafe_stack_allocation
from dataframe.expr import (
    SUM,
    MEAN,
    MIN,
    MAX,
    COUNT,
    LEN,
    NULL_COUNT,
    FIRST,
    LAST,
    ANY,
    ALL,
    ARG_MIN,
    ARG_MAX,
    VAR,
    STD,
    SKEW,
    KURTOSIS,
    COV,
    CORR,
)
from dataframe_metal.accelerator.rows import (
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_MIDPOINT,
)
from .rows import integer_fits, unsigned_type
from . import floating as f
from . import decimal
from .decimal_descriptors import DECIMAL_SUM, DECIMAL_MEAN
from dataframe_metal.accelerator.decimal_feed import BATCH_ROWS, DESCRIPTOR_FLAG

comptime STATE_WORDS = 8
comptime DESCRIPTOR_WORDS = 5


def _statistical(op: Int64) -> Bool:
    return (
        op == VAR
        or op == STD
        or op == SKEW
        or op == KURTOSIS
        or op == COV
        or op == CORR
    )


def _better(a: UInt64, b: UInt64, op: Int64, dtype: Int64) -> Bool:
    var maximum = op == MAX or op == ARG_MAX
    if dtype == 1 or dtype == 11:
        var x = f.word(a, dtype)
        var y = f.word(b, dtype)
        if op == MIN or op == MAX:
            if f.is_nan(x):
                return maximum and not f.is_nan(y)
            if f.is_nan(y):
                return not maximum
        return f.less(y, x) if maximum else f.less(x, y)
    if unsigned_type(dtype) or dtype == 4:
        return a > b if maximum else a < b
    return (
        a.cast[DType.int64]()
        > b.cast[DType.int64]() if maximum else a.cast[DType.int64]()
        < b.cast[DType.int64]()
    )


struct State(Copyable):
    var word: UInt64
    var high: UInt64
    var count: UInt64
    var index: UInt64
    var present: UInt64
    var flags: UInt64
    var error: UInt64
    var nan_index: UInt64

    def __init__(out self):
        self.word = 0
        self.high = 0
        self.count = 0
        self.index = UInt64.MAX
        self.present = 0
        self.flags = 0
        self.error = 0
        self.nan_index = UInt64.MAX

    def add(
        mut self,
        value: UInt64,
        valid: Bool,
        error: UInt32,
        row: UInt64,
        op: Int64,
        dtype: Int64,
        high: UInt64 = 0,
        parameter: Int64 = 0,
    ):
        var decimal_sum = op == DECIMAL_SUM or op == DECIMAL_MEAN
        var action = (
            SUM if op == DECIMAL_SUM else MEAN if op == DECIMAL_MEAN else op
        )
        self.error |= UInt64(error)
        self.count += UInt64(valid)
        self.flags |= (
            UInt64(1) if value != 0 else UInt64(2)
        ) if valid else UInt64(4)
        if action == FIRST or action == LAST:
            if self.index == UInt64.MAX or (
                row < self.index if action == FIRST else row > self.index
            ):
                self.word = value
                self.high = high if dtype == 12 else UInt64(0)
                self.index = row
                self.present = UInt64(valid)
            return
        if not valid:
            return
        if action == SUM or action == MEAN:
            if dtype == 1 or dtype == 11:
                self.word = f.add(self.word, f.word(value, dtype))
            else:
                var low = self.word + value
                var upper = high if dtype == 12 else (
                    UInt64.MAX if not unsigned_type(dtype)
                    and dtype != 4
                    and value >> 63 else UInt64(0)
                )
                self.high += upper + UInt64(low < self.word)
                self.word = low
                if decimal_sum:
                    var precision = UInt32((UInt64(parameter) >> 32) & 63)
                    if not decimal.cast(
                        decimal.Wide(self.word, self.high),
                        0,
                        0,
                        precision,
                        128,
                        False,
                        False,
                    ).valid:
                        self.error |= 8
        elif (
            action == MIN
            or action == MAX
            or action == ARG_MIN
            or action == ARG_MAX
        ):
            if (dtype == 1 or dtype == 11) and f.is_nan(f.word(value, dtype)):
                self.nan_index = min(self.nan_index, row)
                if action == ARG_MIN or action == ARG_MAX:
                    return
            var better = decimal.signed_less(
                decimal.Wide(self.word, self.high), decimal.Wide(value, high)
            ) if action == MAX or action == ARG_MAX else decimal.signed_less(
                decimal.Wide(value, high), decimal.Wide(self.word, self.high)
            )
            if self.index == UInt64.MAX or (
                better if dtype
                == 12 else _better(value, self.word, action, dtype)
            ):
                self.word = value
                self.high = high if dtype == 12 else UInt64(0)
                self.index = row
                self.present = 1

    def merge(
        mut self, other: Self, op: Int64, dtype: Int64, parameter: Int64 = 0
    ):
        var decimal_sum = op == DECIMAL_SUM or op == DECIMAL_MEAN
        var action = (
            SUM if op == DECIMAL_SUM else MEAN if op == DECIMAL_MEAN else op
        )
        self.flags |= other.flags
        self.error |= other.error
        self.count += other.count
        self.nan_index = min(self.nan_index, other.nan_index)
        if action == SUM or action == MEAN:
            if dtype == 1 or dtype == 11:
                if other.count:
                    self.word = f.add(self.word, other.word)
            else:
                var low = self.word + other.word
                self.high += other.high + UInt64(low < self.word)
                self.word = low
                if decimal_sum and other.count:
                    var precision = UInt32((UInt64(parameter) >> 32) & 63)
                    if not decimal.cast(
                        decimal.Wide(self.word, self.high),
                        0,
                        0,
                        precision,
                        128,
                        False,
                        False,
                    ).valid:
                        self.error |= 8
        elif action == FIRST or action == LAST:
            if other.index != UInt64.MAX and (
                self.index == UInt64.MAX
                or (
                    other.index
                    < self.index if op
                    == FIRST else other.index
                    > self.index
                )
            ):
                self.word = other.word
                self.high = other.high
                self.index = other.index
                self.present = other.present
        elif (
            action == MIN
            or action == MAX
            or action == ARG_MIN
            or action == ARG_MAX
        ):
            if other.index == UInt64.MAX:
                return
            var maximum = action == MAX or action == ARG_MAX
            var better = decimal.signed_less(
                decimal.Wide(self.word, self.high),
                decimal.Wide(other.word, other.high),
            ) if maximum else decimal.signed_less(
                decimal.Wide(other.word, other.high),
                decimal.Wide(self.word, self.high),
            )
            var take = self.index == UInt64.MAX or (
                better if dtype
                == 12 else _better(other.word, self.word, action, dtype)
            )
            var equal = (
                self.word == other.word
                and self.high
                == other.high if dtype
                == 12 else not _better(self.word, other.word, action, dtype)
            )
            if not take and equal:
                take = other.index < self.index
            if take:
                self.word = other.word
                self.high = other.high
                self.index = other.index
                self.present = other.present

    def finish(
        self,
        op: Int64,
        dtype: Int64,
        target: Int64,
        minimum: Int64,
        rows: UInt64,
    ) -> Tuple[UInt64, UInt8, UInt32]:
        var value = self.word
        var valid = True
        var error = UInt32(self.error)
        if op == DECIMAL_SUM:
            valid = self.count >= (UInt64(minimum) & 0xFFFFFFFF)
        elif op == DECIMAL_MEAN:
            valid = self.count != 0
            if valid:
                value = decimal.mean128(
                    self.word,
                    self.high,
                    self.count,
                    UInt32((UInt64(minimum) >> 38) & 63),
                )
        elif op == SUM:
            valid = self.count >= UInt64(minimum)
            if dtype == 1:
                value = UInt64(f.to_float32(value))
            elif dtype != 11 and valid:
                var upper = (
                    UInt64(0) if unsigned_type(target)
                    or value >> 63 == 0 else UInt64.MAX
                )
                if self.high != upper or not integer_fits(value, target):
                    error |= 2
        elif op == MEAN:
            valid = self.count != 0
            if valid:
                if dtype != 1 and dtype != 11:
                    value = f.from_wide(value, self.high, unsigned_type(dtype))
                value = f.div(value, f.from_unsigned(self.count))
        elif op == COUNT:
            value = self.count
        elif op == NULL_COUNT:
            value = rows - self.count
        elif op == LEN:
            value = rows
        elif op == FIRST or op == LAST:
            valid = self.index != UInt64.MAX and self.present != 0
        elif op == MIN or op == MAX:
            valid = self.index != UInt64.MAX
        elif op == ARG_MIN or op == ARG_MAX:
            value = self.nan_index if self.index == UInt64.MAX else self.index
            valid = value != UInt64.MAX
        elif op == ANY or op == ALL:
            var yes = (self.flags & 1) != 0
            var no = (self.flags & 2) != 0
            var unknown = (self.flags & 4) != 0
            value = UInt64(yes if op == ANY else not no)
            valid = minimum != 0 or not unknown or (yes if op == ANY else no)
        return (value if valid else UInt64(0), UInt8(valid), error)

    def result_high(self, op: Int64, target: Int64, valid: UInt8) -> UInt64:
        return (
            self.high if target == 12
            and valid != 0
            and op != DECIMAL_MEAN else UInt64(0)
        )

    def store(self, output: Pointer[UInt64, MutAnyOrigin], at: Int):
        output[unsafe_offset=at] = self.word
        output[unsafe_offset=at + 1] = self.high
        output[unsafe_offset=at + 2] = self.count
        output[unsafe_offset=at + 3] = self.index
        output[unsafe_offset=at + 4] = self.present
        output[unsafe_offset=at + 5] = self.flags
        output[unsafe_offset=at + 6] = self.error
        output[unsafe_offset=at + 7] = self.nan_index

    @staticmethod
    def load(input: Pointer[UInt64, ImmutAnyOrigin], at: Int) -> Self:
        var state = Self()
        state.word = input[unsafe_offset=at]
        state.high = input[unsafe_offset=at + 1]
        state.count = input[unsafe_offset=at + 2]
        state.index = input[unsafe_offset=at + 3]
        state.present = input[unsafe_offset=at + 4]
        state.flags = input[unsafe_offset=at + 5]
        state.error = input[unsafe_offset=at + 6]
        state.nan_index = input[unsafe_offset=at + 7]
        return state^


def _decimal_feed_worker(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    slot: Int,
    stride: Int,
    slots: Int,
    start: Int,
    end: Int,
    op: Int64,
    dtype: Int64,
    parameter: Int64,
) -> State:
    var partial = State()
    for offset in range(start, end, BATCH_ROWS):
        var stop = min(end, offset + BATCH_ROWS)
        # CPU _feed evaluates the whole operand batch before reducer.update.
        # The certificate permits exactly one fallible arithmetic occurrence,
        # so the first failing row is an unambiguous operand observation.
        for row in range(offset, stop):
            var error = errors[unsafe_offset=slot * stride + row]
            if error:
                partial.error = UInt64(error)
                return partial^
        for row in range(offset, stop):
            var at = slot * stride + row
            partial.add(
                words[unsafe_offset=at],
                validity[unsafe_offset=at] != 0,
                0,
                UInt64(row),
                op,
                dtype,
                words[unsafe_offset=stride * slots + at] if dtype
                == 12 else UInt64(0),
                parameter,
            )
            if partial.error:
                return partial^
    return partial^


def reduce_partials(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    partials: Pointer[UInt64, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
    floating_tree: UInt32 = 0,
):
    var work = global_idx.x
    if work >= Int(outputs) * Int(groups):
        return
    var column = work // Int(groups)
    var group = work % Int(groups)
    var slot = Int(descriptors[unsafe_offset=column * DESCRIPTOR_WORDS])
    var dtype = descriptors[unsafe_offset=column * DESCRIPTOR_WORDS + 1]
    var op = descriptors[unsafe_offset=column * DESCRIPTOR_WORDS + 3]
    if _statistical(op) or (
        op >= ROW_QUANTILE_LINEAR and op <= ROW_QUANTILE_MIDPOINT
    ):
        return
    if (
        floating_tree
        and (op == SUM or op == MEAN)
        and (dtype == 1 or dtype == 11)
    ):
        return
    var state = State()
    var parameter = descriptors[unsafe_offset=column * DESCRIPTOR_WORDS + 4]
    var decimal_sum = op == DECIMAL_SUM or op == DECIMAL_MEAN
    if decimal_sum:
        if group == 0:
            var workers = max(
                1,
                min(Int((UInt64(parameter) >> 44) & 65535), Int(rows) // 65536),
            )
            var step = (
                ((Int(rows) + workers - 1) // workers + 8191) // 8192 * 8192
            )
            if UInt64(parameter) & DESCRIPTOR_FLAG:
                # CPU joins every worker, observing worker failures in order,
                # before it merges any successful partials. A separate replay
                # pass preserves that precedence without another allocation.
                for worker in range(workers):
                    var partial = _decimal_feed_worker(
                        words,
                        validity,
                        errors,
                        slot,
                        Int(stride),
                        Int(slots),
                        min(Int(rows), worker * step),
                        min(Int(rows), (worker + 1) * step),
                        op,
                        dtype,
                        parameter,
                    )
                    if partial.error:
                        partial.store(partials, work * STATE_WORDS)
                        return
                for worker in range(workers):
                    var partial = _decimal_feed_worker(
                        words,
                        validity,
                        errors,
                        slot,
                        Int(stride),
                        Int(slots),
                        min(Int(rows), worker * step),
                        min(Int(rows), (worker + 1) * step),
                        op,
                        dtype,
                        parameter,
                    )
                    state.merge(partial, op, dtype, parameter)
                    if state.error:
                        state.store(partials, work * STATE_WORDS)
                        return
                state.store(partials, work * STATE_WORDS)
                return
            for worker in range(workers):
                var partial = State()
                for row in range(
                    min(Int(rows), worker * step),
                    min(Int(rows), (worker + 1) * step),
                ):
                    var at = slot * Int(stride) + row
                    partial.add(
                        words[unsafe_offset=at],
                        validity[unsafe_offset=at] != 0,
                        errors[unsafe_offset=at],
                        UInt64(row),
                        op,
                        dtype,
                        words[
                            unsafe_offset=Int(stride) * Int(slots) + at
                        ] if dtype
                        == 12 else UInt64(0),
                        parameter,
                    )
                state.merge(partial, op, dtype, parameter)
        state.store(partials, work * STATE_WORDS)
        return
    for row in range(group, Int(rows), Int(groups)):
        var at = slot * Int(stride) + row
        state.add(
            words[unsafe_offset=at],
            validity[unsafe_offset=at] != 0,
            errors[unsafe_offset=at],
            UInt64(row),
            op,
            dtype,
            words[unsafe_offset=Int(stride) * Int(slots) + at] if dtype
            == 12 else UInt64(0),
        )
    state.store(partials, work * STATE_WORDS)


def reduce_float_partials(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    partials: Pointer[UInt64, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    """Preserve native scalar SUM/MEAN's 256-lane floating association."""
    var work = block_idx.x
    var column = work // Int(groups)
    if column >= Int(outputs):
        return
    var base = column * DESCRIPTOR_WORDS
    var dtype = descriptors[unsafe_offset=base + 1]
    var op = descriptors[unsafe_offset=base + 3]
    # The condition is uniform across the block: every participating lane
    # reaches every shared-memory barrier below.
    if (op != SUM and op != MEAN) or (dtype != 1 and dtype != 11):
        return
    var group = work % Int(groups)
    var lane = thread_idx.x
    var slot = Int(descriptors[unsafe_offset=base])
    var state = State()
    for row in range(group * 256 + lane, Int(rows), Int(groups) * 256):
        var at = slot * Int(stride) + row
        state.add(
            words[unsafe_offset=at],
            validity[unsafe_offset=at] != 0,
            errors[unsafe_offset=at],
            UInt64(row),
            op,
            dtype,
        )
    var sums = unsafe_stack_allocation[
        256, DType.uint64, address_space=AddressSpace.SHARED
    ]()
    var counts = unsafe_stack_allocation[
        256, DType.uint64, address_space=AddressSpace.SHARED
    ]()
    var failures = unsafe_stack_allocation[
        256, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    sums[unsafe_offset=lane] = state.word
    counts[unsafe_offset=lane] = state.count
    failures[unsafe_offset=lane] = UInt32(state.error)
    barrier()
    var distance = 128
    while distance:
        if lane < distance:
            sums[unsafe_offset=lane] = f.add(
                sums[unsafe_offset=lane], sums[unsafe_offset=lane + distance]
            )
            counts[unsafe_offset=lane] += counts[unsafe_offset=lane + distance]
            failures[unsafe_offset=lane] |= failures[
                unsafe_offset=lane + distance
            ]
        barrier()
        distance //= 2
    if lane == 0:
        state.word = sums[unsafe_offset=0]
        state.count = counts[unsafe_offset=0]
        state.error = UInt64(failures[unsafe_offset=0])
        state.store(partials, work * STATE_WORDS)


def reduce_finish(
    partials: Pointer[UInt64, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    paired: UInt32 = 0,
):
    var column = global_idx.x
    if column >= Int(outputs):
        return
    var base = column * DESCRIPTOR_WORDS
    var dtype = descriptors[unsafe_offset=base + 1]
    var target = descriptors[unsafe_offset=base + 2]
    var op = descriptors[unsafe_offset=base + 3]
    var minimum = descriptors[unsafe_offset=base + 4]
    if _statistical(op) or (
        op >= ROW_QUANTILE_LINEAR and op <= ROW_QUANTILE_MIDPOINT
    ):
        return
    var state = State()
    for group in range(Int(groups)):
        state.merge(
            State.load(partials, (column * Int(groups) + group) * STATE_WORDS),
            op,
            dtype,
            minimum,
        )
    var result = state.finish(op, dtype, target, minimum, UInt64(rows))
    words[unsafe_offset=column] = result[0]
    if paired:
        words[unsafe_offset=Int(outputs) + column] = state.result_high(
            op, target, result[1]
        )
    validity[unsafe_offset=column] = result[1]
    errors[unsafe_offset=column] = result[2]
