"""Mergeable Mojo GPU reduction states over dataframe words and validity.

Integer sums use paired UInt64 words, exact for every supported row count.
Only final typed sums can overflow. Floating arithmetic uses the shared Mojo
IEEE primitives; extrema retain source words and choose the earliest tie.
"""
from max.gpu import global_idx
from std.memory import Pointer
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
from dataframe.accelerator.rows import (
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_MIDPOINT,
)
from .rows import integer_fits, unsigned_type
from . import floating as f

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
    ):
        self.error |= UInt64(error)
        self.count += UInt64(valid)
        self.flags |= (
            UInt64(1) if value != 0 else UInt64(2)
        ) if valid else UInt64(4)
        if op == FIRST or op == LAST:
            if self.index == UInt64.MAX or (
                row < self.index if op == FIRST else row > self.index
            ):
                self.word = value
                self.index = row
                self.present = UInt64(valid)
            return
        if not valid:
            return
        if op == SUM or op == MEAN:
            if dtype == 1 or dtype == 11:
                self.word = f.add(self.word, f.word(value, dtype))
            else:
                var low = self.word + value
                var high = (
                    UInt64.MAX if not unsigned_type(dtype)
                    and dtype != 4
                    and value >> 63 else UInt64(0)
                )
                self.high += high + UInt64(low < self.word)
                self.word = low
        elif op == MIN or op == MAX or op == ARG_MIN or op == ARG_MAX:
            if (dtype == 1 or dtype == 11) and f.is_nan(f.word(value, dtype)):
                self.nan_index = min(self.nan_index, row)
                if op == ARG_MIN or op == ARG_MAX:
                    return
            if self.index == UInt64.MAX or _better(value, self.word, op, dtype):
                self.word = value
                self.index = row
                self.present = 1

    def merge(mut self, other: Self, op: Int64, dtype: Int64):
        self.flags |= other.flags
        self.error |= other.error
        self.count += other.count
        self.nan_index = min(self.nan_index, other.nan_index)
        if op == SUM or op == MEAN:
            if dtype == 1 or dtype == 11:
                if other.count:
                    self.word = f.add(self.word, other.word)
            else:
                var low = self.word + other.word
                self.high += other.high + UInt64(low < self.word)
                self.word = low
        elif op == FIRST or op == LAST:
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
                self.index = other.index
                self.present = other.present
        elif op == MIN or op == MAX or op == ARG_MIN or op == ARG_MAX:
            if other.index == UInt64.MAX:
                return
            var take = self.index == UInt64.MAX or _better(
                other.word, self.word, op, dtype
            )
            if not take and not _better(self.word, other.word, op, dtype):
                take = other.index < self.index
            if take:
                self.word = other.word
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
        if op == SUM:
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
    var state = State()
    for row in range(group, Int(rows), Int(groups)):
        var at = slot * Int(stride) + row
        state.add(
            words[unsafe_offset=at],
            validity[unsafe_offset=at] != 0,
            errors[unsafe_offset=at],
            UInt64(row),
            op,
            dtype,
        )
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
        )
    var result = state.finish(op, dtype, target, minimum, UInt64(rows))
    words[unsafe_offset=column] = result[0]
    validity[unsafe_offset=column] = result[1]
    errors[unsafe_offset=column] = result[2]
