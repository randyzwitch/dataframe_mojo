"""GPU statistical reductions sharing the basic reduction buffer layouts.

Each kernel ignores descriptors for other operations, allowing mixed scalar
selections to use the same partial and result buffers. Inputs are Float64
words, as established by the shared row-plan lowering.
"""

from max.gpu import global_idx
from std.memory import Pointer
from dataframe.expr import VAR, STD, SKEW, KURTOSIS, COV, CORR
from .moments import VarianceState, CentralMomentState, CovarianceState

comptime STATE_WORDS = 8
comptime DESCRIPTOR_WORDS = 5


def supported(op: Int64) -> Bool:
    return (
        op == VAR
        or op == STD
        or op == SKEW
        or op == KURTOSIS
        or op == COV
        or op == CORR
    )


struct State(Copyable):
    var count: UInt64
    var a: UInt64
    var b: UInt64
    var c: UInt64
    var d: UInt64
    var e: UInt64
    var error: UInt64

    def __init__(out self):
        self.count = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.error = 0

    def add(
        mut self, x: UInt64, y: UInt64, valid: Bool, error: UInt32, op: Int64
    ):
        self.error |= UInt64(error)
        if not valid:
            return
        if op == VAR or op == STD:
            var state = VarianceState(self.count, self.a, self.b)
            state.add(x, retain_nan_accumulator=True)
            self.count = state.count
            self.a = state.mean
            self.b = state.m2
        elif op == SKEW or op == KURTOSIS:
            var state = CentralMomentState(
                self.count, self.a, self.b, self.c, self.d
            )
            state.add(x)
            self.count = state.count
            self.a = state.mean
            self.b = state.m2
            self.c = state.m3
            self.d = state.m4
        else:
            var state = CovarianceState(
                self.count, self.a, self.b, self.c, self.d, self.e
            )
            state.add(x, y, retain_nan_accumulator=True)
            self.count = state.count
            self.a = state.mean_x
            self.b = state.mean_y
            self.c = state.cxy
            self.d = state.m2x
            self.e = state.m2y

    def merge(mut self, other: Self, op: Int64):
        self.error |= other.error
        if op == VAR or op == STD:
            var state = VarianceState(self.count, self.a, self.b)
            state.merge(VarianceState(other.count, other.a, other.b))
            self.count = state.count
            self.a = state.mean
            self.b = state.m2
        elif op == SKEW or op == KURTOSIS:
            var state = CentralMomentState(
                self.count, self.a, self.b, self.c, self.d
            )
            state.merge(
                CentralMomentState(
                    other.count, other.a, other.b, other.c, other.d
                )
            )
            self.count = state.count
            self.a = state.mean
            self.b = state.m2
            self.c = state.m3
            self.d = state.m4
        else:
            var state = CovarianceState(
                self.count, self.a, self.b, self.c, self.d, self.e
            )
            state.merge(
                CovarianceState(
                    other.count, other.a, other.b, other.c, other.d, other.e
                )
            )
            self.count = state.count
            self.a = state.mean_x
            self.b = state.mean_y
            self.c = state.cxy
            self.d = state.m2x
            self.e = state.m2y

    def finish(self, op: Int64, minimum: Int64) -> Tuple[UInt64, UInt8, UInt32]:
        # The upper bits contain workers and the right input slot, not ddof.
        var parameter = minimum.cast[DType.uint64]() & 0x7FFFFFFF
        var valid: Bool
        var value = UInt64(0)
        if op == VAR or op == STD:
            var state = VarianceState(self.count, self.a, self.b)
            valid = state.valid(parameter)
            if valid:
                value = state.finish(parameter, op == STD)
        elif op == SKEW or op == KURTOSIS:
            var state = CentralMomentState(
                self.count, self.a, self.b, self.c, self.d
            )
            valid = state.valid(op == KURTOSIS, parameter)
            if valid:
                value = state.finish(op == KURTOSIS, parameter)
        else:
            var state = CovarianceState(
                self.count, self.a, self.b, self.c, self.d, self.e
            )
            valid = state.valid(op == CORR)
            if valid:
                value = state.finish(parameter, op == CORR)
        return (value, UInt8(valid), UInt32(self.error))

    def store(self, output: Pointer[UInt64, MutAnyOrigin], at: Int):
        output[unsafe_offset=at] = self.count
        output[unsafe_offset=at + 1] = self.a
        output[unsafe_offset=at + 2] = self.b
        output[unsafe_offset=at + 3] = self.c
        output[unsafe_offset=at + 4] = self.d
        output[unsafe_offset=at + 5] = self.e
        output[unsafe_offset=at + 6] = self.error
        output[unsafe_offset=at + 7] = 0

    @staticmethod
    def load(input: Pointer[UInt64, ImmutAnyOrigin], at: Int) -> Self:
        var state = Self()
        state.count = input[unsafe_offset=at]
        state.a = input[unsafe_offset=at + 1]
        state.b = input[unsafe_offset=at + 2]
        state.c = input[unsafe_offset=at + 3]
        state.d = input[unsafe_offset=at + 4]
        state.e = input[unsafe_offset=at + 5]
        state.error = input[unsafe_offset=at + 6]
        return state^


def empty_result(op: Int64, minimum: Int64) -> Tuple[UInt64, UInt8, UInt32]:
    """Apply the normal finalizer to an input with no rows or device work."""
    return State().finish(op, minimum)


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
    var base = column * DESCRIPTOR_WORDS
    var op = descriptors[unsafe_offset=base + 3]
    if not supported(op):
        return
    var slot = Int(descriptors[unsafe_offset=base])
    var pair = op == COV or op == CORR
    var right = Int(
        (descriptors[unsafe_offset=base + 4].cast[DType.uint64]() >> 31)
        & 0xFFFF
    )
    var state = State()
    # Statistical association follows CPU materialization and the former
    # native kernels: contiguous 8192-row-aligned worker ranges, followed
    # by worker-order merges. The basic-reduction grid only supplies storage
    # for these states; it does not determine statistical row membership.
    var configured = Int(
        descriptors[unsafe_offset=base + 4].cast[DType.uint64]() >> 47
    )
    var workers = max(1, min(configured, Int(rows) // 65536))
    var step = ((Int(rows) + workers - 1) // workers + 8191) // 8192 * 8192
    var first = group
    var last = min(group + 1, workers)
    # Extremely large inputs can request more workers than the shared grid
    # has partials. Preserve their association in the first partial instead
    # of clamping the user's worker count or expanding the shared buffers.
    if workers > Int(groups):
        first = 0
        last = workers if group == 0 else 0
    for worker in range(first, last):
        var current = State()
        var begin = min(Int(rows), worker * step)
        var end = min(Int(rows), (worker + 1) * step)
        for row in range(begin, end):
            var at = slot * Int(stride) + row
            var valid = validity[unsafe_offset=at] != 0
            var error = errors[unsafe_offset=at]
            var y = UInt64(0)
            if pair:
                var right_at = right * Int(stride) + row
                valid &= validity[unsafe_offset=right_at] != 0
                error |= errors[unsafe_offset=right_at]
                y = words[unsafe_offset=right_at]
            current.add(words[unsafe_offset=at], y, valid, error, op)
        state.merge(current, op)
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
    var op = descriptors[unsafe_offset=base + 3]
    if not supported(op):
        return
    var state = State()
    for group in range(Int(groups)):
        state.merge(
            State.load(partials, (column * Int(groups) + group) * STATE_WORDS),
            op,
        )
    var result = state.finish(op, descriptors[unsafe_offset=base + 4])
    words[unsafe_offset=column] = result[0]
    validity[unsafe_offset=column] = result[1]
    errors[unsafe_offset=column] = result[2]
