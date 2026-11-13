"""Segment statistics and their zone-map checks.

A `BaseStatistics` describes the valid values of one stretch of a column:
whether it can hold a null and a non-null, and the least and greatest value.
Numbers keep their bounds in one of two domains: `Int128` for every type
stored as an integer (integers, decimals at the column's scale, dates,
datetimes, durations, times), `Float64` for floats. Strings keep the first
`MAX_STRING_MINMAX_SIZE` bytes of their least and greatest value, marked
exact or truncated, with the least and greatest length in bytes.

`check_numeric_zonemap` and `check_string_zonemap` answer whether a
comparison with constants keeps every row, none, or some. A scan can
skip a region only when these bounds prove no row can match.
"""

# What statistics prove about a filter over a stretch of rows.
comptime NO_PRUNING_POSSIBLE = 0
comptime FILTER_ALWAYS_TRUE = 1
comptime FILTER_ALWAYS_FALSE = 2
comptime FILTER_TRUE_OR_NULL = 3
comptime FILTER_FALSE_OR_NULL = 4

# Comparison types a zone map answers.
comptime COMPARE_EQUAL = 0
comptime COMPARE_NOTEQUAL = 1
comptime COMPARE_LESSTHAN = 2
comptime COMPARE_GREATERTHAN = 3
comptime COMPARE_LESSTHANOREQUALTO = 4
comptime COMPARE_GREATERTHANOREQUALTO = 5
comptime COMPARE_DISTINCT_FROM = 6
comptime COMPARE_NOT_DISTINCT_FROM = 7

# What the bounds hold.
comptime BASE_STATS = 0
comptime NUMERIC_STATS = 1
comptime FLOAT_STATS = 2
comptime STRING_STATS = 3

# Bytes of a string bound kept: the inline capacity of a 16-byte string record,
# so a bound never needs a separate allocation. Longer values keep a truncated
# prefix, which still bounds them.
comptime MAX_STRING_MINMAX_SIZE = 12
# Rows per vector and vectors per row group (122,880 / 2,048), the units the
# scan reads in and so the units a zone map can skip.
comptime STATISTICS_VECTOR_SIZE = 2048
comptime ROW_GROUP_VECTORS = 60


def flip_comparison(op: Int) -> Int:
    """The comparison with its sides swapped."""
    if op == COMPARE_LESSTHAN:
        return COMPARE_GREATERTHAN
    if op == COMPARE_GREATERTHAN:
        return COMPARE_LESSTHAN
    if op == COMPARE_LESSTHANOREQUALTO:
        return COMPARE_GREATERTHANOREQUALTO
    if op == COMPARE_GREATERTHANOREQUALTO:
        return COMPARE_LESSTHANOREQUALTO
    return op


@fieldwise_init
struct BaseStatistics(Copyable, ImplicitlyCopyable, Movable):
    """The statistics of one stretch of a column's rows."""

    var stats_type: Int
    # Whether a row can be null, and whether a row can be non-null.
    var has_null: Bool
    var has_no_null: Bool
    # Whether min/max are set: false when no row holds a valid value other
    # than NaN. Float bounds leave NaN out; `has_nan` records it.
    var has_min_max: Bool
    var int_min: Int128
    var int_max: Int128
    var float_min: Float64
    var float_max: Float64
    var str_min: SIMD[DType.uint8, 16]
    var str_max: SIMD[DType.uint8, 16]
    # Bytes held in str_min / str_max, at most MAX_STRING_MINMAX_SIZE.
    var str_min_size: Int
    var str_max_size: Int
    # EXACT_STATS when true, TRUNCATED_STATS when false.
    var min_exact: Bool
    var max_exact: Bool
    var min_string_length: Int
    var max_string_length: Int
    var has_unicode: Bool
    # Whether a valid float value can be NaN.
    var has_nan: Bool

    @staticmethod
    def unknown() -> Self:
        """Statistics that rule nothing out."""
        return Self(
            BASE_STATS,
            True,
            True,
            False,
            0,
            0,
            0.0,
            0.0,
            SIMD[DType.uint8, 16](0),
            SIMD[DType.uint8, 16](0),
            0,
            0,
            False,
            False,
            0,
            0,
            True,
            True,
        )

    @staticmethod
    def empty(stats_type: Int) -> Self:
        """Statistics of no rows: merging into them yields the other side."""
        var result = Self.unknown()
        result.stats_type = stats_type
        result.has_null = False
        result.has_no_null = False
        result.has_unicode = False
        result.has_nan = False
        result.min_string_length = Int.MAX
        return result

    def can_have_null(self) -> Bool:
        return self.has_null

    def can_have_no_null(self) -> Bool:
        return self.has_no_null

    def merge(mut self, other: Self):
        """Statistics covering both stretches."""
        if other.stats_type == BASE_STATS and self.stats_type != BASE_STATS:
            self = Self.unknown()
            return
        self.has_null = self.has_null or other.has_null
        self.has_no_null = self.has_no_null or other.has_no_null
        # A stretch whose only values are NaN has no bounds to merge, but
        # its NaN must reach the merged statistics.
        self.has_nan = self.has_nan or other.has_nan
        if self.stats_type == BASE_STATS:
            return
        if self.stats_type == STRING_STATS:
            self.has_unicode = self.has_unicode or other.has_unicode
            if other.has_no_null:
                self.min_string_length = min(
                    self.min_string_length, other.min_string_length
                )
                self.max_string_length = max(
                    self.max_string_length, other.max_string_length
                )
        if not other.has_min_max:
            return
        if not self.has_min_max:
            self.has_min_max = True
            self.int_min = other.int_min
            self.int_max = other.int_max
            self.float_min = other.float_min
            self.float_max = other.float_max
            self.str_min = other.str_min
            self.str_max = other.str_max
            self.str_min_size = other.str_min_size
            self.str_max_size = other.str_max_size
            self.min_exact = other.min_exact
            self.max_exact = other.max_exact
            return
        if self.stats_type == NUMERIC_STATS:
            self.int_min = min(self.int_min, other.int_min)
            self.int_max = max(self.int_max, other.int_max)
        elif self.stats_type == FLOAT_STATS:
            self.float_min = min(self.float_min, other.float_min)
            self.float_max = max(self.float_max, other.float_max)
        else:
            # A truncated bound compares as its prefix; keep the smaller min and
            # the larger max, each with its exact/truncated mark.
            if (
                _compare_prefixes(
                    other.str_min,
                    other.str_min_size,
                    self.str_min,
                    self.str_min_size,
                )
                < 0
            ):
                self.str_min = other.str_min
                self.str_min_size = other.str_min_size
                self.min_exact = other.min_exact
            if (
                _compare_prefixes(
                    other.str_max,
                    other.str_max_size,
                    self.str_max,
                    self.str_max_size,
                )
                > 0
            ):
                self.str_max = other.str_max
                self.str_max_size = other.str_max_size
                self.max_exact = other.max_exact


def _compare_prefixes(
    a: SIMD[DType.uint8, 16], a_size: Int, b: SIMD[DType.uint8, 16], b_size: Int
) -> Int:
    """Bytewise comparison of two stored prefixes, the shorter first on a tie,
    as strings order."""
    for i in range(min(a_size, b_size)):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    if a_size == b_size:
        return 0
    return -1 if a_size < b_size else 1


def compare_string_stats(
    input: Span[UInt8, _], bound: SIMD[DType.uint8, 16], size: Int, exact: Bool
) -> Int:
    """A constant against a stored bound; a truncated bound is compared over its
    own length only, since the bytes after it are unknown."""
    var length = len(input)
    if not exact and length > size:
        length = size
    for i in range(min(length, size)):
        if input[i] != bound[i]:
            return -1 if input[i] < bound[i] else 1
    if length == size:
        return 0
    return -1 if length < size else 1


def _range_answer[
    T: Comparable & ImplicitlyCopyable
](op: Int, min_value: T, max_value: T, constant: T) -> Int:
    """What [min_value, max_value] proves about `op` with one constant."""
    if op == COMPARE_EQUAL or op == COMPARE_NOT_DISTINCT_FROM:
        if constant == min_value and constant == max_value:
            return FILTER_ALWAYS_TRUE
        if not (constant < min_value or constant > max_value):
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_NOTEQUAL or op == COMPARE_DISTINCT_FROM:
        if constant < min_value or constant > max_value:
            return FILTER_ALWAYS_TRUE
        if constant == min_value and constant == max_value:
            # A cluster of one value equal to the constant.
            return FILTER_ALWAYS_FALSE
        return NO_PRUNING_POSSIBLE
    if op == COMPARE_GREATERTHANOREQUALTO:
        if min_value >= constant:
            return FILTER_ALWAYS_TRUE
        if max_value >= constant:
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_GREATERTHAN:
        if min_value > constant:
            return FILTER_ALWAYS_TRUE
        if max_value > constant:
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_LESSTHANOREQUALTO:
        if max_value <= constant:
            return FILTER_ALWAYS_TRUE
        if min_value <= constant:
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    # COMPARE_LESSTHAN
    if max_value < constant:
        return FILTER_ALWAYS_TRUE
    if min_value < constant:
        return NO_PRUNING_POSSIBLE
    return FILTER_ALWAYS_FALSE


def check_numeric_zonemap(
    stats: BaseStatistics, op: Int, constants: List[Int128]
) -> Int:
    """What [min, max] proves about `op` over integer-domain constants:
    ALWAYS_TRUE as soon as one constant keeps every row, ALWAYS_FALSE when none
    keeps any."""
    if not stats.has_min_max or stats.stats_type != NUMERIC_STATS:
        return NO_PRUNING_POSSIBLE
    for constant in constants:
        var answer = _range_answer(op, stats.int_min, stats.int_max, constant)
        if answer == NO_PRUNING_POSSIBLE or answer == FILTER_ALWAYS_TRUE:
            return answer
    return FILTER_ALWAYS_FALSE


def _nan_answer(op: Int) -> Int:
    """What a comparison with a number gives a NaN row: comparisons in
    this library follow IEEE 754, so NaN is unequal to every number and
    neither less nor greater than any (`!=` and `is_distinct_from` true,
    every other comparison false)."""
    if op == COMPARE_NOTEQUAL or op == COMPARE_DISTINCT_FROM:
        return FILTER_ALWAYS_TRUE
    return FILTER_ALWAYS_FALSE


def check_float_zonemap(
    stats: BaseStatistics, op: Int, constants: List[Float64]
) -> Int:
    """Check float bounds while tracking NaN separately from numeric bounds.
    `_nan_answer` supplies NaN's comparison result: a stretch holding NaN keeps every row or none
    only when its NaN rows agree. A NaN constant rules nothing out."""
    if stats.stats_type != FLOAT_STATS:
        return NO_PRUNING_POSSIBLE
    for constant in constants:
        if constant != constant:
            return NO_PRUNING_POSSIBLE
    if not stats.has_min_max:
        # No number: every valid row is NaN, or none is valid.
        if stats.has_nan:
            return _nan_answer(op)
        return NO_PRUNING_POSSIBLE
    var result = FILTER_ALWAYS_FALSE
    for constant in constants:
        var answer = _range_answer(
            op, stats.float_min, stats.float_max, constant
        )
        if answer == NO_PRUNING_POSSIBLE or answer == FILTER_ALWAYS_TRUE:
            result = answer
            break
    if stats.has_nan and result != _nan_answer(op):
        return NO_PRUNING_POSSIBLE
    return result


def constants_cover_range(
    stats: BaseStatistics, constants: List[Int128]
) -> Bool:
    """Whether an integer column's whole [min, max] range is listed, so an IN
    list keeps every row."""
    if not stats.has_min_max or stats.stats_type != NUMERIC_STATS:
        return False
    var low = stats.int_min
    var high = stats.int_max
    if high - low >= Int128(len(constants)):
        return False
    var values = List[Int128]()
    for constant in constants:
        if constant >= low and constant <= high:
            values.append(constant)
    if len(values) == 0:
        return False
    sort(values)
    if values[0] != low or values[len(values) - 1] != high:
        return False
    for i in range(1, len(values)):
        if values[i - 1] != values[i] and values[i - 1] + 1 != values[i]:
            return False
    return True


def _string_answer(
    stats: BaseStatistics, op: Int, constant: Span[UInt8, _]
) -> Int:
    """What the string bounds prove about `op` with one constant (min, min_type,
    max, max_type)."""
    var min_value = stats.str_min
    var min_size = stats.str_min_size
    var min_exact = stats.min_exact
    # A truncated min equal to the constant, over strings all longer than
    # it, is greater than the constant: its lower bound is the prefix
    # followed by a zero byte.
    if (
        not min_exact
        and len(constant) == min_size
        and stats.min_string_length > len(constant)
        and compare_string_stats(constant, min_value, min_size, False) == 0
        and min_size < 16
    ):
        min_value[min_size] = 0
        min_size += 1
        min_exact = True
    var min_comp = compare_string_stats(
        constant, min_value, min_size, min_exact
    )
    var max_comp = compare_string_stats(
        constant, stats.str_max, stats.str_max_size, stats.max_exact
    )
    var exact = min_exact and stats.max_exact
    if op == COMPARE_EQUAL or op == COMPARE_NOT_DISTINCT_FROM:
        if min_comp == 0 and max_comp == 0 and exact:
            return FILTER_ALWAYS_TRUE
        if min_comp >= 0 and max_comp <= 0:
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_NOTEQUAL or op == COMPARE_DISTINCT_FROM:
        if min_comp < 0 or max_comp > 0:
            return FILTER_ALWAYS_TRUE
        if min_comp == 0 and max_comp == 0 and exact:
            return FILTER_ALWAYS_FALSE
        return NO_PRUNING_POSSIBLE
    if op == COMPARE_GREATERTHAN:
        if max_comp < 0 or (max_comp == 0 and not stats.max_exact):
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_GREATERTHANOREQUALTO:
        if max_comp <= 0:
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    if op == COMPARE_LESSTHAN:
        if min_comp > 0 or (min_comp == 0 and not min_exact):
            return NO_PRUNING_POSSIBLE
        return FILTER_ALWAYS_FALSE
    # COMPARE_LESSTHANOREQUALTO
    if min_comp >= 0:
        return NO_PRUNING_POSSIBLE
    return FILTER_ALWAYS_FALSE


def check_string_zonemap(
    stats: BaseStatistics, op: Int, constants: List[List[UInt8]]
) -> Int:
    """What the string bounds prove about `op` over several constants."""
    if stats.stats_type != STRING_STATS:
        return NO_PRUNING_POSSIBLE
    for constant in constants:
        var answer = NO_PRUNING_POSSIBLE
        if stats.has_min_max:
            answer = _string_answer(stats, op, Span(constant))
        if answer == NO_PRUNING_POSSIBLE or answer == FILTER_ALWAYS_TRUE:
            return answer
    return FILTER_ALWAYS_FALSE
