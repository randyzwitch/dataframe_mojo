"""Sorted as-of matching, retaining exact integer and temporal key ordering."""
from std.collections import Optional
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .series import Series
from .value import AnyValue


def _less[D: DType](a: Scalar[D], b: Scalar[D]) -> Bool:
    comptime if D.is_floating_point():
        # Match the dataframe's total order: NaNs follow all other values.
        return a < b or (a == a and b != b)
    else:
        return a < b


def _equal[D: DType](a: Scalar[D], b: Scalar[D]) -> Bool:
    return not _less(a, b) and not _less(b, a)


def _tolerance(
    value: Optional[AnyValue], dtype: DataType
) raises -> Tuple[Bool, Int128, Float64]:
    if not value:
        return (False, 0, 0)
    var scalar = value.value().copy()
    if scalar.is_null():
        raise Error("join_asof tolerance cannot be a null scalar; use None")
    var typ = scalar.dtype()
    var integer: Int128
    var floating: Float64
    if typ.is_duration():
        if not (dtype.is_date() or dtype.is_datetime() or dtype.is_duration()):
            raise Error("join_asof Duration tolerance requires a temporal key")
        integer = Int128(scalar.to_physical())
        if integer < 0:
            raise Error("join_asof tolerance must be nonnegative")
        if dtype.is_date():
            integer //= Int128(86400) * Int128(typ.per_second())
        else:
            integer = (
                integer * Int128(dtype.per_second()) // Int128(typ.per_second())
            )
        floating = Float64(integer)
    elif typ.is_integer():
        integer = Int128(
            scalar._int.cast[DType.uint64]()
        ) if typ.is_unsigned() else Int128(scalar._int)
        floating = Float64(integer)
    elif typ.is_float():
        floating = scalar._float
        if floating != floating or floating < 0:
            raise Error("join_asof tolerance must be nonnegative and not NaN")
        # No supported key can have an integer distance above UInt64.MAX.
        integer = (
            Int128(UInt64.MAX) if floating
            >= Float64(UInt64.MAX) else floating.cast[DType.int128]()
        )
    else:
        raise Error("join_asof tolerance must be numeric or a Duration")
    if integer < 0:
        raise Error("join_asof tolerance must be nonnegative")
    return (True, integer, floating)


def _check_sorted[
    D: DType
](
    column: Column[Scalar[D]],
    ids: List[Int],
    groups: Int,
    side: String,
    name: String,
) raises:
    var previous = List[Int](length=groups, fill=-1)
    for row in range(len(column)):
        if not column._valid(row):
            continue
        var group = ids[row] if len(ids) else 0
        var last = previous[group]
        if last >= 0 and _less(column._get(row), column._get(last)):
            raise Error(
                "join_asof "
                + side
                + " column '"
                + name
                + "' is not sorted within its group (row "
                + String(row)
                + ")"
            )
        previous[group] = row


def _scan[
    D: DType, strategy: Int
](
    left: Column[Scalar[D]],
    right: Column[Scalar[D]],
    left_ids: List[Int],
    right_ids: List[Int],
    matchable: List[Bool],
    left_name: String,
    right_name: String,
    exact: Bool,
    tolerance: Tuple[Bool, Int128, Float64],
) raises -> List[Int]:
    var groups = len(matchable)
    _check_sorted(left, left_ids, groups, "left", left_name)
    _check_sorted(right, right_ids, groups, "right", right_name)
    # Stable CSR buckets: total storage and work are linear, even for many
    # groups or long runs of duplicate keys. Nearest retains null positions
    # to preserve Polars' duplicate-run boundaries; nulls cannot be matches.
    var starts = List[Int](length=groups + 1, fill=0)
    for row in range(len(right)):
        var g = right_ids[row] if len(right_ids) else 0
        if (strategy == 2 or right._valid(row)) and matchable[g]:
            starts[g + 1] += 1
    for g in range(groups):
        starts[g + 1] += starts[g]
    var places = starts.copy()
    var rows = List[Int](length=starts[groups], fill=0)
    for row in range(len(right)):
        var g = right_ids[row] if len(right_ids) else 0
        if (strategy == 2 or right._valid(row)) and matchable[g]:
            rows[places[g]] = row
            places[g] += 1
    var ends = List[Int]()
    comptime if strategy == 2:
        ends = List[Int](length=len(rows), fill=0)
        for g in range(groups):
            for i in range(starts[g + 1] - 1, starts[g] - 1, -1):
                ends[i] = (
                    ends[i + 1] if i + 1 < starts[g + 1]
                    and right._valid(rows[i])
                    and right._valid(rows[i + 1])
                    and right._get(rows[i]) == right._get(rows[i + 1]) else i
                )
    var lower = starts.copy()
    var upper = starts.copy()
    var smaller = List[Int](length=groups, fill=-1)
    var result = List[Int](length=len(left), fill=-1)
    for row in range(len(left)):
        if not left._valid(row):
            continue
        var g = left_ids[row] if len(left_ids) else 0
        if not matchable[g]:
            continue
        var value = left._get(row)
        var stop = starts[g + 1]
        var chosen = -1
        comptime if strategy == 0:
            var pos = upper[g]
            while pos < stop:
                var candidate = right._get(rows[pos])
                if _less(candidate, value) or (
                    exact and _equal(candidate, value)
                ):
                    pos += 1
                else:
                    break
            upper[g] = pos
            if pos > starts[g]:
                chosen = rows[pos - 1]
        elif strategy == 1:
            var pos = lower[g]
            while pos < stop:
                var candidate = right._get(rows[pos])
                if _less(value, candidate) or (
                    exact and _equal(candidate, value)
                ):
                    break
                pos += 1
            lower[g] = pos
            if pos < stop:
                chosen = rows[pos]
        else:
            var hi = upper[g]
            while hi < stop:
                if right._valid(rows[hi]) and right._get(rows[hi]) > value:
                    break
                hi += 1
            if (
                exact
                and hi > starts[g]
                and right._valid(rows[hi - 1])
                and right._get(rows[hi - 1]) == value
            ):
                chosen = rows[hi - 1]
            else:
                if hi < stop:
                    hi = ends[hi]
                var lo = lower[g]
                while lo < hi:
                    if right._valid(rows[lo]):
                        if right._get(rows[lo]) >= value:
                            break
                        smaller[g] = rows[lo]
                    lo += 1
                lower[g] = lo
                chosen = smaller[g]
                if hi < stop:
                    var candidate = rows[hi]
                    if chosen < 0:
                        chosen = candidate
                    else:
                        comptime if D.is_floating_point():
                            if abs(
                                Float64(right._get(candidate)) - Float64(value)
                            ) <= abs(
                                Float64(value) - Float64(right._get(chosen))
                            ):
                                chosen = candidate
                        else:
                            if abs(
                                Int128(right._get(candidate)) - Int128(value)
                            ) <= abs(
                                Int128(value) - Int128(right._get(chosen))
                            ):
                                chosen = candidate
            upper[g] = hi
        if chosen >= 0 and tolerance[0]:
            comptime if D.is_floating_point():
                if not (
                    abs(Float64(value) - Float64(right._get(chosen)))
                    <= tolerance[2]
                ):
                    chosen = -1
            else:
                if (
                    abs(Int128(value) - Int128(right._get(chosen)))
                    > tolerance[1]
                ):
                    chosen = -1
        result[row] = chosen
    return result^


def asof_rows(
    left: Series,
    right: Series,
    left_ids: List[Int],
    right_ids: List[Int],
    matchable: List[Bool],
    strategy: String,
    tolerance: Optional[AnyValue],
    allow_exact_matches: Bool,
) raises -> List[Int]:
    var dtype = left.dtype()
    if dtype != right.dtype():
        raise Error(
            "join_asof key dtypes must match: "
            + dtype.name()
            + " and "
            + right.dtype().name()
        )
    if not (
        dtype.is_integer()
        or dtype.is_float()
        or dtype.is_date()
        or dtype.is_datetime()
        or dtype.is_duration()
    ):
        raise Error(
            "join_asof requires integer, float, Date, Datetime or Duration keys"
        )
    if strategy not in ["backward", "forward", "nearest"]:
        raise Error("join_asof strategy must be backward, forward or nearest")
    var bound = _tolerance(tolerance, dtype)
    var lhs = left.rechunk()
    var rhs = right.rechunk()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if lhs._data.isa[Column[Scalar[D]]]():
            comptime for s in range(3):
                if strategy == (
                    "backward" if s == 0 else "forward" if s == 1 else "nearest"
                ):
                    return _scan[D, s](
                        lhs._data[Column[Scalar[D]]],
                        rhs._data[Column[Scalar[D]]],
                        left_ids,
                        right_ids,
                        matchable,
                        left.name(),
                        right.name(),
                        allow_exact_matches,
                        bound,
                    )
    raise Error("Unsupported join_asof key storage")
