"""Scalar positional window models over frozen, independently typed literals."""
from tests.support.key_kernel_reference import cell_equal, cell_less
from tests.support.selection_types_reference import (
    selection_decimal,
    selection_key,
    selection_valid,
)


def extended_group(row: Int) -> Int:
    return 3 if row % 7 == 3 else row % 3


def extended_partitions(
    n: Int, offset: Int, partitioned: Bool
) -> List[List[Int]]:
    var groups = List[List[Int]](
        length=4 if partitioned else 1, fill=List[Int]()
    )
    for i in range(n):
        groups[extended_group(i + offset) if partitioned else 0].append(i)
    return groups^


def extended_sources(
    kind: Int,
    n: Int,
    offset: Int,
    mode: Int,
    partitioned: Bool,
    operation: String,
    reverse: Bool = False,
    distance: Int = 1,
    window: Int = 3,
    minimum: Int = 1,
    limit: Int = -1,
) raises -> List[Int]:
    var result = List[Int](length=n, fill=-1)
    for group in extended_partitions(n, offset, partitioned):
        for j in range(len(group)):
            var destination = group[j]
            if operation == "shift":
                var from_position = j - distance
                if from_position >= 0 and from_position < len(group):
                    result[destination] = group[from_position] + offset
                continue
            if operation == "forward" or operation == "backward":
                # Search original valid neighbors; filled results never become
                # inputs and distance is measured inside the partition.
                var step = 1 if operation == "backward" else -1
                var from_position = j
                while from_position >= 0 and from_position < len(group):
                    var gap = abs(from_position - j)
                    if limit >= 0 and gap > limit:
                        break
                    var source = group[from_position] + offset
                    if selection_valid(source, mode):
                        result[destination] = source
                        break
                    from_position += step
                continue
            var rolling = (
                operation == "rolling_min" or operation == "rolling_max"
            )
            if not rolling and not selection_valid(destination + offset, mode):
                continue
            var first = max(
                0, j - window + 1
            ) if rolling else j if reverse else 0
            var last = j + 1 if rolling or not reverse else len(group)
            var best = -1
            var count = 0
            # Reverse cumulative ties retain the last source in original order.
            for position in range(first, last):
                var source = (
                    group[last - 1 - (position - first)] if reverse
                    and not rolling else group[position]
                )
                source += offset
                if not selection_valid(source, mode):
                    continue
                count += 1
                var maximum = (
                    operation == "cum_max" or operation == "rolling_max"
                )
                var better = best < 0
                if best >= 0:
                    var before = best if maximum else source
                    var after = source if maximum else best
                    better = cell_less(
                        selection_key(kind, before, mode),
                        selection_key(kind, after, mode),
                        False,
                        True,
                    )
                if better:
                    best = source
            if not rolling or count >= minimum:
                result[destination] = best
    return result^


def extended_counts(
    n: Int, offset: Int, mode: Int, partitioned: Bool, reverse: Bool
) -> List[Int64]:
    var result = List[Int64](length=n, fill=0)
    for group in extended_partitions(n, offset, partitioned):
        for j in range(len(group)):
            var count = Int64(0)
            for k in range(
                j if reverse else 0, len(group) if reverse else j + 1
            ):
                count += Int64(selection_valid(group[k] + offset, mode))
            result[group[j]] = count
    return result^


def extended_ranks(
    kind: Int,
    n: Int,
    offset: Int,
    mode: Int,
    partitioned: Bool,
    method: String,
    descending: Bool,
) raises -> List[Float64]:
    var result = List[Float64](length=n, fill=0)
    for group in extended_partitions(n, offset, partitioned):
        for i in range(len(group)):
            var row = group[i] + offset
            if not selection_valid(row, mode):
                continue
            var before = 0
            var equal = 0
            var earlier_equal = 0
            var distinct_before = List[Int]()
            for j in range(len(group)):
                var other = group[j] + offset
                if not selection_valid(other, mode):
                    continue
                if cell_equal(
                    selection_key(kind, row, mode),
                    selection_key(kind, other, mode),
                ):
                    equal += 1
                    earlier_equal += Int(j < i)
                elif cell_less(
                    selection_key(kind, other, mode),
                    selection_key(kind, row, mode),
                    descending,
                    True,
                ):
                    before += 1
                    var seen = False
                    for prior in distinct_before:
                        seen = seen or cell_equal(
                            selection_key(kind, prior, mode),
                            selection_key(kind, other, mode),
                        )
                    if not seen:
                        distinct_before.append(other)
            result[group[i]] = (
                Float64(before + 1) + Float64(equal - 1) / 2 if method
                == "average" else Float64(before + 1) if method
                == "min" else Float64(before + equal) if method
                == "max" else Float64(len(distinct_before) + 1) if method
                == "dense" else Float64(before + earlier_equal + 1)
            )
    return result^


def extended_decimal_totals(
    kind: Int, n: Int, offset: Int, mode: Int, partitioned: Bool, reverse: Bool
) -> List[Int128]:
    var result = List[Int128](length=n, fill=0)
    for group in extended_partitions(n, offset, partitioned):
        for j in range(len(group)):
            var row = group[j]
            if not selection_valid(row + offset, mode):
                continue
            var total = Int128(0)
            for k in range(
                j if reverse else 0, len(group) if reverse else j + 1
            ):
                var source = group[k] + offset
                if selection_valid(source, mode):
                    total += selection_decimal(kind, source)
            result[row] = total
    return result^
