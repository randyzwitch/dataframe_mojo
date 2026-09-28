"""Row sampling for DataFrame.sample and Series.sample.

The generator is SplitMix64 (Steele, Lea and Flood, 2014), chosen because it
is small, fast, passes BigCrush, and is fully specified, so a seed gives the
same rows on every platform and release. Bounded draws use Lemire's
multiply-and-reject method, which has no modulo bias. Distinct indices come
from Floyd's algorithm when few rows are wanted and from a partial
Fisher-Yates shuffle otherwise.
"""
from std.collections import Dict, Optional
from std.time import perf_counter_ns


struct SplitMix64(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def below(mut self, bound: Int) -> Int:
        """Uniform integer in [0, bound); bound must be positive."""
        var limit = UInt128(UInt64(bound))
        var product = UInt128(self.next()) * limit
        var low = UInt64(product & 0xFFFFFFFFFFFFFFFF)
        if low < UInt64(bound):
            var threshold = (0 - UInt64(bound)) % UInt64(bound)
            while low < threshold:
                product = UInt128(self.next()) * limit
                low = UInt64(product & 0xFFFFFFFFFFFFFFFF)
        return Int(product >> 64)


def sample_size(
    height: Int,
    n: Optional[Int],
    fraction: Optional[Float64],
    with_replacement: Bool,
) raises -> Int:
    """Rows to draw: n, or floor(fraction * height), or 1 when neither is
    given. Without replacement the count may not exceed the height."""
    if n and fraction:
        raise Error("sample() takes n or fraction, not both")
    var count = 1
    if n:
        count = n.value()
        if count < 0:
            raise Error("sample() n must be nonnegative")
    elif fraction:
        var share = fraction.value()
        if share < 0 or share != share:
            raise Error("sample() fraction must be nonnegative")
        count = Int(share * Float64(height))
    if not with_replacement and count > height:
        raise Error(
            "cannot take a larger sample than the total population ("
            + String(count)
            + " > "
            + String(height)
            + ") without replacement"
        )
    if with_replacement and count > 0 and height == 0:
        raise Error("cannot sample from an empty population")
    return count


def sample_indices(
    height: Int,
    count: Int,
    with_replacement: Bool,
    shuffle: Bool,
    seed: Optional[Int],
) -> List[Int]:
    """`count` row indices in [0, height). Ascending unless shuffle, in which
    case the order is uniformly random too."""
    var generator = SplitMix64(
        UInt64(seed.value()) if seed else UInt64(perf_counter_ns())
    )
    var chosen = List[Int](capacity=count)
    if with_replacement:
        for _ in range(count):
            chosen.append(generator.below(height))
    elif 2 * count >= height:
        # Partial Fisher-Yates: the first `count` slots are a uniform
        # random ordered sample.
        var slots = List[Int](capacity=height)
        for i in range(height):
            slots.append(i)
        for i in range(count):
            var j = i + generator.below(height - i)
            var swap = slots[i]
            slots[i] = slots[j]
            slots[j] = swap
        for i in range(count):
            chosen.append(slots[i])
        if shuffle:
            return chosen^
    else:
        # Floyd's algorithm: `count` distinct values with `count` draws.
        var seen = Dict[Int, Bool]()
        for j in range(height - count, height):
            var pick = generator.below(j + 1)
            if pick in seen:
                pick = j
            seen[pick] = True
            chosen.append(pick)
    if shuffle:
        for i in range(len(chosen) - 1, 0, -1):
            var j = generator.below(i + 1)
            var swap = chosen[i]
            chosen[i] = chosen[j]
            chosen[j] = swap
    else:
        sort(chosen)
    return chosen^
