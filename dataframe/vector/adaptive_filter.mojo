"""DuckDB 2.0's AdaptiveFilter (execution/adaptive_filter.cpp): the order a
filter's conjuncts run in, adapted at run time from how long each ordering
takes.

After a warmup of 5 chunks, every 20 chunks one adjacent pair of conjuncts
is swapped at random (weighted by how often swapping that pair helped
before), the swap is observed for 10 chunks, and kept only when the mean
time per chunk went down; a reverted swap halves its pair's likeliness. A
filter with a conjunct that can raise keeps its order, as DuckDB disables
permutations for a conjunct that `CanThrow`.
"""
from std.time import perf_counter_ns


struct AdaptiveFilter(Copyable, Movable):
    var permutation: List[Int]
    var swap_likeliness: List[Int]
    var disabled: Bool
    var observe_interval: Int
    var execute_interval: Int
    var warmup: Bool
    var observe: Bool
    var iteration_count: Int
    var runtime_sum: Float64
    var prev_mean: Float64
    var swap_idx: Int
    var right_random_border: Int
    var seed: UInt64

    def __init__(out self, conjuncts: Int, can_throw: Bool):
        self.permutation = List[Int](capacity=conjuncts)
        self.swap_likeliness = List[Int]()
        for i in range(conjuncts):
            self.permutation.append(i)
            if i != conjuncts - 1:
                self.swap_likeliness.append(100)
        self.disabled = can_throw or conjuncts <= 1
        self.observe_interval = 10
        self.execute_interval = 20
        self.warmup = True
        self.observe = False
        self.iteration_count = 0
        self.runtime_sum = 0.0
        self.prev_mean = 0.0
        self.swap_idx = 0
        self.right_random_border = 100 * max(conjuncts - 1, 0)
        self.seed = 0x9E3779B97F4A7C15

    def begin(self) -> Int:
        """BeginFilter: the start time, 0 when the order never changes."""
        if self.disabled:
            return 0
        return Int(perf_counter_ns())

    def end(mut self, start: Int):
        """EndFilter: adapt to this chunk's duration."""
        if self.disabled:
            return
        self._adapt(Float64(Int(perf_counter_ns()) - start) * 1e-9)

    def _random(mut self, low: Int, high: Int) -> Int:
        """A pseudo-random integer in [low, high] (xorshift64*)."""
        self.seed ^= self.seed >> 12
        self.seed ^= self.seed << 25
        self.seed ^= self.seed >> 27
        var value = self.seed * 0x2545F4914F6CDD1D
        return low + Int(value % UInt64(high - low + 1))

    def _adapt(mut self, duration: Float64):
        self.iteration_count += 1
        self.runtime_sum += duration
        if self.warmup:
            if self.iteration_count == 5:
                self.iteration_count = 0
                self.runtime_sum = 0.0
                self.observe = False
                self.warmup = False
            return
        if self.observe and self.iteration_count == self.observe_interval:
            # Keep the swap if the runtime decreased, else reverse it.
            var trial_mean = self.runtime_sum / Float64(self.iteration_count)
            if self.prev_mean - trial_mean <= 0:
                var held = self.permutation[self.swap_idx]
                self.permutation[self.swap_idx] = self.permutation[
                    self.swap_idx + 1
                ]
                self.permutation[self.swap_idx + 1] = held
                # Less likely next time, but never impossible.
                if self.swap_likeliness[self.swap_idx] > 1:
                    self.swap_likeliness[self.swap_idx] //= 2
            else:
                self.swap_likeliness[self.swap_idx] = 100
            self.observe = False
            self.iteration_count = 0
            self.runtime_sum = 0.0
        elif not self.observe and self.iteration_count == self.execute_interval:
            self.prev_mean = self.runtime_sum / Float64(self.iteration_count)
            # Each adjacent pair owns 100 zero-based probability slots.
            var random_number = self._random(0, self.right_random_border - 1)
            self.swap_idx = random_number // 100
            var likeliness = random_number - 100 * self.swap_idx
            if self.swap_likeliness[self.swap_idx] > likeliness:
                var held = self.permutation[self.swap_idx]
                self.permutation[self.swap_idx] = self.permutation[
                    self.swap_idx + 1
                ]
                self.permutation[self.swap_idx + 1] = held
                self.observe = True
            self.iteration_count = 0
            self.runtime_sum = 0.0
