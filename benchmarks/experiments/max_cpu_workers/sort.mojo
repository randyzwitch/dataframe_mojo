from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true
from dataframe.kernels.comparison_sort import KeyOrder, comparison_arg_sort
from tests.support.lifecycle_jobs import WorkerSetting


@fieldwise_init
struct Keys(KeyOrder):
    var values: List[Int]

    def less(self, a: Int, b: Int) -> Bool:
        return self.values[a] < self.values[b] or (
            self.values[a] == self.values[b] and a < b
        )


def main() raises:
    for rows in [16384, 100000, 1000000]:
        var values = List[Int]()
        var state = UInt64(17)
        for _ in range(rows):
            state = state * 6364136223846793005 + 1442695040888963407
            values.append(Int((state >> 32) % 10007))
        for workers in [1, 2, 4, 8]:
            var setting = WorkerSetting(workers)
            for repetition in range(4):
                var keys = Keys(values.copy())
                var start = perf_counter_ns()
                var result = comparison_arg_sort(keys^, rows)
                var elapsed = perf_counter_ns() - start
                assert_equal(len(result), rows)
                var seen = List[Bool](length=rows, fill=False)
                for i in range(rows):
                    assert_true(not seen[result[i]])
                    seen[result[i]] = True
                    if i > 0:
                        var a = result[i - 1]
                        var b = result[i]
                        assert_true(
                            values[a] < values[b]
                            or (values[a] == values[b] and a < b)
                        )
                if repetition > 0:
                    print(rows, workers, repetition, elapsed)
            _ = setting^
