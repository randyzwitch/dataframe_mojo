"""Seeded sampling has broad distribution checks in addition to row invariants."""
# contract-group: api.inspect
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series


def test_seeded_sampling_does_not_systematically_favor_a_subset_of_rows() raises:
    comptime POPULATION = 17
    comptime SEEDS = 2048
    comptime DRAW = 3
    var frame = DataFrame(
        [
            Series(
                "row", Column[Int64]([Int64(row) for row in range(POPULATION)])
            )
        ]
    )
    for replacement in [False, True]:
        var counts = List[Int](length=POPULATION, fill=0)
        for seed in range(SEEDS):
            var sample = frame.sample(
                DRAW, with_replacement=replacement, seed=seed
            )
            assert_equal(sample.height(), DRAW)
            for position in range(DRAW):
                var row = Int(sample.item(position, "row").int64())
                assert_true(0 <= row and row < POPULATION)
                counts[row] += 1
        # Expected count361.41; independent draws have sigma<19 for either
        # replacement policy. A fixed seed family and >7-sigma tolerance check
        # gross bias without pinning the generator or exact sampled rows.
        for count in counts:
            assert_true(
                abs(count * POPULATION - SEEDS * DRAW) < 140 * POPULATION,
                "seeded sample is systematically biased across rows",
            )
