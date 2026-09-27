"""Paired physical join orientations over size ratios and key distributions.

Input creation and complete ordered-pair validation are outside timing.
This measures row matching/order restoration, not payload gathering.
"""
from std.time import monotonic
from dataframe import Column, Series
from dataframe.frame import _smaller_build_join_rows
from dataframe.join_hash import direct_hash_join_rows


def main() raises:
    print("left_rows,right_rows,duplicates,orientation,rep,ns")
    for left_count in [4096, 131072]:
        for ratio in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            var right_count = left_count * ratio
            if right_count > 8_388_608:
                continue
            for duplicates in [1, 4]:
                var left_values = List[Int64](capacity=left_count)
                var right_values = List[Int64](capacity=right_count)
                for i in range(left_count):
                    left_values.append(Int64(i * 104729))
                for i in range(right_count):
                    # A permutation for these power-of-two input sizes,
                    # followed by a wide mapping; duplicate groups are skewed
                    # toward the lower domain as duplication increases.
                    var key = ((i * 48271) % right_count) // duplicates
                    right_values.append(Int64(key * 104729))
                var left: List[Series] = [
                    Series("k", Column[Int64](left_values^))
                ]
                var right: List[Series] = [
                    Series("k", Column[Int64](right_values^))
                ]
                var expected = direct_hash_join_rows(left, right, False)
                var warm = _smaller_build_join_rows(left, right, False)
                if warm[0] != expected[0] or warm[1] != expected[1]:
                    raise Error("Build orientation changed ordered matches")
                for rep in range(5):
                    for turn in range(2):
                        if (rep + turn) % 2 == 0:
                            var start = monotonic()
                            var rows = direct_hash_join_rows(left, right, False)
                            var elapsed = monotonic() - start
                            if rows[0] != expected[0] or rows[1] != expected[1]:
                                raise Error(
                                    "Right-build ordered matches changed"
                                )
                            print(
                                left_count,
                                right_count,
                                duplicates,
                                "right",
                                rep,
                                elapsed,
                                sep=",",
                            )
                        else:
                            var start = monotonic()
                            var rows = _smaller_build_join_rows(
                                left, right, False
                            )
                            var elapsed = monotonic() - start
                            if rows[0] != expected[0] or rows[1] != expected[1]:
                                raise Error(
                                    "Left-build ordered matches changed"
                                )
                            print(
                                left_count,
                                right_count,
                                duplicates,
                                "left",
                                rep,
                                elapsed,
                                sep=",",
                            )
