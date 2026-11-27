"""The reserved all-null dictionary pair counts toward the observed-group limit."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.execution_observation import DeviceObservation
from dataframe.expr import SUM
from dataframe_metal.accelerator.gpu.target import METAL_TARGET
from dataframe_metal.dictionary_pair_reduce import execute_dictionary_sum
from dataframe_metal.group_layout import GENERAL_GROUP_LIMIT
from dataframe_metal.multi_plan import ReductionOutput


def _capacity_chunk(second: Bool) raises -> DataFrame:
    # In a 256 x 256 domain, reserve (255, 255) as the first over-limit key.
    var a = List[Int32](capacity=GENERAL_GROUP_LIMIT)
    var b = List[Int32](capacity=GENERAL_GROUP_LIMIT)
    var valid = List[Bool](length=GENERAL_GROUP_LIMIT, fill=True)
    var v = List[Int64](capacity=GENERAL_GROUP_LIMIT)
    var large = Int64(1) << 42
    for i in range(GENERAL_GROUP_LIMIT - 1):
        a.append(Int32(i >> 8))
        b.append(Int32(i & 255))
        v.append(-large if second else large + Int64(i - 30000))
    a.append(255)
    b.append(255)
    valid[GENERAL_GROUP_LIMIT - 1] = False
    v.append(large + 7 if second else -large)
    return DataFrame(
        [
            Series("a", Column[Int32](a^, valid.copy())),
            Series("b", Column[Int32](b^, valid^)),
            Series("delta", Column[Int64](v^)),
        ]
    )


# requires: metal
def test_sparse_dictionary_sum_reserves_all_null_within_group_limit() raises:
    comptime if METAL_TARGET:
        var frame = _capacity_chunk(False).vstack(_capacity_chunk(True))
        var output = ReductionOutput(0, "net", SUM, 1)
        var obs = ArcPointer(DeviceObservation())
        var collected = execute_dictionary_sum[False, True](
            frame, output, obs, 256, 256, True
        )
        ref result = collected[0]
        assert_equal(result.height(), GENERAL_GROUP_LIMIT)
        assert_equal(obs[].completed_launches.load(), Int64(1))
        assert_equal(obs[].upload_bytes.load(), Int64(0))
        assert_equal(obs[].download_bytes.load(), Int64(0))
        var left = result.column("a").numeric[DType.int32]()
        var right = result.column("b").numeric[DType.int32]()
        var sums = result.column("net").numeric[DType.int64]()
        for i in range(GENERAL_GROUP_LIMIT - 1):
            assert_equal(left.value(i), Int32(i >> 8))
            assert_equal(right.value(i), Int32(i & 255))
            assert_equal(sums.value(i), Int64(i - 30000))
        var last = GENERAL_GROUP_LIMIT - 1
        assert_true(left.is_null(last))
        assert_true(right.is_null(last))
        assert_equal(sums.value(last), Int64(7))

        # The 65,536th non-null pair still fits its hash table. Counting the
        # reserved all-null tuple makes this a 65,537-group query.
        var extra = DataFrame(
            [
                Series("a", Column[Int32]([255])),
                Series("b", Column[Int32]([255])),
                Series("delta", Column[Int64]([-1])),
            ]
        )
        with assert_raises(contains="grouped reduction exceeded 65536 groups"):
            _ = execute_dictionary_sum[False, True](
                frame.vstack(extra), output, None, 256, 256, True
            )
