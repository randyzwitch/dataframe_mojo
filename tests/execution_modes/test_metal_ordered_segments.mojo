"""Chunk-native ordered groups keep first ties and both group-local rank policies."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_false, assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe.execution_observation import DeviceObservation
from dataframe_metal.ordered_plan import plan_ordered
from dataframe_metal.ordered_reduce import execute_ordered
from dataframe_metal.ordered_segments import segmented_ordered_supported
from dataframe_metal.accelerator.gpu.target import METAL_TARGET


def _parts(
    name: String, raw: List[Int32], valid: List[Bool], ends: List[Int]
) raises -> Series:
    var chunks = List[Series]()
    var start = 0
    for end in ends:
        var prefix = len(chunks) + 1
        var data = List[Int32](length=prefix, fill=0)
        var bits = List[Bool](length=prefix, fill=False)
        for i in range(start, end):
            data.append(raw[i])
            bits.append(valid[i])
        chunks.append(
            Series(name, Column[Int32](data^, bits^)).slice(prefix, end - start)
        )
        start = end
    return Series._from_chunks(chunks^)


def _ordered_inputs() raises -> DataFrame:
    return DataFrame(
        [
            _parts(
                "district",
                [0, -4, 0, 9, -4, 0, 9, -4, 0, -4, 9, 0, 99],
                [
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                ],
                [3, 7, 13],
            ),
            _parts(
                "reading",
                [
                    0,
                    Int32.MAX,
                    Int32.MIN,
                    5,
                    Int32.MAX,
                    Int32.MIN,
                    5,
                    0,
                    8,
                    Int32.MIN,
                    0,
                    0,
                    0,
                ],
                [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                ],
                [2, 8, 13],
            ),
        ]
    )


def test_metal_ordered_segment_admission_depends_on_input_layout() raises:
    var frame = _ordered_inputs()
    var outputs: List[Expr] = [
        col("reading").first().alias("a"),
        col("reading").arg_min().alias("b"),
    ]
    var plan = frame.lazy().group_by("district").agg(outputs)
    assert_true(segmented_ordered_supported(plan, plan_ordered(plan).value()))
    var contiguous = frame.rechunk().lazy().group_by("district").agg(outputs)
    assert_false(
        segmented_ordered_supported(
            contiguous, plan_ordered(contiguous).value()
        )
    )
    var filtered = (
        frame.lazy()
        .filter(col("district") < 50)
        .group_by("district")
        .agg(outputs)
    )
    assert_false(
        segmented_ordered_supported(filtered, plan_ordered(filtered).value())
    )


# requires: metal
def test_metal_ordered_segments_preserve_ties_nulls_and_local_ranks() raises:
    comptime if METAL_TARGET:
        var frame = _ordered_inputs()
        var v = col("reading")
        var outputs: List[Expr] = [
            v.first().alias("first"),
            v.last().alias("last"),
            v.arg_min().alias("minimum"),
            v.arg_max().alias("maximum"),
            v.first(ignore_nulls=True).alias("first_valid"),
            v.last(ignore_nulls=True).alias("last_valid"),
            v.drop_nulls().arg_min().alias("minimum_valid"),
            v.drop_nulls().arg_max().alias("maximum_valid"),
        ]
        var plan = (
            frame.lazy().group_by("district", maintain_order=True).agg(outputs)
        )
        var expected = plan.collect(engine="cpu")
        assert_equal(expected.item(0, "minimum").uint32(), UInt32(1))
        assert_equal(expected.item(0, "minimum_valid").uint32(), UInt32(0))
        assert_equal(expected.item(0, "maximum").uint32(), UInt32(3))
        assert_equal(expected.item(0, "maximum_valid").uint32(), UInt32(2))
        assert_true(expected.item(3, "minimum").is_null())
        for wrap in [True, False]:
            var observation = ArcPointer(DeviceObservation())
            plan._plan._device_observation = observation
            var actual = execute_ordered(plan, wrap)
            assert_true(actual[0].equals(expected))
            assert_equal(
                observation[].completed_launches.load(),
                Int64(15 if wrap else 3),
            )
            if wrap:
                assert_equal(observation[].upload_bytes.load(), Int64(0))
        var endpoints: List[Expr] = [
            v.first(ignore_nulls=True).alias("lo"),
            v.last().alias("hi"),
        ]
        var ends = (
            frame.lazy()
            .group_by("district", maintain_order=True)
            .agg(endpoints)
        )
        var cpu = ends.collect(engine="cpu")
        var observed = ArcPointer(DeviceObservation())
        ends._plan._device_observation = observed
        assert_true(execute_ordered(ends)[0].equals(cpu))
        assert_equal(observed[].completed_launches.load(), Int64(5))
        var positions: List[Expr] = [
            v.first().alias("begin"),
            v.last().alias("end"),
        ]
        var positional = (
            frame.lazy()
            .group_by("district", maintain_order=True)
            .agg(positions)
        )
        var positional_cpu = positional.collect(engine="cpu")
        var scanned_keys = ArcPointer(DeviceObservation())
        positional._plan._device_observation = scanned_keys
        assert_true(execute_ordered(positional)[0].equals(positional_cpu))
        assert_equal(scanned_keys[].completed_launches.load(), Int64(3))
        var by_key = (
            frame.lazy()
            .group_by("district", maintain_order=True)
            .agg([col("district").arg_min().alias("index")])
        )
        var key_cpu = by_key.collect(engine="cpu")
        var key_observed = ArcPointer(DeviceObservation())
        by_key._plan._device_observation = key_observed
        assert_true(execute_ordered(by_key)[0].equals(key_cpu))
        assert_equal(key_observed[].completed_launches.load(), Int64(9))
