"""Dictionary calibration separates physical code spans and sampled tuples."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_false
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe_metal.dictionary_group import plan_dictionary_group
from dataframe_metal.dictionary_properties import dictionary_properties
from dataframe_metal.handoff import layout_keys, region_key
from tests.execution_modes.test_metal_dictionary_group import _coded_key


def test_metal_dictionary_layout_tracks_code_and_validity_spans() raises:
    var source = _coded_key(
        "l", ["a", "b"], [0, 1, 1, 0], [True, False, True, True]
    )
    var other = _coded_key(
        "l", ["a", "b"], [0, 1, 1, 0], [True, False, True, True]
    )
    var shared = Series._from_chunks([source.slice(0, 2), source.slice(2, 2)])
    var split = Series._from_chunks([source.slice(0, 2), other.slice(2, 2)])
    var frame = DataFrame(
        [shared^, source.renamed("r"), Series("v", Column[Int32]([1, 2, 3, 4]))]
    )
    var independent = frame.with_column(split^)
    var total = col("v").sum()
    var one = frame.lazy().group_by("l").agg([total.copy()])
    var two = independent.lazy().group_by("l").agg([total.copy()])
    var joined = layout_keys(one)
    var separate = layout_keys(two)
    assert_false(joined[0] == separate[0])
    assert_equal(joined[1], separate[1])
    assert_equal(
        dictionary_properties(frame, plan_dictionary_group(one).value()), ""
    )

    # The pre-existing dense-pair calibration sees the same logical layout.
    var names: List[String] = ["l", "r"]
    var old_plan = frame.lazy().group_by(names).agg([total.copy()])
    var old_other = independent.lazy().group_by(names).agg([total.copy()])
    var old_operator = (
        "dictionary_pair_sum:dense:v1:"
        + DataType.INT32.name()
        + ":labels2:labels2:min_count0"
    )
    assert_equal(region_key(old_plan), old_operator)
    var unchanged = layout_keys(old_plan)
    assert_equal(unchanged[0], layout_keys(old_other)[0])
    assert_false(String(":code_spans") in unchanged[0])
    assert_equal(
        dictionary_properties(frame, plan_dictionary_group(old_plan).value()),
        "",
    )


def test_metal_sparse_dictionary_calibration_samples_code_tuples() raises:
    var labels = List[String]()
    for i in range(256):
        labels.append(String(i))
    var left = _coded_key("l", labels, [0, 0, 0, 0], [True, True, True, True])
    var right = _coded_key("r", labels, [0, 0, 0, 0], [True, True, True, True])
    var frame = DataFrame(
        [left^, right^, Series("v", Column[Int32]([1, 2, 3, 4]))]
    )
    var names: List[String] = ["l", "r"]
    var plan = frame.lazy().group_by(names).agg([col("v").sum()])
    var matched = plan_dictionary_group(plan)
    assert_equal(
        dictionary_properties(frame, matched.value()),
        ":code_sample:v1:philox:groups1:hot8:null0",
    )
    var varied = frame.with_column(
        _coded_key("r", labels, [1, 2, 3, 1], [True, True, False, True])
    )
    assert_equal(
        dictionary_properties(varied, matched.value()),
        ":code_sample:v1:philox:groups4:hot4:null2",
    )
