"""Physical layout and sampled tuple distribution for dictionary SUM costs."""
from std.bit import next_power_of_two
from dataframe import DataFrame, DataType, Series
from .accelerator.gpu.typed_storage import coalesced_native_chunks
from .dictionary_group import DictionaryGroup, _signed_group_codes
from .dictionary_pair_reduce import _combine_dictionary_stripes
from .accelerator.gpu.dictionary_pair import SIMD_WIDTH
from .group_properties import sample_group_tuples


def dictionary_layout(source: Series) raises -> String:
    """Measure the same code and validity windows that wrapped execution uses."""
    var codes = _signed_group_codes(source)
    var spans = coalesced_native_chunks[DType.int32](codes)
    var layout = String(":code_spans[")
    for span in spans:
        # Express lengths as proportions to preserve the calibration row scale.
        layout += String((len(span) * 16 + len(source) // 2) // len(source))
        layout += ","
    return layout + "]"


def dictionary_stripe_key(
    frame: DataFrame, plan: DictionaryGroup
) raises -> String:
    """Use the execution selector and the wrapped key/value intersection."""
    if frame.height() <= SIMD_WIDTH:
        return ":kernel=direct"
    var codes = _signed_group_codes(frame.column(plan.keys[0]))
    var values = frame.column(plan.value)
    var lengths = List[Int]()
    if not codes.is_chunked() and not values.is_chunked():
        lengths.append(frame.height())
    else:
        var keys = coalesced_native_chunks[DType.int32](codes)
        var inputs: List[Series]
        if values.dtype() == DataType.INT32:
            inputs = coalesced_native_chunks[DType.int32](values)
        else:
            inputs = coalesced_native_chunks[DType.int64](values)
        # These native windows are precisely the inputs to intersect_numeric_views.
        # Work with row counts here so costing neither maps nor copies a device buffer.
        var k = 0
        var v = 0
        var key_end = len(keys[0])
        var value_end = len(inputs[0])
        var start = 0
        while start < frame.height():
            var end = min(key_end, value_end)
            lengths.append(end - start)
            start = end
            if start < frame.height():
                if start == key_end:
                    k += 1
                    key_end += len(keys[k])
                if start == value_end:
                    v += 1
                    value_end += len(inputs[v])
    return ":kernel=striped" if _combine_dictionary_stripes(
        codes, lengths
    ) else ":kernel=direct"


def dictionary_properties(
    frame: DataFrame, plan: DictionaryGroup
) raises -> String:
    """Sample sparse numeric code tuples without reading dictionary strings."""
    if (
        len(plan.keys) != 2
        or frame.height() <= 0
        or frame.height() >= Int(Int32.MAX)
        or plan.dense_dictionary(frame.height())
    ):
        return String()
    var codes = List[Series]()
    for key in plan.keys:
        codes.append(_signed_group_codes(frame.column(key)))
    # At most 4096 positions contribute. Group count and skew describe only
    # the sample and never determine the table's capacity or result.
    var sample = sample_group_tuples(DataFrame(codes^), plan.keys)
    return (
        ":code_sample:v1:philox:groups"
        + String(next_power_of_two(sample.distinct))
        + ":hot"
        + String((sample.hottest * 8 + sample.samples - 1) // sample.samples)
        + ":null"
        + String((sample.nulls * 8 + sample.samples - 1) // sample.samples)
    )
