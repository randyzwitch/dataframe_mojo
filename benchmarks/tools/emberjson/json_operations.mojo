from std.time import perf_counter_ns
from std.testing import assert_equal
from dataframe.dtype import DataType
from dataframe.json.reader import JsonReader, json_bytes
from dataframe.json.decode import JsonDecoder, JsonDecodeOptions
from dataframe.json.number import validate_json_numbers
from dataframe.json.path import JsonPath, append_json


def main() raises:
    var text = String(
        '{"id":17,"name":"caf\\u00e9","score":12.5,"active":true}'
    )
    var dtype = DataType.struct(
        ["id", "name", "score", "active"],
        [DataType.INT64, DataType.STRING, DataType.FLOAT64, DataType.BOOL],
    )
    var reader = JsonReader(json_bytes(text))
    var count = 20000
    for run in range(8):
        var decoder = JsonDecoder(dtype, count, JsonDecodeOptions(io=True))
        var start = perf_counter_ns()
        for _ in range(count):
            reader.parse(json_bytes(text))
            validate_json_numbers(reader)
            decoder.append(reader)
        var result = decoder.finish()
        var elapsed = perf_counter_ns() - start
        assert_equal(len(result), count)
        if run:
            print("decode", elapsed)
    var path = JsonPath("$.name")
    var scratch = List[Int]()
    var output = List[UInt8]()
    for run in range(8):
        var start = perf_counter_ns()
        var checksum = 0
        for _ in range(count):
            reader.parse(json_bytes(text))
            validate_json_numbers(reader, False)
            var found = path.first(reader, scratch)
            output.clear()
            reader.append_string(reader.nodes[found].start, output)
            checksum += len(output)
        var elapsed = perf_counter_ns() - start
        assert_equal(checksum, count * 5)
        if run:
            print("jsonpath", elapsed)
