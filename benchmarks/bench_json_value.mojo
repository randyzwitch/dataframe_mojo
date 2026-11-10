"""Width scaling for JSON construction, all-field lookup, and canonicalization.

mojo run -I . benchmarks/bench_json_value.mojo
Times are minimum nanoseconds across three repetitions; setup is untimed.
"""
from std.time import perf_counter_ns
from std.testing import assert_equal
from dataframe.json.value import JsonValue, json_quote


def main() raises:
    print("fields parse_ns lookup_ns canonical_ns")
    for count in [100, 200, 400, 800, 1600, 3200]:
        var source = String("{")
        var keys = List[String]()
        for i in range(count):
            if i:
                source += ","
            var key = String(10000 + count - i)
            keys.append(key)
            source += json_quote(key) + ":" + String(i)
        source += "}"
        var parse_ns = Int.MAX
        var lookup_ns = Int.MAX
        var canonical_ns = Int.MAX
        for _ in range(3):
            var start = perf_counter_ns()
            var value = JsonValue(source)
            parse_ns = min(parse_ns, Int(perf_counter_ns() - start))
            var checksum = 0
            start = perf_counter_ns()
            for key in keys:
                checksum += Int(value.get(key).text)
            lookup_ns = min(lookup_ns, Int(perf_counter_ns() - start))
            assert_equal(checksum, count * (count - 1) // 2)
            start = perf_counter_ns()
            var canonical = value.canonical()
            canonical_ns = min(canonical_ns, Int(perf_counter_ns() - start))
            assert_equal(JsonValue(canonical).get(keys[0]).text, "0")
        print(count, parse_ns, lookup_ns, canonical_ns)
