"""Independent Polars/Python first-match JSONPath corpus and invalid inputs."""
# contract-group: kernel.jsonpath
from std.collections import Optional
from std.testing import assert_equal, assert_true
from dataframe.json_path import JsonPath, append_json
from dataframe.json_reader import JsonReader, json_bytes
from dataframe.json_number import validate_json_numbers


def _unhex(text: StringSlice) -> String:
    var data = text.as_bytes()
    var out = List[UInt8](capacity=len(data) // 2)
    for i in range(0, len(data), 2):
        var a = Int(data[i])
        var b = Int(data[i + 1])
        a = a - 48 if a <= 57 else a - 87
        b = b - 48 if b <= 57 else b - 87
        out.append(UInt8((a << 4) | b))
    return String(unsafe_from_utf8=out)


def _run(text: String, query: String) raises -> Optional[String]:
    var path = JsonPath(query)
    var reader = JsonReader(json_bytes(text))
    try:
        reader.parse(json_bytes(text))
        validate_json_numbers(reader, False)
    except:
        return None
    var scratch = List[Int]()
    var selected = path.first(reader, scratch)
    if selected < 0 or reader.nodes[selected].kind == 110:
        return None
    var bytes = List[UInt8]()
    if reader.nodes[selected].kind == 34:
        reader.append_string(reader.nodes[selected].start, bytes)
    else:
        append_json(reader, selected, bytes)
    return String(unsafe_from_utf8=bytes)


def test_jsonpath_matches_independent_polars_corpus() raises:
    var corpus = open("tests/fixtures/json_strings/jsonpath.tsv", "r").read()
    var failures = 0
    var cases = 0
    var by_query = Dict[String, Int]()
    var reference_cases = 0
    for line in corpus.split("\n"):
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var fields = line.split("\t")
        var query = _unhex(fields[0])
        var text = _unhex(fields[1])
        var threw = False
        var result: Optional[String] = None
        try:
            result = _run(text, query)
        except:
            threw = True
        var expected_error = fields[2] == "ERROR"
        if len(fields) > 3 and fields[3] == "PYTHON_REFERENCE_UPSTREAM_PANIC":
            assert_equal(query, "$[?(@.a[*] > 1)]")
            reference_cases += 1
        var matches = threw == expected_error
        if matches and not threw:
            if fields[2] == "NULL":
                matches = not result
            else:
                matches = Bool(result) and result.value() == _unhex(fields[2])
        if not matches:
            failures += 1
            if query not in by_query:
                by_query[query] = 0
            by_query[query] += 1
            if failures <= 30:
                print(
                    "MISMATCH",
                    query,
                    text,
                    "got",
                    "ERROR" if threw else (
                        result.value().copy() if result else String("NULL")
                    ),
                    "expected",
                    fields[2],
                )
        cases += 1
    print("JSONPATH_CORPUS", cases, "cases", failures, "mismatches")
    print("UPSTREAM_PANIC_REFERENCE_CASES", reference_cases)
    for query in by_query:
        print("QUERY_MISMATCHES", query, by_query[query])
    assert_true(cases > 1500)
    assert_equal(failures, 0)
