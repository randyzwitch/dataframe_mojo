"""Independent CPU key-domain/hash/worker contracts for grouped GPU planning."""
# contract-group: public.apple_parity
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    from_wkb,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.categorical import decode
from dataframe.string_column import StringCodes
from dataframe.partition import _hash_column, _mix, low_cardinality
from dataframe.accelerator.grouped_key_domains import (
    KeyDomain,
    direct_category_codes,
    coded_string_dtype,
    kind,
    STRING,
    CATEGORY,
    CODED_STRING,
    BINARY,
    GEOMETRY,
)
from tests.support.lifecycle_jobs import WorkerSetting
from dataframe.parallel import configured_workers


def _coded(source: Series) raises -> Series:
    # decode() alone deliberately produces plain text views. Parquet-style
    # dictionary provenance is an independent owner beside those views.
    var result = decode(source)
    var codes = source.uint32()
    var values = List[UInt32](capacity=len(source))
    for row in range(len(source)):
        values.append(codes._get(row))
    result._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(values^), source.dtype())
    )
    result._field_metadata = source._field_metadata
    assert_true(Bool(coded_string_dtype(result)))
    assert_true(Bool(result._dictionary_codes()))
    assert_equal(kind(result), CODED_STRING)
    return result^


def _key(domain: Int, values: List[Int64], valid: List[Bool]) raises -> Series:
    if domain == CATEGORY or domain == CODED_STRING:
        var largest = Int64(0)
        for row in range(len(values)):
            if valid[row]:
                largest = max(largest, values[row])
        var dictionary = CategoricalDictionary()
        for code in range(Int(largest) + 1):
            # Swapped labels distinguish original code hashing from decoded
            # text hashing: code0's label hashes AFTER code4's at4/8buckets.
            dictionary.append(
                "k4" if code == 0 else "k0" if code == 4 else "k" + String(code)
            )
        var codes = List[UInt32]()
        for value in values:
            codes.append(UInt32(value))
        var key = Series("k", Column[UInt32](codes^, valid)).with_dtype(
            DataType.categorical(dictionary^)
        )
        return _coded(key) if domain == CODED_STRING else key^
    if domain == STRING:
        var text = List[String]()
        for value in values:
            text.append("k" + String(value))
        return Series("k", StringColumn(text, valid))
    var rows = List[List[UInt8]]()
    for value in values:
        var bytes = List[UInt8]()
        if domain == GEOMETRY:
            bytes.extend([UInt8(1), UInt8(1), UInt8(0), UInt8(0), UInt8(0)])
        var word = bitcast[DType.uint64](
            Float64(value)
        ) if domain == GEOMETRY else UInt64(value)
        for byte in range(8):
            bytes.append(UInt8((word >> UInt64(8 * byte)) & UInt64(255)))
        if domain == GEOMETRY:
            bytes.extend(List[UInt8](length=8, fill=0))
        rows.append(bytes^)
    var key = Series.binary("k", rows, valid)
    return from_wkb(key, '{"crs":"EPSG:4326"}') if domain == GEOMETRY else key^


def _effective(source: Series) raises -> Series:
    if source.dtype() == DataType.STRING:
        var codes = source._dictionary_codes()
        if codes:
            return codes.take()
    return source.copy()


def _hashes(source: Series) raises -> List[UInt64]:
    var contiguous = source.rechunk()
    var result = List[UInt64](length=len(source), fill=0)
    _hash_column(contiguous, 0, len(source), Int(result.unsafe_ptr()), True)
    return result^


def test_grouped_key_domain_original_categorical_and_coded_string_identity() raises:
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    dictionary.append("unused")
    var source = Series(
        "k",
        Column[UInt32](
            [0, 1, 0, 1, UInt32.MAX], [True, True, True, True, False]
        ),
    ).with_dtype(DataType.categorical(dictionary^))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    var decoded = _coded(source)
    decoded._field_metadata = source._field_metadata
    var text = Series(
        "k",
        StringColumn(
            ["same", "same", "same", "same", ""],
            [True, True, True, True, False],
        ),
    )
    var raw: List[Int64] = [10, 20, 30, 40, 50]
    for workers in List[Int]([1, 2, 4, 8]):
        var setting = WorkerSetting(workers)
        assert_equal(configured_workers(), workers)
        for input in List[Series]([source.copy(), decoded.copy(), text.copy()]):
            var categorical = input.dtype().is_categorical() or Bool(
                coded_string_dtype(input)
            )
            var grouped = (
                DataFrame(
                    [input.copy(), Series("x", Column[Int64](raw.copy()))]
                )
                .group_by("k", maintain_order=True)
                .agg(col("x").implode().alias("xs"))
            )
            assert_equal(grouped.height(), 3 if categorical else 2)
            var lists = grouped.column("xs").list_column()
            var offsets: List[Int64] = [0, 2, 4, 5]
            var children: List[Int64] = [10, 30, 20, 40, 50]
            if not categorical:
                offsets = [0, 4, 5]
                children = [10, 20, 30, 40, 50]
            assert_equal(lists._offsets[], offsets)
            assert_equal(lists.child().int64().to_list(), children)
            var domain = KeyDomain(input)
            assert_equal(
                domain.kind,
                CATEGORY if input.dtype().is_categorical() else CODED_STRING if categorical else STRING,
            )
            if categorical:
                assert_true(
                    domain.encoded._field_metadata.value().ptr()
                    == source._field_metadata.value().ptr()
                )
            var restored = domain.restore(domain.encoded)
            assert_true(restored.equals(input))
            if categorical:
                assert_equal(domain.encoded.uint32().value(0), UInt32(0))
                assert_equal(domain.encoded.uint32().value(1), UInt32(1))
                assert_equal(
                    domain.provenance.values[],
                    List[UInt64]([UInt64(0), UInt64(1), UInt64(2)]),
                )
                assert_false(domain.provenance.string_hash)
                assert_true(
                    domain.provenance.dtype.dictionary().ptr()
                    == source.dtype().dictionary().ptr()
                )
                var planning = KeyDomain(input, metadata_only=True)
                assert_equal(len(planning.encoded), 0)
                assert_equal(len(planning.provenance.values[]), 0)
                assert_true(
                    planning.provenance.dtype.dictionary().ptr()
                    == source.dtype().dictionary().ptr()
                )
        _ = setting^
    # Different StringCodes dictionaries disable CPU GroupBy._coded_keys.
    var a = CategoricalDictionary()
    a.append("same")
    a.append("unusedA")
    var b = CategoricalDictionary()
    b.append("same")
    b.append("unusedB")
    var left = _coded(
        Series("k", Column[UInt32]([0])).with_dtype(DataType.categorical(a^))
    )
    var right = _coded(
        Series("k", Column[UInt32]([0])).with_dtype(DataType.categorical(b^))
    )
    var mixed = Series._from_chunks([left^, right^])
    assert_false(Bool(coded_string_dtype(mixed)))
    assert_equal(kind(mixed), STRING)
    var result = (
        DataFrame([mixed.copy(), Series("x", Column[Int64]([7, 8]))])
        .group_by("k")
        .agg(col("x").implode().alias("xs"))
    )
    assert_equal(result.height(), 1)
    assert_equal(
        result.column("xs").list_column().child().int64().to_list(),
        List[Int64]([Int64(7), Int64(8)]),
    )


def test_grouped_key_domain_independent_raw_hashes_restore_and_null_poison() raises:
    var vectors: List[Int64] = [0, 4, 32, 65537, 262176]
    var valid = List[Bool](length=len(vectors), fill=True)
    var anchors: List[List[UInt64]] = [
        [
            0x34352C3AE5DBA3CD,
            0xD0D837314FA010C8,
            0xE57A7A3C79AB1838,
            0x05314E8EEC2A4875,
            0x20AB2F17736265D0,
        ],
        [
            0,
            0xB7A4712C74562914,
            0xADFB1EBB497FAD45,
            0x1644D33451F3F4FB,
            0x445902154BCFF2E9,
        ],
        [
            0,
            0xB7A4712C74562914,
            0xADFB1EBB497FAD45,
            0x1644D33451F3F4FB,
            0x445902154BCFF2E9,
        ],
        [
            0x01CEFC995DE0A3A5,
            0xF3377D872BCB705D,
            0x8425A841817F35B2,
            0x655873F165A116C4,
            0x9FFD2BDE88B5395D,
        ],
        [
            0xC3021B077580A591,
            0x3AC25D7541CEC3A9,
            0xC16983516B87C10D,
            0x307114069BFC78F1,
            0x3CB0916DE48B1530,
        ],
    ]
    for domain in range(STRING, GEOMETRY + 1):
        var source = _key(domain, vectors, valid)
        var effective = _effective(source)
        var hashes = _hashes(effective)
        assert_equal(hashes, anchors[domain - STRING])
        if domain == CODED_STRING:
            # Direct source text hashes differ; GroupBy's effective code key
            # is authoritative before route/sample/partition selection.
            var decoded_hashes = _hashes(source)
            assert_equal(decoded_hashes[0], UInt64(0xD0D837314FA010C8))
            assert_equal(hashes[0], UInt64(0))
        var transported = KeyDomain(source)
        var codes = transported.encoded.uint32()
        for row in range(len(source)):
            assert_equal(
                _mix(transported.provenance.values[][Int(codes.value(row))]),
                anchors[domain - STRING][row],
            )
        var selected = List[Int]([4, 1, 0, 1])
        assert_true(
            transported.restore(transported.encoded.take(selected)).equals(
                source.take(selected)
            )
        )
    # Binary key bytes are never interpreted as UTF8; per-field domains retain
    # embedded NUL, invalid sequences, empty valid bytes and null independently.
    var binary_values: List[List[UInt8]] = [
        [255],
        [195, 40],
        [97, 0, 98],
        [],
        [255],
    ]
    var binary = Series.binary(
        "k",
        binary_values,
        List[Bool]([True, True, True, True, False]),
    )
    var domain = KeyDomain(binary)
    assert_true(domain.restore(domain.encoded).equals(binary))
    var text = Series("k", StringColumn(["", "a\x00b", "é"]))
    var text_domain = KeyDomain(text)
    assert_true(text_domain.restore(text_domain.encoded).equals(text))
    var empty_dictionary = CategoricalDictionary()
    var null_key = Series(
        "k", Column[UInt32]([UInt32.MAX], [False])
    ).with_dtype(DataType.categorical(empty_dictionary^))
    var null_domain = KeyDomain(null_key)
    assert_equal(len(null_domain.provenance.values[]), 0)
    assert_false(null_domain.encoded.uint32().is_valid(0))
    assert_true(null_domain.restore(null_domain.encoded).equals(null_key))
    var hidden: List[UInt32] = [UInt32.MAX, 0]
    var dictionary = CategoricalDictionary()
    dictionary.append("x")
    var nullable = Series(
        "k", Column[UInt32](hidden^, List[Bool]([False, True]))
    ).with_dtype(DataType.categorical(dictionary^))
    var staged = direct_category_codes(nullable)
    assert_false(staged[0].uint32().is_valid(0))
    assert_equal(staged[0].uint32().value(1), UInt32(0))


def test_grouped_key_domain_cpu_worker_bucket_heavy_fault_order() raises:
    # Literal hashes above show0 precedes4 for text/binary/category, while
    # Geometry point4's hash precedes point0's. At8workers the40%hot bucket is
    # heavy and runs before light buckets regardless of its numeric bucket ID.
    var n = 262145
    var hot = 2 * n // 5
    var values = List[Int64]()
    for row in range(n):
        values.append(
            Int64(4) if row
            < hot else Int64(0) if row
            == hot else Int64(row + 32)
        )
    var valid = List[Bool](length=n, fill=True)
    var integers = List[Int64](length=n, fill=0)
    integers[hot] = 128
    var exponents = List[Int64](length=n, fill=0)
    exponents[0] = -1
    for domain in range(STRING, GEOMETRY + 1):
        var source = _key(domain, values, valid)
        var effective = _effective(source)
        assert_false(low_cardinality([effective.copy()]))
        var hashes = _hashes(effective)
        var counts = List[Int](length=16, fill=0)
        for hash in hashes:
            counts[Int(hash >> UInt64(60))] += 1
        var hot_buckets: List[Int] = [13, 11, 11, 15, 3]
        var zero_buckets: List[Int] = [3, 0, 0, 0, 12]
        var hot_counts: List[Int] = [114794, 114684, 114684, 114781, 114808]
        var zero_counts: List[Int] = [9732, 9870, 9870, 10048, 9809]
        assert_equal(
            counts[hot_buckets[domain - STRING]], hot_counts[domain - STRING]
        )
        assert_equal(
            counts[zero_buckets[domain - STRING]], zero_counts[domain - STRING]
        )
        var heavy = 0
        for count in counts:
            heavy += Int(count > max(n // 8, 4 * n // 16))
        assert_equal(heavy, 1)
        for workers in List[Int]([1, 2, 4, 8]):
            var setting = WorkerSetting(workers)
            assert_equal(configured_workers(), workers)
            var frame = DataFrame(
                [
                    source.copy(),
                    Series("u", Column[Int64](integers.copy())),
                    Series(
                        "base", Column[Int64](List[Int64](length=n, fill=2))
                    ),
                    Series("exp", Column[Int64](exponents.copy())),
                ]
            )
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    [
                        col("u").cast(DataType.INT8).implode().alias("first"),
                        col("base").pow(col("exp")).implode().alias("later"),
                    ]
                )
            )
            var power_first = workers == 8 or (
                domain == GEOMETRY and workers > 1
            )
            with assert_raises(
                contains="nonnegative exponent" if power_first else "cast from int64 to int8 failed"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            _ = setting^
