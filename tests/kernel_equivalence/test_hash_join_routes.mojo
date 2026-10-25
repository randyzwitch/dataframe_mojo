"""Prepared hash, bounded membership and physical build routes retain typed rows."""
# contract-group: kernel.order_keys
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, StringColumn
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.string_view import StringViewBuilder
from dataframe.join_hash import (
    prepare_hash_index,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
    direct_hash_join_rows,
    direct_hash_semi_anti_rows,
    prefer_left_build,
    prepare_progression_index,
)
from dataframe.parallel import worker_count
from tests.support.lifecycle_jobs import WorkerSetting
from dataframe.join_fuse import FusedJoin
from dataframe.frame import (
    _join_range_possible,
    _join_range_rows,
    _smaller_build_join_rows,
    _bounded_int64_join_rows,
    _range_int64_membership_rows,
    _range_int64_membership_domain,
    _range_join_table_capacity,
    _left_builds,
)


def numeric_probes[D: DType]() raises:
    var left_values: List[Scalar[D]] = [
        9,
        Scalar[D].MAX,
        Scalar[D].MIN,
        5,
        5,
        0,
        77,
    ]
    var left_valid: List[Bool] = [True, True, True, True, True, False, True]
    var right_values: List[Scalar[D]] = [
        5,
        Scalar[D].MIN,
        5,
        Scalar[D].MAX,
        9,
        0,
    ]
    var right_valid: List[Bool] = [True, True, True, True, True, False]
    for offset in [0, 1, 7]:
        var padded_left = List[Scalar[D]](length=offset, fill=Scalar[D].MAX)
        var padded_right = List[Scalar[D]](length=offset, fill=Scalar[D].MAX)
        var lv = List[Bool](length=offset, fill=False)
        var rv = lv.copy()
        padded_left.extend(left_values.copy())
        padded_right.extend(right_values.copy())
        lv.extend(left_valid.copy())
        rv.extend(right_valid.copy())
        var left = Series("k", Column[Scalar[D]](padded_left^, lv)).slice(
            offset, 7
        )
        var right = Series("k", Column[Scalar[D]](padded_right^, rv)).slice(
            offset, 6
        )
        for chunked in [False, True]:
            var probe = left.copy()
            var build = right.copy()
            if chunked:
                probe = Series._from_chunks(
                    [left.slice(0, 1), left.slice(1, 6)]
                )
                build = Series._from_chunks(
                    [right.slice(0, 3), right.slice(3, 3)]
                )
            var prepared = prepare_hash_index(
                [build.copy()], allow_progression=False
            )
            assert_true(not prepared.progression)
            assert_equal(len(prepared.probe([probe.copy()]).left_hashes[]), 0)
            for outer in [False, True]:
                var expected_left = List[Int]()
                var expected_right = List[Int]()
                for i in range(7):
                    var found = False
                    for j in range(6):
                        if (
                            left_valid[i]
                            and right_valid[j]
                            and left_values[i] == right_values[j]
                        ):
                            expected_left.append(i)
                            expected_right.append(j)
                            found = True
                    if outer and not found:
                        expected_left.append(i)
                        expected_right.append(-1)
                var actual = prepared_hash_join_rows(
                    [probe.copy()], prepared, outer
                )
                assert_equal(actual[0], expected_left)
                assert_equal(actual[1], expected_right)
            assert_equal(
                prepared_hash_semi_anti_rows([probe.copy()], prepared, True),
                [0, 1, 2, 3, 4],
            )
            assert_equal(
                prepared_hash_semi_anti_rows([probe.copy()], prepared, False),
                [5, 6],
            )


def test_all_integer_widths_prepared_probes_hash_on_read_and_preserve_extreme_null_duplicate_row_pairs() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            numeric_probes[D]()


def test_range_samples_match_literal_rows_for_offsets_empty_chunks_nulls_and_signed_extremes() raises:
    for n in [0, 1, 127, 257, 100003]:
        for offset in [0, 3]:
            var values = List[Int64](length=offset, fill=Int64.MAX)
            var valid = List[Bool](length=offset, fill=False)
            for row in range(n):
                values.append(
                    Int64.MIN if row % 17
                    == 0 else Int64.MAX if row % 19
                    == 0 else Int64(row % 11 - 5)
                )
                valid.append(row % 23 != 3)
            var key = Series(
                "k", Column[Int64](values.copy(), valid.copy())
            ).slice(offset, n)
            var chunks = Series._from_chunks(
                [
                    key.slice(0, 0),
                    key.slice(0, n // 3),
                    key.slice(n // 3, n - n // 3),
                    key.slice(n, 0),
                ]
            )
            var bounds: List[Tuple[Int64, Int64]] = [
                (Int64(-2), Int64(3)),
                (Int64.MIN, Int64.MIN),
                (Int64.MAX, Int64.MAX),
                (Int64.MIN, Int64.MAX),
            ]
            for bound in bounds:
                var expected = 0
                var samples = min(256, n)
                for i in range(samples):
                    var row = offset + i * (n // samples)
                    expected += Int(
                        valid[row]
                        and values[row] >= bound[0]
                        and values[row] <= bound[1]
                    )
                assert_equal(
                    _join_range_possible(key, bound[0], bound[1]), expected
                )
                assert_equal(
                    _join_range_possible(chunks, bound[0], bound[1]), expected
                )


def compact_numeric_probes[D: DType]() raises:
    var keys: List[Scalar[D]] = [Scalar[D].MIN, Scalar[D].MAX, 5, 0, 9]
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var n = 4099
    for row in range(n):
        values.append(keys[row % len(keys)])
        valid.append(row % 17 != 3)
    var build = Series("k", Column[Scalar[D]](values.copy(), valid.copy()))
    var probe_values = keys.copy()
    probe_values.append(77)
    probe_values.append(0)
    var probe_valid = List[Bool](length=len(probe_values), fill=True)
    probe_valid[len(probe_values) - 1] = False
    var probe = Series(
        "k", Column[Scalar[D]](probe_values.copy(), probe_valid.copy())
    )
    var index = prepare_hash_index([build^], allow_progression=False)
    var grouped = 0
    for bucket in index.indexes[]:
        grouped += len(bucket.groups)
    assert_true(grouped > 0, "exercise duplicate-group hash reconstruction")
    for outer in [False, True]:
        var expected_left = List[Int]()
        var expected_right = List[Int]()
        for i in range(len(probe_values)):
            var matched = False
            for j in range(n):
                if probe_valid[i] and valid[j] and probe_values[i] == values[j]:
                    expected_left.append(i)
                    expected_right.append(j)
                    matched = True
            if outer and not matched:
                expected_left.append(i)
                expected_right.append(-1)
        var actual = prepared_hash_join_rows([probe.copy()], index, outer)
        assert_equal(actual[0], expected_left)
        assert_equal(actual[1], expected_right)
    assert_equal(
        prepared_hash_semi_anti_rows([probe.copy()], index, True),
        [0, 1, 2, 3, 4],
    )
    assert_equal(
        prepared_hash_semi_anti_rows([probe.copy()], index, False), [5, 6]
    )


def test_integer_duplicate_group_compaction_reconstructs_full_width_hashes_and_preserves_order() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            compact_numeric_probes[D]()


def text_key(
    values: List[String], valid: List[Bool], native: Bool
) raises -> Series:
    if not native:
        return Series("s", StringColumn(values.copy(), valid.copy()))
    var builder = StringViewBuilder(len(values))
    for row in range(len(values)):
        if valid[row]:
            builder.append(StringSlice(values[row]))
        else:
            builder.append_null()
    return Series("s", StringColumn(builder^.finish()))


def test_native_and_offsets_strings_prepared_duplicate_groups_reuse_build_and_compare_full_utf8_nul_bytes() raises:
    var words: List[String] = [
        "",
        "abcdefghijkl",
        "abcdefghijklm",
        "same-prefix-abcdefgh-A",
        "same-prefix-abcdefgh-B",
        "é",
        "é",
        "a\x00b",
        "missing",
    ]
    var n = 12013
    var values = List[String](length=7, fill="padding")
    var valid = List[Bool](length=7, fill=False)
    var codes = List[UInt64](length=7, fill=UInt64.MAX)
    for row in range(n):
        values.append(words[(row * 3 + row // 8) % 8].copy())
        valid.append(row % 17 != 2)
        codes.append((UInt64(1) << 63) + UInt64(row % 3))
    var probe_valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    var probe_words = words.copy()
    probe_words.append("abcdefghijkl")
    var probe_codes = List[UInt64]()
    for row in range(10):
        probe_codes.append((UInt64(1) << 63) + UInt64(row % 3))
    for left_native in [False, True]:
        for right_native in [False, True]:
            var left = text_key(probe_words, probe_valid, left_native)
            var right = text_key(values, valid, right_native).slice(7, n)
            assert_equal(
                Bool(left._data[StringColumn]._view_storage), left_native
            )
            assert_equal(
                Bool(right._data[StringColumn]._view_storage), right_native
            )
            for compound in [False, True]:
                var build: List[Series] = [right.copy()]
                var probe: List[Series] = [left.copy()]
                if compound:
                    build.append(
                        Series("u", Column[UInt64](codes.copy())).slice(7, n)
                    )
                    probe.append(
                        Series("u", Column[UInt64](probe_codes.copy()))
                    )
                var index = prepare_hash_index(build, allow_progression=False)
                assert_equal(
                    len(index.probe(probe).left_hashes[]),
                    len(left) if compound else 0,
                )
                var grouped_rows = 0
                var slots = 0
                for bucket in index.indexes[]:
                    grouped_rows += len(bucket.groups)
                    slots += len(bucket.slots)
                assert_true(
                    grouped_rows > 0,
                    "actual duplicate-group representation required",
                )
                assert_true(
                    slots < n // 2,
                    "duplicate groups must compact the actual hash index",
                )
                var shared = index.copy()
                assert_equal(
                    Int(index.indexes.ptr()), Int(shared.indexes.ptr())
                )
                for outer in [False, True]:
                    var expected_left = List[Int]()
                    var expected_right = List[Int]()
                    var semi = List[Int]()
                    var anti = List[Int]()
                    for i in range(10):
                        var found = False
                        for j in range(n):
                            if (
                                probe_valid[i]
                                and valid[j + 7]
                                and probe_words[i] == values[j + 7]
                                and (
                                    not compound
                                    or probe_codes[i] == codes[j + 7]
                                )
                            ):
                                expected_left.append(i)
                                expected_right.append(j)
                                found = True
                        if found:
                            semi.append(i)
                        else:
                            anti.append(i)
                            if outer:
                                expected_left.append(i)
                                expected_right.append(-1)
                    for _ in range(2):
                        var actual = prepared_hash_join_rows(
                            probe, shared, outer, omit_identity=True
                        )
                        assert_true(not actual[2])
                        assert_equal(actual[0], expected_left)
                        assert_equal(actual[1], expected_right)
                    var direct = direct_hash_join_rows(probe, build, outer)
                    assert_equal(direct[0], expected_left)
                    assert_equal(direct[1], expected_right)
                    assert_equal(
                        prepared_hash_semi_anti_rows(probe, index, True), semi
                    )
                    assert_equal(
                        prepared_hash_semi_anti_rows(probe, index, False), anti
                    )
                    assert_equal(
                        direct_hash_semi_anti_rows(probe, build, True), semi
                    )
                    assert_equal(
                        direct_hash_semi_anti_rows(probe, build, False), anti
                    )


def test_join_range_guards_use_exact_bounds_and_bounded_membership_keeps_each_literal_row() raises:
    var n = 8192
    for low in [Int64.MIN + 1, Int64.MAX - 5]:
        for dtype in [
            DataType.INT64,
            DataType.DATE,
            DataType.datetime("ns", "America/New_York"),
        ]:
            var values = List[Int64](length=n + 7, fill=0)
            var valid = List[Bool](length=n + 7, fill=True)
            for row in [9, 10, 1003, n - 1]:
                values[row + 7] = low + Int64(row % 3)
            valid[17] = False
            var right = (
                Series("k", Column[Int64](values^, valid))
                .with_dtype(dtype)
                .slice(7, n)
            )
            right = Series._from_chunks(
                [right.slice(0, 4093), right.slice(4093, n - 4093)]
            )
            var left = Series(
                "k", Column[Int64]([low, low + 2, 0], [True, True, False])
            ).with_dtype(dtype)
            var selected = _join_range_rows(left, right)
            assert_true(Bool(selected))
            assert_equal(selected.value(), [9, 1003, n - 1])
            var lrows = List[Int]()
            var rrows = List[Int]()
            var unfused = FusedJoin()
            assert_true(
                _bounded_int64_join_rows(
                    right, left, True, lrows, rrows, unfused
                )
            )
            assert_equal(len(lrows), n)
            assert_equal(len(rrows), n)
            for row in range(n):
                assert_equal(lrows[row], row)
                var expected = -1
                if row in [9, 1003, n - 1] and row % 3 != 1:
                    expected = 0 if row % 3 == 0 else 1
                assert_equal(rrows[row], expected)
    var left = Series("k", Column[Int64]([7]))
    var misleading = List[Int64](length=n, fill=7)
    for row in range(0, n, n // 256):
        misleading[row] = 0
    assert_true(
        not _join_range_rows(left, Series("k", Column[Int64](misleading^)))
    )
    var wide = Series("k", Column[Int64]([Int64.MIN, Int64.MAX]))
    assert_true(not _join_range_rows(wide, left))
    for input in [
        Series("k", Column[Int64]([])),
        Series("k", Column[Int64]([Int64.MAX], [False])),
    ]:
        assert_equal(len(_join_range_rows(input, left).value()), 0)
    var float_key = Series("k", Column[Float64]([7.0]))
    assert_true(not _join_range_rows(float_key, float_key))
    var build = Series(
        "k", Column[Int64]([-3, -1, -1, 2, 0], [True, True, True, True, False])
    )
    var probes = List[Int64]()
    var validity = List[Bool]()
    var semi = List[Int]()
    var anti = List[Int]()
    for row in range(131073):
        probes.append(Int64(row % 9 - 4))
        validity.append(row % 13 != 2)
        if row % 13 != 2 and (row % 9 - 4 in [-3, -1, 2]):
            semi.append(row)
        else:
            anti.append(row)
    var probe = Series("k", Column[Int64](probes^, validity))
    for matched in [False, True]:
        var actual = _range_int64_membership_rows(probe, build, matched)
        assert_true(actual[0])
        assert_equal(actual[1], semi if matched else anti)
    var over_budget = List[Int64](length=400000, fill=0)
    over_budget[399999] = 12000000
    var domain = _range_int64_membership_domain(
        Series("k", Column[Int64](over_budget^))
    )
    assert_true(not domain.supported)
    assert_equal(len(domain.present), 0)
    for slot_bytes in [1, 8, 16]:
        assert_equal(_range_join_table_capacity(0, slot_bytes), 0)
        assert_equal(
            _range_join_table_capacity(Int.MAX, slot_bytes),
            128 * 1024 * 1024 // slot_bytes,
        )


def test_smaller_build_order_restoration_capacity_boundary_and_compact_progression_have_complete_rows() raises:
    for left_rows in [-1, 0, 1, 16384, 16385, Int(Int32.MAX) + 1]:
        for right_rows in [-1, 524287, 524288, Int.MAX]:
            assert_equal(
                prefer_left_build(left_rows, right_rows),
                right_rows >= 524288
                and left_rows > 0
                and left_rows <= Int(Int32.MAX)
                and left_rows <= right_rows // 32,
            )
    var n = 524288
    var keys = List[Int64]()
    var texts = List[String]()
    var ids = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        keys.append(Int64(3 + 4 * (row % 3)))
        texts.append("other")
        ids.append(Int64(row))
        valid.append(row % 97 != 2)
    for row in [13, 29, 1003, 65537, n - 1]:
        keys[row] = 7 if row % 2 else 3
        texts[row] = "a" if row % 3 else "b"
    var left_keys: List[Int64] = [7, 3, 7, 11, 0]
    var left_valid: List[Bool] = [True, True, True, True, False]
    var left_texts: List[String] = ["a", "a", "b", "missing", "a"]
    var left = DataFrame(
        [
            Series("k", Column[Int64](left_keys.copy(), left_valid.copy())),
            Series("s", Column[String](left_texts.copy())),
            Series("l", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](keys.copy(), valid.copy())),
            Series("s", Column[String](texts.copy())),
            Series("r", Column[Int64](ids^)),
        ]
    )
    assert_true(prefer_left_build(left.height(), right.height()))
    assert_true(not _join_range_rows(left.column("k"), right.column("k")))
    for outer in [False, True]:
        var expected_left = List[Int]()
        var expected_right = List[Int]()
        for i in range(5):
            var found = False
            for j in range(n):
                if (
                    left_valid[i]
                    and valid[j]
                    and left_keys[i] == keys[j]
                    and left_texts[i] == texts[j]
                ):
                    expected_left.append(i)
                    expected_right.append(j)
                    found = True
            if outer and not found:
                expected_left.append(i)
                expected_right.append(-1)
        var pairs = _smaller_build_join_rows(
            [left.column("k"), left.column("s")],
            [right.column("k"), right.column("s")],
            outer,
        )
        assert_equal(pairs[0], expected_left)
        assert_equal(pairs[1], expected_right)
        var actual = left.join(right, ["k", "s"], "left" if outer else "inner")
        assert_equal(actual.height(), len(expected_left))
        for row in range(actual.height()):
            var i = expected_left[row]
            var j = expected_right[row]
            assert_equal(actual.item(row, "l").int64(), Int64(i))
            assert_equal(actual.item(row, "k").is_null(), not left_valid[i])
            if left_valid[i]:
                assert_equal(actual.item(row, "k").int64(), left_keys[i])
            assert_equal(actual.item(row, "s").string(), left_texts[i])
            assert_equal(actual.item(row, "r").is_null(), j < 0)
            if j >= 0:
                assert_equal(actual.item(row, "r").int64(), Int64(j))
    var progression = List[Int64]()
    for row in range(n):
        progression.append(Int64(row) * 3 + 5)
    var compact = prepare_progression_index(
        [Series("k", Column[Int64](progression^))]
    )
    assert_true(Bool(compact))
    assert_true(compact.value().progression)
    assert_equal(len(compact.value().indexes[]), 0)
    assert_equal(compact.value().progression_row(8, n), 1)
    assert_equal(compact.value().progression_row(7, n), -1)
    var nan = Float64(0) / Float64(0)
    var float_pairs = _smaller_build_join_rows(
        [
            Series("f", Column[Float64]([nan, -0.0, 0.0, 8])),
            Series("s", Column[String](["a", "b", "a", "missing"])),
        ],
        [
            Series(
                "f",
                Column[Float64](
                    [0.0, nan, nan, -0.0, 0], [True, True, True, True, False]
                ),
            ),
            Series("s", Column[String](["a", "a", "a", "b", "missing"])),
        ],
        True,
    )
    assert_equal(float_pairs[0], [0, 0, 1, 2, 3])
    assert_equal(float_pairs[1], [1, 2, 3, 0, -1])


def test_build_key_bits_reject_only_absent_keys_in_joins_semi_and_anti() raises:
    # A sparse Int64 build (every 37th value over a span, duplicates, a
    # null, negative values) probed by more rows than it has: the build
    # keys' bits are made, and a probe key below, inside or past the span,
    # null or not, matches exactly as a scalar scan says.
    var build_values = List[Int64]()
    var build_valid = List[Bool]()
    for i in range(400):
        build_values.append(Int64(i * 37 - 5000))
        build_valid.append(i % 97 != 5)
    build_values.append(Int64(37 * 3 - 5000))
    build_valid.append(True)
    var build = Series(
        "k", Column[Int64](build_values.copy(), build_valid.copy())
    )
    var probe_values = List[Int64]()
    var probe_valid = List[Bool]()
    for i in range(20_000):
        probe_values.append(Int64((i * 7919) % 30_000 - 12_000))
        probe_valid.append(i % 211 != 7)
    probe_values.append(Int64.MIN)
    probe_valid.append(True)
    probe_values.append(Int64.MAX)
    probe_valid.append(True)
    var probe = Series(
        "k", Column[Int64](probe_values.copy(), probe_valid.copy())
    )
    var prepared = prepare_hash_index(
        [build.copy()], allow_progression=False, probe_rows=len(probe)
    )
    assert_true(
        len(prepared.filter[]) > 0, "the build keys' bits were not made"
    )
    var expected_pairs = List[Int]()
    var semi = List[Int]()
    var anti = List[Int]()
    for i in range(len(probe_values)):
        var count = 0
        if probe_valid[i]:
            for j in range(len(build_values)):
                if build_valid[j] and build_values[j] == probe_values[i]:
                    count += 1
        expected_pairs.append(count)
        if count > 0:
            semi.append(i)
        else:
            anti.append(i)
    var rows = prepared_hash_join_rows([probe.copy()], prepared, False)
    var seen = List[Int](length=len(probe_values), fill=0)
    for k in range(len(rows[1])):
        var left = rows[0][k]
        var right = rows[1][k]
        assert_equal(build_values[right], probe_values[left])
        assert_true(build_valid[right])
        seen[left] += 1
    for i in range(len(probe_values)):
        assert_equal(seen[i], expected_pairs[i])
    assert_equal(
        prepared_hash_semi_anti_rows([probe.copy()], prepared, True), semi
    )
    assert_equal(
        prepared_hash_semi_anti_rows([probe.copy()], prepared, False), anti
    )
    # A build with more rows than probe it makes no bits.
    var small = prepare_hash_index(
        [build.copy()], allow_progression=False, probe_rows=10
    )
    assert_equal(len(small.filter[]), 0)


def test_eager_join_builds_on_the_side_of_fewer_rows_times_columns() raises:
    # DuckDB's build side: fewer rows times columns builds, the right on a
    # tie, with no minimum size. A 300-row left against a 20,000-row right
    # builds on the left and still returns left-major rows in right order,
    # nulls and duplicate keys on both sides included.
    var narrow = DataFrame([Series("k", Column[Int64]([1, 2]))])
    var wide = DataFrame(
        [Series("k", Column[Int64]([1])), Series("v", Column[Int64]([2]))]
    )
    # Two rows of one column against one row of two: a tie either way.
    assert_true(not _left_builds(narrow, wide))
    assert_true(not _left_builds(wide, narrow))
    assert_true(not _left_builds(narrow.slice(0, 0), wide))
    var left_rows = 300
    var right_rows = 20000
    var left_keys = List[Int64]()
    var left_valid = List[Bool]()
    for i in range(left_rows):
        left_keys.append(Int64(i % 250))
        left_valid.append(i % 41 != 3)
    var right_keys = List[Int64]()
    var right_valid = List[Bool]()
    var right_ids = List[Int64]()
    for j in range(right_rows):
        right_keys.append(Int64((j * 7) % 400))
        right_valid.append(j % 37 != 5)
        right_ids.append(Int64(j))
    var left_ids = List[Int64]()
    for i in range(left_rows):
        left_ids.append(Int64(i))
    var left = DataFrame(
        [
            Series("k", Column[Int64](left_keys.copy(), left_valid.copy())),
            Series("l", Column[Int64](left_ids^)),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys.copy(), right_valid.copy())),
            Series("r", Column[Int64](right_ids^)),
        ]
    )
    assert_true(_left_builds(left, right))
    assert_true(not _left_builds(right, left))
    assert_true(not prefer_left_build(left.height(), right.height()))
    for outer in [False, True]:
        var expected_left = List[Int]()
        var expected_right = List[Int]()
        for i in range(left_rows):
            var found = False
            for j in range(right_rows):
                if (
                    left_valid[i]
                    and right_valid[j]
                    and left_keys[i] == right_keys[j]
                ):
                    expected_left.append(i)
                    expected_right.append(j)
                    found = True
            if outer and not found:
                expected_left.append(i)
                expected_right.append(-1)
        var actual = left.join(right, ["k"], "left" if outer else "inner")
        assert_equal(actual.height(), len(expected_left))
        for row in range(actual.height()):
            assert_equal(
                actual.item(row, "l").int64(), Int64(expected_left[row])
            )
            var j = expected_right[row]
            assert_equal(actual.item(row, "r").is_null(), j < 0)
            if j >= 0:
                assert_equal(actual.item(row, "r").int64(), Int64(j))


def test_parallel_left_major_restore_matches_the_serial_order() raises:
    # Many pairs a left row: the restore counts and scatters on several
    # workers, and every left group keeps its pairs in probe order, as the
    # serial counting sort does. Duplicate keys on both sides.
    var left_keys = List[Int64]()
    for i in range(500):
        left_keys.append(Int64(i % 400))
    var right_keys = List[Int64]()
    for j in range(300_000):
        right_keys.append(Int64((j * 7919) % 450))
    var left: List[Series] = [Series("k", Column[Int64](left_keys^))]
    var right: List[Series] = [Series("k", Column[Int64](right_keys^))]
    var one = WorkerSetting(1)
    var serial = _smaller_build_join_rows(left, right, False)
    _ = one^
    var four = WorkerSetting(4)
    assert_true(worker_count(len(serial[0])) > 1)
    var parallel = _smaller_build_join_rows(left, right, False)
    _ = four^
    assert_equal(parallel[0], serial[0])
    assert_equal(parallel[1], serial[1])
