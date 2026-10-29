"""Parquet interchange against independent files and observed native I/O."""
# contract-group: io.parquet
from std.ffi import external_call
from std.os import getenv, mkdir
from std.os.path import exists
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true
from dataframe import (
    DataFrame,
    Series,
    DataType,
    Expr,
    col,
    read_parquet,
    scan_parquet,
    write_parquet,
    parquet_row_group_statistics,
    ArrowArray,
    ArrowSchema,
    export_arrow_series,
    import_arrow_series,
)
from std.memory import Pointer
from tests.support.arrow_foreign import (
    packed_metadata,
    text_bytes,
    array_release,
    schema_release,
)
from tests.support.lifecycle_stream import at
from dataframe.io.parquet import (
    _ParquetBatches,
    parquet_library_candidates,
    parquet_backend_version,
)
from tests.support.parquet_runtime import Observer, set_setting
from tests.support.lifecycle_jobs import WorkerSetting


def fixture(name: String) -> String:
    return "tests/fixtures/parquet/" + name + ".parquet"


def assert_core_rows(
    frame: DataFrame, rows: List[Int], names: List[String]
) raises:
    assert_equal(frame.columns(), names)
    assert_equal(frame.height(), len(rows))
    var integers: List[Int64] = [0, 1, 2, 3, 100, 101, 102, 200, 201]
    var strings: List[String] = [
        "red",
        "",
        "blue",
        "",
        "red",
        "green",
        "red",
        "blue",
        "last",
    ]
    for row in range(len(rows)):
        var original = rows[row]
        for name in names:
            if name == "x":
                assert_equal(frame.item(row, name).int64(), integers[original])
            elif name == "s":
                assert_equal(frame.column(name).dtype(), DataType.STRING)
                if original == 3:
                    assert_true(frame.item(row, name).is_null())
                else:
                    assert_equal(
                        frame.item(row, name).string(), strings[original]
                    )
            else:
                if original == 2:
                    assert_true(frame.item(row, name).is_null())
                else:
                    assert_equal(
                        frame.item(row, name).float64(), Float64(original + 1)
                    )


def test_parquet_projection_group_order_duplicates_and_empty_selection_match_typed_rows() raises:
    var observer = Observer()
    for threads in [False, True]:
        observer.reset()
        var frame = read_parquet(
            fixture("plain"),
            columns=["s", "x"],
            row_groups=List[Int]([3, 0, 2, 0]),
            use_threads=threads,
        )
        assert_core_rows(
            frame, [7, 8, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3], ["s", "x"]
        )
        assert_equal(observer.groups(), List[Int]([3, 0, 2, 0]))
        assert_equal(observer.projection(), List[String](["s", "x"]))
        assert_equal(observer.get(3), Int64(4))
        assert_equal(observer.get(4), Int64(13))
        observer.assert_balanced()
        observer.reset()
        var empty = read_parquet(
            fixture("plain"), columns=["x", "s"], row_groups=List[Int]()
        )
        assert_core_rows(empty, [], ["x", "s"])
        assert_equal(observer.get(11), Int64(0))
        assert_equal(observer.get(4), Int64(0))
        observer.assert_balanced()


def test_parquet_pruning_counts_native_decoded_groups_and_preserves_unknown_bounds() raises:
    var observer = Observer()
    var predicates: List[Expr] = [
        col("x") < 100,
        (col("x") >= 100) & (col("x") < 200),
        (col("x") < 100) | (col("x") >= 200),
    ]
    var expected_groups: List[List[Int]] = [[0, 1], [1, 2], [0, 1, 3]]
    var expected_rows: List[List[Int]] = [
        [0, 1, 2, 3],
        [4, 5, 6],
        [0, 1, 2, 3, 7, 8],
    ]
    for index in range(len(predicates)):
        observer.reset()
        var frame = (
            scan_parquet(fixture("plain"))
            .filter(predicates[index])
            .select(["x"])
            .collect(streaming=True, batch_size=2, engine="cpu")
        )
        assert_core_rows(frame, expected_rows[index], ["x"])
        assert_equal(observer.get(1), Int64(1))
        assert_equal(observer.groups(), expected_groups[index])
        # A separate initial schema probe may decode the first native batch;
        # this record counts only the final execution stream's actual batches.
        assert_equal(
            observer.last_stream(2), Int64(len(expected_groups[index]))
        )
        assert_equal(observer.last_stream(3), Int64(len(expected_rows[index])))
        observer.assert_balanced()
    observer.reset()
    var unknown = (
        scan_parquet(fixture("unknown-statistics"))
        .filter(col("x") == 201)
        .select(["x"])
        .collect(streaming=True, batch_size=2, engine="cpu")
    )
    assert_core_rows(unknown, [8], ["x"])
    if observer.get(11) >= 0:
        assert_equal(observer.groups(), List[Int]([0, 1, 2, 3]))
    assert_equal(observer.last_stream(2), Int64(4))
    assert_equal(observer.last_stream(3), Int64(9))
    observer.assert_balanced()
    # The final x column's data page is corrupt while its footer is intact.
    # Success below proves the excluded page was never decoded.
    observer.reset()
    var pruned_corrupt = (
        scan_parquet(fixture("late-corrupt"))
        .filter(col("x") < 100)
        .select(["x"])
        .collect(streaming=True, batch_size=2, engine="cpu")
    )
    assert_core_rows(pruned_corrupt, [0, 1, 2, 3], ["x"])
    assert_equal(observer.groups(), List[Int]([0, 1]))
    assert_equal(observer.last_stream(2), Int64(2))
    assert_equal(observer.get(12), Int64(0))
    observer.assert_balanced()


def test_parquet_dictionary_and_plain_encodings_keep_strings_nulls_and_group_semantics() raises:
    var observer = Observer()
    for file in ["dictionary", "plain"]:
        observer.reset()
        var frame = read_parquet(fixture(file))
        assert_core_rows(
            frame, [0, 1, 2, 3, 4, 5, 6, 7, 8], ["x", "s", "score"]
        )
        assert_equal(observer.get(3), Int64(4))
        observer.assert_balanced()
        for batch in [1, 3, 17]:
            observer.reset()
            var result = (
                scan_parquet(fixture(file))
                .group_by("s")
                .agg(col("x").sum())
                .collect(streaming=True, batch_size=batch, engine="cpu")
                .sort("x")
            )
            assert_equal(result.height(), 6)
            var seen_keys = List[String]()
            # Compare each group by its key independently of hash-group order.
            for row in range(result.height()):
                var value = result.item(row, "s")
                var key = String(
                    "<null>"
                ) if value.is_null() else value.string()
                for previous in seen_keys:
                    assert_true(
                        key != previous, "each dictionary group appears once"
                    )
                seen_keys.append(key)
                if value.is_null():
                    assert_equal(result.item(row, "x").int64(), Int64(3))
                elif value.string() == "":
                    assert_equal(result.item(row, "x").int64(), Int64(1))
                elif value.string() == "red":
                    assert_equal(result.item(row, "x").int64(), Int64(202))
                elif value.string() == "blue":
                    assert_equal(result.item(row, "x").int64(), Int64(202))
                elif value.string() == "green":
                    assert_equal(result.item(row, "x").int64(), Int64(101))
                else:
                    assert_equal(value.string(), "last")
                    assert_equal(result.item(row, "x").int64(), Int64(201))
            observer.assert_balanced()


def test_parquet_empty_groups_and_empty_typed_files_preserve_schema_and_eof_cleanup() raises:
    var observer = Observer()
    var statistics = parquet_row_group_statistics(fixture("plain"))
    assert_equal(statistics.height(), 4)
    var rows: List[Int64] = [4, 0, 3, 2]
    for row in range(4):
        assert_equal(statistics.item(row, "row_group").int32(), Int32(row))
        assert_equal(statistics.item(row, "rows").int64(), rows[row])
    observer.reset()
    var only_empty = read_parquet(
        fixture("plain"), columns=["x", "s"], row_groups=List[Int]([1])
    )
    assert_core_rows(only_empty, [], ["x", "s"])
    assert_equal(observer.last_stream(2), Int64(1))
    assert_equal(observer.get(4), Int64(0))
    observer.assert_balanced()
    observer.reset()
    var empty = read_parquet(fixture("empty"))
    assert_equal(empty.height(), 0)
    assert_equal(empty.width(), 13)
    assert_equal(
        empty.column("stamp").dtype(),
        DataType.datetime("ns", "America/New_York"),
    )
    assert_equal(empty.column("duration").dtype(), DataType.duration("us"))
    assert_true(empty.column("list").dtype().is_list())
    assert_true(empty.column("struct").dtype().is_struct())
    observer.assert_balanced()


def test_parquet_late_corruption_raises_after_valid_batches_and_column_projection_avoids_it() raises:
    var observer = Observer()
    var cursor = _ParquetBatches(fixture("late-corrupt"), ["x"], None, 2)
    var seen = List[Int64]()
    var rejected = False
    try:
        while True:
            var batch = cursor.next()
            if not batch:
                break
            for row in range(batch.value().height()):
                seen.append(batch.value().item(row, "x").int64())
    except error:
        if String(error).find("read_parquet stream:") < 0:
            raise error ^
        rejected = True
    assert_true(
        rejected, "late corrupted data page must propagate its native error"
    )
    var expected: List[Int64] = [0, 1, 2, 3, 100, 101, 102]
    assert_equal(seen, expected)
    assert_equal(observer.get(3), Int64(3))
    assert_equal(observer.get(12), Int64(1))
    observer.assert_balanced()
    _ = cursor ^
    observer.assert_balanced()
    observer.reset()
    var projected = read_parquet(fixture("late-corrupt"), columns=["s"])
    assert_core_rows(projected, [0, 1, 2, 3, 4, 5, 6, 7, 8], ["s"])
    assert_equal(observer.projection(), List[String](["s"]))
    assert_equal(observer.get(12), Int64(0))
    observer.assert_balanced()
    observer.reset()
    var public_rejected = False
    try:
        _ = (
            scan_parquet(fixture("late-corrupt"))
            .select(["x"])
            .collect(streaming=True, batch_size=2, engine="cpu")
        )
    except error:
        if String(error).find("read_parquet stream:") < 0:
            raise error ^
        public_rejected = True
    assert_true(public_rejected)
    observer.assert_balanced()


def consume_first_native_batch(path: String, fail: Bool) raises -> DataFrame:
    var cursor = _ParquetBatches(path, ["x", "s"], None, 1)
    var first = cursor.next().value().copy()
    if fail:
        raise Error("deliberate parquet consumer stop")
    return first ^


def test_parquet_early_close_consumer_error_and_public_head_release_actual_native_exports() raises:
    var observer = Observer()
    var first = consume_first_native_batch(fixture("plain"), False)
    assert_core_rows(first, [0], ["x", "s"])
    assert_equal(observer.get(3), Int64(1))
    assert_equal(observer.get(4), Int64(4))
    observer.assert_balanced()
    observer.reset()
    var rejected = False
    try:
        _ = consume_first_native_batch(fixture("plain"), True)
    except error:
        if String(error).find("deliberate parquet consumer stop") < 0:
            raise error ^
        rejected = True
    assert_true(rejected)
    assert_equal(observer.get(3), Int64(1))
    observer.assert_balanced()
    observer.reset()
    var setting = WorkerSetting(1)
    var head = (
        scan_parquet(fixture("late-corrupt"))
        .select(["x"])
        .head(1)
        .collect(streaming=True, batch_size=1, engine="cpu")
    )
    assert_core_rows(head, [0], ["x"])
    assert_equal(observer.last_stream(2), Int64(1))
    assert_equal(observer.get(12), Int64(0))
    observer.assert_balanced()
    _ = setting ^


def assert_typed_rows(frame: DataFrame) raises:
    assert_equal(frame.height(), 3)
    assert_equal(frame.width(), 13)
    assert_equal(frame.item(0, "i").int8(), Int8(-2))
    assert_equal(frame.item(2, "i").int8(), Int8(7))
    assert_equal(frame.item(0, "u").uint64(), UInt64.MAX)
    assert_equal(frame.item(2, "u").uint64(), UInt64(0))
    assert_equal(frame.item(0, "f").float32(), Float32(1.25))
    assert_equal(frame.item(2, "f").float32(), Float32(-2.5))
    assert_equal(frame.item(0, "flag").bool(), True)
    assert_equal(frame.item(2, "flag").bool(), False)
    assert_equal(frame.item(0, "text").string(), "")
    assert_equal(frame.item(2, "text").string(), 'café\nquoted "x"')
    var binary: List[UInt8] = [255, 0]
    assert_equal(frame.item(0, "binary").bytes(), binary)
    assert_equal(frame.item(2, "binary").bytes(), List[UInt8]())
    assert_equal(frame.item(0, "day").to_physical(), Int64(-1))
    assert_equal(frame.item(2, "day").to_physical(), Int64(19782))
    assert_equal(
        frame.column("stamp").dtype(),
        DataType.datetime("ns", "America/New_York"),
    )
    assert_equal(frame.item(0, "stamp").to_physical(), Int64(-1))
    assert_equal(frame.item(2, "stamp").to_physical(), Int64(42))
    assert_equal(frame.column("duration").dtype(), DataType.duration("us"))
    assert_equal(frame.item(0, "duration").to_physical(), Int64(-1))
    assert_equal(frame.item(2, "duration").to_physical(), Int64(42))
    assert_equal(frame.item(2, "time").to_physical(), Int64(42))
    assert_equal(String(frame.item(0, "decimal")), "-12.34")
    assert_equal(String(frame.item(2, "decimal")), "56.78")
    var list = frame.item(0, "list").list()
    assert_equal(len(list), 3)
    assert_equal(list.get(0).int64(), Int64(1))
    assert_true(list.get(1).is_null())
    assert_equal(list.get(2).int64(), Int64(3))
    assert_equal(len(frame.item(1, "list").list()), 0)
    assert_true(frame.item(2, "list").is_null())
    assert_equal(frame.item(0, "struct").struct_field("id").int64(), Int64(7))
    assert_equal(frame.item(0, "struct").struct_field("name").string(), "a")
    assert_equal(frame.item(2, "struct").struct_field("id").int64(), Int64(9))
    assert_true(frame.item(2, "struct").struct_field("name").is_null())
    for name in frame.columns():
        if name != "list":
            assert_true(frame.item(1, name).is_null())


def with_independent_field_metadata(frame: DataFrame) raises -> DataFrame:
    var fields = List[Series]()
    var encoded = packed_metadata(
        [text_bytes("application")], [text_bytes("independent")]
    )
    for name in frame.columns():
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(frame.column(name), array, schema)
        # The independent foreign C consumer supplies this metadata before
        # importing the released public export into a new owned Series.
        schema.metadata = Int(encoded.unsafe_ptr())
        fields.append(import_arrow_series(array, schema))
    _ = encoded ^
    return DataFrame(fields ^)


# expected-failure: Public Parquet read drops arbitrary field metadata
# expected-failure: independently stored by PyArrow; verified via public Arrow
# expected-failure: export after read. Writer tested separately with explicit
# expected-failure: foreign C metadata, and independent consumer verifies its
# expected-failure: preservation.
# expected-diagnostic: Parquet arbitrary field metadata must survive import
def test_parquet_reader_preserves_independent_arbitrary_field_metadata() raises:
    var source = read_parquet(fixture("typed"))
    var expected = packed_metadata(
        [text_bytes("application")], [text_bytes("independent")]
    )
    for name in source.columns():
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(source.column(name), array, schema)
        var present = schema.metadata != 0
        if present:
            for byte in range(len(expected)):
                assert_equal(
                    at[UInt8](schema.metadata)[unsafe_offset=byte],
                    expected[byte],
                )
        array_release(array)
        schema_release(schema)
        assert_true(
            present, "Parquet arbitrary field metadata must survive import"
        )


def test_parquet_public_writer_preserves_independently_produced_types_metadata_codecs_and_bounds() raises:
    var observer = Observer()
    var source = with_independent_field_metadata(read_parquet(fixture("typed")))
    assert_typed_rows(source)
    for codec in ["zstd", "snappy", "uncompressed"]:
        for size in [1, 2, 17]:
            var path = (
                getenv("COVERAGE_TEST_TMP")
                + "/parquet-written-"
                + codec
                + "-"
                + String(size)
                + ".parquet"
            )
            observer.reset()
            write_parquet(source, path, compression=codec, row_group_size=size)
            observer.assert_balanced(0)
            var result = read_parquet(path)
            assert_typed_rows(result)
            var stats = parquet_row_group_statistics(path)
            for group in range(stats.height()):
                assert_true(stats.item(group, "rows").int64() <= Int64(size))
    var empty = source.head(0)
    var empty_path = (
        getenv("COVERAGE_TEST_TMP") + "/parquet-written-empty.parquet"
    )
    write_parquet(empty, empty_path)
    var empty_result = read_parquet(empty_path)
    assert_equal(empty_result.height(), 0)
    assert_equal(empty_result.columns(), empty.columns())
    for name in empty.columns():
        assert_equal(
            empty_result.column(name).dtype(), empty.column(name).dtype()
        )
    assert_typed_rows(source)
    # parquet_oracle.py --check-written consumes these actual public writer
    # outputs independently, validating cells/types/metadata/codecs/group bounds.


def test_parquet_library_discovery_uses_actual_explicit_prefix_fallback_and_reports_failures() raises:
    from tests.support.parquet_runtime import Locations, WorkingDirectory

    var previous_library = getenv("DATAFRAME_PARQUET_LIBRARY")
    var previous_delegate = getenv("DATAFRAME_TEST_PARQUET_DELEGATE")
    var previous_prefix = getenv("CONDA_PREFIX")
    var observer = Observer()
    var locations = Locations()
    var expected_version = parquet_backend_version()
    assert_true(expected_version != "")
    var root = getenv("COVERAGE_TEST_TMP") + "/parquet-prefix"
    var library_directory = root + "/lib"
    if not exists(root):
        mkdir(root)
    if not exists(library_directory):
        mkdir(library_directory)
    var name = String(
        "libdfparquet.dylib" if CompilationTarget.is_macos() else "libdfparquet.so"
    )
    var alias = library_directory + "/" + name
    if not exists(alias):
        var c_source = observer.path + "\0"
        var c_alias = alias + "\0"
        assert_equal(
            external_call["link", Int32](
                c_source.unsafe_ptr(), c_alias.unsafe_ptr()
            ),
            Int32(0),
        )
        _ = c_source ^
        _ = c_alias ^
    set_setting("DATAFRAME_PARQUET_LIBRARY", observer.path)
    set_setting("CONDA_PREFIX", root)
    assert_equal(
        parquet_library_candidates(),
        List[String]([observer.path, alias, "build/dfparquet/" + name]),
    )
    assert_equal(parquet_backend_version(), expected_version)
    set_setting("DATAFRAME_PARQUET_LIBRARY", root + "/missing-library")
    assert_equal(parquet_backend_version(), expected_version)
    var broken = root + "/not-a-library"
    with open(broken, "w") as file:
        file.write("This is an existing invalid dynamic library.\n")
    set_setting("DATAFRAME_PARQUET_LIBRARY", broken)
    var rejected = False
    try:
        _ = parquet_backend_version()
    except error:
        if String(error).find("read_parquet could not load " + broken) < 0:
            raise error ^
        rejected = True
    assert_true(rejected)
    # Prefix and default paths are absent in this isolated working directory.
    # An explicitly missing library must report the paths it actually searched.
    set_setting("DATAFRAME_PARQUET_LIBRARY", root + "/missing-library")
    set_setting("CONDA_PREFIX", "")
    var working = WorkingDirectory(root)
    rejected = False
    try:
        _ = parquet_backend_version()
    except error:
        if (
            String(error).find("read_parquet needs libdfparquet; looked for")
            < 0
        ):
            raise error ^
        rejected = True
    assert_true(rejected)
    _ = working ^
    _ = locations ^
    # Locations captured the observer override. Keep the observer alive until
    # after that inner guard restores its snapshot, then restore the caller.
    _ = observer ^
    assert_equal(getenv("DATAFRAME_PARQUET_LIBRARY"), previous_library)
    assert_equal(getenv("DATAFRAME_TEST_PARQUET_DELEGATE"), previous_delegate)
    assert_equal(getenv("CONDA_PREFIX"), previous_prefix)


def test_parquet_invalid_requests_and_native_writer_io_errors_leave_sources_and_exports_valid() raises:
    var observer = Observer()
    var source = read_parquet(fixture("typed"))
    for failure in range(6):
        observer.reset()
        var rejected = False
        try:
            if failure == 0:
                _ = read_parquet(fixture("plain"), columns=["x", "x"])
            elif failure == 1:
                _ = read_parquet(fixture("plain"), row_groups=List[Int]([-1]))
            elif failure == 2:
                _ = read_parquet(fixture("plain"), row_groups=List[Int]([99]))
            elif failure == 3:
                _ = read_parquet(fixture("plain"), columns=["absent"])
            elif failure == 4:
                write_parquet(
                    source,
                    getenv("COVERAGE_TEST_TMP") + "/invalid-options.parquet",
                    compression="unknown",
                )
            else:
                write_parquet(
                    source,
                    getenv("COVERAGE_TEST_TMP") + "/invalid-options.parquet",
                    row_group_size=0,
                )
        except error:
            var message = String(error)
            var valid = (
                message.find("read_parquet") >= 0
                or message.find("write_parquet") >= 0
            )
            if not valid:
                raise error ^
            rejected = True
        assert_true(rejected)
        observer.assert_balanced(0)
        assert_typed_rows(source)
    observer.reset()
    var failed_write = False
    try:
        write_parquet(
            source,
            getenv("COVERAGE_TEST_TMP")
            + "/parquet-nonexistent-parent/output.parquet",
        )
    except error:
        if String(error).find("write_parquet:") < 0:
            raise error ^
        failed_write = True
    assert_true(failed_write)
    assert_equal(observer.get(6), Int64(1))
    assert_equal(observer.get(8), Int64(1))
    observer.assert_balanced(0)
    assert_typed_rows(source)
