"""Build/use the Mojo ABI observer and scope native-library selection."""
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true
from dataframe.io.parquet import parquet_library_candidates

comptime ResetFn = def() thin abi("C") -> None
comptime GetFn = def(Int32) thin abi("C") -> Int64


def set_setting(name: String, value: String):
    var key = name + "\0"
    var contents = value + "\0"
    if value == "":
        _ = external_call["unsetenv", Int32](Int(key.unsafe_ptr()))
    else:
        _ = external_call["setenv", Int32](
            Int(key.unsafe_ptr()), Int(contents.unsafe_ptr()), Int32(1)
        )
    _ = key^
    _ = contents^


def shell_quote(text: String) -> String:
    var result = String("'")
    for byte in text.as_bytes():
        if byte == 39:
            result += "'\\''"
        else:
            var one: List[UInt8] = [byte]
            result += String(unsafe_from_utf8=one)
    return result + "'"


struct Observer(Movable):
    var previous_library: String
    var previous_delegate: String
    var path: String
    var delegate: String
    var handle: Int
    var reset_entry: Int
    var get_entry: Int

    def __init__(out self) raises:
        self.previous_library = getenv("DATAFRAME_PARQUET_LIBRARY")
        self.previous_delegate = getenv("DATAFRAME_TEST_PARQUET_DELEGATE")
        self.path = (
            getenv("COVERAGE_TEST_TMP")
            + "/parquet-observer"
            + (".dylib" if CompilationTarget.is_macos() else ".so")
        )
        self.delegate = ""
        self.handle = 0
        self.reset_entry = 0
        self.get_entry = 0
        for candidate in parquet_library_candidates():
            if exists(candidate):
                self.delegate = candidate.copy()
                break
        assert_true(
            self.delegate != "",
            "Parquet integration requires the existing native libdfparquet backend",
        )
        if not exists(self.path):
            # The observer is Mojo, built from source for the current compiler;
            # no checked-in executable or Python harness is needed at runtime.
            var command = (
                "mojo build --emit shared-lib tests/support/parquet_observer.mojo -o "
                + shell_quote(self.path)
                + "\0"
            )
            assert_equal(
                external_call["system", Int32](command.unsafe_ptr()), Int32(0)
            )
            _ = command^
        var c_path = self.path + "\0"
        self.handle = external_call["dlopen", Int](
            c_path.unsafe_ptr(), Int32(2)
        )
        _ = c_path^
        assert_true(self.handle != 0)
        var reset = String("df_test_reset\0")
        var get = String("df_test_get\0")
        self.reset_entry = external_call["dlsym", Int](
            self.handle, reset.unsafe_ptr()
        )
        self.get_entry = external_call["dlsym", Int](
            self.handle, get.unsafe_ptr()
        )
        assert_true(self.reset_entry != 0 and self.get_entry != 0)
        _ = reset^
        _ = get^
        set_setting("DATAFRAME_TEST_PARQUET_DELEGATE", self.delegate)
        set_setting("DATAFRAME_PARQUET_LIBRARY", self.path)
        self.reset()

    def __deinit__(deinit self):
        set_setting("DATAFRAME_PARQUET_LIBRARY", self.previous_library)
        set_setting("DATAFRAME_TEST_PARQUET_DELEGATE", self.previous_delegate)
        if self.handle:
            _ = external_call["dlclose", Int32](self.handle)

    def reset(self):
        var entry = self.reset_entry
        Pointer(to=entry).unsafe_bitcast[ResetFn]()[]()

    def get(self, index: Int) -> Int64:
        var entry = self.get_entry
        return Pointer(to=entry).unsafe_bitcast[GetFn]()[](Int32(index))

    def groups(self) -> List[Int]:
        var result = List[Int]()
        for index in range(max(0, Int(self.get(11)))):
            result.append(Int(self.get(16 + index)))
        return result^

    def projection(self) -> List[String]:
        var result = List[String]()
        for index in range(Int(self.get(10))):
            var bytes = List[UInt8]()
            for byte in range(31):
                var character = self.get(64 + 32 * index + byte)
                if character == 0:
                    break
                bytes.append(UInt8(character))
            result.append(String(unsafe_from_utf8=bytes^))
        return result^

    def last_stream(self, field: Int) -> Int64:
        return self.get(256 + 16 * (Int(self.get(128)) - 1) + field)

    def assert_balanced(self, streams: Int = -1) raises:
        assert_equal(
            self.get(5), self.get(128) if streams < 0 else Int64(streams)
        )
        assert_equal(self.get(6), self.get(7))
        assert_equal(self.get(8), self.get(9))


struct Locations(Movable):
    var library: String
    var prefix: String

    def __init__(out self):
        self.library = getenv("DATAFRAME_PARQUET_LIBRARY")
        self.prefix = getenv("CONDA_PREFIX")

    def __deinit__(deinit self):
        set_setting("DATAFRAME_PARQUET_LIBRARY", self.library)
        set_setting("CONDA_PREFIX", self.prefix)


struct WorkingDirectory(Movable):
    var previous: String

    def __init__(out self, path: String) raises:
        var bytes = List[UInt8](length=4096, fill=0)
        assert_true(external_call["getcwd", Int](bytes.unsafe_ptr(), 4096) != 0)
        var size = 0
        while bytes[size] != 0:
            size += 1
        var actual = List[UInt8]()
        for index in range(size):
            actual.append(bytes[index])
        self.previous = String(unsafe_from_utf8=actual^)
        var target = path + "\0"
        assert_equal(
            external_call["chdir", Int32](target.unsafe_ptr()), Int32(0)
        )
        _ = target^

    def __deinit__(deinit self):
        var target = self.previous + "\0"
        _ = external_call["chdir", Int32](target.unsafe_ptr())
        _ = target^
