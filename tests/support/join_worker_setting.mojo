"""Scoped worker setting without redeclaring libc getenv's pointer ABI."""
from std.ffi import external_call
from std.os import getenv


struct JoinWorkers(Movable):
    var previous: String
    var existed: Bool

    def __init__(out self, count: Int) raises:
        # Worker settings are decimal integers or empty. This impossible
        # setting is only a stdlib default for an absent environment entry.
        comptime ABSENT = "dataframe-test-worker-setting-absent"
        self.previous = getenv("DATAFRAME_THREADS", ABSENT)
        self.existed = self.previous != ABSENT
        var name = String("DATAFRAME_THREADS\x00")
        var value = String(count) + "\x00"
        if (
            external_call["setenv", Int32](
                Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
            )
            != 0
        ):
            raise Error("cannot install worker setting")

    def __deinit__(deinit self):
        var name = String("DATAFRAME_THREADS\x00")
        if self.existed:
            var previous = self.previous + "\x00"
            _ = external_call["setenv", Int32](
                Int(name.unsafe_ptr()), Int(previous.unsafe_ptr()), Int32(1)
            )
        else:
            _ = external_call["unsetenv", Int32](Int(name.unsafe_ptr()))
