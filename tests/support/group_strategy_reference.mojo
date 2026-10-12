"""Independent finite-domain group models and scoped C-string environment settings."""
from std.ffi import external_call
from std.os import getenv


struct CStringArgument(Movable):
    var bytes: List[UInt8]

    def __init__(out self, value: String):
        self.bytes = List[UInt8](value.as_bytes())
        self.bytes.append(0)


struct ScopedEnvironment(Movable):
    var name: String
    var previous: String
    var existed: Bool

    def __init__(out self, name: String, value: String) raises:
        self.name = name.copy()
        var key = CStringArgument(name)
        # Environment values cannot contain NUL, so this sentinel separates
        # an absent setting from a present empty value using the stdlib ABI.
        self.previous = getenv(name, "\x00missing group setting")
        self.existed = self.previous != "\x00missing group setting"
        var text = CStringArgument(value)
        var result = external_call["setenv", Int32](
            Int(key.bytes.unsafe_ptr()), Int(text.bytes.unsafe_ptr()), Int32(1)
        )
        _ = key^
        _ = text^
        if result != 0:
            raise Error("Cannot configure group strategy test environment")

    def __deinit__(deinit self):
        var key = CStringArgument(self.name)
        if self.existed:
            var text = CStringArgument(self.previous)
            _ = external_call["setenv", Int32](
                Int(key.bytes.unsafe_ptr()),
                Int(text.bytes.unsafe_ptr()),
                Int32(1),
            )
            _ = text^
        else:
            _ = external_call["unsetenv", Int32](Int(key.bytes.unsafe_ptr()))
        _ = key^


struct StrategyTrace(Movable):
    """Capture the existing opt-in production path trace, restoring stderr."""

    var file: Int
    var previous: Int32

    def __init__(out self) raises:
        self.file = external_call["tmpfile", Int]()
        self.previous = external_call["dup", Int32](Int32(2))
        if self.file == 0 or self.previous < 0:
            if self.file != 0:
                _ = external_call["fclose", Int32](self.file)
            if self.previous >= 0:
                _ = external_call["close", Int32](self.previous)
            raise Error("Cannot capture streaming strategy trace")
        var descriptor = external_call["fileno", Int32](self.file)
        if external_call["dup2", Int32](descriptor, Int32(2)) < 0:
            _ = external_call["close", Int32](self.previous)
            _ = external_call["fclose", Int32](self.file)
            raise Error("Cannot redirect streaming strategy trace")

    def text(self) raises -> String:
        _ = external_call["fflush", Int32](self.file)
        if external_call["fseek", Int32](self.file, Int64(0), Int32(0)) != 0:
            raise Error("Cannot read streaming strategy trace")
        var bytes = List[UInt8]()
        while True:
            var byte = external_call["fgetc", Int32](self.file)
            if byte < 0:
                break
            bytes.append(UInt8(byte))
        return String(unsafe_from_utf8=bytes^)

    def __deinit__(deinit self):
        _ = external_call["dup2", Int32](self.previous, Int32(2))
        _ = external_call["close", Int32](self.previous)
        _ = external_call["fclose", Int32](self.file)


struct GroupSample(Copyable, Movable):
    var group: Int
    var key: Int64
    var key_valid: Bool
    var string_key: String
    var string_key_valid: Bool
    var dimension: Int64
    var integer: Int64
    var integer_valid: Bool
    var floating: Float64
    var float_valid: Bool
    var text: String
    var text_valid: Bool

    def __init__(
        out self, row: Int, groups: Int, stride: Int = 1, phase: Int = 0
    ):
        self.group = (row // stride + groups - 3) % groups
        var visit = row // (stride * groups)
        self.key_valid = self.group != 0
        self.key = Int64(self.group) if self.key_valid else Int64.MAX
        self.string_key_valid = self.group % 11 != 0
        self.string_key = (
            "é:"
            + String(self.group % 7) if self.string_key_valid else "hidden key"
        )
        self.dimension = Int64((row + phase) % stride)
        self.integer_valid = row % 13 != 0 and self.group != groups - 1
        self.integer = Int64(
            self.group % 5 - 2 if self.group % 2 == 0 else visit % 7 - 3
        ) if self.integer_valid else Int64.MAX
        self.float_valid = row % 17 != 0 and self.group != groups - 1
        var code = visit % 7
        self.floating = -0.0 if visit % 2 == 0 else 0.0
        if self.group % 2:
            self.floating = (
                Float64(0) / Float64(0) if code
                == 0 else -0.0 if code
                == 1 else 0.0 if code
                == 2 else 1.0 if code
                == 3 else -1.0 if code
                == 4 else Float64(1) / Float64(0) if code
                == 5 else -Float64(1) / Float64(0)
            )
        if not self.float_valid:
            self.floating = Float64(1) / Float64(0)
        self.text_valid = row % 19 != 0 and self.group != groups - 1
        var tokens: List[String] = [
            "",
            "é",
            "a\x00b",
            "a longer Unicode value 東京",
        ]
        self.text = "固定" if self.group % 2 == 0 else tokens[visit % 4].copy()
        if not self.text_valid:
            self.text = "hidden text"


struct ScalarGroup(Copyable, Movable):
    var rows: Int
    var count: Int
    var total: Int128
    var histogram: List[Int]
    var integers: List[Int64]
    var floats: List[Float64]
    var words: List[String]
    var integer_null: Bool
    var float_null: Bool
    var word_null: Bool
    var first: Int64
    var last: Int64
    var first_valid: Bool
    var last_valid: Bool
    var float_first: Float64
    var float_last: Float64
    var float_first_valid: Bool
    var float_last_valid: Bool

    def __init__(out self):
        self.rows = 0
        self.count = 0
        self.total = 0
        self.histogram = List[Int](length=7, fill=0)
        self.integers = []
        self.floats = []
        self.words = []
        self.integer_null = False
        self.float_null = False
        self.word_null = False
        self.first = 0
        self.last = 0
        self.first_valid = False
        self.last_valid = False
        self.float_first = 0
        self.float_last = 0
        self.float_first_valid = False
        self.float_last_valid = False

    def add(mut self, sample: GroupSample):
        if self.rows == 0:
            self.first = sample.integer
            self.first_valid = sample.integer_valid
            self.float_first = sample.floating
            self.float_first_valid = sample.float_valid
        self.rows += 1
        self.last = sample.integer
        self.last_valid = sample.integer_valid
        self.float_last = sample.floating
        self.float_last_valid = sample.float_valid
        if sample.integer_valid:
            self.total += Int128(sample.integer)
            self.count += 1
            self.histogram[Int(sample.integer) + 3] += 1
            if sample.integer not in self.integers:
                self.integers.append(sample.integer)
        else:
            self.integer_null = True
        if sample.float_valid:
            var seen = False
            for previous in self.floats:
                seen = (
                    seen
                    or previous == sample.floating
                    or (
                        previous != previous
                        and sample.floating != sample.floating
                    )
                )
            if not seen:
                self.floats.append(sample.floating)
        else:
            self.float_null = True
        if sample.text_valid:
            if sample.text not in self.words:
                self.words.append(sample.text.copy())
        else:
            self.word_null = True

    def median(self) -> Float64:
        var lower = (self.count - 1) // 2
        var upper = self.count // 2
        var cursor = 0
        var a = 0
        var b = 0
        for bin in range(7):
            var next = cursor + self.histogram[bin]
            if cursor <= lower and lower < next:
                a = bin - 3
            if cursor <= upper and upper < next:
                b = bin - 3
            cursor = next
        return Float64(a + b) / 2
