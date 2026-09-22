"""CSV schema, options, and input mapping shared by the single reader."""
from std.collections import Dict
from std.ffi import external_call
from std.memory import Pointer
from .dtype import DataType
from .frame import DataFrame


comptime CSV_INT64 = DataType.INT64
comptime CSV_FLOAT64 = DataType.FLOAT64
comptime CSV_BOOL = DataType.BOOL
comptime CSV_STRING = DataType.STRING


struct CsvField(Copyable):
    """One CSV output field. Names and dtypes are always explicit.

    Temporal fields take an optional strftime-style format; an empty format
    means ISO 8601 (see dataframe/temporal.mojo).
    """

    var name: String
    var dtype: DataType
    var nullable: Bool
    var format: String

    def __init__(
        out self,
        name: String,
        dtype: DataType,
        nullable: Bool = True,
        format: String = "",
    ):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
        self.format = format

    @staticmethod
    def int64(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_INT64, nullable)

    @staticmethod
    def float64(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_FLOAT64, nullable)

    @staticmethod
    def bool(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_BOOL, nullable)

    @staticmethod
    def string(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_STRING, nullable)

    @staticmethod
    def date(
        name: String, format: String = "", nullable: Bool = True
    ) -> CsvField:
        return CsvField(name, DataType.DATE, nullable, format)

    @staticmethod
    def datetime(
        name: String,
        unit: String = "us",
        format: String = "",
        nullable: Bool = True,
    ) raises -> CsvField:
        return CsvField(name, DataType.datetime(unit), nullable, format)

    @staticmethod
    def time(
        name: String, format: String = "", nullable: Bool = True
    ) -> CsvField:
        return CsvField(name, DataType.TIME, nullable, format)


struct CsvSchema(Copyable, Sized):
    """An ordered, non-empty set of uniquely named CSV fields."""

    var _fields: List[CsvField]

    def __init__(out self, var fields: List[CsvField]) raises:
        if len(fields) == 0:
            raise Error("CSV schema must contain at least one field")
        var names = Dict[String, Bool]()
        for field in fields:
            if field.name in names:
                raise Error("Duplicate CSV field name: " + field.name)
            names[field.name] = True
        self._fields = fields^

    def __len__(self) -> Int:
        return len(self._fields)

    @staticmethod
    def of(frame: DataFrame) raises -> CsvSchema:
        """A nullable schema matching a frame's names and dtypes."""
        var fields = List[CsvField](capacity=frame.width())
        for field in frame.schema():
            fields.append(CsvField(field.name, field.dtype, True))
        return CsvSchema(fields^)

    def field(self, index: Int) raises -> CsvField:
        if index < 0 or index >= len(self):
            raise Error("CSV schema index out of bounds")
        return self._fields[index].copy()


@fieldwise_init
struct CsvOptions(Copyable):
    """Validated reader options; see read_csv for their meaning."""

    var separator: String
    var quote_char: String
    var comment_prefix: String
    var skip_rows: Int
    var n_rows: Int
    var null_values: List[String]
    var ignore_errors: Bool
    var truncate_ragged_lines: Bool
    var encoding: String

    def validate(self) raises:
        if self.separator.byte_length() != 1:
            raise Error("CSV separator must be a single byte")
        var sep = self.separator.as_bytes()[0]
        if sep == 10 or sep == 13:
            raise Error("CSV separator cannot be CR or LF")
        if self.quote_char.byte_length() > 1:
            raise Error("CSV quote_char must be one byte or empty")
        if self.quote_char.byte_length() == 1:
            var q = self.quote_char.as_bytes()[0]
            if q == sep or q == 10 or q == 13:
                raise Error("CSV quote_char cannot be the separator, CR, or LF")
        for b in self.comment_prefix.as_bytes():
            if b == 10 or b == 13:
                raise Error("comment_prefix cannot contain CR or LF")
        if self.skip_rows < 0:
            raise Error("skip_rows must be nonnegative")
        if self.n_rows < -1:
            raise Error("n_rows must be nonnegative or -1")
        if self.encoding != "utf8" and self.encoding != "utf8-lossy":
            raise Error("encoding must be 'utf8' or 'utf8-lossy'")


def _options(
    separator: String,
    quote_char: String,
    comment_prefix: String,
    skip_rows: Int,
    n_rows: Int,
    null_values: List[String],
    ignore_errors: Bool,
    truncate_ragged_lines: Bool,
    encoding: String,
    buffer_size: Int,
) raises -> CsvOptions:
    if buffer_size <= 0:
        raise Error("CSV buffer_size must be positive")
    var options = CsvOptions(
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        n_rows,
        null_values.copy(),
        ignore_errors,
        truncate_ragged_lines,
        encoding,
    )
    options.validate()
    return options^


def _projection(schema: CsvSchema, columns: List[String]) raises -> List[Bool]:
    var keep = List[Bool](length=len(schema), fill=len(columns) == 0)
    var requested = Dict[String, Bool]()
    for name in columns:
        if name in requested:
            raise Error("CSV column listed twice: " + name)
        requested[name] = True
        var found = False
        for i in range(len(schema)):
            if schema._fields[i].name == name:
                keep[i] = True
                found = True
        if not found:
            raise Error("Unknown CSV column: " + name)
    return keep^


comptime _PROT_READ = 1
comptime _MAP_PRIVATE = 2
comptime _SEEK_END = 2  # whence for seeking to the end


struct _Mapping(Movable):
    """A file mapped for reading, unmapped when it goes out of scope."""

    var address: Int
    var length: Int

    def __init__(out self, address: Int, length: Int):
        self.address = address
        self.length = length

    def __deinit__(deinit self):
        if self.address != 0:
            _ = external_call["munmap", Int32](self.address, self.length)

    def span(self) -> Span[UInt8, ImmutAnyOrigin]:
        return Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=self.address
            )
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=self.length,
        )


def _map_file(path: String) -> _Mapping:
    """Map `path` whole, or return a mapping with address 0 if it cannot be.

    A pipe or a character device has no length to map. Reporting that as a
    value rather than an exception keeps the caller's fallback to reading in
    blocks separate from a decoding error, which must not be swallowed.
    """
    try:
        return _try_map(path)
    except:
        return _Mapping(0, 0)


def _try_map(path: String) raises -> _Mapping:
    with open(path, "r") as file:
        # The handle's own seek, not lseek through FFI: the standard library
        # already declares that symbol with another signature.
        var length = Int(file.seek(0, _SEEK_END))
        if length <= 0:
            raise Error("nothing to map")
        var address = external_call["mmap", Int](
            0,
            length,
            Int32(_PROT_READ),
            Int32(_MAP_PRIVATE),
            Int32(file._get_raw_fd()),
            0,
        )
        if address == 0 or address == -1:
            raise Error("mmap failed")
        # The mapping outlives the descriptor, so closing the file here is
        # fine and is what leaving this block does.
        return _Mapping(address, length)
