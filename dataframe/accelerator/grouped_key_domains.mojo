"""Original CPU GroupBy key domains for GPU primitive grouped List producers.

Categorical and explicitly admitted dictionary-backed String keys preserve
original code identity, including duplicate labels. Ordinary text and opaque
keys use separate per-field byte dictionaries: their hashes remain original
raw bytes, and invalid UTF8 from another field cannot enter String restoration.
This module stages keys only; it never groups or evaluates query operands.
"""
from std.memory import ArcPointer
from std.sys import size_of
from dataframe import Column, DataType, Series, StringColumn
from dataframe.categorical import decode
from .window_keys import WindowKey
from .opaque_dictionary import ByteDomain
from .memory import checked_add, checked_mul

comptime FIXED = 0
comptime STRING = 1
comptime CATEGORY = 2
comptime CODED_STRING = 3
comptime BINARY = 4
comptime GEOMETRY = 5


def coded_string_dtype(source: Series) -> Optional[DataType]:
    """Same metadata-only dictionary test as GroupBy._coded_keys.

    Do not invoke _dictionary_codes during describe: selected/gathered String
    arrays can allocate/write codes there. CPU requires EVERY physical chunk
    to carry one equal dictionary, including empty chunks.
    """
    if source.dtype() != DataType.STRING:
        return None
    var dtype = Optional[DataType]()
    for part in source.chunks():
        if not part._data.isa[StringColumn]():
            return None
        ref strings = part._data[StringColumn]
        if not strings._codes:
            return None
        var current = strings._codes.value()[].dtype
        if dtype and current != dtype.value():
            return None
        if not dtype:
            dtype = current
    return dtype^


def kind(source: Series) -> Int:
    var dtype = source.dtype()
    if dtype.is_categorical():
        return CATEGORY
    if dtype == DataType.STRING:
        return CODED_STRING if coded_string_dtype(source) else STRING
    if dtype._is_bytes():
        return BINARY if dtype == DataType.BINARY else GEOMETRY
    return FIXED


def _validate_codes(source: Series, dtype: DataType) raises:
    if not dtype.is_categorical() or not dtype.has_dictionary():
        raise Error("Categorical key requires its original dictionary")
    var domain = UInt64(len(dtype.dictionary()[]))
    for part in source.chunks():
        if not part._data.isa[Column[UInt32]]():
            raise Error("Categorical key has invalid physical code storage")
        ref codes = part._data[Column[UInt32]]
        for row in range(len(part)):
            if codes._valid(row) and UInt64(codes._get(row)) >= domain:
                raise Error("Categorical code exceeds its dictionary")


def direct_category_codes(
    source: Series, *, metadata_only: Bool = False
) raises -> Tuple[Series, WindowKey]:
    """Return original UInt32 codes and explicit original-code provenance.

    For OVER, call this ONLY when its captured WindowKey requires categorical
    code semantics. GroupBy additionally invokes it for same-dictionary String
    codes, exactly as CPU GroupBy._coded_keys does. Text-reading expressions
    must decode separately and continue to hash their original String bytes.
    """
    var original = source.dtype()
    var dtype = Optional[DataType](
        original
    ) if original.is_categorical() else coded_string_dtype(source)
    if not dtype or not dtype.value().has_dictionary():
        raise Error("Original categorical code provenance is missing")
    var key = WindowKey(source.name(), dtype.value(), False)
    var encoded: Series
    if metadata_only:
        encoded = Series.full_null(source.name(), DataType.UINT32, 0)
    else:
        if original.is_categorical():
            encoded = source.copy()
        else:
            var held = source._dictionary_codes()
            if not held:
                raise Error("String key lost its original code dictionary")
            encoded = held.take()
        _validate_codes(encoded, dtype.value())
        encoded = encoded.with_dtype(DataType.UINT32)
        var values = List[UInt64](
            length=len(dtype.value().dictionary()[]), fill=0
        )
        for code in range(len(values)):
            values[code] = UInt64(code)
        key.values = ArcPointer(values^)
    encoded._field_metadata = source._field_metadata
    return (encoded^, key^)


def reservation(source: Series) raises -> Int:
    """Check this conservative construction envelope BEFORE staging values.

    Original immutable input/code/dictionary owners are caller-owned. Count
    new source code arrays, explicit byte-domain String/vector reservations,
    hash copies, temporary validity and possible gathered StringCodes.
    """
    var domain = kind(source)
    if domain == FIXED:
        return 0
    var rows = len(source)
    if domain == CATEGORY or domain == CODED_STRING:
        var dtype = Optional[DataType](
            source.dtype()
        ) if domain == CATEGORY else coded_string_dtype(source)
        if not dtype or not dtype.value().has_dictionary():
            raise Error("Original categorical code provenance is missing")
        var lookup = checked_mul(len(dtype.value().dictionary()[]), 8)
        if domain == CATEGORY:
            return lookup
        # Some gathered/chunked StringCodes share codes; others own a compact
        # code array and validity copy. Reserve both without reading payloads.
        var bitmap = checked_add(rows, checked_mul(source.n_chunks(), 7)) // 8
        return checked_add(lookup, checked_add(checked_mul(rows, 4), bitmap))
    var payload = 0
    for part in source.chunks():
        ref values = part._data[StringColumn]
        for row in range(len(part)):
            if values._valid(row):
                payload = checked_add(payload, len(values._row_bytes(row)))
    # Match ByteDomain's explicit entry capacities and ASCII key construction;
    # add one captured WindowKey raw-hash owner (8*D <=8*N).
    return checked_add(
        checked_add(
            checked_mul(payload, 12),
            checked_mul(rows, 4 * size_of[String]() + 32),
        ),
        checked_add(
            checked_mul(rows, 5),
            checked_add(checked_mul((rows + 7) // 8, 2), 8),
        ),
    )


struct KeyDomain(Movable):
    var kind: Int
    var prototype: Series
    var encoded: Series
    var provenance: WindowKey
    var dictionary: Optional[ByteDomain]
    var staging_bytes: Int

    def __init__(
        out self, source: Series, *, metadata_only: Bool = False
    ) raises:
        # Prepare complete local owners before publishing any out-self fields.
        # A raising encode must never clean up a partly replaced nested Arc.
        var domain = kind(source)
        var prototype = source.slice(0, 0)
        var encoded = prototype.copy() if metadata_only else source.copy()
        var provenance = WindowKey(source.name(), source.dtype(), False)
        var dictionary = Optional[ByteDomain]()
        var staging = 0 if metadata_only else reservation(source)
        if domain == CATEGORY or domain == CODED_STRING:
            var coded = direct_category_codes(
                source, metadata_only=metadata_only
            )
            encoded = coded[0].copy()
            provenance = coded[1].copy()
        elif domain != FIXED:
            provenance = WindowKey(source.name(), source.dtype(), True)
            if metadata_only:
                encoded = Series.full_null(source.name(), DataType.UINT32, 0)
            else:
                # Each field's dictionary holds only its original byte domain;
                # invalid bytes cannot enter another String field's restoration.
                var bytes = source.with_dtype(DataType.BINARY)
                var prepared = ByteDomain([bytes.copy()])
                encoded = prepared.encode(bytes)
                provenance.values = ArcPointer(prepared.hashes.copy())
                dictionary = prepared^
            encoded._field_metadata = source._field_metadata
        self.kind = domain
        self.prototype = prototype^
        self.encoded = encoded^
        self.provenance = provenance^
        self.dictionary = dictionary^
        self.staging_bytes = staging

    def restore(self, codes: Series) raises -> Series:
        if not len(codes):
            return self.prototype.renamed(codes.name())
        var output = codes.copy()
        if self.kind == CATEGORY:
            output = codes.with_dtype(self.prototype.dtype())
        elif self.kind == CODED_STRING:
            output = decode(codes.with_dtype(self.provenance.dtype))
        elif self.kind != FIXED:
            if not self.dictionary:
                raise Error("Metadata-only key domain cannot restore values")
            output = self.dictionary.value().restore(
                codes, self.prototype.with_dtype(DataType.BINARY)
            )
            if self.kind == STRING:
                # This per-field domain contains ONLY bytes captured from an
                # original validated String. Restore its proven logical type;
                # public Binary->String retagging remains forbidden, and no
                # bytes from other opaque fields can enter this domain.
                output._dtype = DataType.STRING
            else:
                output = output.with_dtype(self.prototype.dtype())
        output._field_metadata = self.prototype._field_metadata
        return output^

    def restoration_bytes(self, codes: Series) raises -> Int:
        """Extra typed restoration peak; caller separately retains input result."""
        if not len(codes) or self.kind == FIXED or self.kind == CATEGORY:
            return 0
        var rows = len(codes)
        if self.kind == CODED_STRING:
            var dictionary = self.provenance.dtype.dictionary()
            return checked_add(
                checked_mul(rows, 24),
                checked_add(
                    checked_mul(
                        checked_add(
                            len(dictionary[].bytes),
                            checked_mul(len(dictionary[].offsets), 8),
                        ),
                        2,
                    ),
                    checked_mul(len(dictionary[]), 16),
                ),
            )
        if not self.dictionary or codes.dtype() != DataType.UINT32:
            raise Error("Key domain restoration schema differs")
        var payload = 0
        for part in codes.chunks():
            ref column = part._data[Column[UInt32]]
            for row in range(len(part)):
                if column._valid(row):
                    var code = Int(column._get(row))
                    if code >= len(self.dictionary.value().keys):
                        raise Error("Key result is outside original domain")
                    payload = checked_add(
                        payload,
                        Int(
                            self.dictionary.value().offsets[code + 1]
                            - self.dictionary.value().offsets[code]
                        ),
                    )
        return checked_add(
            checked_mul(payload, 2),
            checked_add(
                checked_mul(rows, size_of[List[UInt8]]() + 8 + size_of[Bool]()),
                checked_add(8, checked_add(rows, 7) // 8),
            ),
        )
