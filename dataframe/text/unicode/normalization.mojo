"""Unicode 17 normalization with static tables and reusable row scratch space."""
from dataframe.text.unicode.normalization_tables import (
    properties,
    decomposition,
    decomposition_value,
    composition,
)
from dataframe.series import Series
from dataframe.storage.string import StringColumn, StringBuilder
from dataframe.text.bytes import _Bytes

comptime _Row = Span[UInt8, ImmutAnyOrigin]


@always_inline
def _codepoint(row: _Row, mut at: Int) -> Int:
    var p = row.unsafe_ptr()
    var cp = Int(p[unsafe_offset=at])
    at += 1
    if cp < 128:
        return cp
    var size = 1 if cp < 224 else (2 if cp < 240 else 3)
    cp &= 31 if size == 1 else (15 if size == 2 else 7)
    for _ in range(size):
        cp = (cp << 6) | (Int(p[unsafe_offset=at]) & 63)
        at += 1
    return cp


@always_inline
def _append_utf8(mut out: List[UInt8], cp: Int):
    if cp < 128:
        out.append(UInt8(cp))
    elif cp < 2048:
        out.append(UInt8(192 | (cp >> 6)))
        out.append(UInt8(128 | (cp & 63)))
    elif cp < 65536:
        out.append(UInt8(224 | (cp >> 12)))
        out.append(UInt8(128 | ((cp >> 6) & 63)))
        out.append(UInt8(128 | (cp & 63)))
    else:
        out.append(UInt8(240 | (cp >> 18)))
        out.append(UInt8(128 | ((cp >> 12) & 63)))
        out.append(UInt8(128 | ((cp >> 6) & 63)))
        out.append(UInt8(128 | (cp & 63)))


def _quick(row: _Row, shift: Int, mask: Int) -> Bool:
    var at = 0
    var previous = 0
    while at < len(row):
        if (
            at + 8 <= len(row)
            and (
                row.unsafe_ptr()
                .unsafe_offset(at)
                .unsafe_bitcast[UInt64]()
                .unsafe_load()
                & 0x8080808080808080
            )
            == 0
        ):
            at += 8
            previous = 0
            continue
        var cp = _codepoint(row, at)
        var value = properties(cp)
        if ((value >> shift) & mask) != 0:
            return False
        var ccc = value & 255
        if ccc != 0 and previous > ccc:
            return False
        previous = ccc
    return True


def _decompose(mut out: List[Int], cp: Int, compatible: Bool):
    if cp >= 0xAC00 and cp < 0xAC00 + 11172:
        var syllable = cp - 0xAC00
        out.append(0x1100 + syllable // 588)
        out.append(0x1161 + (syllable % 588) // 28)
        if syllable % 28 != 0:
            out.append(0x11A7 + syllable % 28)
        return
    if cp >= 128:
        var mapped = decomposition(cp, compatible)
        if mapped[1] != 0:
            for k in range(mapped[1]):
                _decompose(out, decomposition_value(mapped[0] + k), compatible)
            return
    out.append(cp)


def _sort_marks(mut values: List[Int], start: Int, end: Int):
    # Short runs use bounded insertion; long adversarial runs use a stable
    # counting sort over the 256 canonical classes, avoiding quadratic work.
    if end - start <= 32:
        for i in range(start + 1, end):
            var value = values[i]
            var ccc = properties(value) & 255
            var j = i
            while j > start and (properties(values[j - 1]) & 255) > ccc:
                values[j] = values[j - 1]
                j -= 1
            values[j] = value
        return
    var positions = List[Int](length=256, fill=0)
    for i in range(start, end):
        positions[properties(values[i]) & 255] += 1
    var total = 0
    for i in range(256):
        var count = positions[i]
        positions[i] = total
        total += count
    var sorted = List[Int](unsafe_uninit_length=end - start)
    for i in range(start, end):
        var ccc = properties(values[i]) & 255
        sorted[positions[ccc]] = values[i]
        positions[ccc] += 1
    for i in range(start, end):
        values[i] = sorted[i - start]


def _order(mut values: List[Int]):
    var start = 0
    var previous = 0
    var unsorted = False
    for i in range(len(values)):
        var ccc = properties(values[i]) & 255
        if ccc == 0:
            if unsorted:
                _sort_marks(values, start, i)
            start = i + 1
            unsorted = False
        elif previous > ccc:
            unsorted = True
        previous = ccc
    if unsorted:
        _sort_marks(values, start, len(values))


@always_inline
def _compose(a: Int, b: Int) -> Int:
    if a >= 0x1100 and a < 0x1100 + 19 and b >= 0x1161 and b < 0x1161 + 21:
        return 0xAC00 + ((a - 0x1100) * 21 + b - 0x1161) * 28
    if (
        a >= 0xAC00
        and a < 0xAC00 + 11172
        and (a - 0xAC00) % 28 == 0
        and b > 0x11A7
        and b < 0x11A7 + 28
    ):
        return a + b - 0x11A7
    return composition(a, b)


def _recompose(mut values: List[Int]) -> Int:
    var write = 0
    var starter = -1
    var previous = 0
    for i in range(len(values)):
        var cp = values[i]
        var ccc = properties(cp) & 255
        var combined = -1
        if cp >= 128 and starter >= 0 and (previous == 0 or previous < ccc):
            combined = _compose(values[starter], cp)
        if combined >= 0:
            values[starter] = combined
        else:
            if ccc == 0:
                starter = write
            values[write] = cp
            write += 1
            previous = ccc
    return write


def normalize_series(input: Series, form: String) raises -> Series:
    """Normalize UTF-8 with static tables and reusable decomposition scratch.

    An entirely normalized chunk keeps its original buffers. Changed chunks
    use one output builder and reuse decomposition scratch across rows.
    """
    if input.is_chunked():
        var parts = List[Series]()
        for chunk in input.chunks():
            parts.append(normalize_series(chunk, form))
        return Series._from_chunks(parts)
    ref column = input._data[StringColumn]
    var shift = 8 if form == "NFC" else (
        10 if form == "NFD" else (11 if form == "NFKC" else 13)
    )
    var mask = 3 if form == "NFC" or form == "NFKC" else 1
    var compose = form == "NFC" or form == "NFKC"
    var compatible = form == "NFKC" or form == "NFKD"
    var rows = _Bytes(column)
    var nulls = column.null_count() != 0
    var first = len(column)
    for i in range(len(column)):
        if (not nulls or column._valid(i)) and not _quick(
            rows.get(i), shift, mask
        ):
            first = i
            break
    if first == len(column):
        return Series("", column.copy())
    var out = StringBuilder(len(column), column._value_bytes())
    var scratch = List[Int]()
    for i in range(len(column)):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        if i < first or _quick(row, shift, mask):
            out._bytes.extend(row)
        else:
            scratch.clear()
            var at = 0
            while at < len(row):
                var cp = _codepoint(row, at)
                if cp < 128:
                    scratch.append(cp)
                else:
                    _decompose(scratch, cp, compatible)
            _order(scratch)
            var size = _recompose(scratch) if compose else len(scratch)
            for k in range(size):
                _append_utf8(out._bytes, scratch[k])
        out._offsets.append(Int64(len(out._bytes)))
        out._push_bit(True)
    return Series("", out^.finish())
