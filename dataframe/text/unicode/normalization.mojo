"""Dataframe storage adapter for the standalone Unicode normalizer."""
from unicode_mojo import Normalizer, QuickCheck
from dataframe.series import Series
from dataframe.storage.string import StringColumn, StringBuilder
from dataframe.text.bytes import _Bytes


def normalize_series(input: Series, form: String) raises -> Series:
    """Normalize columns while retaining unchanged buffers and shared output."""
    var normalizer = Normalizer(form)
    if input.is_chunked():
        var parts = List[Series]()
        for chunk in input.chunks():
            parts.append(normalize_series(chunk, form))
        return Series._from_chunks(parts)
    ref column = input._data[StringColumn]
    var rows = _Bytes(column)
    var nulls = column.null_count() != 0
    var first = len(column)
    for i in range(len(column)):
        if (not nulls or column._valid(i)) and normalizer.quick_check(
            StringSpan(unsafe_from_utf8=rows.get(i))
        ) != QuickCheck.YES:
            first = i
            break
    if first == len(column):
        return Series("", column.copy())
    var out = StringBuilder(len(column), column._value_bytes())
    for i in range(len(column)):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        if i < first:
            out._bytes.extend(row)
        else:
            normalizer.normalize_into(
                StringSpan(unsafe_from_utf8=row), out._bytes
            )
        out._offsets.append(Int64(len(out._bytes)))
        out._push_bit(True)
    return Series("", out^.finish())
