"""Concatenate nested Arrow arrays once per level, including sliced children.

Like Polars' concatenate_list/concatenate_struct, gather child arrays first
and merge each child once. Repeated binary append recopies earlier batches.
"""
from .column import Column, _append_validity
from .dtype import DataType, CategoricalDictionary
from .nested_column import ListColumn, StructColumn
from .series import Series


def _child(parts: List[Series]) raises -> Series:
    if parts[0].dtype().is_categorical():
        var same = True
        for part in parts:
            if part.dtype() != parts[0].dtype():
                same = False
                break
        if not same:
            # Build the complete dictionary before remapping any codes.
            # Binary unions would repeatedly remap all earlier children.
            var dictionary = CategoricalDictionary()
            var seen = Dict[String, UInt32]()
            var length = 0
            for part in parts:
                parts[0]._check_metadata(part)
                length += len(part)
                var labels = part.dtype().dictionary()
                for i in range(len(labels[])):
                    var label = String(labels[].get(i))
                    if label not in seen:
                        if len(dictionary) > Int(UInt32.MAX):
                            raise Error(
                                "Categorical dictionary exceeds UInt32 codes"
                            )
                        seen[label] = UInt32(len(dictionary))
                        dictionary.append(label)
            var target = DataType.categorical(dictionary^, distinct=True)
            var codes = List[UInt32](capacity=length)
            var bits = List[UInt8](capacity=(length + 7) // 8)
            var rows = 0
            for part in parts:
                var flat = part.rechunk() if part.is_chunked() else part.copy()
                var labels = flat.dtype().dictionary()
                var mapping = List[UInt32](capacity=len(labels[]))
                for i in range(len(labels[])):
                    mapping.append(seen[String(labels[].get(i))])
                ref column = flat._data[Column[UInt32]]
                for row in range(len(column)):
                    codes.append(
                        mapping[Int(column._get(row))] if column._valid(
                            row
                        ) else UInt32(0)
                    )
                _append_validity(
                    bits, rows, column._bits[], column._offset, len(column)
                )
                rows += len(column)
            var result = Series(
                parts[0].name(), Column[UInt32](values=codes^, bits=bits^)
            ).with_dtype(target)
            result._field_metadata = parts[0]._field_metadata
            return result^
    return Series._from_chunks(parts).rechunk()


def concatenate_nested(parts: List[Series]) raises -> Series:
    var length = 0
    for part in parts:
        length += len(part)
    var bits = List[UInt8](capacity=(length + 7) // 8)
    var rows = 0
    if parts[0].dtype().is_struct():
        var fields = List[Series]()
        var count = parts[0]._data[StructColumn].field_count()
        for i in range(count):
            var children = List[Series](capacity=len(parts))
            for part in parts:
                children.append(part._data[StructColumn].field(i))
            fields.append(_child(children))
        for part in parts:
            ref column = part._data[StructColumn]
            _append_validity(
                bits, rows, column._bits[], column._offset, len(column)
            )
            rows += len(column)
        return Series(
            parts[0].name(), StructColumn(fields^, bits^, length=length)
        )
    var offsets = List[Int64](capacity=length + 1)
    offsets.append(0)
    var children = List[Series](capacity=len(parts))
    var total = Int64(0)
    for part in parts:
        ref column = part._data[ListColumn]
        var begin = column._start(0)
        var end = column._start(len(column))
        children.append(column.child().slice(begin, end - begin))
        for row in range(len(column)):
            var size = Int64(column._physical_count(row))
            if size > Int64.MAX - total:
                raise Error("Concatenated list child length overflows")
            total += size
            offsets.append(total)
        _append_validity(
            bits, rows, column._bits[], column._offset, len(column)
        )
        rows += len(column)
    var child = _child(children)
    return Series(parts[0].name(), ListColumn(offsets^, child^, bits^))
