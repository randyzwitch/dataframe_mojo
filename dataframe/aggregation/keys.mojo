"""Struct and categorical grouping key normalization."""
from dataframe.bool_column import (
    BoolColumn,
)
from dataframe.canonical_labels import (
    canonical_labels,
)
from dataframe.column import (
    _pack_bits,
)
from dataframe.expr import (
    decode_name,
    encode_name,
    null,
)
from dataframe.expr_kernels import (
    validity,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.nested_column import (
    StructColumn,
)
from dataframe.series import (
    Series,
)


# A struct key groups, joins and deduplicates by its fields plus its own
# validity, so a null struct is one group and a struct of nulls another.
# The expanded columns are named this separator, the struct's name, the
# separator again, and the field name (or _KEY_VALID), the names escaped
# (`encode_name`) so neither holds a raw separator; _pack_struct_keys packs
# them back. The leading separator marks them: a name without it is never
# taken apart, whatever it holds (#438).
comptime _KEY_SEP = "\x1f"
comptime _KEY_VALID = "\x1evalid"


def _expand_struct_keys(columns: List[Series]) raises -> List[Series]:
    """Replace each struct key column by its fields and validity."""
    var out = List[Series](capacity=len(columns))
    for column in columns:
        if not column.dtype().is_struct():
            out.append(column.copy())
            continue
        var structs = column.struct_column()
        for i in range(structs.field_count()):
            var field = structs.field(i)
            if field.dtype().is_nested():
                raise Error(
                    "struct keys with nested fields are not supported yet: "
                    + column.name()
                    + "."
                    + field.name()
                )
            out.append(
                field.renamed(
                    _KEY_SEP
                    + encode_name(column.name())
                    + _KEY_SEP
                    + encode_name(field.name())
                )
            )
        out.append(
            Series(
                _KEY_SEP + encode_name(column.name()) + _KEY_SEP + _KEY_VALID,
                BoolColumn(structs.validity()),
            )
        )
    return out^


def _group_keys(columns: List[Series]) raises -> List[Series]:
    """`_expand_struct_keys` for a grouping. A categorical key whose
    dictionary repeats a label has its codes made canonical, so equal labels
    are equal groups (#943)."""
    var out = _expand_struct_keys(columns)
    for i in range(len(out)):
        out[i] = canonical_labels(out[i])
    return out^


def _key_parts(name: String) -> Tuple[String, String]:
    """An expanded key column's struct name and its field part (escaped, or
    _KEY_VALID); ("", "") for a plain column."""
    if not name.startswith(_KEY_SEP):
        return (String(""), String(""))
    var rest = String(name[byte = _KEY_SEP.byte_length() :])
    var at = rest.find(_KEY_SEP)
    if at < 0:
        return (String(""), String(""))
    return (
        decode_name(String(rest[byte=0:at])),
        String(rest[byte = at + _KEY_SEP.byte_length() :]),
    )


def _key_prefix(name: String) -> String:
    """The struct name an expanded key column belongs to, or "" if plain."""
    return _key_parts(name)[0]


def _pack_struct_keys(frame: DataFrame) raises -> DataFrame:
    """Pack runs of expanded key columns (see _expand_struct_keys) back into
    their struct, in place; other columns are unchanged."""
    var columns = List[Series](capacity=frame.width())
    var k = 0
    while k < frame.width():
        var prefix = _key_prefix(frame._columns[k].name())
        if prefix == "":
            columns.append(frame._columns[k].copy())
            k += 1
            continue
        var fields = List[Series]()
        var bits = List[UInt8]()
        while (
            k < frame.width()
            and _key_prefix(frame._columns[k].name()) == prefix
        ):
            ref column = frame._columns[k]
            var suffix = _key_parts(column.name())[1]
            if suffix == _KEY_VALID:
                # The helper's values (not its bitmap) are the struct's
                # validity.
                var flags = column.bool()
                var values = List[Bool](capacity=len(flags))
                for i in range(len(flags)):
                    values.append(flags._get(i))
                bits = _pack_bits(values)
            else:
                fields.append(column.renamed(decode_name(suffix)))
            k += 1
        columns.append(
            Series(prefix, StructColumn(fields^, bits^, length=frame.height()))
        )
    return DataFrame(columns^, height=frame.height())
