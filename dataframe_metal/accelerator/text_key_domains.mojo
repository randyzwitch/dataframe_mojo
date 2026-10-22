"""Rebuild private text-code partition lookup from original CPU provenance."""
from dataframe.dtype import DataType
from dataframe.partition import _hash_bytes
from dataframe.plan_types import WindowKey
from .code_domains import code


def append_partition_values(
    mut values: List[UInt64], dtype: DataType, key: WindowKey
) raises:
    """Refresh code lookup without changing original key dtype/hash family.

    Only String/categorical keys may call this. Binary/Geometry code domains
    belong to the opaque adapter and must keep their original lookup.
    """
    if key.dtype != DataType.STRING and not key.dtype.is_categorical():
        raise Error(
            "Private text lookup cannot replace opaque/numeric provenance"
        )
    var dictionary = dtype.dictionary()
    var begin = len(values)
    for _ in range(len(dictionary[])):
        values.append(0)
    if key.string_hash:
        for index in range(len(dictionary[])):
            values[begin + index] = _hash_bytes(
                dictionary[].get(index).as_bytes()
            )
    else:
        if not key.dtype.is_categorical():
            raise Error(
                "Private categorical lookup requires original logical dictionary"
            )
        var original = key.dtype.dictionary()
        for index in range(len(original[])):
            values[begin + Int(code(dtype, original[].get(index)))] = UInt64(
                index
            )
