"""Expanded ISO years preserve the full stored Date and Datetime ranges."""
# contract-group: io.json
from dataframe import Column, DataFrame, DataType, Series
from tests.interchange.test_json_roundtrip import _roundtrip


def test_json_iso_temporal_wire_forms_roundtrip_full_int64_storage_bounds() raises:
    var values: List[Int64] = [
        Int64(-9223372036854775807) - 1,
        -1,
        0,
        1,
        9223372036854775807,
    ]
    var fields = List[Series]()
    fields.append(
        Series("date", Column[Int64](values.copy())).with_dtype(DataType.DATE)
    )
    for unit in [String("ns"), String("us"), String("ms")]:
        for zone in [String(""), String("UTC"), String("America/New_York")]:
            fields.append(
                Series(unit + zone, Column[Int64](values.copy())).with_dtype(
                    DataType.datetime(unit, zone)
                )
            )
    _roundtrip(DataFrame(fields^), "json-temporal-bounds")
