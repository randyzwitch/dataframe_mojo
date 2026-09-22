"""Round-trip the public CSV reader through all native integer widths."""
from std.sys import argv
from dataframe import CsvField, CsvSchema, DataType, read_csv, write_csv


def schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField("i8", DataType.INT8),
            CsvField("u8", DataType.UINT8),
            CsvField("i16", DataType.INT16),
            CsvField("u16", DataType.UINT16),
            CsvField("i32", DataType.INT32),
            CsvField("u32", DataType.UINT32),
            CsvField.int64("i64"),
            CsvField("u64", DataType.UINT64),
        ]
    )


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: public_integer_roundtrip INPUT_CSV OUTPUT_CSV")
    var frame = read_csv(String(args[1]), schema())
    write_csv(frame, String(args[2]))
