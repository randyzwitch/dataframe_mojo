"""Original CPU partition-key semantics retained through GPU normalization.

The source dtype owns its original categorical dictionary. Lookup values are
indexed by the query-wide staged dictionary code, never by a source row.
"""
from dataframe.dtype import DataType
from std.memory import ArcPointer


struct WindowKey(Copyable):
    var name: String
    var dtype: DataType
    var string_hash: Bool
    var values: ArcPointer[List[UInt64]]

    def __init__(out self, name: String, dtype: DataType, string_hash: Bool):
        self.name = name
        self.dtype = dtype
        self.string_hash = string_hash
        self.values = ArcPointer(List[UInt64]())
