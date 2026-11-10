"""Compute field offsets and sizes for fixed-layout rows.

Rows store validity, fixed-width fields and references to variable-size
heap data. Nested layouts live in a flat list: node zero describes the
row, and struct fields point to their child nodes. Sharing this list
between collections and allocators avoids copying layout metadata.
"""
from std.memory import ArcPointer

from dataframe.dtype import DataType
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_LIST,
    PHYSICAL_STRUCT,
    PHYSICAL_VARCHAR,
    physical_of,
    physical_size,
)

# Use a 256 KiB allocation less an 8-byte block header. Row
# blocks have this capacity; heap blocks at least this.
comptime TUPLE_DATA_BLOCK_SIZE = 262144 - 8

comptime _FIXED_STORAGE = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
    DType.float64,
    DType.int128,
]


@always_inline
def validity_bytes(count: Int) -> Int:
    """ValidityBytes::SizeInBytes: one bit a value, rounded up to bytes."""
    return (count + 7) // 8


@always_inline
def constant_size(physical: Int) -> Bool:
    """TypeIsConstantSize: fixed-width values (not strings or nested)."""
    return (
        physical != PHYSICAL_VARCHAR
        and physical != PHYSICAL_LIST
        and physical != PHYSICAL_STRUCT
    )


@always_inline
def _align(value: Int) -> Int:
    """AlignValue: the next multiple of 8."""
    return (value + 7) & ~7


def physical_of_dtype(dtype: DataType) raises -> Int:
    """The physical type of a column of `dtype` in a row (LogicalType::
    InternalType): its storage for numeric, decimal and temporal types."""
    if dtype.is_list():
        return PHYSICAL_LIST
    if dtype.is_struct():
        return PHYSICAL_STRUCT
    if dtype == DataType.BOOL:
        return PHYSICAL_BOOL
    if dtype == DataType.STRING or dtype == DataType.BINARY:
        return PHYSICAL_VARCHAR
    var storage = dtype.storage()
    if storage:
        var s = storage.value()
        comptime for k in range(len(_FIXED_STORAGE)):
            comptime D = _FIXED_STORAGE[k]
            if s == D:
                return physical_of[D]()
    raise Error("tuple data: no row layout for " + dtype.name())


struct TupleDataLayoutNode(Copyable, Movable):
    """One layout: a row, or the nested row of a struct column."""

    var types: List[DataType]
    var physicals: List[Int]
    # Byte offset of each column, then of each aggregate, in the row.
    var offsets: List[Int]
    # Per column, the node of a struct column's layout; -1 otherwise.
    var struct_layouts: List[Int]
    var flag_width: Int
    var data_width: Int
    var aggr_width: Int
    var row_width: Int
    var all_constant: Bool
    var variable_columns: List[Int]
    var heap_size_offset: Int
    var can_have_null: Bool

    def __init__(out self):
        self.types = List[DataType]()
        self.physicals = List[Int]()
        self.offsets = List[Int]()
        self.struct_layouts = List[Int]()
        self.flag_width = 0
        self.data_width = 0
        self.aggr_width = 0
        self.row_width = 0
        self.all_constant = True
        self.variable_columns = List[Int]()
        self.heap_size_offset = 0
        self.can_have_null = True

    def column_count(self) -> Int:
        return len(self.types)


struct TupleDataLayout(Copyable, Movable):
    """The layout of a row and of every struct nested in it."""

    # Shared among a collection, its partitions and their allocators: a copy is a
    # reference count, not a copy of every node's lists.
    var nodes: ArcPointer[List[TupleDataLayoutNode]]

    def __init__(
        out self,
        types: List[DataType],
        can_have_null: Bool = True,
        aggregate_sizes: List[Int] = List[Int](),
    ) raises:
        """TupleDataLayout::Initialize(types, aggregates, validity_type):
        `aggregate_sizes` are the payload sizes of the aggregate states
        that follow the columns (each a multiple of 8)."""
        var nodes = List[TupleDataLayoutNode]()
        _ = TupleDataLayout._initialize(
            nodes, types, can_have_null, True, aggregate_sizes
        )
        self.nodes = ArcPointer(nodes^)

    @staticmethod
    def _initialize(
        mut nodes: List[TupleDataLayoutNode],
        types: List[DataType],
        can_have_null: Bool,
        top_level: Bool,
        aggregate_sizes: List[Int],
    ) raises -> Int:
        var index = len(nodes)
        nodes.append(TupleDataLayoutNode())
        var node = TupleDataLayoutNode()
        node.types = types.copy()
        node.can_have_null = can_have_null

        # Null mask at the front, one bit a value.
        node.flag_width = validity_bytes(len(types)) if can_have_null else 0
        node.row_width = node.flag_width

        # Whether all columns are constant size. Structs are stored as a
        # layout of their own (NESTED_STRUCT_LAYOUT, which can hold nulls).
        for c in range(len(types)):
            var physical = physical_of_dtype(types[c])
            node.physicals.append(physical)
            var child = -1
            if physical == PHYSICAL_STRUCT:
                child = TupleDataLayout._initialize(
                    nodes, types[c].field_dtypes(), True, False, List[Int]()
                )
                if not nodes[child].all_constant:
                    node.all_constant = False
                    node.variable_columns.append(c)
            elif not constant_size(physical):
                node.all_constant = False
                node.variable_columns.append(c)
            node.struct_layouts.append(child)

        # A top-level row with a heap records its heap size.
        if top_level and not node.all_constant:
            node.heap_size_offset = node.row_width
            node.row_width += 8

        # Data columns, unaligned.
        for c in range(len(types)):
            node.offsets.append(node.row_width)
            var physical = node.physicals[c]
            if constant_size(physical) or physical == PHYSICAL_VARCHAR:
                node.row_width += physical_size(physical)
            elif physical == PHYSICAL_STRUCT:
                node.row_width += nodes[node.struct_layouts[c]].row_width
            else:
                # A list is a pointer to its data in the heap.
                node.row_width += 8

        # Aggregate states, aligned.
        var align = len(aggregate_sizes) > 0
        if align:
            node.row_width = _align(node.row_width)
        node.data_width = node.row_width - node.flag_width
        for size in aggregate_sizes:
            node.offsets.append(node.row_width)
            node.row_width += size
        node.aggr_width = node.row_width - node.data_width - node.flag_width
        if align:
            node.row_width = _align(node.row_width)

        nodes[index] = node^
        return index

    def column_count(self) -> Int:
        return len(self.nodes[][0].types)

    def types(self) -> List[DataType]:
        return self.nodes[][0].types.copy()

    def row_width(self) -> Int:
        return self.nodes[][0].row_width

    def data_offset(self) -> Int:
        return self.nodes[][0].flag_width

    def data_width(self) -> Int:
        return self.nodes[][0].data_width

    def aggr_offset(self) -> Int:
        return self.nodes[][0].flag_width + self.nodes[][0].data_width

    def offsets(self) -> List[Int]:
        return self.nodes[][0].offsets.copy()

    def all_constant(self) -> Bool:
        return self.nodes[][0].all_constant

    def heap_size_offset(self) -> Int:
        return self.nodes[][0].heap_size_offset

    def can_have_null(self) -> Bool:
        return self.nodes[][0].can_have_null

    def has_nested_types(self) -> Bool:
        for p in self.nodes[][0].physicals:
            if p == PHYSICAL_LIST or p == PHYSICAL_STRUCT:
                return True
        return False
