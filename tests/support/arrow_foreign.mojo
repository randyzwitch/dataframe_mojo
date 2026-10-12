"""Independent Arrow C layout producer/consumer; no dataframe conversion helpers.

The fixture owns raw little-endian bytes, bitmaps and C structs until its
consumer returns. Real C release callbacks count each node. Import must copy
before those callbacks; poisoning producer buffers subsequently proves ownership.
"""
from std.memory import Pointer
from std.testing import assert_equal
from dataframe import ArrowArray, ArrowSchema
from tests.support.lifecycle_stream import (
    at,
    array_release,
    schema_release,
    ArrayRelease,
    SchemaRelease,
)


def text_bytes(text: String, terminated: Bool = False) -> List[UInt8]:
    var result = List[UInt8]()
    result.extend(text.as_bytes())
    if terminated:
        result.append(0)
    return result^


def integers(values: List[Int128], width: Int, pad: Int = 0) -> List[UInt8]:
    """Author bytes numerically, independently of native alignment and casts."""
    var result = List[UInt8](length=pad, fill=229)
    for value in values:
        var raw = value.cast[DType.uint128]()
        for _ in range(width):
            result.append(UInt8(raw & 255))
            raw = raw >> 8
    return result^


def read_unsigned(address: Int, index: Int, width: Int) -> UInt128:
    var result = UInt128(0)
    for byte in range(width):
        result |= UInt128(
            at[UInt8](address)[unsafe_offset=index * width + byte]
        ) << UInt128(8 * byte)
    return result


def read_signed(address: Int, index: Int, width: Int) -> Int128:
    var raw = read_unsigned(address, index, width)
    if width < 16 and (raw & (UInt128(1) << UInt128(8 * width - 1))) != 0:
        raw |= UInt128.MAX << UInt128(8 * width)
    return raw.cast[DType.int128]()


def c_string(address: Int) -> String:
    var bytes = List[UInt8]()
    var offset = 0
    while at[UInt8](address)[unsafe_offset=offset] != 0:
        bytes.append(at[UInt8](address)[unsafe_offset=offset])
        offset += 1
    return String(unsafe_from_utf8=Span(bytes))


def bitmap_valid(address: Int, row: Int) -> Bool:
    return (
        address == 0
        or (at[UInt8](address)[unsafe_offset=row // 8] & UInt8(1 << (row % 8)))
        != 0
    )


def packed_metadata(
    keys: List[List[UInt8]], values: List[List[UInt8]]
) -> List[UInt8]:
    var result = integers([Int128(len(keys))], 4)
    for index in range(len(keys)):
        result.extend(integers([Int128(len(keys[index]))], 4))
        result.extend(keys[index].copy())
        result.extend(integers([Int128(len(values[index]))], 4))
        result.extend(values[index].copy())
    return result^


def release_foreign_array(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    ref node = at[ForeignNode](array[].private_data)[]
    node.array_releases += 1
    for address in node.array_children:
        array_release(at[ArrowArray](address)[])
    if node.dictionary_array:
        array_release(at[ArrowArray](node.dictionary_array)[])
    array[].release = 0


def release_foreign_schema(schema: Pointer[ArrowSchema, MutAnyOrigin]) abi("C"):
    ref node = at[ForeignNode](schema[].private_data)[]
    node.schema_releases += 1
    for address in node.schema_children:
        schema_release(at[ArrowSchema](address)[])
    if node.dictionary_schema:
        schema_release(at[ArrowSchema](node.dictionary_schema)[])
    schema[].release = 0


struct ForeignNode(Movable):
    var array: ArrowArray
    var schema: ArrowSchema
    var format: List[UInt8]
    var name: List[UInt8]
    var metadata: List[UInt8]
    var bytes: List[List[UInt8]]
    var skips: List[Int]
    var buffers: List[Int]
    var children: List[Int]
    var array_children: List[Int]
    var schema_children: List[Int]
    var dictionary: Int
    var dictionary_array: Int
    var dictionary_schema: Int
    var array_releases: Int
    var schema_releases: Int

    def __init__(out self):
        self.array = ArrowArray()
        self.schema = ArrowSchema()
        self.format = []
        self.name = []
        self.metadata = []
        self.bytes = []
        self.skips = []
        self.buffers = []
        self.children = []
        self.array_children = []
        self.schema_children = []
        self.dictionary = -1
        self.dictionary_array = 0
        self.dictionary_schema = 0
        self.array_releases = 0
        self.schema_releases = 0

    def describe(
        mut self,
        format: String,
        name: String,
        length: Int,
        offset: Int = 0,
        nulls: Int = 0,
    ):
        self.format = text_bytes(format, True)
        self.name = text_bytes(name, True)
        self.array.length = Int64(length)
        self.array.offset = Int64(offset)
        self.array.null_count = Int64(nulls)
        self.schema.flags = 2  # ARROW_FLAG_NULLABLE

    def buffer(mut self, var bytes: List[UInt8], skip: Int = 0):
        self.bytes.append(bytes^)
        self.skips.append(skip)


struct ForeignArrow(Movable):
    var nodes: List[ForeignNode]

    def __init__(out self, count: Int):
        self.nodes = List[ForeignNode](capacity=count)
        for _ in range(count):
            self.nodes.append(ForeignNode())

    def seal(mut self):
        # Call only after building all nodes. Node storage and owned bytes stay
        # stable until import/export callbacks have returned.
        var release_array: ArrayRelease = release_foreign_array
        var release_schema: SchemaRelease = release_foreign_schema
        for index in range(len(self.nodes)):
            ref node = self.nodes[index]
            for buffer in range(len(node.bytes)):
                node.buffers.append(
                    Int(node.bytes[buffer].unsafe_ptr())
                    + node.skips[buffer] if len(node.bytes[buffer]) else 0
                )
            for child in node.children:
                node.array_children.append(
                    Int(Pointer(to=self.nodes[child].array))
                )
                node.schema_children.append(
                    Int(Pointer(to=self.nodes[child].schema))
                )
            node.array.n_buffers = Int64(len(node.buffers))
            node.array.buffers = Int(node.buffers.unsafe_ptr())
            node.array.n_children = Int64(len(node.children))
            node.array.children = Int(node.array_children.unsafe_ptr())
            node.array.release = Pointer(to=release_array).unsafe_bitcast[
                Int
            ]()[]
            node.array.private_data = Int(Pointer(to=node))
            node.schema.format = Int(node.format.unsafe_ptr())
            node.schema.name = Int(node.name.unsafe_ptr())
            node.schema.metadata = Int(node.metadata.unsafe_ptr()) if len(
                node.metadata
            ) else 0
            node.schema.n_children = Int64(len(node.children))
            node.schema.children = Int(node.schema_children.unsafe_ptr())
            node.schema.release = Pointer(to=release_schema).unsafe_bitcast[
                Int
            ]()[]
            node.schema.private_data = Int(Pointer(to=node))
            if node.dictionary >= 0:
                node.array.dictionary = Int(
                    Pointer(to=self.nodes[node.dictionary].array)
                )
                node.schema.dictionary = Int(
                    Pointer(to=self.nodes[node.dictionary].schema)
                )
                node.dictionary_array = node.array.dictionary
                node.dictionary_schema = node.schema.dictionary

    def assert_released(self) raises:
        for index in range(len(self.nodes)):
            ref node = self.nodes[index]
            assert_equal(node.array_releases, 1)
            assert_equal(node.schema_releases, 1)
            assert_equal(node.array.release, 0)
            assert_equal(node.schema.release, 0)

    def poison(mut self):
        for index in range(len(self.nodes)):
            ref node = self.nodes[index]
            for buffer in range(len(node.bytes)):
                for byte in range(len(node.bytes[buffer])):
                    node.bytes[buffer][byte] = 173
            for byte in range(len(node.metadata)):
                node.metadata[byte] = 173
