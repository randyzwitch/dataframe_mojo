"""A counted foreign Arrow producer for the production stream/import lifecycle."""
from std.memory import Pointer
from dataframe import ArrowArray, ArrowSchema
from dataframe.parquet import _ArrowArrayStream

comptime ArrayRelease = def(Pointer[ArrowArray, MutAnyOrigin]) thin abi(
    "C"
) -> None
comptime SchemaRelease = def(Pointer[ArrowSchema, MutAnyOrigin]) thin abi(
    "C"
) -> None
comptime StreamGet = def(Int, Int) thin abi("C") -> Int32
comptime StreamRelease = def(Int) thin abi("C") -> None
comptime StreamError = def(Int) thin abi("C") -> Int


def at[T: AnyType](address: Int) -> Pointer[T, MutAnyOrigin]:
    return Pointer[T, MutAnyOrigin](unsafe_from_address=address)


def array_release(mut array: ArrowArray):
    if array.release:
        var callback = array.release
        Pointer(to=callback).unsafe_bitcast[ArrayRelease]()[](
            at[ArrowArray](Int(Pointer(to=array)))
        )


def schema_release(mut schema: ArrowSchema):
    if schema.release:
        var callback = schema.release
        Pointer(to=callback).unsafe_bitcast[SchemaRelease]()[](
            at[ArrowSchema](Int(Pointer(to=schema)))
        )


struct StreamState(Movable):
    """The test owns this state until every borrowed C callback has finished.

    mode 0 succeeds; 1 fails get_schema; 2 exports an unsupported format;
    3 fails get_next after one valid batch; 4 publishes an array then fails.
    Counters count actual callback entries, including recursive child releases.
    """

    var mode: Int
    var batch_count: Int
    var next_calls: Int
    var schema_calls: Int
    var stream_releases: Int
    var root_array_releases: Int
    var child_array_releases: Int
    var root_schema_releases: Int
    var child_schema_releases: Int
    var emitted: Int
    var values: List[Int64]
    var format: List[UInt8]
    var root_format: List[UInt8]
    var field_name: List[UInt8]
    var error: List[UInt8]
    var child_array: ArrowArray
    var child_schema: ArrowSchema
    var array_children: List[Int]
    var schema_children: List[Int]
    var value_buffers: List[Int]
    var root_buffers: List[Int]

    def __init__(out self, mode: Int = 0, batch_count: Int = 2):
        self.mode = mode
        self.batch_count = batch_count
        self.next_calls = 0
        self.schema_calls = 0
        self.stream_releases = 0
        self.root_array_releases = 0
        self.child_array_releases = 0
        self.root_schema_releases = 0
        self.child_schema_releases = 0
        self.emitted = 0
        self.values = [1, 2]
        self.format = [108, 0]
        if mode == 2:
            self.format = [43, 108, 0]
        self.root_format = [43, 115, 0]
        self.field_name = [120, 0]
        self.error = []
        self.error.extend(
            String("deliberate lifecycle producer failure").as_bytes()
        )
        self.error.append(0)
        self.child_array = ArrowArray()
        self.child_schema = ArrowSchema()
        self.array_children = [0]
        self.schema_children = [0]
        self.value_buffers = [0, 0]
        self.root_buffers = [0]


def state_of(address: Int) -> ref[MutAnyOrigin] StreamState:
    return at[StreamState](address)[]


def release_child_array(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    ref state = state_of(array[].private_data)
    state.child_array_releases += 1
    array[].release = 0


def release_root_array(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    ref state = state_of(array[].private_data)
    state.root_array_releases += 1
    array_release(state.child_array)
    array[].release = 0


def release_child_schema(schema: Pointer[ArrowSchema, MutAnyOrigin]) abi("C"):
    ref state = state_of(schema[].private_data)
    state.child_schema_releases += 1
    schema[].release = 0


def release_root_schema(schema: Pointer[ArrowSchema, MutAnyOrigin]) abi("C"):
    ref state = state_of(schema[].private_data)
    state.root_schema_releases += 1
    schema_release(state.child_schema)
    schema[].release = 0


def get_next(stream_address: Int, output: Int) abi("C") -> Int32:
    ref stream = at[_ArrowArrayStream](stream_address)[]
    ref state = state_of(stream.private_data)
    state.next_calls += 1
    ref array = at[ArrowArray](output)[]
    if state.mode == 3 and state.emitted == 1:
        return 1
    if state.emitted >= state.batch_count:
        array.release = 0
        return 0
    state.values[0] = Int64(2 * state.emitted + 1)
    state.values[1] = Int64(2 * state.emitted + 2)
    state.value_buffers[0] = 0
    state.value_buffers[1] = Int(state.values.unsafe_ptr())
    state.child_array.length = 2
    state.child_array.null_count = 0
    state.child_array.offset = 0
    state.child_array.n_buffers = 2
    state.child_array.n_children = 0
    state.child_array.buffers = Int(state.value_buffers.unsafe_ptr())
    state.child_array.children = 0
    state.child_array.dictionary = 0
    var child_callback: ArrayRelease = release_child_array
    state.child_array.release = Pointer(to=child_callback).unsafe_bitcast[
        Int
    ]()[]
    state.child_array.private_data = stream.private_data
    state.array_children[0] = Int(Pointer(to=state.child_array))
    array.length = 2
    array.null_count = 0
    array.offset = 0
    array.n_buffers = 1
    array.n_children = 1
    array.buffers = Int(state.root_buffers.unsafe_ptr())
    array.children = Int(state.array_children.unsafe_ptr())
    array.dictionary = 0
    var callback: ArrayRelease = release_root_array
    array.release = Pointer(to=callback).unsafe_bitcast[Int]()[]
    array.private_data = stream.private_data
    state.emitted += 1
    return 1 if state.mode == 4 else 0


def get_schema(stream_address: Int, output: Int) abi("C") -> Int32:
    ref stream = at[_ArrowArrayStream](stream_address)[]
    ref state = state_of(stream.private_data)
    state.schema_calls += 1
    state.child_schema.format = Int(state.format.unsafe_ptr())
    state.child_schema.name = Int(state.field_name.unsafe_ptr())
    state.child_schema.metadata = 0
    state.child_schema.flags = 2
    state.child_schema.n_children = 0
    state.child_schema.children = 0
    state.child_schema.dictionary = 0
    var child_callback: SchemaRelease = release_child_schema
    state.child_schema.release = Pointer(to=child_callback).unsafe_bitcast[
        Int
    ]()[]
    state.child_schema.private_data = stream.private_data
    state.schema_children[0] = Int(Pointer(to=state.child_schema))
    ref schema = at[ArrowSchema](output)[]
    schema.format = Int(state.root_format.unsafe_ptr())
    schema.name = 0
    schema.metadata = 0
    schema.flags = 0
    schema.n_children = 1
    schema.children = Int(state.schema_children.unsafe_ptr())
    schema.dictionary = 0
    var callback: SchemaRelease = release_root_schema
    schema.release = Pointer(to=callback).unsafe_bitcast[Int]()[]
    schema.private_data = stream.private_data
    return 1 if state.mode == 1 else 0


def last_error(stream_address: Int) abi("C") -> Int:
    ref stream = at[_ArrowArrayStream](stream_address)[]
    return Int(state_of(stream.private_data).error.unsafe_ptr())


def release_stream(stream_address: Int) abi("C"):
    ref stream = at[_ArrowArrayStream](stream_address)[]
    ref state = state_of(stream.private_data)
    state.stream_releases += 1
    stream.release = 0


def counted_stream(mut state: StreamState) -> _ArrowArrayStream:
    var stream = _ArrowArrayStream()
    stream.private_data = Int(Pointer(to=state))
    var next_callback: StreamGet = get_next
    var schema_callback: StreamGet = get_schema
    var error_callback: StreamError = last_error
    var release_callback: StreamRelease = release_stream
    stream.get_next = Pointer(to=next_callback).unsafe_bitcast[Int]()[]
    stream.get_schema = Pointer(to=schema_callback).unsafe_bitcast[Int]()[]
    stream.get_last_error = Pointer(to=error_callback).unsafe_bitcast[Int]()[]
    stream.release = Pointer(to=release_callback).unsafe_bitcast[Int]()[]
    return stream^
