"""Independent Arrow extension producer and consumer for the Mojo oracle."""
import pyarrow as pa


class TaggedInt(pa.ExtensionType):
    def __init__(self, payload=b"\x00\xffregistered\x80"):
        self.payload = payload
        super().__init__(pa.int64(), "dataframe_mojo.test.tagged_int")

    def __arrow_ext_serialize__(self):
        return self.payload

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        assert storage_type == pa.int64()
        return cls(serialized)


pa.register_extension_type(TaggedInt())
CUSTOM = {b"binary\x00key\xff": b"\x80\xff\x00", b"empty": b""}


def make(registered):
    if registered:
        array = pa.ExtensionArray.from_storage(TaggedInt(), pa.array([3, None, 1, 2]))
        field = pa.field("x", array.type, metadata=CUSTOM)
    else:
        array = pa.array([3, None, 1, 2])
        field = pa.field("x", array.type, metadata={
            **CUSTOM,
            b"ARROW:extension:name": b"unknown.external.extension",
            b"ARROW:extension:metadata": b"\x00\xfeopaque\xff",
        })
    return pa.RecordBatch.from_arrays([array], schema=pa.schema([field]))


def check(batch, registered, rows):
    expected = make(registered)
    batch.validate(full=True)
    assert batch.schema.equals(expected.schema, check_metadata=True), (batch.schema, expected.schema)
    expected_values = expected.column(0).to_pylist()
    assert batch.column(0).to_pylist() == [expected_values[i] for i in rows]
    if registered:
        assert batch.column(0).type.payload == TaggedInt().payload
    return True


def make_nested():
    struct_type = pa.struct([pa.field("value", pa.int64(), metadata=CUSTOM)])
    list_type = pa.large_list(pa.field("item", pa.int64(), metadata=CUSTOM))
    return pa.RecordBatch.from_arrays(
        [pa.array([{"value": 1}, None, {"value": None}], type=struct_type),
         pa.array([[1, None], None, []], type=list_type)],
        schema=pa.schema([pa.field("record", struct_type, metadata=CUSTOM),
                          pa.field("list", list_type, metadata=CUSTOM)]),
    )


def check_nested(back, rows):
    original = make_nested()
    assert back.schema.equals(original.schema, check_metadata=True)
    assert back.to_pydict() == original.take(pa.array(rows, type=pa.int64())).to_pydict()
    return True
