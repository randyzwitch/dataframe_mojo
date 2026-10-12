"""Regenerate standalone Mojo fixtures from independent pyarrow C exports.

Run: .pixi/envs/oracle/bin/python tests/support/arrow_oracle.py [--check]
The generated module needs neither Python nor pyarrow at test runtime. Exported
buffers are snapshotted before their real pyarrow release callbacks are called.
"""
import ctypes as C
from decimal import Decimal
from pathlib import Path
import sys
import pyarrow as pa


class Array(C.Structure):
    pass


class Schema(C.Structure):
    pass


Array._fields_ = [
    (name, C.c_int64)
    for name in ("length", "null_count", "offset", "n_buffers", "n_children")
] + [
    ("buffers", C.POINTER(C.c_void_p)),
    ("children", C.POINTER(C.POINTER(Array))),
    ("dictionary", C.POINTER(Array)),
    ("release", C.c_void_p),
    ("private_data", C.c_void_p),
]
Schema._fields_ = [
    ("format", C.c_char_p),
    ("name", C.c_char_p),
    ("metadata", C.c_void_p),
    ("flags", C.c_int64),
    ("n_children", C.c_int64),
    ("children", C.POINTER(C.POINTER(Schema))),
    ("dictionary", C.POINTER(Schema)),
    ("release", C.c_void_p),
    ("private_data", C.c_void_p),
]


def metadata(address):
    if not address:
        return b""
    start = address
    count = C.c_int32.from_address(address).value
    address += 4
    for _ in range(2 * count):
        size = C.c_int32.from_address(address).value
        address += 4 + size
    return C.string_at(start, address - start)


def buffers_of(array):
    result = {b.address: b.size for b in array.buffers() if b is not None}
    if pa.types.is_dictionary(array.type):
        result.update(buffers_of(array.dictionary))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        result.update(buffers_of(array.values))
    if pa.types.is_struct(array.type):
        for index in range(array.type.num_fields):
            result.update(buffers_of(array.field(index)))
    return result


def snapshot(array, field):
    array.validate(full=True)
    exported = Array()
    schema = Schema()
    # Export the field separately to include arbitrary binary field metadata.
    array._export_to_c(C.addressof(exported))
    field._export_to_c(C.addressof(schema))
    registry = buffers_of(array)
    nodes = []

    def visit(a, s):
        index = len(nodes)
        node = dict(
            format=s.format.decode(),
            name=(s.name or b"").decode(),
            length=a.length,
            offset=a.offset,
            nulls=a.null_count,
            metadata=metadata(s.metadata),
            children=[],
            dictionary=-1,
            buffers=[
                C.string_at(a.buffers[k], registry[a.buffers[k]]) if a.buffers[
                    k
                ] else b""
                for k in range(a.n_buffers)
            ],
        )
        nodes.append(node)
        for k in range(a.n_children):
            node["children"].append(
                visit(a.children[k].contents, s.children[k].contents)
            )
        if a.dictionary:
            node["dictionary"] = visit(
                a.dictionary.contents, s.dictionary.contents
            )
        return index

    try:
        visit(exported, schema)
    finally:
        C.CFUNCTYPE(None, C.POINTER(Array))(exported.release)(C.byref(exported))
        C.CFUNCTYPE(None, C.POINTER(Schema))(schema.release)(C.byref(schema))
        assert not exported.release and not schema.release
    return nodes


def cases():
    arrays = [
        pa.array([999] * 9 + [11, None, -33, 44], type=pa.int64()).slice(9, 4),
        pa.array([False] * 9 + [True, None, False, True]).slice(9, 4),
        pa.array(["old", "pre", "", "hé", None, "z"]).slice(2, 4),
        pa.array([b"old", b"pre", b"", b"\xff\x00\x01", None, b"z"]).slice(
            2, 4
        ),
        pa.DictionaryArray.from_arrays(
            pa.array([1, None, 2, 1], type=pa.int32()),
            pa.array(["red", "blue", None]),
        ),
        pa.array(
            [
                [{"id": 7, "name": "a"}],
                [{"id": 8, "name": "b"}, None],
                [],
                None,
            ],
            type=pa.list_(
                pa.struct([("id", pa.int64()), ("name", pa.string())])
            ),
        ).slice(1, 3),
        pa.array(
            [Decimal("-12.34"), None, Decimal("56.78")], type=pa.decimal32(9, 2)
        ),
        pa.array(
            [Decimal("-1.234"), None, Decimal("5.678")],
            type=pa.decimal64(18, 3),
        ),
        pa.array(
            [Decimal("-0.1234"), None, Decimal("0.5678")],
            type=pa.decimal128(38, 4),
        ),
        pa.array([-1, None, 42], type=pa.timestamp("ns", "America/New_York")),
        pa.StructArray.from_arrays(
            [
                pa.array([999, 11, None, -33, 44], type=pa.int64()),
                pa.array(["prefix", "", "hé", None, "z"]),
            ],
            names=["integer", "text"],
        ).slice(1, 3),
    ]
    for array in arrays:
        yield snapshot(
            array,
            pa.field(
                "value", array.type, metadata={b"\x00\xffkey": b"value\x00\xfe"}
            ),
        )


def quoted(text):
    import json

    return json.dumps(text, ensure_ascii=False)


def byte_list(data):
    return "[" + ", ".join(str(x) for x in data) + "]"


def generate():
    lines = [
        f'"""Generated by arrow_oracle.py from pyarrow {pa.__version__}; do not hand edit."""',
        "from tests.support.arrow_foreign import ForeignArrow",
        "",
        "",
        "def pyarrow_producer(fixture_id: Int) raises -> ForeignArrow:",
    ]
    for fixture_id, nodes in enumerate(cases()):
        lines += [
            f"    {'if' if fixture_id == 0 else 'elif'} fixture_id == {fixture_id}:",
            f"        var producer = ForeignArrow({len(nodes)})",
        ]
        for index, node in enumerate(nodes):
            target = f"producer.nodes[{index}]"
            lines.append(
                f"        {target}.describe({quoted(node['format'])}, {quoted(node['name'])}, {node['length']}, {node['offset']}, {node['nulls']})"
            )
            if node["metadata"]:
                lines.append(
                    f"        {target}.metadata = {byte_list(node['metadata'])}"
                )
            for data in node["buffers"]:
                # Offset all nonempty buffers by one byte to exercise unaligned
                # loading, keeping Arrow logical offsets from pyarrow intact.
                if data:
                    lines.append(
                        f"        {target}.buffer({byte_list(bytes([229]) + data)}, 1)"
                    )
                else:
                    lines.append(f"        {target}.buffer([])")
            if node["children"]:
                lines.append(f"        {target}.children = {node['children']}")
            if node["dictionary"] >= 0:
                lines.append(
                    f"        {target}.dictionary = {node['dictionary']}"
                )
        lines += ["        producer.seal()", "        return producer^"]
    lines += ['    raise Error("unknown independent pyarrow fixture")', ""]
    import subprocess
    import tempfile

    formatter = Path(__file__).parents[2] / ".pixi/envs/default/bin/mblack"
    with tempfile.TemporaryDirectory(prefix="arrow-fixtures-") as temporary:
        source = Path(temporary) / "fixtures.mojo"
        source.write_text("\n".join(lines))
        result = subprocess.run(
            [
                str(formatter),
                "--target-version",
                "mojo",
                "--line-length",
                "80",
                str(source),
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return source.read_text()


if __name__ == "__main__":
    target = Path(__file__).with_name("arrow_pyarrow_fixtures.mojo")
    contents = generate()
    if "--check" in sys.argv:
        assert (
            target.read_text() == contents
        ), "regenerate Arrow oracle fixture module"
    else:
        target.write_text(contents)
    print(
        f"Verified {len(list(cases()))} independent pyarrow C export fixtures ({pa.__version__})"
    )
