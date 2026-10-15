"""Scoped primitive and persistent-List fields before nested row operators.

A logical nested field owns its List descriptors and child Table explicitly;
it is never assigned a fictional primitive matrix slot. Recursive List/Struct
field variants can extend this owner without changing primitive Table storage.
"""
from dataframe import Series
from .residency import Scope, Table
from .device_lists import DeviceLists, validate


@fieldwise_init
struct ResidentField(Copyable):
    var prototype: Series
    var table: Optional[Table]
    var lists: Optional[DeviceLists]

    @staticmethod
    def primitive(value: Table) raises -> Self:
        if value.schema.width() != 1:
            raise Error("Resident primitive field requires one logical column")
        return Self(value.schema._columns[0].copy(), value.copy(), None)

    @staticmethod
    def list(value: DeviceLists) -> Self:
        return Self(value.prototype.copy(), None, value.copy())


def validate_field(field: ResidentField, scope: Scope, rows: Int) raises:
    if Bool(field.table) == Bool(field.lists):
        raise Error("Resident field requires exactly one physical owner")
    if len(field.prototype):
        raise Error("Resident field prototype must be metadata only")
    if field.lists:
        ref value = field.lists.value()
        validate(value, scope)
        if (
            value.rows != rows
            or value.prototype.dtype() != field.prototype.dtype()
        ):
            raise Error("Resident List field rows or logical dtype differ")
        return
    ref value = field.table.value()
    if (
        value.rows != rows
        or value.schema.width() != 1
        or value.schema._columns[0].dtype() != field.prototype.dtype()
    ):
        raise Error("Resident primitive field rows or logical dtype differ")
    if field.prototype.dtype().is_nested():
        raise Error("Resident primitive field cannot carry nested storage")
    if value.storage:
        var owner = value.matrix()
        if not owner[].scope.same_query(scope):
            raise Error("Resident primitive field belongs to another query")
        if (
            len(value.columns) != 1
            or value.columns[0] < 0
            or value.columns[0] >= Int64(owner[].slots)
            or value.offset < 0
            or value.offset > owner[].capacity
            or rows > owner[].capacity - value.offset
        ):
            raise Error("Resident primitive field lies outside its matrix")
    elif rows:
        raise Error("Resident primitive field requires device-backed storage")


@fieldwise_init
struct ResidentFrame(Copyable):
    var scope: Scope
    var rows: Int
    var fields: List[ResidentField]


def validate_frame(frame: ResidentFrame, scope: Scope) raises:
    if (
        not frame.scope.same_query(scope)
        or frame.rows < 0
        or frame.rows > Int(Int32.MAX)
    ):
        raise Error(
            "Resident fields belong to another query or invalid row range"
        )
    var names = List[String]()
    for field in frame.fields:
        for name in names:
            if name == field.prototype.name():
                raise Error("Resident field name is duplicated")
        names.append(field.prototype.name())
        validate_field(field, scope, frame.rows)
