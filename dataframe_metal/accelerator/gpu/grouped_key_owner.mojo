"""Query-scoped certificates for original grouped key domains.

The certificate retains the exact prepared plan, original KeyQuery, staged
matrix view and its staging charge. A UInt32 column alone is never a key
certificate, including when its values happen to fit a dictionary.
"""
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from dataframe import DataType, Column, Series
from dataframe_metal.accelerator.grouped_key_queries import KeyQuery
from dataframe_metal.accelerator.grouped_lists import GroupedListsPlan
from dataframe.key_domains import (
    FIXED,
    STRING,
    BINARY,
    GEOMETRY,
    CATEGORY,
    CODED_STRING,
)
from dataframe.byte_domain import hex_key
from .residency import Scope, Table, checked_add, checked_mul
from .device_lists import HostReservation


def staging_bytes(query: KeyQuery) raises -> Int:
    var total = query.staging_bytes
    for index in range(len(query.domains)):
        ref domain = query.domains[index]
        if domain.kind != FIXED and query.rows:
            total = checked_add(
                total,
                checked_add(
                    checked_mul(query.rows, 16),
                    checked_add(
                        checked_mul(len(domain.provenance.values[]), 4),
                        checked_mul(
                            2 * domain.encoded.n_chunks(), size_of[Series]()
                        ),
                    ),
                ),
            )
    return total


struct GroupedKeyOwner(Movable):
    var query: ArcPointer[KeyQuery]
    var plan: ArcPointer[GroupedListsPlan]
    var source: Table
    var scope: Scope
    var staging: ArcPointer[HostReservation]
    var publication_maps: List[List[UInt32]]

    def __init__(
        out self,
        query: ArcPointer[KeyQuery],
        plan: ArcPointer[GroupedListsPlan],
        source: Table,
        scope: Scope,
        staging: ArcPointer[HostReservation],
    ) raises:
        if not staging[].scope.same_query(scope):
            raise Error("Grouped key staging belongs to another query")
        if staging[].bytes != staging_bytes(query[]):
            raise Error(
                "Grouped key staging charge differs from original owner"
            )
        if query[].rows != plan[].rows or source.rows != plan[].rows:
            raise Error("Grouped key source row certificate differs")
        if query[].schema.width() != len(query[].key_outputs):
            raise Error("Grouped key original publication width differs")
        if query[].names != plan[].names or len(query[].domains) != len(
            plan[].names
        ):
            raise Error("Grouped key original domain certificate differs")
        for field in source.grouped_keys:
            if field:
                raise Error(
                    "Grouped key source must retain its original scan certificate"
                )
        if source.schema.width() != len(plan[].source.outputs):
            raise Error("Grouped key source schema certificate differs")
        for index in range(source.schema.width()):
            ref expected = plan[].source.outputs[index]
            ref actual = source.schema._columns[index]
            if (
                actual.name() != expected.name
                or actual.dtype() != expected.dtype
            ):
                raise Error("Grouped key source transport schema differs")
        if source.storage:
            var matrix = source.matrix()
            if not matrix[].scope.same_query(scope):
                raise Error("Grouped key source belongs to another query")
            if (
                source.offset < 0
                or source.offset > matrix[].capacity
                or source.rows > matrix[].capacity - source.offset
            ):
                raise Error("Grouped key source view is outside its matrix")
            if len(source.columns) != source.schema.width():
                raise Error("Grouped key source physical width differs")
            for slot in source.columns:
                if slot < 0 or slot >= Int64(matrix[].slots):
                    raise Error(
                        "Grouped key source column is outside its matrix"
                    )
        elif source.rows:
            raise Error("Grouped key source requires its staged matrix")
        if (
            source.identity_storage
            and not source.identity_storage.value()[].scope.same_query(scope)
        ):
            raise Error("Grouped key identity source belongs to another query")
        var maps = List[List[UInt32]]()
        for index in range(len(query[].domains)):
            ref domain = query[].domains[index]
            var mapping = List[UInt32]()
            if domain.kind != FIXED and query[].rows:
                mapping = List[UInt32](
                    length=len(domain.provenance.values[]), fill=UInt32.MAX
                )
                var original = domain.encoded.rechunk()
                var published = (
                    plan[].source.source.column(query[].names[index]).rechunk()
                )
                if published.dtype() != DataType.UINT32 or len(
                    published
                ) != len(original):
                    raise Error(
                        "Grouped key publication transport geometry differs"
                    )
                ref old = original._data[Column[UInt32]]
                ref new = published._data[Column[UInt32]]
                for row in range(len(original)):
                    if old._valid(row) != new._valid(row):
                        raise Error("Grouped key publication validity differs")
                    if not old._valid(row):
                        continue
                    var code = Int(old._get(row))
                    if code >= len(mapping):
                        raise Error(
                            "Grouped key publication original code is outside its domain"
                        )
                    var transport = new._get(row)
                    if (
                        mapping[code] != UInt32.MAX
                        and mapping[code] != transport
                    ):
                        raise Error(
                            "Grouped key publication remap is inconsistent"
                        )
                    mapping[code] = transport
                for code in range(len(mapping)):
                    var transport = mapping[code]
                    if transport == UInt32.MAX:
                        continue
                    if domain.kind == BINARY or domain.kind == GEOMETRY:
                        if plan[].publication_bytes:
                            ref published_domain = (
                                plan[].publication_bytes.value()[]
                            )
                            if (
                                Int(transport) >= len(published_domain.keys)
                                or published_domain.keys[Int(transport)]
                                != domain.dictionary.value().keys[code]
                            ):
                                raise Error(
                                    "Grouped opaque publication remap changes original bytes"
                                )
                        elif transport != UInt32(code):
                            raise Error(
                                "Grouped opaque publication domain is missing"
                            )
                    elif domain.kind == STRING:
                        if plan[].source.source_domain:
                            var dictionary = (
                                plan[].source.source_domain.value().dictionary()
                            )
                            if (
                                Int(transport) >= len(dictionary[])
                                or hex_key(
                                    String(dictionary[].get(Int(transport)))
                                )
                                != domain.dictionary.value().keys[code]
                            ):
                                raise Error(
                                    "Grouped String publication remap changes original bytes"
                                )
                        elif transport != UInt32(code):
                            raise Error(
                                "Grouped String publication domain is missing"
                            )
                if domain.kind == CATEGORY or domain.kind == CODED_STRING:
                    var position = source.schema._index(query[].names[index])
                    if source.categories[position]:
                        ref category = (
                            source.categories[position].value().owner[]
                        )
                        if (
                            category.dtype != domain.provenance.dtype
                            or category.dtype.dictionary().ptr()
                            != domain.provenance.dtype.dictionary().ptr()
                        ):
                            raise Error(
                                "Grouped categorical publication dictionary differs"
                            )
                        var row = 0
                        for chunk in category.codes:
                            for at in range(len(chunk)):
                                if (
                                    row >= len(original)
                                    or chunk._valid(at) != old._valid(row)
                                    or (
                                        chunk._valid(at)
                                        and chunk._get(at) != old._get(row)
                                    )
                                ):
                                    raise Error(
                                        "Grouped categorical publication original code differs"
                                    )
                                row += 1
                        if row != len(original):
                            raise Error(
                                "Grouped categorical publication original rows differ"
                            )
                    elif plan[].source.source_domain:
                        raise Error(
                            "Grouped categorical publication original code owner is missing"
                        )
            maps.append(mapping^)
        self.query = query.copy()
        self.plan = plan.copy()
        self.source = source.copy()
        self.scope = scope.copy()
        self.staging = staging.copy()
        self.publication_maps = maps^

    def publication(self, domain: Int) raises -> Series:
        if domain < 0 or domain >= len(self.query[].domains):
            raise Error("Grouped key field domain is outside its owner")
        for index in range(len(self.query[].key_outputs)):
            if self.query[].key_outputs[index] == domain:
                return self.query[].schema._columns[index].slice(0, 0)
        raise Error("Grouped key original publication field is missing")

    def validate(
        self, plan: GroupedListsPlan, source: Table, scope: Scope
    ) raises:
        if not self.scope.same_query(scope):
            raise Error("Grouped key owner belongs to another query")
        if Int(Pointer(to=plan)) != Int(Pointer(to=self.plan[])):
            raise Error("Grouped key owner prepared plan differs")
        if (
            source.rows != self.source.rows
            or source.offset != self.source.offset
            or source.columns != self.source.columns
        ):
            raise Error("Grouped key owner source view differs")
        if Bool(source.storage) != Bool(self.source.storage):
            raise Error("Grouped key owner source matrix differs")
        if len(source.text) != len(self.source.text) or len(
            source.categories
        ) != len(self.source.categories):
            raise Error("Grouped key owner source sidecar width differs")
        for index in range(len(source.text)):
            if Bool(source.text[index]) != Bool(self.source.text[index]):
                raise Error("Grouped key owner source text domain differs")
            if (
                source.text[index]
                and source.text[index].value().domain.ptr()
                != self.source.text[index].value().domain.ptr()
            ):
                raise Error("Grouped key owner source text domain differs")
            if Bool(source.categories[index]) != Bool(
                self.source.categories[index]
            ):
                raise Error("Grouped key owner source category domain differs")
            if (
                source.categories[index]
                and source.categories[index].value().owner.ptr()
                != self.source.categories[index].value().owner.ptr()
            ):
                raise Error("Grouped key owner source category domain differs")
        if (
            source.storage
            and source.storage.value().ptr()
            != self.source.storage.value().ptr()
        ):
            raise Error("Grouped key owner source matrix differs")


@fieldwise_init
struct GroupedKeyField(Copyable):
    var owner: ArcPointer[GroupedKeyOwner]
    var domain: Int

    def logical_dtype(self) raises -> DataType:
        if self.domain < 0 or self.domain >= len(self.owner[].query[].domains):
            raise Error("Grouped key field domain is outside its owner")
        return self.owner[].publication(self.domain).dtype()

    def validate(self, table: Table, scope: Scope) raises:
        if not self.owner[].scope.same_query(scope):
            raise Error("Grouped key field belongs to another query")
        if table.storage and not table.matrix()[].scope.same_query(scope):
            raise Error("Grouped key field matrix belongs to another query")
        if (
            table.identity_storage
            and not table.identity_storage.value()[].scope.same_query(scope)
        ):
            raise Error("Grouped key field identity belongs to another query")
        _ = self.logical_dtype()
        ref domain = self.owner[].query[].domains[self.domain]
        if (
            table.schema.width() != 1
            or table.schema._columns[0].dtype() != domain.encoded.dtype()
        ):
            raise Error("Grouped key field transport dtype differs")
        if (
            domain.kind != FIXED
            and table.schema._columns[0].dtype() != DataType.UINT32
        ):
            raise Error("Grouped key field requires explicit code transport")
