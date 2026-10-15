"""Transport immutable primitive/List source arrays into scoped field owners.

Only source storage is copied. No row expression, grouping, expansion or child
query is evaluated on CPU. Private chunk merges bypass the caller's cache and
are charged until upload completes; clean metadata prototypes retain no newly
merged CPU payload after publication.
"""
from dataframe import DataFrame, ListColumn, Series
from dataframe.accelerator.source_lists import fixed, child_prototype
from .residency import Scope, Table, checked_add, checked_mul
from .resident_fields import ResidentField, ResidentFrame, validate_frame
from .host_list_reference import list_ref, stage_child
from .device_lists import HostReservation, stage_descriptors, empty_grouped
from .runtime import _width


def _primitive_prototype(source: Series) raises -> Series:
    var result = Series.full_null(source.name(), source.dtype(), 0)
    result._field_metadata = source._field_metadata
    return result^


def _prototype(source: Series) raises -> Series:
    if not source.dtype().is_list():
        return _primitive_prototype(source)
    var original = child_prototype(source)
    var child = _primitive_prototype(original).renamed("item")
    var offsets: List[Int64] = [0]
    var result = Series(source.name(), ListColumn(offsets^, child^))
    result._field_metadata = source._field_metadata
    return result^


def preflight(source: DataFrame) raises:
    if (
        source.height() < 0
        or source.height() > Int(Int32.MAX)
        or source.width() > 256
    ):
        raise Error("Resident source fields exceed fixed-width GPU dimensions")
    for field in source._columns:
        var dtype = field.dtype()
        if dtype.is_list():
            if not fixed(dtype.inner()):
                raise Error(
                    "Resident source List requires primitive fixed-width children"
                )
            _ = child_prototype(field)
            if source.height():
                for part in field.chunks():
                    if len(part.list_column().child()) > Int(Int32.MAX):
                        raise Error(
                            "Resident source List child exceeds GPU dimensions"
                        )
        elif not fixed(dtype):
            raise Error(
                "Resident source ordinary fields require primitive fixed-width storage"
            )


def staging_bytes(source: DataFrame) raises -> Int:
    """Private CPU chunk-merge envelope, excluding immutable caller buffers."""
    preflight(source)
    if not source.height():
        return 0
    var total = 0
    for field in source._columns:
        if not field.is_chunked():
            continue
        var rows = len(field)
        var bytes = checked_add(
            checked_mul(rows, _width(field.dtype())), (rows + 7) // 8
        ) if not field.dtype().is_list() else 0
        if field.dtype().is_list():
            var children = 0
            for part in field.chunks():
                var lists = part.list_column()
                if len(lists):
                    children = checked_add(
                        children, lists._end(len(lists) - 1) - lists._start(0)
                    )
            if children > Int(Int32.MAX):
                raise Error("Resident merged List child exceeds GPU dimensions")
            bytes = checked_add(
                checked_add(checked_mul(rows + 1, 8), rows), (rows + 7) // 8
            )
            bytes = checked_add(
                bytes,
                checked_add(
                    checked_mul(children, _width(field.dtype().inner())),
                    (children + 7) // 8,
                ),
            )
        # List append retains the previous private child while compacting its
        # first segment and reserving the replacement: three payload owners can
        # coexist. Parent Bool validity scratch is included above. Ordinary
        # primitive merge needs compact+replacement only. This is explicit
        # requested workspace, not allocator spare capacity or overhead.
        total = checked_add(
            total, checked_mul(bytes, 3 if field.dtype().is_list() else 2)
        )
    return total


def stage(source: DataFrame, scope: Scope) raises -> ResidentFrame:
    preflight(source)
    var bytes = staging_bytes(source)
    var reservation = HostReservation(scope, bytes)
    var fields = List[ResidentField]()
    for input in source._columns:
        var prototype = _prototype(input)
        # _merge_chunks builds private storage without populating the immutable
        # caller's shared _MergedCache. Empty metadata never merges payloads.
        var field = (
            input._merge_chunks() if source.height()
            and input.is_chunked() else input.copy()
        )
        if field.dtype().is_list():
            var child_prototype = prototype.list_column().child()
            if not len(field):
                fields.append(
                    ResidentField.list(
                        empty_grouped(child_prototype, scope, field.name())
                    )
                )
            else:
                var child = stage_child(field.list_column().child(), scope)
                child.schema = DataFrame([child_prototype.copy()])
                var reference = list_ref(field, child, scope)
                var value = stage_descriptors(reference, scope)
                value.prototype = prototype.copy()
                fields.append(ResidentField.list(value))
                _ = reference^
                _ = child^
                _ = value^
        else:
            var value = stage_child(field, scope)
            value.schema = DataFrame([prototype.copy()])
            fields.append(ResidentField.primitive(value))
            _ = value^
        _ = field^
    var result = ResidentFrame(scope.copy(), source.height(), fields^)
    validate_frame(result, scope)
    _ = reservation^
    return result^
