"""Publish certified grouped fields as scoped GPU owners, without List export."""
from std.memory import ArcPointer
from .grouped_key_owner import GroupedKeyOwner, GroupedKeyField
from dataframe.accelerator.grouped_key_domains import CODED_STRING
from dataframe.parallel import configured_workers
from dataframe import DataFrame, Series
from dataframe.accelerator.grouped_lists import GroupedListsPlan
from .residency import Scope, Table, TextDomain
from .runtime import execute_plan
from .device_lists import (
    DeviceLists,
    grouped_implode,
    empty_grouped,
    consume,
)
from .list_groups import build_groups
from .grouped_phase_groups import PhaseGroups
from .grouped_feed_runtime import whole_geometry, execute as execute_feed
from .grouped_schedule import build as build_schedule, read_phases, gather_phase
from .grouped_list_targets import (
    ListTarget,
    build_destinations,
    create_target,
    append_phase,
    finish_target,
)

from .resident_fields import ResidentField, ResidentFrame, validate_frame


def _column(source: Table, name: String) raises -> Table:
    var result = source.copy()
    for index in range(source.schema.width()):
        if source.schema._columns[index].name() == name:
            result.schema = DataFrame([source.schema._columns[index].copy()])
            if (
                not source.rows
                and not source.storage
                and not source.identity_storage
            ):
                return Table.empty(result.schema.copy())
            result.columns = [source.columns[index]]
            result.text = [source.text[index].copy()]
            result.categories = [source.categories[index].copy()]
            result.category_leases = source.category_leases.copy()
            result.grouped_keys = [source.grouped_keys[index].copy()]
            return result^
    raise Error("Grouped IMPLODE output column is missing")


def _validate_source(
    plan: GroupedListsPlan,
    source: Table,
    scope: Scope,
    typed: Optional[ArcPointer[GroupedKeyOwner]] = None,
) raises:
    if typed:
        typed.value()[].validate(plan, source, scope)
    if not typed and (
        len(plan.key_provenance)
        or plan.source.source_domain
        or len(plan.source.source_text)
    ):
        raise Error(
            "Resident grouped publication requires explicit typed-domain owner for encoded keys"
        )
    if source.rows != plan.rows or source.schema.width() != len(
        plan.source.outputs
    ):
        raise Error("Resident grouped producer source geometry differs")
    for index in range(source.schema.width()):
        ref expected = plan.source.outputs[index]
        ref field = source.schema._columns[index]
        if field.name() != expected.name or field.dtype() != expected.dtype:
            raise Error("Resident grouped producer source schema differs")
    if source.rows:
        if not source.storage:
            raise Error(
                "Resident grouped producer needs a device-backed source"
            )
        var matrix = source.matrix()
        if not matrix[].scope.same_query(scope):
            raise Error(
                "Resident grouped producer source belongs to another query"
            )
        if (
            source.offset < 0
            or source.offset > matrix[].capacity
            or source.rows > matrix[].capacity - source.offset
            or len(source.columns) != source.schema.width()
        ):
            raise Error("Resident grouped producer source view is invalid")
        for slot in source.columns:
            if slot < 0 or slot >= Int64(matrix[].slots):
                raise Error(
                    "Resident grouped producer source physical column is invalid"
                )
    elif source.storage and not source.matrix()[].scope.same_query(scope):
        raise Error("Resident grouped producer source belongs to another query")


def _slice(
    frame: ResidentFrame, offset: Int, length: Int
) raises -> ResidentFrame:
    if length < -1:
        raise Error("Slice length must be nonnegative")
    var lo = min(offset, frame.rows) if offset >= 0 else max(
        0, frame.rows + offset
    )
    var rows = frame.rows - lo if length < 0 else min(length, frame.rows - lo)
    var fields = frame.fields.copy()
    for index in range(len(fields)):
        if fields[index].lists:
            var value = fields[index].lists.value().copy()
            value.offset += lo
            value.rows = rows
            fields[index].lists = value^
        else:
            var value = fields[index].table.value().copy()
            value.offset += lo
            value.rows = rows
            fields[index].table = value^
    var result = ResidentFrame(frame.scope.copy(), rows, fields^)
    validate_frame(result, frame.scope)
    return result^


def produce(
    plan: GroupedListsPlan,
    source: Table,
    scope: Scope,
    profiling: Bool = False,
    *,
    typed: Optional[ArcPointer[GroupedKeyOwner]] = None,
) raises -> ResidentFrame:
    """Complete the certified grouped producer without any List/value export.

    Caller owns the prepared plan and already-staged source through this call.
    It records original scan execution once when staging. This function never
    reconstructs keys or groups on CPU, never creates a new Scope, and cannot
    grant an original-source ordering certificate to an arbitrary resident AST.
    """
    _validate_source(plan, source, scope, typed)
    source.observe_errors()
    var layout = build_groups(source, plan.names, scope)
    var text_domain = Optional[ArcPointer[TextDomain]]()
    for field in source.text:
        if field:
            text_domain = field.value().domain.copy()
            break
    var key_result = execute_plan(
        plan.keys,
        scope,
        Optional(source.copy()),
        profiling,
        export_values=False,
        text_domain=text_domain,
    )
    var keys = key_result[0].copy()
    keys.observe_errors()
    var lists = List[DeviceLists]()
    var scalars = List[Table]()
    var indices = List[Int]()
    if plan.scheduled:
        var schedule = build_schedule(
            source,
            plan.names,
            scope,
            configured_workers(),
            provenance=plan.key_provenance,
        )
        var phases = read_phases(schedule, scope)
        var destinations = build_destinations(layout.value(), scope)
        var targets = List[ListTarget]()
        for index in range(len(plan.producers)):
            if plan.list_outputs[index]:
                ref descriptor = plan.producers[index][].outputs[0]
                var child = Series.full_null(
                    descriptor.name, descriptor.dtype, 0
                )
                child._field_metadata = descriptor.field_metadata
                indices.append(len(targets))
                targets.append(
                    create_target(
                        destinations, schedule, child, scope, child.name()
                    )
                )
            else:
                # Admission excludes input/feed/state/association faults on
                # this path. Exact infallible scalar siblings can be computed
                # independently, while checked List operands obey CPU phases.
                var produced = execute_plan(
                    plan.producers[index][],
                    scope,
                    Optional(source.copy()),
                    profiling,
                    export_values=False,
                )
                var output = _column(
                    produced[0],
                    plan.grouped_schema._columns[
                        len(plan.names) + index
                    ].name(),
                )
                output.observe_errors()
                indices.append(len(scalars))
                scalars.append(output^)
        # CPU job priority precedes projection priority: heavy buckets first,
        # then light buckets, each in ascending bucket order. A private target
        # only publishes after every successful phase filled its stable slots.
        for phase in range(Int(schedule[].facts[4])):
            var bucket = gather_phase(source, schedule, phases, phase, scope)
            for index in range(len(plan.producers)):
                if not plan.list_outputs[index]:
                    continue
                var produced = execute_plan(
                    plan.producers[index][],
                    scope,
                    Optional(bucket.copy()),
                    profiling,
                    export_values=False,
                )
                var output = _column(
                    produced[0],
                    plan.grouped_schema._columns[
                        len(plan.names) + index
                    ].name(),
                )
                append_phase(
                    targets[indices[index]], output, phases, phase, scope
                )
        # Pop in reverse, then restore original List output order.
        var reversed = List[DeviceLists]()
        while len(targets):
            reversed.append(finish_target(targets.pop(), scope))
        while len(reversed):
            lists.append(reversed.pop())
    else:
        var geometry = Optional[PhaseGroups]()
        var feed_count = 0
        for descriptor in plan.feed_plans:
            feed_count += Int(Bool(descriptor))
        if feed_count and layout:
            var destinations = build_destinations(layout.value(), scope)
            geometry = whole_geometry(destinations, scope)
        for index in range(len(plan.producers)):
            if plan.feed_plans[index]:
                var output = execute_feed(
                    plan.feed_plans[index].value(),
                    source,
                    geometry.value(),
                    scope,
                ) if geometry else Table.empty(
                    DataFrame([plan.feed_plans[index].value().prototype.copy()])
                )
                indices.append(len(scalars))
                scalars.append(output^)
                continue
            var produced = execute_plan(
                plan.producers[index][],
                scope,
                Optional(source.copy()),
                profiling,
                export_values=False,
            )
            var output = _column(
                produced[0],
                plan.grouped_schema._columns[len(plan.names) + index].name(),
            )
            # CPU serial grouped route completes each original aggregate before
            # preparing the next one's states/operand. No multi-output region can
            # prepare a later reduction ahead of an earlier checked List operand.
            if plan.list_outputs[index]:
                var value = grouped_implode(
                    output,
                    layout.value(),
                    scope,
                    output.schema._columns[0].name(),
                ) if layout else empty_grouped(
                    output.schema._columns[0],
                    scope,
                    output.schema._columns[0].name(),
                )
                indices.append(len(lists))
                lists.append(value^)
            else:
                output.observe_errors()
                indices.append(len(scalars))
                scalars.append(output.copy())
            _ = output^
            _ = produced^
    # Lists cross the aggregate -> row-consumer boundary as device owners.
    # No offsets or child values are exported/reuploaded for len/get.
    var final_tables = List[Table]()
    var final_lists = List[DeviceLists]()
    var final_indices = List[Int]()
    var final_kinds = List[Bool]()
    for request in plan.consumers:
        if request.kind:
            var producer = request.source - len(plan.names)
            var value = lists[indices[producer]].copy()
            if request.kind == 1:
                value.prototype = value.prototype.renamed(request.name)
                final_indices.append(len(final_lists))
                final_lists.append(value^)
                final_kinds.append(True)
            else:
                var output = consume[True](
                    value, scope, request.name
                ) if request.kind == 2 else consume[False](
                    value, scope, request.name, request.index
                )
                output.observe_errors()
                final_indices.append(len(final_tables))
                final_tables.append(output^)
                final_kinds.append(False)
        else:
            var table = (
                _column(
                    keys, plan.grouped_schema._columns[request.source].name()
                ) if request.source
                < len(plan.names) else scalars[
                    indices[request.source - len(plan.names)]
                ].copy()
            )
            table.schema = DataFrame(
                [table.schema._columns[0].renamed(request.name)]
            )
            if typed and request.source < len(plan.names):
                table.grouped_keys = [
                    GroupedKeyField(typed.value().copy(), request.source)
                ]
                ref original = typed.value()[].query[]
                var logical = typed.value()[].publication(request.source)
                table.schema._columns[
                    0
                ]._field_metadata = logical._field_metadata
                if (
                    original.domains[request.source].kind == CODED_STRING
                    or not logical.dtype().is_categorical()
                ):
                    table.categories[0] = None
            final_indices.append(len(final_tables))
            final_tables.append(table^)
            final_kinds.append(False)
    for slice in plan.slices:
        if slice.source_failure:
            raise Error(slice.source_failure)
        if slice.length < -1:
            raise Error("Slice length must be nonnegative")
    var fields = List[ResidentField](capacity=len(plan.consumers))
    for index in range(len(plan.consumers)):
        if final_kinds[index]:
            var value = final_lists[final_indices[index]].copy()
            value.prototype._field_metadata = plan.schema._columns[
                index
            ]._field_metadata
            fields.append(ResidentField.list(value))
        else:
            var table = final_tables[final_indices[index]].copy()
            table.schema._columns[0]._field_metadata = plan.schema._columns[
                index
            ]._field_metadata
            if table.grouped_keys[0]:
                ref certificate = table.grouped_keys[0].value()
                var publication = certificate.owner[].publication(
                    certificate.domain
                )
                table.schema._columns[
                    0
                ]._field_metadata = publication._field_metadata
            var field = ResidentField.primitive(table)
            if table.grouped_keys[0]:
                ref certificate = table.grouped_keys[0].value()
                field.prototype = (
                    certificate.owner[]
                    .publication(certificate.domain)
                    .renamed(plan.schema._columns[index].name())
                )
            fields.append(field^)
    var result = ResidentFrame(scope.copy(), keys.rows, fields^)
    validate_frame(result, scope)
    for slice in plan.slices:
        result = _slice(result, slice.offset, slice.length)
    # Existing helpers synchronize before publication. Explicit final source/
    # layout uses also retain every input owner through their last submission.
    _ = layout^
    _ = keys^
    _ = lists^
    _ = scalars^
    _ = final_tables^
    _ = final_lists^
    _ = typed^
    return result^
