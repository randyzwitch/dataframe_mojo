"""Row relations implemented with scalar scans, independent of the dataframe."""
from tests.support.reference_model import (
    ReferenceRow,
    ReferenceTable,
    equal_keys,
    field_valid,
    reference_groups,
)


def distinct_indices(
    model: ReferenceTable, subset: List[String], keep: String
) raises -> List[Int]:
    var keys = model.names.copy() if not subset else subset.copy()
    if not keys or keep not in ["any", "first", "last", "none"]:
        raise Error("Invalid reference distinct options")
    var groups = reference_groups(model, keys)
    var selected = List[Bool](length=len(model.rows), fill=False)
    for group in groups:
        if keep == "none":
            if len(group) == 1:
                selected[group[0]] = True
        else:
            selected[
                group[len(group) - 1] if keep == "last" else group[0]
            ] = True
    var indices = List[Int]()
    for index in range(len(selected)):
        if selected[index]:
            indices.append(index)
    return indices^


def nonnull_indices(
    model: ReferenceTable, subset: List[String]
) raises -> List[Int]:
    var keys = model.names.copy() if not subset else subset.copy()
    _ = model.project(keys)
    var indices = List[Int]()
    for index in range(len(model.rows)):
        var valid = True
        for key in keys:
            valid = valid and field_valid(model.rows[index], key)
        if valid:
            indices.append(index)
    return indices^


def _assign_field(
    mut target: ReferenceRow, source: ReferenceRow, name: String
) raises:
    if name == "i":
        target.i = source.i
        target.iv = source.iv
    elif name == "f":
        target.f = source.f
        target.fv = source.fv
    elif name == "s":
        target.s = source.s.copy()
        target.sv = source.sv
    elif name == "b":
        target.b = source.b
        target.bv = source.bv
    else:
        raise Error("Unknown reference field")


def concatenated_model(
    frames: List[ReferenceTable], how: String
) raises -> ReferenceTable:
    if not frames or how not in ["vertical", "diagonal", "horizontal"]:
        raise Error("Invalid reference concat options")
    var names = List[String]()
    for frame in frames:
        _ = frame.project(frame.names)
        if how == "vertical" and frame.names != frames[0].names:
            raise Error("Reference vertical schema mismatch")
        if how == "horizontal" and len(frame.rows) != len(frames[0].rows):
            raise Error("Reference horizontal height mismatch")
        for name in frame.names:
            if how == "horizontal" and name in names:
                raise Error("Reference horizontal name collision")
            if name not in names:
                names.append(name.copy())
    var rows = List[ReferenceRow]()
    if how == "horizontal":
        for index in range(len(frames[0].rows)):
            var row = ReferenceRow(0, 0, "", False, False, False, False, False)
            for frame in frames:
                for name in frame.names:
                    _assign_field(row, frame.rows[index], name)
            rows.append(row^)
    else:
        for frame in frames:
            for source in frame.rows:
                var row = ReferenceRow(
                    0, 0, "", False, False, False, False, False
                )
                for name in frame.names:
                    _assign_field(row, source, name)
                rows.append(row^)
    return ReferenceTable(names^, rows^)


@fieldwise_init
struct PivotModel(Copyable, Movable):
    var representatives: List[Int]
    var labels: List[String]
    # Row-major lists of source rows in each cell. Empty means a missing cell,
    # which is distinct from an existing cell whose values are all null.
    var cells: List[List[Int]]


def pivot_model(
    model: ReferenceTable, index: List[String], sort_columns: Bool
) raises -> PivotModel:
    var rows = reference_groups(model, index)
    var columns = reference_groups(model, ["s"])
    var labels = List[String]()
    var order = List[Int]()
    for c in range(len(columns)):
        ref row = model.rows[columns[c][0]]
        var label = row.s.copy() if row.sv else String("null")
        if label in index or label in labels:
            raise Error("Reference pivot name collision")
        labels.append(label^)
        order.append(c)
    if sort_columns:
        # Scalar string ordering, null after every valid key.
        for c in range(1, len(order)):
            var value = order[c]
            var at = c
            ref next = model.rows[columns[value][0]]
            while at > 0:
                ref previous = model.rows[columns[order[at - 1]][0]]
                if not next.sv or (previous.sv and next.s >= previous.s):
                    break
                order[at] = order[at - 1]
                at -= 1
            order[at] = value
    var sorted_labels = List[String]()
    for c in order:
        sorted_labels.append(labels[c].copy())
    var representatives = List[Int]()
    var cells = List[List[Int]]()
    for group in rows:
        representatives.append(group[0])
        for c in order:
            var matches = List[Int]()
            for source in group:
                if equal_keys(
                    model.rows[source], model.rows[columns[c][0]], ["s"], True
                ):
                    matches.append(source)
            cells.append(matches^)
    return PivotModel(representatives^, sorted_labels^, cells^)
