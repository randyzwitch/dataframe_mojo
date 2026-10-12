"""Small scalar reference semantics. This module must never import dataframe.

Rows carry typed payloads and independent validity for each cell. The initial
schema vocabulary is i:Int64, f:Float64, s:String, b:Bool. Extended dtypes are
explicit open obligations in the contract catalog, not silently approximated.
"""


@fieldwise_init
struct ReferenceRow(Copyable, Movable):
    var i: Int64
    var f: Float64
    var s: String
    var b: Bool
    var iv: Bool
    var fv: Bool
    var sv: Bool
    var bv: Bool


@fieldwise_init
struct ReferenceTable(Copyable, Movable):
    var names: List[String]
    var rows: List[ReferenceRow]

    def project(self, names: List[String]) raises -> Self:
        for index in range(len(names)):
            var found = False
            for existing in self.names:
                found = found or existing == names[index]
            if not found:
                raise Error("Unknown reference column")
            for earlier in range(index):
                if names[earlier] == names[index]:
                    raise Error("Duplicate reference column")
        return Self(names.copy(), self.rows.copy())

    def take(self, indices: List[Int]) raises -> Self:
        var rows = List[ReferenceRow]()
        for index in indices:
            if index < 0 or index >= len(self.rows):
                raise Error("Reference index out of bounds")
            rows.append(self.rows[index].copy())
        return Self(self.names.copy(), rows^)

    def slice(self, offset: Int, length: Int = -1) raises -> Self:
        if length < -1:
            raise Error("Invalid reference slice length")
        var start = offset if offset >= 0 else len(self.rows) + offset
        start = min(max(start, 0), len(self.rows))
        var count = len(self.rows) - start
        if length >= 0:
            count = min(count, length)
        var indices = List[Int]()
        for index in range(start, start + count):
            indices.append(index)
        return self.take(indices)

    def filter_i_gt(self, minimum: Int64) -> Self:
        var rows = List[ReferenceRow]()
        for row in self.rows:
            if row.iv and row.i > minimum:
                rows.append(row.copy())
        return Self(self.names.copy(), rows^)

    def double_i(self) raises -> Self:
        var rows = self.rows.copy()
        for index in range(len(rows)):
            if rows[index].iv:
                var doubled = Int128(rows[index].i) * 2
                if doubled < -(Int128(1) << 63) or doubled >= (Int128(1) << 63):
                    raise Error("Reference Int64 arithmetic overflow")
                rows[index].i = Int64(doubled)
        return Self(self.names.copy(), rows^)

    def sort_i(self) -> Self:
        # Stable insertion sort, ascending with nulls last. Deliberately simple;
        # no production sorting, encoding, gathering or comparison helpers.
        var rows = self.rows.copy()
        for index in range(1, len(rows)):
            var value = rows[index].copy()
            var position = index
            while position > 0:
                ref previous = rows[position - 1]
                var precedes = value.iv and (
                    not previous.iv or value.i < previous.i
                )
                if not precedes:
                    break
                rows[position] = previous.copy()
                position -= 1
            rows[position] = value^
        return Self(self.names.copy(), rows^)

    def int_sum(self) -> Int128:
        var total = Int128(0)
        for row in self.rows:
            if row.iv:
                total += Int128(row.i)
        return total

    def int_count(self) -> Int64:
        var count = Int64(0)
        for row in self.rows:
            if row.iv:
                count += 1
        return count


def generated_model(
    n: Int, offset: Int = 0, null_mode: Int = 0
) -> ReferenceTable:
    """Logical data only; offset chooses row identities, never a storage layout."""
    var rows = List[ReferenceRow]()
    for index in range(offset, offset + n):
        rows.append(
            ReferenceRow(
                Int64((index * 17 + 3) % 31 - 15),
                Float64((index * 7) % 19 - 9) / 4,
                [
                    "",
                    "é",
                    'a,"b"',
                    "line\nbreak",
                    "a long string beyond inline storage",
                ][index % 5],
                index % 3 == 0,
                null_mode == 0 or (null_mode == 1 and index % 7 != 0),
                null_mode == 0 or (null_mode == 1 and index % 5 != 1),
                null_mode == 0 or (null_mode == 1 and index % 3 != 2),
                null_mode == 0 or (null_mode == 1 and index % 11 != 0),
            )
        )
    return ReferenceTable(["i", "f", "s", "b"], rows^)


def field_valid(row: ReferenceRow, name: String) raises -> Bool:
    if name == "i":
        return row.iv
    if name == "f":
        return row.fv
    if name == "s":
        return row.sv
    if name == "b":
        return row.bv
    raise Error("Unknown reference field")


def compare_field(
    left: ReferenceRow, right: ReferenceRow, name: String
) raises -> Int:
    """Compare valid values; NaN follows numbers, both zero signs tie."""
    if name == "i":
        return -1 if left.i < right.i else 1 if left.i > right.i else 0
    if name == "f":
        var ln = left.f != left.f
        var rn = right.f != right.f
        if ln or rn:
            return 0 if ln == rn else 1 if ln else -1
        return -1 if left.f < right.f else 1 if left.f > right.f else 0
    if name == "s":
        return -1 if left.s < right.s else 1 if left.s > right.s else 0
    if name == "b":
        return (
            -1 if not left.b and right.b else 1 if left.b and not right.b else 0
        )
    raise Error("Unknown reference field")


def precedes(
    left: ReferenceRow,
    right: ReferenceRow,
    keys: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> Bool:
    for k in range(len(keys)):
        var lv = field_valid(left, keys[k])
        var rv = field_valid(right, keys[k])
        if lv != rv:
            return lv if nulls_last[k] else rv
        if not lv:
            continue
        # NaN remains after numbers in either direction, as specified.
        if keys[k] == "f":
            var ln = left.f != left.f
            var rn = right.f != right.f
            if ln != rn:
                return not ln
        var order = compare_field(left, right, keys[k])
        if order:
            return order > 0 if descending[k] else order < 0
    return False


def sorted_model(
    model: ReferenceTable,
    keys: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> ReferenceTable:
    if not keys or len(keys) != len(descending) or len(keys) != len(nulls_last):
        raise Error("Invalid reference sort options")
    _ = model.project(keys)
    var rows = model.rows.copy()
    for index in range(1, len(rows)):
        var value = rows[index].copy()
        var position = index
        while position > 0 and precedes(
            value, rows[position - 1], keys, descending, nulls_last
        ):
            rows[position] = rows[position - 1].copy()
            position -= 1
        rows[position] = value^
    return ReferenceTable(model.names.copy(), rows^)


def equal_keys(
    left: ReferenceRow,
    right: ReferenceRow,
    keys: List[String],
    null_equal: Bool,
) raises -> Bool:
    for key in keys:
        var lv = field_valid(left, key)
        var rv = field_valid(right, key)
        if not lv or not rv:
            if not null_equal or lv != rv:
                return False
        elif compare_field(left, right, key) != 0:
            return False
    return True


def reference_groups(
    model: ReferenceTable, keys: List[String]
) raises -> List[List[Int]]:
    """First-occurrence groups using scalar comparisons, without hash encoding."""
    _ = model.project(keys)
    var groups = List[List[Int]]()
    for index in range(len(model.rows)):
        var found = False
        for g in range(len(groups)):
            if equal_keys(
                model.rows[index], model.rows[groups[g][0]], keys, True
            ):
                groups[g].append(index)
                found = True
                break
        if not found:
            groups.append([index])
    return groups^


@fieldwise_init
struct RowPair(Copyable, Movable):
    var left: Int
    var right: Int


def reference_join(
    left: ReferenceTable, right: ReferenceTable, keys: List[String], how: String
) raises -> List[RowPair]:
    """Nested-loop join. -1 denotes the missing side, never a source row."""
    if how != "cross":
        if not keys:
            raise Error("Reference join needs keys")
        _ = left.project(keys)
        _ = right.project(keys)
    var pairs = List[RowPair]()
    if how == "right":
        for r in range(len(right.rows)):
            var matched = False
            for l in range(len(left.rows)):
                if equal_keys(left.rows[l], right.rows[r], keys, False):
                    pairs.append(RowPair(l, r))
                    matched = True
            if not matched:
                pairs.append(RowPair(-1, r))
        return pairs^
    if (
        how != "inner"
        and how != "left"
        and how != "full"
        and how != "semi"
        and how != "anti"
        and how != "cross"
    ):
        raise Error("Unknown reference join mode")
    var seen = List[Bool](length=len(right.rows), fill=False)
    for l in range(len(left.rows)):
        var matched = False
        for r in range(len(right.rows)):
            if how == "cross" or equal_keys(
                left.rows[l], right.rows[r], keys, False
            ):
                matched = True
                seen[r] = True
                if how != "semi" and how != "anti":
                    pairs.append(RowPair(l, r))
        if (
            (how == "semi" and matched)
            or (how == "anti" and not matched)
            or ((how == "left" or how == "full") and not matched)
        ):
            pairs.append(RowPair(l, -1))
    if how == "full":
        for r in range(len(right.rows)):
            if not seen[r]:
                pairs.append(RowPair(-1, r))
    return pairs^
