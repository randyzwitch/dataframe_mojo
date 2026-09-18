"""An eager CPU dataframe with runtime schema and positional row semantics."""
from std.collections import Dict
from .column import Column
from .series import Series
from .kernels import checked_add


@fieldwise_init
struct Field(Copyable):
    var name: String
    var dtype: String


struct DataFrame(Copyable):
    """Own equal-length, uniquely named columns; transformations copy storage."""

    var _columns: List[Series]
    var _height: Int

    def __init__(
        out self, var columns: List[Series], *, height: Int = -1
    ) raises:
        if height < -1:
            raise Error("Height must be nonnegative or inferred")
        var inferred = 0
        if len(columns) > 0:
            inferred = len(columns[0])
        elif height >= 0:
            inferred = height
        if height >= 0 and inferred != height:
            raise Error("Explicit height does not match columns")
        var names = Dict[String, Bool]()
        for column in columns:
            if len(column) != inferred:
                raise Error("DataFrame columns must have equal lengths")
            var name = column.name()
            if name in names:
                raise Error("Duplicate column name: " + name)
            names[name] = True
        self._columns = columns^
        self._height = inferred

    def height(self) -> Int:
        return self._height

    def width(self) -> Int:
        return len(self._columns)

    def schema(self) -> List[Field]:
        var fields = List[Field](capacity=self.width())
        for column in self._columns:
            fields.append(Field(column.name(), column.dtype()))
        return fields^

    def _index(self, name: String) raises -> Int:
        for i in range(self.width()):
            if self._columns[i].name() == name:
                return i
        raise Error("Unknown column: " + name)

    def column(self, name: String) raises -> Series:
        """Return an owned copy. Column lookup is linear in the schema width."""
        return self._columns[self._index(name)].copy()

    def select(self, names: List[String]) raises -> Self:
        var columns = List[Series](capacity=len(names))
        for name in names:
            columns.append(self.column(name))
        return Self(columns^, height=self._height)

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            if i < 0 or i >= self._height:
                raise Error("Row index out of bounds")
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.take(indices))
        return Self(columns^, height=len(indices))

    def filter(self, mask: Column[Bool]) raises -> Self:
        """Keep true rows, dropping false and null mask entries, in input order."""
        if len(mask) != self._height:
            raise Error("Filter mask must match dataframe height")
        var indices = List[Int]()
        for i in range(self._height):
            if not mask.is_null(i) and mask.value(i):
                indices.append(i)
        return self.take(indices)

    def with_column(self, var column: Series) raises -> Self:
        """Replace by name or append; the input dataframe is unchanged."""
        if len(column) != self._height:
            raise Error("New column must match dataframe height")
        var columns = self._columns.copy()
        for i in range(len(columns)):
            if columns[i].name() == column.name():
                columns[i] = column^
                return Self(columns^, height=self._height)
        columns.append(column^)
        return Self(columns^, height=self._height)

    def group_by_sum(
        self, key: String, value: String, output: String = "sum"
    ) raises -> Self:
        """Sum one numeric column by one string key, in first-seen order.

        Null keys form a group. Empty/all-null groups produce null sums.
        Int64 sums are checked, Float64 sums accumulate in input order.
        """
        if output == key:
            raise Error(
                "Aggregate output name must differ from the grouping key"
            )
        var key_index = self._index(key)
        var value_index = self._index(value)
        var keys = self._columns[key_index].string()
        var dtype = self._columns[value_index].dtype()
        if dtype != "int64" and dtype != "float64":
            raise Error("Grouped sum requires an int64 or float64 value column")
        var lookup = Dict[String, Int]()
        var representatives = List[Int]()
        var groups = List[Int](capacity=self._height)
        var null_group = -1
        for i in range(self._height):
            var group: Int
            if keys.is_null(i):
                if null_group == -1:
                    null_group = len(representatives)
                    representatives.append(i)
                group = null_group
            else:
                var label = keys.value(i)
                if label not in lookup:
                    lookup[label] = len(representatives)
                    representatives.append(i)
                group = lookup[label]
            groups.append(group)
        var result = List[Series]()
        result.append(Series(key, keys.take(representatives)))
        var valid = List[Bool](length=len(representatives), fill=False)
        if dtype == "int64":
            var values = self._columns[value_index].int64()
            var totals = List[Int64](length=len(representatives), fill=0)
            for i in range(self._height):
                if not values.is_null(i):
                    var g = groups[i]
                    totals[g] = checked_add(totals[g], values.value(i))
                    valid[g] = True
            result.append(Series(output, Column[Int64](totals^, valid)))
        else:
            var values = self._columns[value_index].float64()
            var totals = List[Float64](length=len(representatives), fill=0)
            for i in range(self._height):
                if not values.is_null(i):
                    var g = groups[i]
                    totals[g] += values.value(i)
                    valid[g] = True
            result.append(Series(output, Column[Float64](totals^, valid)))
        return Self(result^)

    def sort(
        self, by: String, descending: Bool = False, nulls_last: Bool = True
    ) raises -> Self:
        """Stable single-column sort. NaNs follow non-null numbers."""
        return self.take(
            self._columns[self._index(by)].argsort(descending, nulls_last)
        )

    def join(
        self,
        right: Self,
        on: String,
        how: String = "inner",
        suffix: String = "_right",
    ) raises -> Self:
        """Hash join on one shared string column; 'inner' and 'left' supported.

        Null keys never match. Emit left rows in input order and each row's
        right matches in right input order. Duplicate keys produce all pairs.
        Overlapping right names gain suffix; remaining collisions raise.
        """
        if how != "inner" and how != "left":
            raise Error("Join how must be 'inner' or 'left'")
        var left_key_index = self._index(on)
        var right_key_index = right._index(on)
        var left_keys = self._columns[left_key_index].string()
        var right_keys = right._columns[right_key_index].string()
        var names = Dict[String, Bool]()
        for column in self._columns:
            names[column.name()] = True
        var left_names = names.copy()
        var right_names = List[String]()
        for i in range(right.width()):
            if i == right_key_index:
                continue
            var name = right._columns[i].name()
            if name in left_names:
                name += suffix
            # Validate against both original left and preceding output names.
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_names.append(name)
        var lookup = Dict[String, List[Int]]()
        for i in range(right.height()):
            if not right_keys.is_null(i):
                var key = right_keys.value(i)
                if key not in lookup:
                    lookup[key] = List[Int]()
                lookup[key].append(i)
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for i in range(self.height()):
            var matched = False
            if not left_keys.is_null(i):
                var key = left_keys.value(i)
                if key in lookup:
                    for j in lookup[key]:
                        left_rows.append(i)
                        right_rows.append(j)
                    matched = True
            if not matched and how == "left":
                left_rows.append(i)
                right_rows.append(-1)
        var columns = List[Series]()
        for column in self._columns:
            columns.append(column.take(left_rows))
        var name_index = 0
        for i in range(right.width()):
            if i != right_key_index:
                columns.append(
                    right._columns[i]
                    .take_or_null(right_rows)
                    .renamed(right_names[name_index])
                )
                name_index += 1
        return Self(columns^, height=len(left_rows))
