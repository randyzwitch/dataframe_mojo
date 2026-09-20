"""Row indices per group, for callers that want the rows rather than an
aggregate.

`group_by(...).agg(...)` reduces each group to one row. A caller that wants
the rows themselves -- one chart panel per region, one series per category --
needs the grouping, not the reduction. `DataFrame.group_indices` gives the
grouping and stops there: what a caller builds from it, whether sub-frames
through `take` or a direct read of the shared buffers, is its own business.

Groups are numbered in first-occurrence order, so group 0 contains the first
row. Null keys form their own group, matching `group_by`.
"""


struct GroupIndices(Movable):
    """Which rows belong to which group, plus a representative row per group.

    `ids()[i]` is row i's group. `representative(g)` is the first row of
    group g, which is where to read that group's key values from.
    """

    var _ids: List[Int]
    var _representatives: List[Int]

    def __init__(out self, var ids: List[Int], var representatives: List[Int]):
        self._ids = ids^
        self._representatives = representatives^

    def count(self) -> Int:
        """The number of groups."""
        return len(self._representatives)

    def height(self) -> Int:
        """The number of rows these groups cover."""
        return len(self._ids)

    def ids(self) -> List[Int]:
        """Each row's group id, in row order."""
        return self._ids.copy()

    def group_of(self, row: Int) raises -> Int:
        """Row's group id."""
        if row < 0 or row >= len(self._ids):
            raise Error("Row index out of bounds")
        return self._ids[row]

    def representative(self, group: Int) raises -> Int:
        """The first row of this group: read the group's key values there."""
        if group < 0 or group >= len(self._representatives):
            raise Error("Group index out of bounds")
        return self._representatives[group]

    def representatives(self) -> List[Int]:
        """The first row of every group, in group order."""
        return self._representatives.copy()

    def rows(self, group: Int) raises -> List[Int]:
        """One group's rows, in row order.

        This scans every row, so use `all_rows` when you want every group.
        """
        if group < 0 or group >= len(self._representatives):
            raise Error("Group index out of bounds")
        var out = List[Int]()
        for i in range(len(self._ids)):
            if self._ids[i] == group:
                out.append(i)
        return out^

    def all_rows(self) -> List[List[Int]]:
        """Every group's rows, in group order then row order, in one pass."""
        var out = List[List[Int]](
            length=len(self._representatives), fill=List[Int]()
        )
        for i in range(len(self._ids)):
            var id = self._ids[i]
            if id >= 0:
                out[id].append(i)
        return out^

    def sizes(self) -> List[Int]:
        """Each group's row count, in group order."""
        var out = List[Int](length=len(self._representatives), fill=0)
        for id in self._ids:
            if id >= 0:
                out[id] += 1
        return out^
