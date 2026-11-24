"""As-of joins run as rounds of tasks of the query's schedule rather than as
one task that runs whole-frame kernels.

An as-of join matches each left row to one right row of its `by` group:
the last right key at or below the left key (backward), the first at or
above it (forward), or the nearest. The right side is the build side: the
serial step numbers the `by` keys of both sides and lists each group's
right rows in key order, leaving out null keys. Rows already in key order
keep their order, so nothing is sorted for sorted input.

Backward and forward matches are then a binary search over the group's
keys, so each left row is matched on its own: a round splits the left
rows into ranges, a task a range, and the left side needs no order at
all. A nearest match also depends on where the null keys sit among the
right rows, and is found by one ordered scan of both sides per group, in
a round of one task.

A last round gathers the matched rows of each right column, a column a
task. The left columns are kept as they are.
"""
from std.collections import Dict, Optional

from dataframe.aggregation.keys import _expand_struct_keys
from dataframe.asof import asof_buckets, asof_key_checks, asof_probe, asof_rows
from dataframe.categorical import unify
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.frame import DataFrame
from dataframe.kernels.hashing import encode_rows
from dataframe.parallel import (
    Rounds,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.planning.model import PlanNode
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.value import AnyValue

# The round that ran last.
comptime _START = 0
comptime _MATCH = 1
comptime _GATHER = 2
comptime _DONE = 3


def asof_outputs(
    left: DataFrame,
    right: DataFrame,
    left_key: String,
    right_key: String,
    by: List[String],
    strategy: String,
    tolerance: Optional[AnyValue],
    suffix: String,
) raises -> List[Tuple[Int, String]]:
    """Check an as-of join of `left` and `right`, raising what the request
    gets wrong, and return the right columns it outputs (their positions
    and output names). Right keys with a different name are kept; common
    keys and `by` columns appear once, and other names the left side has
    take `suffix`."""
    var lhs = left.column(left_key)
    var rhs = right.column(right_key)
    var seen = Dict[String, Bool]()
    for name in by:
        if name in seen or name == left_key or name == right_key:
            raise Error("join_asof duplicate or ordered key in by: " + name)
        seen[name] = True
        var l = left.column(name)
        var r = right.column(name)
        if l.dtype().is_categorical() or r.dtype().is_categorical():
            unify(l, r)
        if l.dtype() != r.dtype():
            raise Error("join_asof by dtypes must match: " + name)
    _ = asof_key_checks(lhs, rhs, strategy, tolerance)
    var names = Dict[String, Bool]()
    for name in left.columns():
        names[name] = True
    var left_names = names.copy()
    var outputs = List[Tuple[Int, String]]()
    for i in range(right.width()):
        var name = right._columns[i].name()
        if name in seen or (name == right_key and left_key == right_key):
            continue
        if name in left_names:
            name += suffix
        if name in names:
            raise Error("join_asof output name collision: " + name)
        names[name] = True
        outputs.append((i, name))
    return outputs^


def _asof_by(node: PlanNode) -> List[String]:
    var by = List[String](capacity=node.asof_group_count())
    for i in range(node.asof_group_count()):
        by.append(node.asof_group(i))
    return by^


struct AsofSink(Movable, Rounds):
    """ASOF node `node` over all rows of `left` and `right`, as rounds (see
    the module docstring); `take_result` once no round is left."""

    var left: DataFrame
    var right: DataFrame
    var node: PlanNode
    var stage: Int
    var outputs: List[Tuple[Int, String]]
    # The keys as one array each, each row's `by` group (none without
    # `by`), and whether a group can match (no null `by` value).
    var left_key: Series
    var right_key: Series
    var left_ids: List[Int]
    var right_ids: List[Int]
    var matchable: List[Bool]
    # Backward and forward: each group's first position in `order` (then
    # the end), the right rows a match may pick in key order, and their
    # keys; the left row ranges of the match round.
    var starts: List[Int]
    var order: List[Int]
    var keys: Series
    var ranges: List[Int]
    var tolerance: Tuple[Bool, Int128, Float64]
    # The right row each left row matched (-1: none), and the gathered
    # right columns.
    var matched: List[Int]
    var gathered: List[Series]
    var width: Int
    var result: DataFrame

    def __init__(
        out self, var left: DataFrame, var right: DataFrame, node: PlanNode
    ) raises:
        self.left = left^
        self.right = right^
        self.node = node.copy()
        self.stage = _START
        self.outputs = List[Tuple[Int, String]]()
        self.left_key = Series("", Column[Int64](List[Int64]()))
        self.right_key = Series("", Column[Int64](List[Int64]()))
        self.left_ids = List[Int]()
        self.right_ids = List[Int]()
        self.matchable = List[Bool]([True])
        self.starts = List[Int]()
        self.order = List[Int]()
        self.keys = Series("", Column[Int64](List[Int64]()))
        self.ranges = List[Int]()
        self.tolerance = (False, Int128(0), Float64(0))
        self.matched = List[Int]()
        self.gathered = List[Series]()
        self.width = 0
        self.result = DataFrame(List[Series]())

    def _strategy(self) -> String:
        return self.node.text

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            return self._build()
        if self.stage == _MATCH:
            return self._gather()
        if self.stage == _GATHER:
            self._assemble()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _MATCH:
            if self._strategy() == "nearest":
                self.matched = asof_rows(
                    self.left_key,
                    self.right_key,
                    self.left_ids,
                    self.right_ids,
                    self.matchable,
                    self._strategy(),
                    self.node.asof_tolerance,
                    self.node.asof_allow_exact_matches(),
                )
                return
            for r in range(task, len(self.ranges) - 1, self.width):
                self._probe(self.ranges[r], self.ranges[r + 1])
        elif self.stage == _GATHER:
            for c in range(task, len(self.outputs), self.width):
                var output = self.outputs[c]
                self.gathered[c] = (
                    self.right._columns[output[0]]
                    .take_or_null(self.matched)
                    .renamed(output[1])
                )

    def _build(mut self) raises -> Int:
        """The checks, the `by` groups, each group's right rows in key
        order, and the match round."""
        var left_on = self.node.join_left_keys()[0]
        var right_on = self.node.join_right_keys()[0]
        var by = _asof_by(self.node)
        self.outputs = asof_outputs(
            self.left,
            self.right,
            left_on,
            right_on,
            by,
            self._strategy(),
            self.node.asof_tolerance,
            self.node.join_suffix(),
        )
        self.tolerance = asof_key_checks(
            self.left.column(left_on),
            self.right.column(right_on),
            self._strategy(),
            self.node.asof_tolerance,
        )
        if len(by):
            var stacked = List[Series](capacity=len(by))
            for name in by:
                var l = self.left.column(name)
                var r = self.right.column(name)
                if l.dtype().is_categorical() or r.dtype().is_categorical():
                    unify(l, r)
                stacked.append(l.append(r))
            var keys = encode_rows(_expand_struct_keys(stacked), True)
            self.matchable = List[Bool](length=keys.count(), fill=True)
            for g in range(keys.count()):
                for key in stacked:
                    if key.get(keys.representatives[g]).is_null():
                        self.matchable[g] = False
            var n = self.left.height()
            self.left_ids = List[Int](capacity=n)
            for row in range(n):
                self.left_ids.append(keys.ids[row])
            self.right_ids = List[Int](capacity=self.right.height())
            for row in range(self.right.height()):
                self.right_ids.append(keys.ids[n + row])
        self.left_key = self.left.column(left_on).rechunk()
        self.right_key = self.right.column(right_on).rechunk()
        self.stage = _MATCH
        if self._strategy() == "nearest":
            self.width = 1
            return 1
        var buckets = asof_buckets(
            self.right_key, self.right_ids, self.matchable
        )
        self.starts = buckets[0].copy()
        self.order = buckets[1].copy()
        self.keys = self.right_key.take(self.order)
        var n = self.left.height()
        self.matched = List[Int](length=n, fill=-1)
        self.ranges = partitions(n, worker_count(n), 64)
        self.width = round_width(len(self.ranges) - 1)
        return self.width

    def _probe(mut self, first: Int, end: Int) raises:
        var forward = self._strategy() == "forward"
        var exact = self.node.asof_allow_exact_matches()
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if self.left_key._data.isa[Column[Scalar[D]]]():
                ref left = self.left_key._data[Column[Scalar[D]]]
                ref keys = self.keys._data[Column[Scalar[D]]]
                if forward:
                    asof_probe[D, True](
                        left,
                        keys,
                        self.left_ids,
                        self.starts,
                        self.order,
                        self.matchable,
                        exact,
                        self.tolerance,
                        self.matched,
                        first,
                        end,
                    )
                else:
                    asof_probe[D, False](
                        left,
                        keys,
                        self.left_ids,
                        self.starts,
                        self.order,
                        self.matchable,
                        exact,
                        self.tolerance,
                        self.matched,
                        first,
                        end,
                    )
                return
        raise Error("Unsupported join_asof key storage")

    def _gather(mut self) raises -> Int:
        """The round that gathers each right column's matched rows."""
        self.order = List[Int]()
        self.right_ids = List[Int]()
        self.left_ids = List[Int]()
        self.gathered = List[Series](
            length=len(self.outputs),
            fill=Series("", Column[Int64](List[Int64]())),
        )
        self.stage = _GATHER
        self.width = round_width(len(self.outputs))
        if self.width == 0:
            self._assemble()
        return self.width

    def _assemble(mut self) raises:
        var columns = self.left._columns.copy()
        for column in self.gathered:
            columns.append(column.copy())
        self.result = DataFrame(columns^, height=self.left.height())
        self.gathered = List[Series]()
        self.matched = List[Int]()
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The output, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^


def asof_join(
    var left: DataFrame, var right: DataFrame, node: PlanNode
) raises -> DataFrame:
    """ASOF node `node` over `left` and `right`, its rounds run here: for a
    node the plan's executor runs over its executed inputs, which may be
    inside a pipeline's task, where no schedule may start."""
    var sink = AsofSink(left^, right^, node)
    run_rounds(sink)
    return sink.take_result()
