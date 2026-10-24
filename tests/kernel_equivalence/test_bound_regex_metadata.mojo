"""Cloning expression occurrences retains patterns at their new node indices."""
# contract-group: expr.regex
from std.memory import Pointer
from std.testing import assert_equal, assert_true
from dataframe.binding import bind
from dataframe.column import Column
from dataframe.execution import evaluate
from dataframe.expr import AND, COL, LIT_INT, STR_CONTAINS, Expr, Node, _node
from dataframe.series import Series
from dataframe_metal.accelerator.ordered_rows import occurrences


def test_cloned_bound_regex_patterns_follow_occurrences() raises:
    var columns: List[Series] = [
        Series("s", Column[String](["alpha", "beta", "a"]))
    ]
    # The unused literal disappears and the shared regex node is visited
    # twice: original node 2 becomes cloned nodes 1 and 3.
    var nodes: List[Node] = [
        _node(LIT_INT, integer=7),
        _node(COL, text="s"),
        _node(STR_CONTAINS, left=1, text="^a", integer=1),
        _node(AND, left=2, right=2),
    ]
    var bound = bind(Expr(nodes^, "both"), columns)
    var cached = bound._regex(2)
    var original = cached.take()
    var cloned = occurrences(bound, 128)
    assert_equal(len(cloned[0].expr._nodes), 5)
    assert_true(Bool(cloned[0].regexes))
    assert_equal(len(cloned[0].regexes.value()[]), 2)
    for index in [1, 3]:
        cached = cloned[0]._regex(index)
        var regex = cached.take()
        assert_equal(
            Int(Pointer(to=regex._compiled[])),
            Int(Pointer(to=original._compiled[])),
        )
    assert_true(not cloned[0]._regex(2))
    var result = evaluate(cloned[0], columns, 3, batch_size=1).bool()
    assert_true(result._get(0))
    assert_true(not result._get(1))
    assert_true(result._get(2))
