"""Row transforms preserve schema, order and clipped height, including width zero."""
from std.testing import assert_equal, assert_raises
from dataframe import DataFrame, Series
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model
from tests.support.reference_model import generated_model


def test_row_selection_clipping_order_and_errors() raises:
    for n in [0, 1, 9, 65]:
        for validity in range(3):
            var model = generated_model(n, 3, validity)
            for width_zero in range(2):
                var expected = (
                    model.copy() if width_zero == 0 else model.project([])
                )
                for layout in range(3):
                    var frame = frame_from_model(
                        expected, 7 if layout == 1 else 0
                    )
                    if layout == 2:
                        frame = chunk_layout(frame)
                    var reverse = List[Int]()
                    for index in range(n):
                        reverse.append(n - index - 1)
                    assert_model(frame.reverse(), expected.take(reverse))
                    assert_model(frame.clear(), expected.slice(0, 0))
                    assert_model(frame.head(), expected.slice(0, 5))
                    assert_model(frame.tail(), expected.slice(max(n - 5, 0)))
                    for count in [-100, -1, 0, 1, 5, 100]:
                        var head_count = min(count, n) if count >= 0 else max(
                            n + count, 0
                        )
                        var tail_start = max(
                            n - count, 0
                        ) if count >= 0 else min(-count, n)
                        assert_model(
                            frame.head(count), expected.slice(0, head_count)
                        )
                        assert_model(
                            frame.limit(count), expected.slice(0, head_count)
                        )
                        assert_model(
                            frame.tail(count), expected.slice(tail_start)
                        )
                    for index in [-1, n]:
                        with assert_raises():
                            _ = frame.take([index])
                    with assert_raises():
                        _ = frame.slice(0, -2)
                    if n:
                        assert_model(
                            frame.take([n - 1, 0, n - 1]),
                            expected.take([n - 1, 0, n - 1]),
                        )
                    assert_model(frame.take(List[Int]()), expected.take([]))
