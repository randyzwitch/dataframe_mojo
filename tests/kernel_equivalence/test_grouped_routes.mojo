"""Grouped native intervals and mixed indexed/gathered preparation have scalar oracles."""
# contract-group: kernel.reduce
from std.math import isnan
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.binding import bind
from dataframe.execution import _ReduceJob, _direct_grouped_sum
from dataframe.frame import _bind_all


def native_intervals[D: DType]() raises:
    for n in [0, 1, 9, 65, 2057]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var values = List[Scalar[D]]()
                var valid = List[Bool]()
                for physical in range(n + offset):
                    var present = mode == 0 or mode == 1 and physical % 5 != 2
                    comptime if D.is_floating_point():
                        values.append(
                            Scalar[D](physical % 17) / 8
                            - 1 if present else Scalar[D].MAX
                        )
                    else:
                        values.append(
                            Scalar[D].MAX if not present
                            or physical % 3
                            == 0 else Scalar[D].MIN if physical % 3
                            == 1 else Scalar[D](physical % 100)
                        )
                    valid.append(present)
                var flat = Series("v", Column[Scalar[D]](values^, valid)).slice(
                    offset, n
                )
                for chunked in [False, True]:
                    var source = flat.copy()
                    if chunked and n > 1:
                        var cut = min(9, n - 1)
                        source = Series._from_chunks(
                            [flat.slice(0, cut), flat.slice(cut, n - cut)]
                        )
                    for mean in [False, True]:
                        var expression = (
                            col("v").mean() if mean else col("v").sum()
                        )
                        var columns: List[Series] = [source.copy()]
                        var bound = bind(expression, columns)
                        var node = expression._nodes[
                            len(expression._nodes) - 1
                        ].copy()
                        var ids = List[Int]()
                        for row in range(n):
                            ids.append((row * 7 + row // 5) % 4)
                        for interval in range(3):
                            var start = 0 if interval == 0 else min(3, n)
                            var end = (
                                n if interval
                                == 0 else max(start, n - 4) if interval
                                == 1 else start
                            )
                            var job = _ReduceJob[8](
                                bound,
                                columns,
                                List[Series](),
                                node,
                                start,
                                end,
                                17,
                                True,
                                ArcPointer(ids.copy()),
                                5,
                            )
                            assert_true(
                                _direct_grouped_sum(
                                    job.reducer,
                                    bound,
                                    columns,
                                    node,
                                    start,
                                    end,
                                    True,
                                    ids,
                                )
                            )
                            var counts = List[Int64](length=5, fill=0)
                            var integers = List[Int128](length=5, fill=0)
                            var floats = List[Float64](length=5, fill=0)
                            for row in range(start, end):
                                var physical = row + offset
                                if not (
                                    mode == 0 or mode == 1 and physical % 5 != 2
                                ):
                                    continue
                                var group = ids[row]
                                counts[group] += 1
                                comptime if D.is_floating_point():
                                    floats[group] += (
                                        Float64(physical % 17) / 8 - 1
                                    )
                                else:
                                    integers[group] += (
                                        Int128(Scalar[D].MAX) if physical % 3
                                        == 0 else Int128(
                                            Scalar[D].MIN
                                        ) if physical
                                        % 3
                                        == 1 else Int128(physical % 100)
                                    )
                            for group in range(5):
                                comptime if D.is_floating_point():
                                    assert_equal(
                                        job.reducer.float_sums[group].total,
                                        floats[group],
                                    )
                                    assert_equal(
                                        job.reducer.float_sums[group].count,
                                        counts[group],
                                    )
                                else:
                                    assert_equal(
                                        job.reducer.int_sums[group].total,
                                        integers[group],
                                    )
                                    assert_equal(
                                        job.reducer.int_sums[group].count,
                                        counts[group],
                                    )


def test_grouped_native_sum_mean_all_ten_widths_chunk_intersections_empty_groups_and_wide_totals_match_scalar_states() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        native_intervals[NUMERIC_DTYPES[k]]()


def test_grouped_native_eligibility_rejects_computed_logical_and_ungrouped_inputs_without_changing_state() raises:
    var sources: List[Series] = [
        Series("v", Column[Int64]([1, 2, 3])),
        Series("v", Column[Int32]([1, 2, 3])).with_dtype(
            DataType.decimal(8, 2, 32)
        ),
        Series("v", Column[Int64]([1, 2, 3])).with_dtype(DataType.DATE),
    ]
    var ids: List[Int] = [0, 1, 0]
    for source in sources:
        var expressions = List[Expr]()
        expressions.append(col("v").max())
        if source.dtype() == DataType.INT64:
            expressions.append((col("v") + 1).sum())
            expressions.append(col("v").sum())
        elif source.dtype().is_decimal():
            expressions.append(col("v").sum())
        for expression in expressions:
            var columns: List[Series] = [source.copy()]
            var bound = bind(expression, columns)
            var node = expression._nodes[len(expression._nodes) - 1].copy()
            var grouped = not (
                source.dtype() == DataType.INT64
                and len(expression._nodes) == 2
                and expression._nodes[1].op == col("v").sum()._nodes[1].op
            )
            var job = _ReduceJob[8](
                bound,
                columns,
                List[Series](),
                node,
                0,
                3,
                17,
                grouped,
                ArcPointer(ids.copy()),
                2,
            )
            var counts_before = job.reducer.counts.copy()
            var seen_before = job.reducer.seen.copy()
            assert_true(
                not _direct_grouped_sum(
                    job.reducer, bound, columns, node, 0, 3, grouped, ids
                )
            )
            assert_equal(job.reducer.counts, counts_before)
            assert_equal(job.reducer.seen, seen_before)
            for state in job.reducer.int_sums:
                assert_equal(state.total, Int128(0))
                assert_equal(state.count, Int64(0))
            for state in job.reducer.float_sums:
                assert_equal(state.total, Float64(0))
                assert_equal(state.count, Int64(0))


def test_partitioned_indexed_and_gathered_reducers_keep_distinct_source_owners_and_complete_literal_groups() raises:
    var n = 32017
    var groups = 2003
    var ks = List[Int64]()
    var xs = List[Int64]()
    var xv = List[Bool]()
    var ys = List[Float64]()
    var texts = List[String]()
    var members = List[List[Int]](length=groups, fill=List[Int]())
    var words: List[String] = ["z", "a", "external-thirteen\x00é", "m"]
    for row in range(n):
        var group = (row * 37) % groups
        ks.append(Int64(group))
        xs.append(Int64(row % 31 - 15) if row % 11 != 2 else Int64.MAX)
        xv.append(row % 11 != 2)
        ys.append(
            Float64(group % 5) if row // groups > 1 else Float64(100 + row % 7)
        )
        texts.append(words[row % 4].copy())
        members[group].append(row)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](ks^)),
            Series("x", Column[Int64](xs^, xv)),
            Series("y", Column[Float64](ys^)),
            Series("s", Column[String](texts^)).cast("categorical"),
        ]
    )
    var x = frame.column("x")
    frame = frame.with_column(
        Series._from_chunks([x.slice(0, 61), x.slice(61, n - 61)])
    )
    var y = frame.column("y")
    frame = frame.with_column(
        Series._from_chunks([y.slice(0, 127), y.slice(127, n - 127)])
    )
    var expressions: List[Expr] = [
        col("x").sum().alias("sum"),
        (col("x") + 1).mean().alias("computed"),
        col("x").first().alias("first"),
        col("y").median().alias("median"),
        col("y").mode().alias("mode"),
        col("y").n_unique().alias("unique"),
        col("s").min().alias("min"),
        col("s").max().alias("max"),
        col("s").first().alias("first_s"),
        col("s").n_unique().alias("unique_s"),
    ]
    var grouped = frame.group_by("k", maintain_order=True)
    var bound = _bind_all(expressions, frame._columns)
    var preparation = grouped._partitioned_references(bound)
    assert_equal(
        preparation[0],
        [True, True, True, True, False, True, False, False, False, False],
    )
    assert_equal(len(preparation[1]), 2)
    assert_equal(len(preparation[2]), 2)
    assert_equal(preparation[1][0].name(), "x")
    assert_equal(preparation[1][1].name(), "y")
    assert_equal(preparation[2][0].name(), "y")
    assert_equal(preparation[2][1].name(), "s")
    assert_true(
        Int(preparation[1][0]._chunked.value().ptr())
        == Int(frame.column("x")._chunked.value().ptr())
    )
    var actual = grouped._agg_partitioned(
        expressions, bound, 127, 4, whole=False
    )
    assert_equal(actual.height(), groups)
    var firsts = List[Int]()
    for group in range(groups):
        firsts.append(members[group][0])
    sort(firsts)
    for at in range(groups):
        var group = (firsts[at] * 37) % groups
        assert_equal(actual.item(at, "k").int64(), Int64(group))
        var sum = Int64(0)
        var count = 0
        var values = List[Float64]()
        var distinct = List[Float64]()
        var distinct_text = List[String]()
        var lo = words[members[group][0] % 4].copy()
        var hi = lo.copy()
        for row in members[group]:
            if row % 11 != 2:
                sum += Int64(row % 31 - 15)
                count += 1
            var value = Float64(group % 5) if row // groups > 1 else Float64(
                100 + row % 7
            )
            values.append(value)
            if value not in distinct:
                distinct.append(value)
            var text = words[row % 4].copy()
            lo = min(lo, text)
            hi = max(hi, text)
            if text not in distinct_text:
                distinct_text.append(text.copy())
        sort(values)
        var median = (
            values[len(values) // 2] if len(values)
            % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2])
            / 2
        )
        assert_equal(actual.item(at, "sum").int64(), sum)
        assert_equal(
            actual.item(at, "computed").float64(),
            Float64(sum + Int64(count)) / Float64(count),
        )
        var first = members[group][0]
        assert_equal(actual.item(at, "first").is_null(), first % 11 == 2)
        if first % 11 != 2:
            assert_equal(
                actual.item(at, "first").int64(), Int64(first % 31 - 15)
            )
        assert_equal(actual.item(at, "median").float64(), median)
        var modes = actual.item(at, "mode").list()
        assert_equal(len(modes), 1)
        assert_equal(modes.get(0).float64(), Float64(group % 5))
        assert_equal(actual.item(at, "unique").int64(), Int64(len(distinct)))
        for name in ["min", "max", "first_s"]:
            assert_true(actual.column(name).dtype() == DataType.STRING)
        assert_equal(actual.item(at, "min").string(), lo)
        assert_equal(actual.item(at, "max").string(), hi)
        assert_equal(actual.item(at, "first_s").string(), words[first % 4])
        assert_equal(
            actual.item(at, "unique_s").int64(), Int64(len(distinct_text))
        )
