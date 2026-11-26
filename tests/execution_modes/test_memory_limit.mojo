"""The memory limit: settings, the count of memory held, and the clean
`out of memory` error raised in place of being killed by the system."""
# contract-group: execution.lifecycle
from std.os import makedirs
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    Series,
    col,
    memory_limit,
    memory_usage,
    set_memory_limit,
)
from dataframe.storage.allocation import release_retained
from dataframe.storage.memory_limit import (
    UNLIMITED,
    available_memory,
    cgroup_memory_limit,
    parse_memory_limit,
)


def _frame(rows: Int, offset: Int = 0) raises -> DataFrame:
    var keys = List[Int64](capacity=rows)
    var values = List[Int64](capacity=rows)
    for i in range(rows):
        keys.append(Int64(i + offset))
        values.append(Int64(i % 7))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^)),
        ]
    )


def _is_out_of_memory(message: String) -> Bool:
    return message.startswith("out of memory")


def test_memory_limit_parses_sizes_shares_and_none() raises:
    assert_equal(parse_memory_limit("1000"), 1000)
    assert_equal(parse_memory_limit("4GB"), 4_000_000_000)
    assert_equal(parse_memory_limit(" 512 MiB "), 512 * 1024 * 1024)
    assert_equal(parse_memory_limit("1.5gib"), 3 * 512 * 1024 * 1024)
    assert_equal(parse_memory_limit("2kb"), 2000)
    assert_equal(parse_memory_limit("none"), 0)
    assert_equal(parse_memory_limit("-1"), 0)
    var half = parse_memory_limit("50%")
    assert_true(half > 0 and half <= available_memory())
    for bad in ["GB", "4 parsecs", "150%"]:
        var raised = False
        try:
            _ = parse_memory_limit(bad)
        except e:
            raised = String(e).startswith("invalid memory limit")
        assert_true(raised, bad)


def test_memory_limit_setting_round_trips() raises:
    var default = memory_limit()
    assert_true(default > 0)
    if available_memory() > 0 and default != Int(UNLIMITED):
        # The default leaves a fifth of available memory as headroom.
        assert_true(default <= available_memory())
        assert_true(default >= available_memory() // 100 * 79)
    set_memory_limit(1 << 30)
    assert_equal(memory_limit(), 1 << 30)
    set_memory_limit("2GiB")
    assert_equal(memory_limit(), 2 << 30)
    set_memory_limit("none")
    assert_equal(memory_limit(), Int(UNLIMITED))
    set_memory_limit(0)
    assert_equal(memory_limit(), Int(UNLIMITED))
    set_memory_limit(default)


def _write(path: String, text: String) raises:
    with open(path, "w") as file:
        file.write(text)


def test_cgroup_limit_takes_the_smallest_limit_up_the_tree() raises:
    var base = String("build/memory-limit-cgroup-test")
    # cgroup v2: the scope has no limit of its own, its parent does.
    makedirs(base + "/v2/user.slice/app.scope", exist_ok=True)
    _write(base + "/proc_v2", "0::/user.slice/app.scope\n")
    _write(base + "/v2/user.slice/app.scope/memory.max", "max\n")
    _write(base + "/v2/user.slice/memory.max", "3221225472\n")
    assert_equal(
        cgroup_memory_limit(base + "/proc_v2", base + "/v2"), 3221225472
    )
    # A lower limit on the group itself wins.
    _write(base + "/v2/user.slice/app.scope/memory.max", "1073741824\n")
    assert_equal(
        cgroup_memory_limit(base + "/proc_v2", base + "/v2"), 1073741824
    )
    # cgroup v1: the memory controller's line names the group.
    makedirs(base + "/v1/memory/docker/abc", exist_ok=True)
    _write(
        base + "/proc_v1",
        "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n",
    )
    _write(base + "/v1/memory/docker/abc/memory.limit_in_bytes", "536870912\n")
    assert_equal(
        cgroup_memory_limit(base + "/proc_v1", base + "/v1"), 536870912
    )
    # No group file, or no limit anywhere: none.
    _write(base + "/v2/user.slice/memory.max", "max\n")
    _write(base + "/v2/user.slice/app.scope/memory.max", "max\n")
    assert_equal(cgroup_memory_limit(base + "/proc_v2", base + "/v2"), 0)
    assert_equal(cgroup_memory_limit(base + "/missing", base + "/v2"), 0)


def test_memory_usage_counts_column_buffers_while_they_live() raises:
    release_retained()
    var before = memory_usage()
    var frame = _frame(1_000_000)
    var held = memory_usage() - before
    # Two Int64 columns of a million rows.
    assert_true(held >= 16_000_000, String(held))
    _ = frame^
    release_retained()
    assert_true(memory_usage() - before < 1_000_000)


def _group_by_raises(frame: DataFrame, lazy: Bool) raises -> String:
    try:
        if lazy:
            _ = (
                frame.lazy()
                .group_by("k")
                .agg(col("v").sum().alias("s"))
                .collect()
            )
        else:
            _ = frame.group_by("k").agg(col("v").sum().alias("s"))
    except e:
        return String(e)
    return String()


def test_group_by_over_the_limit_raises_and_the_process_goes_on() raises:
    var frame = _frame(300_000)
    var default = memory_limit()
    for lazy in [False, True]:
        # Freed blocks kept for reuse would be freed before an error is
        # raised; free them now so the limit below is what the query has.
        release_retained()
        var before = memory_usage()
        # Room for a fraction of the 300,000 groups' output.
        set_memory_limit(before + 1_000_000)
        var message = _group_by_raises(frame, lazy)
        set_memory_limit(default)
        assert_true(_is_out_of_memory(message), message)
        assert_true("set_memory_limit" in message, message)
        # What the failed query held was freed.
        release_retained()
        assert_true(memory_usage() - before < 1_000_000)
        # The same query runs once the limit allows it.
        assert_equal(_group_by_raises(frame, lazy), String())


def test_join_over_the_limit_raises_eagerly_and_lazily() raises:
    var left = _frame(200_000)
    var right = _frame(200_000, 100_000)
    var default = memory_limit()
    for lazy in [False, True]:
        # Freed blocks kept for reuse would be freed before an error is
        # raised; free them now so the limit below is what the query has.
        release_retained()
        var before = memory_usage()
        set_memory_limit(before + 500_000)
        var message = String()
        try:
            if lazy:
                _ = left.lazy().join(right.lazy(), on="k").collect()
            else:
                _ = left.join(right, on="k")
        except e:
            message = String(e)
        set_memory_limit(default)
        assert_true(_is_out_of_memory(message), message)
        var joined = left.join(right, on="k")
        assert_equal(joined.height(), 100_000)


def test_tables_above_the_limit_fail_the_next_query() raises:
    var frame = _frame(200_000)
    var default = memory_limit()
    set_memory_limit(memory_usage() // 2)
    var message = String()
    try:
        _ = frame.sort("v")
    except e:
        message = String(e)
    set_memory_limit(default)
    assert_true(_is_out_of_memory(message), message)
    assert_equal(frame.sort("v").height(), 200_000)
