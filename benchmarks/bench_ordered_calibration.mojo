"""Capture and audit grouped first/last and arg-min/arg-max on an identified Mac."""
from std.sys import argv
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe.json.value import json_quote
from dataframe_metal.calibration_capture import capture
from dataframe_metal.handoff import layout_key
from dataframe_metal.metal import MetalRuntime


def _case(
    rows: Int, chunks: Int, skew: Bool, held_out: Bool
) raises -> Tuple[LazyFrame, DataFrame]:
    comptime groups = 32
    var counts = List[UInt32](length=groups, fill=0)
    var first = List[Int32](length=groups, fill=0)
    var last = List[Int32](length=groups, fill=0)
    var minimum = List[Int32](length=groups, fill=Int32.MAX)
    var maximum = List[Int32](length=groups, fill=Int32.MIN)
    var arg_min = List[UInt32](length=groups, fill=0)
    var arg_max = List[UInt32](length=groups, fill=0)
    var order = List[Int]()
    var key_parts = List[Series]()
    var value_parts = List[Series]()
    for part in range(chunks):
        var start = rows * part // chunks
        var end = rows * (part + 1) // chunks
        var keys = List[Int32](capacity=end - start)
        var values = List[Int32](capacity=end - start)
        var valid = List[Bool](capacity=end - start)
        for row in range(start, end):
            var k = row % groups
            if skew:
                k = 0 if row % 3 else 1 + (row // 3) % (groups - 1)
            var v = Int32((row * 73 + row // 97) % 503 - 251)
            if held_out:
                v = Int32((row * 67 + row // 131) % 503 - 251)
            var present = row % 11 != 0 and k != groups - 1
            valid.append(present)
            keys.append(Int32(300 - 7 * k if held_out else k))
            values.append(v)
            # Scalar oracle retains the first tied extremum in each group's
            # encounter order. Chunk boundaries never reset its local rank.
            if counts[k] == 0:
                first[k] = v if present else Int32.MIN
                order.append(k)
            last[k] = v if present else Int32.MIN
            if present and v < minimum[k]:
                minimum[k] = v
                arg_min[k] = counts[k]
            if present and v > maximum[k]:
                maximum[k] = v
                arg_max[k] = counts[k]
            counts[k] += 1
        key_parts.append(Series("k", Column[Int32](keys^)))
        value_parts.append(Series("v", Column[Int32](values^, valid^)))
    var group_keys = List[Int32]()
    var first_values = List[Int32]()
    var last_values = List[Int32]()
    var min_at = List[UInt32]()
    var max_at = List[UInt32]()
    var first_valid = List[Bool]()
    var last_valid = List[Bool]()
    var any_valid = List[Bool]()
    for k in order:
        first_valid.append(first[k] != Int32.MIN)
        last_valid.append(last[k] != Int32.MIN)
        any_valid.append(minimum[k] != Int32.MAX)
        group_keys.append(Int32(300 - 7 * k if held_out else k))
        first_values.append(first[k])
        last_values.append(last[k])
        min_at.append(arg_min[k])
        max_at.append(arg_max[k])
    var expected = DataFrame(
        [
            Series("k", Column[Int32](group_keys^)),
            Series("first", Column[Int32](first_values^, first_valid^)),
            Series("last", Column[Int32](last_values^, last_valid^)),
            Series("min_at", Column[UInt32](min_at^, any_valid)),
            Series("max_at", Column[UInt32](max_at^, any_valid^)),
        ]
    )
    var frame = DataFrame(
        [Series._from_chunks(key_parts^), Series._from_chunks(value_parts^)]
    )
    var v = col("v")
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                v.first().alias("first"),
                v.last().alias("last"),
                v.arg_min().alias("min_at"),
                v.arg_max().alias("max_at"),
            ]
        )
    )
    return (query^, expected^)


def _collect(plan: LazyFrame, mode: String) raises -> DataFrame:
    if mode == "metal":
        return plan.collect(accelerator=MetalRuntime())
    if mode == "auto":
        return plan.collect(engine="auto")
    return plan.collect(engine="cpu", streaming=mode == "cpu")


def _profile(
    plan: LazyFrame,
    mode: String,
    expected: DataFrame,
    rep: Int,
    report: Bool = True,
) raises -> Bool:
    var observed = plan.profile(
        engine="auto"
    ) if mode == "auto" else plan.profile(accelerator=MetalRuntime())
    if not observed[0].equals(expected):
        raise Error("Profile output differs from scalar ordered reference")
    ref profile = observed[1]
    var launches = Int64(0)
    var complete = Int64(0)
    var metal = False
    var decision = String()
    for row in range(profile.height()):
        var reason = profile.item(row, "fallback_reason").string()
        if reason:
            if mode != "auto" or reason != "cost_model_prefers_cpu":
                raise Error("Unexpected execution fallback: " + reason)
            decision = reason.copy()
        launches += profile.item(row, "gpu_launches").int64()
        complete += profile.item(row, "gpu_completed_launches").int64()
        if profile.item(row, "result_device").string() == "metal":
            metal = True
            if profile.item(row, "device_status").string() != "completed":
                raise Error("Incomplete Metal result")
    if (
        launches != complete
        or metal != (complete > 0)
        or (mode == "metal" and not metal)
    ):
        raise Error("Profile does not confirm the observed device")
    if report:
        print(
            '{"rep":'
            + String(rep)
            + ',"device":'
            + json_quote("metal" if metal else "cpu")
            + ',"cpu_reason":'
            + json_quote(decision)
            + ',"completed_launches":'
            + String(complete)
            + "}"
        )
    return metal


def main() raises:
    var args = argv()
    if len(args) != 6 and len(args) != 7:
        raise Error(
            "Usage: ordered-calibration MODE ROWS CHUNKS balanced|skew REPS [WARMUP_ROWS]"
        )
    var mode = String(args[1])
    var rows = Int(String(args[2]))
    var chunks = Int(String(args[3]))
    var distribution = String(args[4])
    var reps = Int(String(args[5]))
    if (
        rows < 128
        or chunks < 1
        or chunks > rows
        or reps < 1
        or (distribution != "balanced" and distribution != "skew")
    ):
        raise Error(
            "Expected at least 128 rows, 1..rows chunks, balanced|skew and positive repetitions"
        )
    var item = _case(rows, chunks, distribution == "skew", mode != "capture")
    ref plan = item[0]
    ref expected = item[1]
    if len(args) == 7:
        var warmup_rows = Int(String(args[6]))
        if mode == "capture" or warmup_rows < max(chunks, 128):
            raise Error("Prewarming requires a supported audit shape")
        var warmup = _case(warmup_rows, chunks, distribution == "skew", True)
        if layout_key(warmup[0]) != layout_key(plan):
            raise Error("Warmup and target have different calibration keys")
        if not _profile(warmup[0], "auto", warmup[1], 0, False):
            raise Error("Warmup did not execute the calibrated shape on Metal")
    if mode == "export":
        var result = plan.collect(engine="auto")
        if not result.equals(expected):
            raise Error(
                "Automatic result differs from scalar ordered reference"
            )
        for row in range(result.height()):
            var cells = List[String]()
            for name in result.columns():
                var cell = result.item(row, name)
                if cell.is_null():
                    cells.append("null")
                elif name == "min_at" or name == "max_at":
                    cells.append(String(cell.uint32()))
                else:
                    cells.append(String(cell.int32()))
            print("[" + String(",").join(cells) + "]")
        return
    if mode == "capture":
        print(capture(plan, expected, reps, distribution == "skew"))
        return
    if mode == "profile-auto" or mode == "profile-metal":
        for rep in range(reps):
            _ = _profile(
                plan,
                "auto" if mode == "profile-auto" else "metal",
                expected,
                rep,
            )
        return
    if mode not in ["cpu", "materialized", "metal", "auto"]:
        raise Error("Unknown measurement mode: " + mode)
    for rep in range(reps):
        var started = Int(perf_counter_ns())
        var result = _collect(plan, mode)
        var elapsed = Int(perf_counter_ns()) - started
        if not result.equals(expected):
            raise Error(
                "Collected result differs from scalar ordered reference"
            )
        print('{"rep":' + String(rep) + ',"ns":' + String(elapsed) + "}")
