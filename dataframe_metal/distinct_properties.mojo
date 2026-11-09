"""Bounded cardinality hints for placement, never for the exact GPU answer."""
from std.memory import bitcast
from dataframe import DataFrame, DataType
from .distinct_plan import DistinctPlan
from .accelerator.gpu.distinct_value import distinct_canonical
from .ordered_plan import ordered_tag


@always_inline
def _mix(value: UInt64) -> UInt64:
    var x = value
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def cardinality_key(plan: DistinctPlan, frame: DataFrame) raises -> String:
    # Sampling is a cost hint. Exhausted GPU tables always retry/fallback;
    # neither a sample nor its fingerprints determine any distinct count.
    var samples = min(frame.height(), 4096)
    if samples == 0:
        return ":distinct_cardinality=empty"
    var sets = List[Dict[UInt64, Bool]]()
    for field in plan.fields:
        sets.append(Dict[UInt64, Bool]())
    var state = UInt64(0x51A7D153)
    for i in range(samples):
        state = state * 6364136223846793005 + 1442695040888963407
        var row = Int((state >> 16) % UInt64(frame.height()))
        if i < min(16, samples):
            row = i
        elif i >= samples - 16:
            row = frame.height() - (samples - i)
        var group = UInt64(0xCBF29CE484222325)
        if plan.keys:
            for key in range(len(plan.keys.value().names)):
                var value = frame.item(row, plan.keys.value().names[key])
                var bits = UInt64(0xD1B54A32D192ED03)
                if not value.is_null():
                    var integer = value.int64() if plan.keys.value().dtypes[
                        key
                    ] == DataType.INT64 else Int64(value.int32())
                    bits = bitcast[DType.uint64](integer) ^ 0xA24BAED4963EE407
                group = _mix(group ^ _mix(bits))
        for field in range(len(plan.fields)):
            ref definition = plan.fields[field]
            var value = frame.item(row, definition.column)
            var bits = UInt64(0)
            var present = not value.is_null()
            if present:
                if definition.dtype == DataType.INT64:
                    bits = bitcast[DType.uint64](value.int64())
                elif definition.dtype == DataType.INT32:
                    bits = UInt64(bitcast[DType.uint32](value.int32()))
                elif definition.dtype == DataType.FLOAT32:
                    bits = UInt64(bitcast[DType.uint32](value.float32()))
                else:
                    bits = UInt64(value.bool())
                bits = distinct_canonical(
                    bits, UInt32(ordered_tag(definition.dtype))
                )
            var fingerprint = _mix(
                group
                ^ _mix(bits)
                ^ (
                    UInt64(0xA24BAED4963EE407) if present else UInt64(
                        0xD1B54A32D192ED03
                    )
                )
            )
            sets[field][fingerprint] = True
    var result = String()
    for values in sets:
        var count = len(values)
        if samples >= 256 and count * 4 >= samples * 3:
            result += ":distinct_cardinality=uncertain"
        else:
            var bucket = 1
            while bucket < count:
                bucket *= 2
            result += ":distinct_cardinality=" + String(bucket)
    return result^
