"""Group a dictionary-encoded string key using the native integer group scan."""
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType, LazyFrame
from dataframe.expr import COL, SUM
from dataframe.lazy import AGG, SCAN_FRAME
from dataframe.storage.string import StringColumn
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .group_layout import GENERAL_GROUP_LIMIT
from .group_reduce import execute_grouped


@fieldwise_init
struct DictionaryGroup(Copyable, Movable):
    var key: String
    var value: String
    var dictionary: DataType


def plan_dictionary_group(
    region: LazyFrame,
) raises -> Optional[DictionaryGroup]:
    if (
        len(region._plan._nodes) != 2
        or region._plan._nodes[0].kind != SCAN_FRAME
    ):
        return None
    ref root = region._plan._nodes[1]
    if (
        root.kind != AGG
        or root.left != 0
        or len(root.group_keys()) != 1
        or len(root.exprs) != 1
    ):
        return None
    ref expression = root.exprs[0]
    if len(expression._nodes) != 2:
        return None
    ref leaf = expression._nodes[0]
    ref aggregate = expression._nodes[1]
    if leaf.op != COL or aggregate.op != SUM or aggregate.left != 0:
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var key = root.group_keys()[0].copy()
    if (
        key == expression._name
        or key not in frame.columns()
        or leaf.text not in frame.columns()
    ):
        return None
    var source = frame.column(key)
    if source.dtype() != DataType.STRING or frame.column(
        leaf.text
    ).dtype() not in [DataType.INT32, DataType.INT64]:
        return None
    # Capability checks inspect chunk metadata only. Codes behind a gather
    # would require an additional materialization. Repeated dictionary labels
    # would require canonicalizing codes before integer grouping.
    var dictionary = Optional[DataType]()
    for chunk in source.chunks():
        if not chunk._data.isa[StringColumn]():
            return None
        ref strings = chunk._data[StringColumn]
        if not strings._codes:
            return None
        ref stored = strings._codes.value()[]
        if stored.rows or not stored.dtype.has_dictionary():
            return None
        var labels = stored.dtype.dictionary()
        if (
            labels[].grows
            or labels[].canonical
            or len(labels[]) >= GENERAL_GROUP_LIMIT
        ):
            return None
        if dictionary and dictionary.value() != stored.dtype:
            return None
        dictionary = stored.dtype
    if not dictionary:
        return None
    return DictionaryGroup(key^, leaf.text.copy(), dictionary.take())


def execute_dictionary_group(
    region: LazyFrame, plan: DictionaryGroup, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var started = Int(perf_counter_ns())
    ref source = region._plan._frames[region._plan._nodes[0].source_slot()]
    ref root = region._plan._nodes[1]
    var coded = source.column(plan.key)._dictionary_codes()
    var integers = coded.take().with_dtype(DataType.UINT32).cast(DataType.INT32)
    var frame = DataFrame([integers^, source.column(plan.value)])
    var lowered = (
        frame.lazy()
        .group_by(plan.key, maintain_order=root.maintain_order)
        .agg(root.exprs)
    )
    lowered._plan._device_observation = region._plan._device_observation
    var preparation = Int(perf_counter_ns()) - started
    var reduced = execute_grouped(lowered, zero_copy)
    started = Int(perf_counter_ns())
    var labels = (
        reduced[0]
        .column(plan.key)
        .cast(DataType.UINT32)
        .with_dtype(plan.dictionary)
        .cast(DataType.STRING)
    )
    var result = reduced[0].with_column(labels^)
    var timings = reduced[1].copy()
    timings.upload_ns += preparation
    timings.finish_ns += Int(perf_counter_ns()) - started
    return (result^, timings^)
