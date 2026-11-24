"""Group dictionary-encoded string keys using the native integer group scan."""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import COL, SUM
from dataframe.lazy import AGG, SCAN_FRAME
from dataframe.storage.buffer import _Buffer
from dataframe.storage.string import StringColumn
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .group_layout import GENERAL_GROUP_LIMIT
from .group_reduce import execute_grouped
from .dictionary_pair_reduce import execute_dictionary_pair


@fieldwise_init
struct DictionaryGroup(Copyable, Movable):
    var keys: List[String]
    var value: String
    var dictionaries: List[DataType]

    def dense_pair(self, rows: Int) -> Bool:
        if len(self.keys) != 2 or rows == 0 or rows >= Int(Int32.MAX):
            return False
        var left = len(self.dictionaries[0].dictionary()[]) + 1
        var right = len(self.dictionaries[1].dictionary()[]) + 1
        return left * right <= GENERAL_GROUP_LIMIT


def _group_dictionary(source: Series) raises -> Optional[DataType]:
    if source.dtype() != DataType.STRING:
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
    return dictionary^


def plan_dictionary_group(
    region: LazyFrame,
) raises -> Optional[DictionaryGroup]:
    if (
        len(region._plan._nodes) != 2
        or region._plan._nodes[0].kind != SCAN_FRAME
    ):
        return None
    ref root = region._plan._nodes[1]
    ref keys = root.group_keys()
    if (
        root.kind != AGG
        or root.left != 0
        or len(keys) == 0
        or len(keys) > 2
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
    if leaf.text not in frame.columns() or frame.column(
        leaf.text
    ).dtype() not in [DataType.INT32, DataType.INT64]:
        return None
    if expression._name in keys or (len(keys) == 2 and keys[0] == keys[1]):
        return None
    var dictionaries = List[DataType]()
    for key in keys:
        if key not in frame.columns():
            return None
        var dictionary = _group_dictionary(frame.column(key))
        if not dictionary:
            return None
        dictionaries.append(dictionary.take())
    return DictionaryGroup(keys.copy(), leaf.text.copy(), dictionaries^)


def _signed_group_codes(source: Series) raises -> Series:
    # Every valid code indexes a dictionary smaller than GENERAL_GROUP_LIMIT.
    # Its UInt32 bits are already the Int32 key the grouped scan expects.
    var parts = List[Series]()
    for chunk in source._joined_windows().chunks():
        ref strings = chunk._data[StringColumn]
        ref codes = strings._codes.value()[].codes
        var data = _Buffer[Int32].borrowed(
            Int(codes.unsafe_ptr()), len(codes), codes._shared_owner()
        )
        parts.append(
            Series(
                source.name(),
                Column[Int32]._from_buffers(
                    data, strings._bits, strings._offset, len(strings)
                ),
            )
        )
    return Series._from_chunks(parts)


def execute_dictionary_group(
    region: LazyFrame, plan: DictionaryGroup, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var started = Int(perf_counter_ns())
    ref source = region._plan._frames[region._plan._nodes[0].source_slot()]
    ref root = region._plan._nodes[1]
    var inputs = List[Series]()
    for key in plan.keys:
        inputs.append(_signed_group_codes(source.column(key)))
    inputs.append(source.column(plan.value))
    var frame = DataFrame(inputs^)
    var reduced: Tuple[DataFrame, HandoffTimings]
    var preparation = Int(perf_counter_ns()) - started
    if plan.dense_pair(frame.height()):
        reduced = execute_dictionary_pair(
            region,
            frame,
            len(plan.dictionaries[0].dictionary()[]),
            len(plan.dictionaries[1].dictionary()[]),
            zero_copy,
        )
    else:
        var lowered = (
            frame.lazy()
            .group_by(plan.keys, maintain_order=root.maintain_order)
            .agg(root.exprs)
        )
        lowered._plan._device_observation = region._plan._device_observation
        preparation = Int(perf_counter_ns()) - started
        reduced = execute_grouped(lowered, zero_copy)
    started = Int(perf_counter_ns())
    var result = reduced[0].copy()
    for i in range(len(plan.keys)):
        var labels = (
            result.column(plan.keys[i])
            .cast(DataType.UINT32)
            .with_dtype(plan.dictionaries[i])
            .cast(DataType.STRING)
        )
        result = result.with_column(labels^)
    var timings = reduced[1].copy()
    timings.upload_ns += preparation
    timings.finish_ns += Int(perf_counter_ns()) - started
    return (result^, timings^)
