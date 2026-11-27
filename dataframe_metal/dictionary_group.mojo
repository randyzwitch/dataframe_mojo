"""Group dictionary-encoded string keys using the native integer group scan."""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col
from dataframe.expr import COL, COUNT, LEN, SUM
from dataframe.lazy import AGG, SCAN_FRAME
from dataframe.storage.buffer import _Buffer
from dataframe.storage.string import StringColumn
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .group_layout import GENERAL_GROUP_LIMIT
from .group_reduce import execute_grouped
from .group_properties import sample_group_tuples
from .multi_plan import MAX_FIELDS, ReductionOutput
from .dictionary_pair_reduce import execute_dictionary_sum
from .dictionary_projection import project_dictionary


@fieldwise_init
struct DictionaryGroup(Copyable, Movable):
    var keys: List[String]
    var inputs: List[String]
    var single_sum: Bool
    var dictionaries: List[DataType]

    def dense_dictionary(self, rows: Int) -> Bool:
        if rows <= 0 or rows >= Int(Int32.MAX):
            return False
        if len(self.keys) == 1:
            return True
        var left = len(self.dictionaries[0].dictionary()[]) + 1
        var right = len(self.dictionaries[1].dictionary()[]) + 1
        return left * right <= GENERAL_GROUP_LIMIT

    def grouped_key(self, region: LazyFrame) raises -> String:
        # Aliases do not affect cost. Reused input bindings share GPU state.
        ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
        var identity = String("dictionary_grouped:packed:v2:")
        var names = self.keys.copy()
        for name in self.inputs:
            if name not in self.keys:
                names.append(name.copy())
        for labels in self.dictionaries:
            identity += ":labels" + String(len(labels.dictionary()[]))
        for expr in region._plan._nodes[1].exprs:
            ref leaf = expr._nodes[0]
            ref aggregate = expr._nodes[1]
            if aggregate.op == LEN:
                identity += ":rows"
                continue
            identity += (
                ":input"
                + String(names.index(leaf.text))
                + ":"
                + frame.column(leaf.text).dtype().name()
                + ":op"
                + String(aggregate.op)
                + ":min_count"
                + String(aggregate.min_count)
            )
        var codes = List[Series]()
        for key in self.keys:
            codes.append(_signed_group_codes(frame.column(key)))
        identity += sample_group_tuples(DataFrame(codes^), self.keys).key()
        return identity^


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
        or not root.exprs
        or (len(keys) == 2 and keys[0] == keys[1])
    ):
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var inputs = List[String]()
    var names = List[String]()
    for expression in root.exprs:
        if (
            len(expression._nodes) != 2
            or expression._name in keys
            or expression._name in names
        ):
            return None
        ref leaf = expression._nodes[0]
        ref aggregate = expression._nodes[1]
        if (
            leaf.op != COL
            or aggregate.left != 0
            or aggregate.op not in [COUNT, LEN, SUM]
            or leaf.text not in frame.columns()
        ):
            return None
        if aggregate.op != LEN:
            var dtype = frame.column(leaf.text).dtype()
            if dtype not in [DataType.INT32, DataType.INT64] and not (
                aggregate.op == COUNT
                and (dtype == DataType.FLOAT32 or leaf.text in keys)
            ):
                return None
            if leaf.text not in keys and leaf.text not in inputs:
                inputs.append(leaf.text.copy())
                if len(inputs) > MAX_FIELDS:
                    return None
        names.append(expression._name.copy())
    var dictionaries = List[DataType]()
    for key in keys:
        if key not in frame.columns():
            return None
        var dictionary = _group_dictionary(frame.column(key))
        if not dictionary:
            return None
        dictionaries.append(dictionary.take())
    return DictionaryGroup(
        keys.copy(),
        inputs^,
        len(root.exprs) == 1 and root.exprs[0]._nodes[1].op == SUM,
        dictionaries^,
    )


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
    for name in plan.inputs:
        if name not in plan.keys:
            inputs.append(source.column(name))
    var frame = DataFrame(inputs^)
    var reduced: Tuple[DataFrame, HandoffTimings]
    var preparation = Int(perf_counter_ns()) - started
    ref expr = root.exprs[0]
    var output = ReductionOutput(
        0, expr._name.copy(), SUM, expr._nodes[1].min_count
    )
    if frame.height() == 0 or frame.height() >= Int(Int32.MAX):
        # The integer grouped reducer shares scans and state between outputs.
        # Dictionary code views retain the original key validity and offsets.
        var expressions = root.exprs.copy()
        for i in range(len(expressions)):
            if expressions[i]._nodes[1].op == LEN:
                # Any existing column has the same row count. Keep LEN-only
                # operands out of the lowered frame and its physical bindings.
                expressions[i] = (
                    col(plan.keys[0]).len().alias(root.exprs[i]._name)
                )
        var lowered = (
            frame.lazy()
            .group_by(plan.keys, maintain_order=root.maintain_order)
            .agg(expressions)
        )
        lowered._plan._device_observation = region._plan._device_observation
        preparation = Int(perf_counter_ns()) - started
        reduced = execute_grouped(lowered, zero_copy)
    elif not plan.single_sum:
        var projection = project_dictionary(plan.keys, frame, root.exprs)
        var left_labels = len(plan.dictionaries[0].dictionary()[])
        var observation = region._plan._device_observation
        preparation = Int(perf_counter_ns()) - started
        if len(plan.keys) == 1:
            reduced = execute_dictionary_sum[True, False, True](
                frame,
                output,
                observation,
                left_labels,
                0,
                zero_copy,
                projection^,
            )
        else:
            var right_labels = len(plan.dictionaries[1].dictionary()[])
            if plan.dense_dictionary(frame.height()):
                reduced = execute_dictionary_sum[False, False, True](
                    frame,
                    output,
                    observation,
                    left_labels,
                    right_labels,
                    zero_copy,
                    projection^,
                )
            else:
                reduced = execute_dictionary_sum[False, True, True](
                    frame,
                    output,
                    observation,
                    left_labels,
                    right_labels,
                    zero_copy,
                    projection^,
                )
    elif plan.dense_dictionary(frame.height()) and len(plan.keys) == 1:
        reduced = execute_dictionary_sum[True](
            frame,
            output,
            region._plan._device_observation,
            len(plan.dictionaries[0].dictionary()[]),
            0,
            zero_copy,
        )
    elif plan.dense_dictionary(frame.height()):
        reduced = execute_dictionary_sum[False](
            frame,
            output,
            region._plan._device_observation,
            len(plan.dictionaries[0].dictionary()[]),
            len(plan.dictionaries[1].dictionary()[]),
            zero_copy,
        )
    else:
        reduced = execute_dictionary_sum[False, True](
            frame,
            output,
            region._plan._device_observation,
            len(plan.dictionaries[0].dictionary()[]),
            len(plan.dictionaries[1].dictionary()[]),
            zero_copy,
        )
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
