"""Segmented CUT/QCUT control/classification kernels, not public admission.

The caller supplies exact partition ranges and CPU run/group priorities. These
kernels preserve within-partition source order and have disjoint selection
scratch. Checked operand/run observation must precede host label formatting;
this module never invents a global observation boundary for partitioned work.
"""
from max.gpu import global_idx
from std.memory import Pointer
from . import floating as f
from .quantiles import reduce_selected
from dataframe.accelerator.rows import ROW_QUANTILE_LINEAR
from .qcuts import (
    NO_FAILURE,
    INVALID_EDGES,
    INVALID_PROBABILITY,
    UNSUPPORTED_NAN_PROBABILITY,
)


def segmented_controls(
    words: Pointer[UInt64, ImmutAnyOrigin],
    sample_mask: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    specifications: Pointer[UInt64, ImmutAnyOrigin],
    edges: Pointer[UInt64, MutAnyOrigin],
    label_indices: Pointer[UInt32, MutAnyOrigin],
    controls: Pointer[UInt32, MutAnyOrigin],
    source_base: UInt64,
    groups: UInt32,
    count: UInt32,
    qcut: UInt32,
    allow_duplicates: UInt32,
):
    """One thread per group, original q-loop/control error association.

    starts has G+1 entries. Each group owns scratch[begin:end], K edge
    slots, K+1 label-index slots and two control words. Zero-edge buffers use
    a max(1,K) stride; all unused edge words are initialized for safe metadata
    readback. The caller handles the empty-input virtual partition separately.
    """
    var group = global_idx.x
    if group >= Int(groups):
        return
    var first = Int(starts[unsafe_offset=group])
    var last = Int(starts[unsafe_offset=group + 1])
    var edge_base = group * max(1, Int(count))
    var label_base = group * (Int(count) + 1)
    var used = 0
    var failure = NO_FAILURE
    for at in range(max(1, Int(count))):
        edges[unsafe_offset=edge_base + at] = 0
    label_indices[unsafe_offset=label_base] = 0
    for at in range(Int(count)):
        var value = specifications[unsafe_offset=at]
        var present = UInt8(1)
        if qcut:
            if f.is_nan(value):
                failure = UNSUPPORTED_NAN_PROBABILITY
                break
            if f.less(value, 0) or f.less(0x3FF0000000000000, value):
                failure = INVALID_PROBABILITY
                break
            # The mask is a compact row plane, independent of the ranked
            # matrix target slot. Shift values and use base0 for both views.
            var selected = reduce_selected[True](
                words.unsafe_offset(Int(source_base)),
                sample_mask,
                index,
                scratch,
                0,
                first,
                last,
                first,
                value,
                ROW_QUANTILE_LINEAR,
            )
            value = selected[0]
            present = selected[1]
        if not present:
            continue
        if used > 0 and not f.less(
            edges[unsafe_offset=edge_base + used - 1], value
        ):
            if (
                not qcut
                or not allow_duplicates
                or not f.equal(value, edges[unsafe_offset=edge_base + used - 1])
            ):
                failure = INVALID_EDGES
                break
            label_indices[unsafe_offset=label_base + used] = UInt32(at + 1)
        else:
            edges[unsafe_offset=edge_base + used] = value
            used += 1
            label_indices[unsafe_offset=label_base + used] = UInt32(at + 1)
    controls[unsafe_offset=group * 2] = UInt32(used)
    controls[unsafe_offset=group * 2 + 1] = failure


def segmented_classify(
    words: Pointer[UInt64, ImmutAnyOrigin],
    source_errors: Pointer[UInt32, ImmutAnyOrigin],
    sample_mask: Pointer[UInt8, ImmutAnyOrigin],
    group_by_row: Pointer[UInt32, ImmutAnyOrigin],
    edges: Pointer[UInt64, ImmutAnyOrigin],
    label_indices: Pointer[UInt32, ImmutAnyOrigin],
    label_codes: Pointer[UInt32, ImmutAnyOrigin],
    controls: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    source_base: UInt64,
    category_base: UInt64,
    breakpoint_base: UInt64,
    cells: UInt64,
    rows: UInt32,
    count: UInt32,
    left_closed: UInt32,
    include_breaks: UInt32,
    default_labels: UInt32,
):
    """Publish only after the exact CPU phase observer reported no fault.

    Custom codes are global K+1 codes indexed by original labels. Default
    codes are G*(K+1) per-partition bin-to-output-domain maps. The output public
    dtype remains String; this UInt32 carrier has explicit external provenance.
    """
    var row = global_idx.x
    if row >= Int(rows):
        return
    var group = Int(group_by_row[unsafe_offset=row])
    var present = sample_mask[unsafe_offset=row]
    var value = words[unsafe_offset=Int(source_base) + row]
    var edge_count = Int(controls[unsafe_offset=group * 2])
    var edge_base = group * max(1, Int(count))
    var label_base = group * (Int(count) + 1)
    var bin = 0
    if present:
        while bin < edge_count:
            var edge = edges[unsafe_offset=edge_base + bin]
            var cross = f.less(edge, value) or (
                left_closed != 0 and f.equal(edge, value)
            )
            if not cross:
                break
            bin += 1
    var label = label_base + bin if default_labels else Int(
        label_indices[unsafe_offset=label_base + bin]
    )
    var target = Int(category_base) + row
    output_words[unsafe_offset=target] = UInt64(
        label_codes[unsafe_offset=label]
    )
    output_words[unsafe_offset=Int(cells) + target] = 0
    output_validity[unsafe_offset=target] = present
    output_errors[unsafe_offset=target] = source_errors[
        unsafe_offset=Int(source_base) + row
    ]
    if include_breaks:
        target = Int(breakpoint_base) + row
        output_words[unsafe_offset=target] = edges[
            unsafe_offset=edge_base + bin
        ] if bin < edge_count else UInt64(0x7FF0000000000000)
        output_words[unsafe_offset=Int(cells) + target] = 0
        output_validity[unsafe_offset=target] = present
        output_errors[unsafe_offset=target] = source_errors[
            unsafe_offset=Int(source_base) + row
        ]
