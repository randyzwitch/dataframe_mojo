"""Bounded compiled-handle cache for fused Int32 kernels."""

from max.gpu.host import DeviceContext, DeviceBuffer
from .target import METAL_TARGET
from . import fused_int32


# Metadata-only targets retain accounting and lease ownership without
# instantiating an unavailable DeviceContext destructor. Real Metal keeps
# the exact context owner type, including its queue lifetime and teardown.
comptime ContextCell = Optional[DeviceContext if METAL_TARGET else Int]

# Keep the exact canonical return types produced by compile_function. The
# aliases are selected only on a Metal compilation target; CPU-only builds
# retain metadata fields without instantiating DeviceFunction specializations.
comptime Partial000Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[False, False, False]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial001Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[False, False, True]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial010Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[False, True, False]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial011Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[False, True, True]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial100Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[True, False, False]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial101Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[True, False, True]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial110Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[True, True, False]
        ]()
    ) if METAL_TARGET else Int
)
comptime Partial111Handle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[True, True, True]
        ]()
    ) if METAL_TARGET else Int
)
comptime FinishHandle = (
    type_of(
        DeviceContext(0, api="metal").compile_function[fused_int32.finish]()
    ) if METAL_TARGET else Int
)


struct FusedPipelineCache(Movable):
    var partial000: Optional[Partial000Handle]
    var partial001: Optional[Partial001Handle]
    var partial010: Optional[Partial010Handle]
    var partial011: Optional[Partial011Handle]
    var partial100: Optional[Partial100Handle]
    var partial101: Optional[Partial101Handle]
    var partial110: Optional[Partial110Handle]
    var partial111: Optional[Partial111Handle]
    var finish: Optional[FinishHandle]
    var hits: Int
    var misses: Int

    def __init__(out self):
        self.partial000 = Optional[Partial000Handle]()
        self.partial001 = Optional[Partial001Handle]()
        self.partial010 = Optional[Partial010Handle]()
        self.partial011 = Optional[Partial011Handle]()
        self.partial100 = Optional[Partial100Handle]()
        self.partial101 = Optional[Partial101Handle]()
        self.partial110 = Optional[Partial110Handle]()
        self.partial111 = Optional[Partial111Handle]()
        self.finish = Optional[FinishHandle]()
        self.hits = 0
        self.misses = 0

    def entries(self) -> Int:
        return (
            (1 if self.partial000 else 0)
            + (1 if self.partial001 else 0)
            + (1 if self.partial010 else 0)
            + (1 if self.partial011 else 0)
            + (1 if self.partial100 else 0)
            + (1 if self.partial101 else 0)
            + (1 if self.partial110 else 0)
            + (1 if self.partial111 else 0)
            + (1 if self.finish else 0)
        )

    def enqueue_partials[
        nullable: Bool, filtered: Bool, arithmetic: Bool
    ](
        mut self,
        context: DeviceContext,
        values: DeviceBuffer[DType.int32],
        validity: DeviceBuffer[DType.uint8],
        validity_offset: UInt32,
        rows: UInt32,
        stride: UInt32,
        threshold: Int32,
        addend: Int32,
        factor: Int32,
        sums: DeviceBuffer[DType.int64],
        counts: DeviceBuffer[DType.uint32],
        faults: DeviceBuffer[DType.uint32],
        partial_count: Int,
    ) raises:
        """Compile once per variant and enqueue on the matching context."""
        comptime if METAL_TARGET:
            comptime if nullable:
                comptime if filtered:
                    comptime if arithmetic:
                        if self.partial111:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial111 = Optional[Partial111Handle](
                                rebind[Partial111Handle](
                                    context.compile_function[
                                        fused_int32.partials[True, True, True]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[True, True, True](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[True, True, True]
                                    ]()
                                )
                            ](self.partial111.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                    else:
                        if self.partial110:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial110 = Optional[Partial110Handle](
                                rebind[Partial110Handle](
                                    context.compile_function[
                                        fused_int32.partials[True, True, False]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[True, True, False](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[True, True, False]
                                    ]()
                                )
                            ](self.partial110.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                else:
                    comptime if arithmetic:
                        if self.partial101:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial101 = Optional[Partial101Handle](
                                rebind[Partial101Handle](
                                    context.compile_function[
                                        fused_int32.partials[True, False, True]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[True, False, True](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[True, False, True]
                                    ]()
                                )
                            ](self.partial101.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                    else:
                        if self.partial100:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial100 = Optional[Partial100Handle](
                                rebind[Partial100Handle](
                                    context.compile_function[
                                        fused_int32.partials[True, False, False]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[True, False, False](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[True, False, False]
                                    ]()
                                )
                            ](self.partial100.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
            else:
                comptime if filtered:
                    comptime if arithmetic:
                        if self.partial011:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial011 = Optional[Partial011Handle](
                                rebind[Partial011Handle](
                                    context.compile_function[
                                        fused_int32.partials[False, True, True]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[False, True, True](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[False, True, True]
                                    ]()
                                )
                            ](self.partial011.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                    else:
                        if self.partial010:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial010 = Optional[Partial010Handle](
                                rebind[Partial010Handle](
                                    context.compile_function[
                                        fused_int32.partials[False, True, False]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[False, True, False](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[False, True, False]
                                    ]()
                                )
                            ](self.partial010.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                else:
                    comptime if arithmetic:
                        if self.partial001:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial001 = Optional[Partial001Handle](
                                rebind[Partial001Handle](
                                    context.compile_function[
                                        fused_int32.partials[False, False, True]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[False, False, True](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[False, False, True]
                                    ]()
                                )
                            ](self.partial001.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
                    else:
                        if self.partial000:
                            self.hits += 1
                        else:
                            self.misses += 1
                            self.partial000 = Optional[Partial000Handle](
                                rebind[Partial000Handle](
                                    context.compile_function[
                                        fused_int32.partials[
                                            False, False, False
                                        ]
                                    ]()
                                )
                            )
                        enqueue_partial_handle[False, False, False](
                            context,
                            rebind[
                                type_of(
                                    context.compile_function[
                                        fused_int32.partials[
                                            False, False, False
                                        ]
                                    ]()
                                )
                            ](self.partial000.value()),
                            values,
                            validity,
                            validity_offset,
                            rows,
                            stride,
                            threshold,
                            addend,
                            factor,
                            sums,
                            counts,
                            faults,
                            partial_count,
                        )
        else:
            raise Error("fused pipeline cache requires a compiled Metal target")

    def enqueue_finish(
        mut self,
        context: DeviceContext,
        sums: DeviceBuffer[DType.int64],
        counts: DeviceBuffer[DType.uint32],
        faults: DeviceBuffer[DType.uint32],
        partial_count: Int,
        output: DeviceBuffer[DType.int64],
    ) raises:
        """Compile once and enqueue the final reduction."""
        comptime if METAL_TARGET:
            if self.finish:
                self.hits += 1
            else:
                self.misses += 1
                self.finish = Optional[FinishHandle](
                    rebind[FinishHandle](
                        context.compile_function[fused_int32.finish]()
                    )
                )
            enqueue_finish_handle(
                context,
                rebind[type_of(context.compile_function[fused_int32.finish]())](
                    self.finish.value()
                ),
                sums,
                counts,
                faults,
                partial_count,
                output,
            )
        else:
            raise Error("fused pipeline cache requires a compiled Metal target")

    def clear(mut self):
        # Compiled DeviceFunctions own context references. Release every handle
        # before the service drops the corresponding cached context.
        self.partial000 = Optional[Partial000Handle]()
        self.partial001 = Optional[Partial001Handle]()
        self.partial010 = Optional[Partial010Handle]()
        self.partial011 = Optional[Partial011Handle]()
        self.partial100 = Optional[Partial100Handle]()
        self.partial101 = Optional[Partial101Handle]()
        self.partial110 = Optional[Partial110Handle]()
        self.partial111 = Optional[Partial111Handle]()
        self.finish = Optional[FinishHandle]()


def enqueue_partial_handle[
    nullable: Bool, filtered: Bool, arithmetic: Bool
](
    context: DeviceContext,
    handle: type_of(
        DeviceContext(0, api="metal").compile_function[
            fused_int32.partials[nullable, filtered, arithmetic]
        ]()
    ),
    values: DeviceBuffer[DType.int32],
    validity: DeviceBuffer[DType.uint8],
    validity_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    threshold: Int32,
    addend: Int32,
    factor: Int32,
    sums: DeviceBuffer[DType.int64],
    counts: DeviceBuffer[DType.uint32],
    faults: DeviceBuffer[DType.uint32],
    partial_count: Int,
) raises:
    context.enqueue_function(
        handle,
        values,
        validity,
        validity_offset,
        rows,
        stride,
        threshold,
        addend,
        factor,
        sums,
        counts,
        faults,
        grid_dim=partial_count,
        block_dim=256,
    )


def enqueue_finish_handle(
    context: DeviceContext,
    handle: type_of(
        DeviceContext(0, api="metal").compile_function[fused_int32.finish]()
    ),
    sums: DeviceBuffer[DType.int64],
    counts: DeviceBuffer[DType.uint32],
    faults: DeviceBuffer[DType.uint32],
    partial_count: Int,
    output: DeviceBuffer[DType.int64],
) raises:
    context.enqueue_function(
        handle,
        sums,
        counts,
        faults,
        UInt32(partial_count),
        output,
        grid_dim=1,
        block_dim=256,
    )
