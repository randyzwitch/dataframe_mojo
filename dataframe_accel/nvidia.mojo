"""Device-owned numeric columns using the shared dataframe buffer layout.

One runtime owns one CUDA context/stream. Copies reuse that context; every
resident column retains its runtime. Transfers are ordered on that stream.
This module contains storage and lifetime management, not query semantics.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Optional
from std.os import abort
from std.sys import size_of
from dataframe.column import Column, _copy_validity
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.lazy import AcceleratorBackend, LazyFrame
from dataframe.frame import DataFrame
from .query import execute, describe


def _drain_before_release(ctx: DeviceContext):
    """Never release host storage while a queued transfer may still use it."""
    try:
        ctx.synchronize()
    except error:
        # Destructors cannot raise. A failed drain cannot establish that raw
        # host pointers are no longer in use; do not turn it into a UAF.
        abort(String("NVIDIA transfer cleanup failed: ", error))


def _require_storage_dtype[D: DType]() raises:
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime if D == NUMERIC_DTYPES[i]:
            return
    raise Error("Unsupported NVIDIA column storage dtype: " + String(D))


struct NvidiaRuntime(AcceleratorBackend):
    """Reusable context for one explicit CUDA device; no global singleton.

    Device IDs are CUDA-visible ordinals. This runtime has one stream and
    is intended for serial host submission, not concurrent host callers.
    Copies share the context. Construct another runtime for another device.
    """

    var _ctx: DeviceContext
    var _device_id: Int

    def __init__(out self, device_id: Int = 0) raises:
        if device_id < 0 or device_id >= Self.device_count():
            raise Error("NVIDIA device is unavailable: " + String(device_id))
        self._ctx = DeviceContext(device_id, api="cuda")
        self._device_id = device_id

    @staticmethod
    def device_count() -> Int:
        """Number of CUDA devices visible to this process."""
        return DeviceContext.number_of_devices(api="cuda")

    def device_id(self) -> Int:
        """The CUDA-visible ordinal selected when this runtime was created."""
        return self._device_id

    def name(self) -> String:
        """Driver-reported device name."""
        return self._ctx.name()

    def shares_context(self, other: Self) -> Bool:
        """Whether both handles use the same context and ordered stream."""
        return self._ctx == other._ctx

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        """Lower and execute a supported float reduction region."""
        return execute(self, plan)

    def describe(self, plan: LazyFrame) -> String:
        """Explain capability without submitting device work."""
        return describe(plan)

    def upload[
        D: DType
    ](self, source: Column[Scalar[D]]) raises -> NvidiaColumn[D]:
        """Enqueue an exact value-window and validity-byte upload.

        The returned owner retains the source until wait/download/destruction.
        No null payloads are inspected, converted, or overwritten.
        """
        return NvidiaColumn[D](self, source)


struct _HostDownload[D: DType](Movable):
    """Keep output lists alive across successful and exceptional transfers."""

    var ctx: DeviceContext
    var values: List[Scalar[Self.D]]
    var bits: List[UInt8]
    var pending: Bool

    def __init__(out self, ctx: DeviceContext, rows: Int, bitmap_bytes: Int):
        self.ctx = ctx
        self.values = List[Scalar[Self.D]](length=rows, fill=0)
        self.bits = List[UInt8](length=bitmap_bytes, fill=0)
        self.pending = False

    def __deinit__(deinit self):
        if self.pending:
            _drain_before_release(self.ctx)

    def finish(mut self, bit_offset: Int) raises -> Column[Scalar[Self.D]]:
        self.ctx.synchronize()
        self.pending = False
        # Reuse the CPU column's bitmap normalization at the host boundary.
        var bits = _copy_validity(self.bits, bit_offset, len(self.values))
        var values = self.values^
        self.values = List[Scalar[Self.D]]()
        return Column[Scalar[Self.D]](values=values^, bits=bits^)


struct NvidiaColumn[D: DType](Movable, Sized):
    """Owned device values/validity plus any source still needed by a transfer.

    Device buffers are not host-accessible. Their host source may coexist
    until synchronization; accessibility and readiness are separate facts.
    Underscored buffers are internal and must only be used on the owner stream.
    """

    var _owner: NvidiaRuntime
    var _source: Optional[Column[Scalar[Self.D]]]
    var _values: DeviceBuffer[Self.D]
    var _bits: DeviceBuffer[DType.uint8]
    var _rows: Int
    var _has_validity: Bool
    var _bit_offset: Int
    var _bitmap_bytes: Int
    var _pending: Bool

    def __init__(
        out self, owner: NvidiaRuntime, source: Column[Scalar[Self.D]]
    ) raises:
        _require_storage_dtype[Self.D]()
        self._owner = owner.copy()
        self._source = Optional(source.copy())
        self._rows = len(source)
        self._has_validity = len(source._bits[]) != 0
        self._bit_offset = (
            source.validity_offset() % 8 if self._has_validity else 0
        )
        self._bitmap_bytes = (
            self._rows // 8 + (self._rows % 8 + self._bit_offset + 7) // 8
        ) if self._has_validity and self._rows > 0 else 0
        if (
            self._rows
            > (Int.MAX - self._bitmap_bytes) // size_of[Scalar[Self.D]]()
        ):
            raise Error("NVIDIA upload size overflows Int")
        # All fields are initialized before any raw host pointer is queued.
        # A failure during allocation only releases SDK-owned device buffers.
        self._values = owner._ctx.enqueue_create_buffer[Self.D](self._rows)
        self._bits = owner._ctx.enqueue_create_buffer[DType.uint8](
            self._bitmap_bytes
        )
        self._pending = True
        try:
            self._enqueue_upload()
        except error:
            # A raising initializer may destroy fields without calling the
            # complete object's destructor. Drain before it unwinds.
            _drain_before_release(self._owner._ctx)
            self._pending = False
            raise error^

    def __deinit__(deinit self):
        if self._pending:
            _drain_before_release(self._owner._ctx)

    def _enqueue_upload(mut self) raises:
        if self._rows:
            self._owner._ctx.enqueue_copy(
                self._values, self._source.value().unsafe_values()
            )
        if self._bitmap_bytes:
            ref source = self._source.value()
            self._owner._ctx.enqueue_copy(
                self._bits,
                source.unsafe_validity().unsafe_offset(
                    source.validity_offset() // 8
                ),
            )

    def __len__(self) -> Int:
        return self._rows

    def device_id(self) -> Int:
        """CUDA-visible ordinal of the owning runtime."""
        return self._owner.device_id()

    def belongs_to(self, runtime: NvidiaRuntime) -> Bool:
        """Check context ownership, not just a matching physical device."""
        return self._owner.shares_context(runtime)

    def is_ready(self) -> Bool:
        """Whether wait/download has confirmed completion for this column.

        This is conservative bookkeeping, not a driver event query. Waiting
        on another column may finish this upload without updating this flag.
        """
        return not self._pending

    def host_accessible(self) -> Bool:
        """False for these device buffers; use download for a CPU column."""
        return False

    def has_validity(self) -> Bool:
        """Whether the input supplied a validity bitmap, including all-valid."""
        return self._has_validity

    def validity_offset(self) -> Int:
        """Row zero's bit index in the uploaded validity bytes (0 through 7)."""
        return self._bit_offset

    def storage_bytes(self) -> Int:
        """Payload and validity bytes allocated, excluding SDK bookkeeping."""
        return self._rows * size_of[Scalar[Self.D]]() + self._bitmap_bytes

    def wait(mut self) raises:
        """Wait for upload completion and release the retained CPU source."""
        if self._pending:
            self._owner._ctx.synchronize()
            self._pending = False
            self._source = Optional[Column[Scalar[Self.D]]]()

    def download(mut self) raises -> Column[Scalar[Self.D]]:
        """Return a synchronized CPU column with identical payload bits.

        The uploaded validity offset is normalized through the existing
        Column helper. Values are copied back without arithmetic/conversion.
        """
        var host = _HostDownload[Self.D](
            self._owner._ctx, self._rows, self._bitmap_bytes
        )
        host.pending = True
        if self._rows:
            self._owner._ctx.enqueue_copy(
                host.values.unsafe_ptr(), self._values
            )
        if self._bitmap_bytes:
            self._owner._ctx.enqueue_copy(host.bits.unsafe_ptr(), self._bits)
        var result = host.finish(self._bit_offset)
        self._pending = False
        self._source = Optional[Column[Scalar[Self.D]]]()
        return result^
