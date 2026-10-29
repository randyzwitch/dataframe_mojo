"""What an accelerator backend tells the planner about its devices.

Placement is written against memory models and capabilities, never against
a vendor. A backend reports each device it can run on (`DeviceInfo`), what
moving data to and from it costs (`TransferModel`), and, per pipeline,
whether it runs it and its estimated time (`pipeline_supports`,
`pipeline_estimate` in `docs/design/accelerator-contract.md`). The planner
adds transfer costs itself, from the transfer model and where each input
buffer already lives (`dataframe/storage/residence.mojo`).

Nothing in this module knows a backend exists.
"""

# How a device's memory relates to the host's.
# Discrete: its own memory, reached over a bus (NVIDIA and AMD discrete GPUs).
comptime MEMORY_DISCRETE = 0
# Unified: the CPU and the device read the same memory (Apple silicon, AMD
# APUs, NVIDIA Grace-class parts).
comptime MEMORY_UNIFIED = 1


@fieldwise_init
struct DeviceInfo(Copyable, Movable):
    """One device a backend can run on. Reported without creating an
    execution context."""

    # The backend's name, as device buffers record it ("nvidia", "amd",
    # "metal"), so devices of different vendors in one machine stay distinct.
    var backend: String
    # The backend's ordinal for this device.
    var ordinal: Int
    # A readable device name for `explain` and profiles.
    var name: String
    # MEMORY_DISCRETE or MEMORY_UNIFIED.
    var memory: Int
    # Bytes the planner may plan to hold on the device at once; -1 when the
    # backend sets no limit (unified memory shared with the CPU).
    var budget_bytes: Int


@fieldwise_init
struct ByteCost(Copyable, Movable):
    """`fixed_ns + ns_per_byte * bytes` nanoseconds; negative fields mean
    not measured."""

    var fixed_ns: Float64
    var ns_per_byte: Float64

    def measured(self) -> Bool:
        return self.fixed_ns >= 0 and self.ns_per_byte >= 0

    def cost(self, bytes: Int) -> Float64:
        return self.fixed_ns + self.ns_per_byte * Float64(bytes)


@fieldwise_init
struct TransferModel(Copyable, Movable):
    """What moving data between the host and one device costs, measured on
    this machine by the backend."""

    var to_device: ByteCost
    var to_host: ByteCost
    # Creating an execution context, paid once per process while none
    # exists; negative when not measured.
    var startup_ns: Float64

    @staticmethod
    def unmeasured() -> Self:
        return Self(ByteCost(-1, -1), ByteCost(-1, -1), -1)

    @staticmethod
    def shared_memory() -> Self:
        """Unified memory: the device reads host buffers in place. Start-up
        is still the backend's to measure."""
        return Self(ByteCost(0, 0), ByteCost(0, 0), -1)

    def measured(self) -> Bool:
        return self.to_device.measured() and self.to_host.measured()


@fieldwise_init
struct PipelineFeatures(Copyable, Movable):
    """Facts the planner knows about a pipeline when it asks for an estimate:
    estimated at plan time, exact when the pipeline is about to start. -1
    means unknown or not applicable."""

    var input_rows: Int
    var input_bytes: Int
    var output_rows: Int
    # Distinct groups of a grouped sink.
    var groups: Int
    # Rows on the build side of a join sink.
    var build_rows: Int

    @staticmethod
    def rows(input_rows: Int) -> Self:
        return Self(input_rows, -1, -1, -1, -1)
