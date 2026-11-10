"""Pipeline sinks consume data chunks using worker-local state.

Each worker updates its own local state while processing input. Once input
is exhausted, combine local states into the global state and finalize the
result. Keeping updates local avoids synchronization on every chunk.
"""
from dataframe.vector.data_chunk import ColumnArena, DataChunk


trait Sink(Deinitable, Movable):
    """A worker's local sink state."""

    def begin(mut self, morsel: Int):
        """Rows from here on are morsel `morsel`'s, numbered from
        `morsel << 32` in the morsel's output order, so a sink that keeps
        first rows orders them the same whichever worker took which
        morsel."""
        ...

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """One chunk of the sink's input: its columns are
        the pipeline's output schema's, read through `arena`."""
        ...

    def release(mut self) raises:
        """The worker's run is over: free what only sinking needs, on the
        worker's own thread."""
        ...

    def combine(var self, mut combined: List[Self]) raises:
        """This local state into the global one."""
        ...
