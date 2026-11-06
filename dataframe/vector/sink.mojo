"""DuckDB 2.0's sink interface (`execution/physical_operator.hpp`): the
operator a pipeline ends in takes the pipeline's data chunks one at a time.

Each worker holds its own local sink state (`GetLocalSinkState`) and feeds
it every chunk the pipeline's last step makes (`Sink`). When the worker's
run ends, its local state is combined into the global state every worker
shares (`Combine`), and the global state is finalized once (`Finalize`).
Here a local sink state is a struct implementing `Sink`, and the global
state is the list its workers' states are combined into: one state that
every other one folds into (an ungrouped aggregate), or every worker's
state kept for a parallel finalize (an aggregate table's partitions, a
top-N's heaps, a materialize sink's batches). Each sink kind's finalize
reads that list.
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
        """One chunk of the sink's input (DuckDB's `Sink`): its columns are
        the pipeline's output schema's, read through `arena`."""
        ...

    def release(mut self) raises:
        """The worker's run is over: free what only sinking needs, on the
        worker's own thread."""
        ...

    def combine(var self, mut combined: List[Self]) raises:
        """This local state into the global one (DuckDB's `Combine`)."""
        ...
