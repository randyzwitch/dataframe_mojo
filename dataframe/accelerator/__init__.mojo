"""CPU-only planning shared by optional accelerator execution providers.

Importing this package never loads a GPU SDK. Runtime providers implement
`dataframe.lazy.AcceleratorBackend` and consume the bound plans below.
"""
from .capabilities import RowCapabilities
from .rows import RowPlan, RowStep, RowOutput, lower_rows
from .memory import MemoryEstimate, RowMemory, row_memory
