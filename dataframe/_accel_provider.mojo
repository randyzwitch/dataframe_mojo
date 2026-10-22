"""The core's door to an accelerator provider.

The provider is chosen at build time by import path. `dataframe_provider` is
the CPU-only provider in the repository root; a build that wants a backend puts
that backend's `dataframe_provider` first on the import path (Metal:
`-I providers/metal -I .`). The six functions are the whole contract; see
docs/design/accelerator-contract.md.
"""
from dataframe_provider import (
    installed,
    execute,
    describe,
    execute_profiled,
    select_auto,
    execute_auto,
)
