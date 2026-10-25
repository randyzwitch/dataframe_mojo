"""The core's door to an accelerator provider.

The provider is chosen at build time by import path. `dataframe_provider` in
the repository root is the Metal provider (`installed()` is False where Metal
cannot run). A build puts another `dataframe_provider` first on the path to
replace it: `-I providers/cpu -I .` compiles no GPU code. The six whole-plan functions
and the two hand-off functions are the whole contract; see
docs/design/accelerator-contract.md.
"""
from dataframe_provider import (
    installed,
    execute,
    describe,
    execute_profiled,
    select_auto,
    execute_auto,
    handoff_supports,
    handoff_execute,
)
