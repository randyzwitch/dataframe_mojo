"""The default accelerator provider: Metal.

`dataframe/_accel_provider.mojo` imports these twelve functions. On a host
without Metal `installed()` is False and collection stays on the CPU. For a
build that compiles no GPU code, put `providers/cpu` first on the import path
(`-I providers/cpu -I .`). Another backend ships its own `dataframe_provider`
package and puts it ahead of this one.
"""
from dataframe_metal.provider import (
    installed,
    execute,
    describe,
    execute_profiled,
    handoff_supports,
    handoff_execute,
    handoff_estimate,
    devices,
    transfer_model,
    pipeline_supports,
    pipeline_estimate,
    run_pipeline,
)
