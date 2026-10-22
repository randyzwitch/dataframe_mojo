"""The Metal accelerator provider for a Metal build.

Put `providers/metal` ahead of the repository root on the import path
(`-I providers/metal -I .`) and `dataframe._accel_provider` resolves here.
"""
from dataframe_metal.provider import (
    installed,
    execute,
    describe,
    execute_profiled,
    select_auto,
    execute_auto,
)
