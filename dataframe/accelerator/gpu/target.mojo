"""Compile-time Metal eligibility without instantiating an unknown GPU target."""
from max.gpu.host import default_accelerator


comptime METAL_TARGET = default_accelerator().is_apple_gpu()
