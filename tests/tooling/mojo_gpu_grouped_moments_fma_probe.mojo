"""Independent exact-bit grouped List/moments sibling regression."""
from tests.execution_modes.test_mojo_gpu_grouped_moments_fma import (
    test_mojo_gpu_grouped_implode_var_std_siblings_local_fma_literal,
)


def main() raises:
    test_mojo_gpu_grouped_implode_var_std_siblings_local_fma_literal()
    print(
        "PASS grouped List VAR/STD local-FMA CPU/GPU literal layouts",
        flush=True,
    )
