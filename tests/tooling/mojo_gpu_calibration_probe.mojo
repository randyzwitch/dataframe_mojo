"""Independent direct Mojo cost hooks and complete CPU/GPU timing probe."""
from tests.execution_modes.test_mojo_gpu_calibration import (
    test_mojo_gpu_calibration_metadata,
    test_mojo_gpu_calibration_isolation,
    test_mojo_gpu_calibration_measurements,
)


def main() raises:
    test_mojo_gpu_calibration_metadata()
    test_mojo_gpu_calibration_isolation()
    test_mojo_gpu_calibration_measurements()
    print(
        "PASS Mojo calibration metadata, curve isolation and real CPU/GPU timing"
    )
