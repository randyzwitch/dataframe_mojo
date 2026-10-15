from tests.execution_modes.test_mojo_gpu_partition_cuts import (
    test_public_partition_cut_qcut_resident_outputs,
    test_public_partition_cut_qcut_eager_fault_order,
    test_public_partition_cuts_original_categorical_keys,
)


def main() raises:
    test_public_partition_cut_qcut_resident_outputs()
    test_public_partition_cut_qcut_eager_fault_order()
    test_public_partition_cuts_original_categorical_keys()
