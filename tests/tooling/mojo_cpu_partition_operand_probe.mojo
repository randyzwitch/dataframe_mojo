"""CPU metadata proof; no Scope or device initialization."""
from tests.public_contracts.test_partition_operand_metadata import (
    test_partition_operand_metadata_scalar_faults_masks_and_numeric_anchor,
)


def main() raises:
    test_partition_operand_metadata_scalar_faults_masks_and_numeric_anchor()
    print("PASS partition operand scalar-literal metadata/masks")
