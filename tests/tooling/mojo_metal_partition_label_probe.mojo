"""Metal-compilation-target host label metadata, no actual device required."""
from tests.public_contracts.test_partition_qcut_label_domains import (
    test_partition_qcut_label_domains_literal_mapping_owner_budget_and_guards,
)


def main() raises:
    test_partition_qcut_label_domains_literal_mapping_owner_budget_and_guards()
    print("PASS partition label literal domains/budget/guards")
