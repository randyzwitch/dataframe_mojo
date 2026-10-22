"""CPU-only regression for optional execution-profile region columns."""
from tests.execution_modes.test_metal_profile_phases import (
    test_metal_profile_regions_union_optional_metrics_in_first_seen_order,
)


def main() raises:
    test_metal_profile_regions_union_optional_metrics_in_first_seen_order()
