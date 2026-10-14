"""Focused prospective private IMPLODE ordered-row certification proof."""
from tests.execution_modes.test_mojo_gpu_ordered_implode import (
    test_mojo_gpu_ordered_implode_checked_categories_batches_and_occurrences,
    test_mojo_gpu_ordered_implode_masks_empty_and_later_eager_states,
    test_mojo_gpu_ordered_implode_resources_provenance_and_foreign_flags,
)


def main() raises:
    test_mojo_gpu_ordered_implode_checked_categories_batches_and_occurrences()
    print("PASS ordered IMPLODE checked categories/batches/occurrences")
    test_mojo_gpu_ordered_implode_masks_empty_and_later_eager_states()
    print("PASS ordered IMPLODE masks/empty/later eager states")
    test_mojo_gpu_ordered_implode_resources_provenance_and_foreign_flags()
    print("PASS ordered IMPLODE resources/provenance/foreign flags")
