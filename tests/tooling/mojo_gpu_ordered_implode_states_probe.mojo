"""Focused prospective post-state IMPLODE ordered-row proof."""
from tests.execution_modes.test_mojo_gpu_ordered_implode_states import (
    test_mojo_gpu_ordered_implode_post_state_ast_categories_and_batches,
    test_mojo_gpu_ordered_implode_prepared_state_order_masks_and_projections,
    test_mojo_gpu_ordered_implode_state_compositions_resources_empty_provenance,
)


def main() raises:
    test_mojo_gpu_ordered_implode_post_state_ast_categories_and_batches()
    print("PASS post-state IMPLODE categories/batches")
    test_mojo_gpu_ordered_implode_prepared_state_order_masks_and_projections()
    print("PASS IMPLODE state preparation/masks/projection order")
    test_mojo_gpu_ordered_implode_state_compositions_resources_empty_provenance()
    print("PASS IMPLODE state compositions/resources/empty/provenance")
