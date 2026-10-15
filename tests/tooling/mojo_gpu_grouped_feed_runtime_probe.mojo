"""Public serial grouped feed runtime contracts; invoked directly, not as a package."""
from tests.execution_modes.test_mojo_gpu_grouped_feeds import (
    test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order,
    test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks,
    test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance,
)


def main() raises:
    test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order()
    print("PASS grouped feed batch/update/finish phases")
    test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks()
    print("PASS grouped feed AST/projection/mask order")
    test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance()
    print("PASS grouped feed wide values/empty/profile/budget/provenance")
