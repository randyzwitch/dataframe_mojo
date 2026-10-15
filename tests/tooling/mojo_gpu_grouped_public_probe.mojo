from std.testing import assert_equal
from tests.execution_modes.test_mojo_gpu_grouped_lists import (
    test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency,
    test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget,
    test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards,
    test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order,
)
from tests.execution_modes.test_mojo_gpu_grouped_feeds import (
    test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order,
    test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks,
    test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance,
)
from tests.execution_modes.test_mojo_gpu_grouped_key_domains import (
    test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts,
    test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority,
    test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission,
)


def main() raises:
    var failures = 0
    print(
        "RUN test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency()
        print(
            "PASS test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget()
        print(
            "PASS test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards()
        print(
            "PASS test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order()
        print(
            "PASS test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order()
        print(
            "PASS test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks()
        print(
            "PASS test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance()
        print(
            "PASS test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts()
        print(
            "PASS test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority()
        print(
            "PASS test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority",
            String(error),
            flush=True,
        )
    print(
        "RUN test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission",
        flush=True,
    )
    try:
        test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission()
        print(
            "PASS test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission",
            String(error),
            flush=True,
        )
    assert_equal(failures, 0)
