from std.testing import assert_equal
from tests.support.resident_grouped_publication_reference import (
    layouts_slices_mixed_siblings_and_resident_consumers,
    empty_foreign_owner_checked_phase_order_and_exact_budget,
)

from tests.execution_modes.test_mojo_gpu_grouped_explode import (
    test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer,
    test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure,
    test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget,
)


def main() raises:
    var failures = 0
    print("RUN layouts_slices_mixed_siblings_and_resident_consumers")
    try:
        layouts_slices_mixed_siblings_and_resident_consumers()
        print("PASS layouts_slices_mixed_siblings_and_resident_consumers")
    except error:
        failures += 1
        print(
            "FAIL layouts_slices_mixed_siblings_and_resident_consumers", error
        )
    print("RUN empty_foreign_owner_checked_phase_order_and_exact_budget")
    try:
        empty_foreign_owner_checked_phase_order_and_exact_budget()
        print("PASS empty_foreign_owner_checked_phase_order_and_exact_budget")
    except error:
        failures += 1
        print(
            "FAIL empty_foreign_owner_checked_phase_order_and_exact_budget",
            error,
        )
    print(
        "RUN test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer"
    )
    try:
        test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer()
        print(
            "PASS test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer"
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer",
            error,
        )
    print(
        "RUN test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure"
    )
    try:
        test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure()
        print(
            "PASS test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure"
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure",
            error,
        )
    print(
        "RUN test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget"
    )
    try:
        test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget()
        print(
            "PASS test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget"
        )
    except error:
        failures += 1
        print(
            "FAIL test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget",
            error,
        )
    assert_equal(failures, 0, "grouped resident+EXPLODE proofs")
