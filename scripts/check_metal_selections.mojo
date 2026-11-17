"""Direct native-device checks for the CPU selection handoff."""
from tests.execution_modes.test_metal_selections import (
    test_metal_selection_input_owners_and_shared_row_lists,
    test_metal_selection_entrypoints_reject_invalid_maps_before_submission,
    test_metal_selected_reductions_types_slices_nulls_duplicates_and_constants,
    test_metal_selected_float_specials_and_integer_faults,
)


def main() raises:
    test_metal_selection_input_owners_and_shared_row_lists()
    test_metal_selection_entrypoints_reject_invalid_maps_before_submission()
    test_metal_selected_reductions_types_slices_nulls_duplicates_and_constants()
    test_metal_selected_float_specials_and_integer_faults()
    print("PASS native selected reductions, ownership, nulls and faults")
