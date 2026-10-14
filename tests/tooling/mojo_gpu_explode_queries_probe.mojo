from tests.execution_modes.test_mojo_gpu_explode import (
    test_mojo_gpu_explode_source_layouts_hidden_spans_high_words_and_metadata,
    test_mojo_gpu_explode_resident_row_join_group_and_complete_cardinality_observation,
    test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery,
    test_mojo_gpu_explode_ieee_words_and_child_metadata_survive_projection,
)


def main() raises:
    test_mojo_gpu_explode_source_layouts_hidden_spans_high_words_and_metadata()
    print("PASS source layouts/high words/metadata")
    test_mojo_gpu_explode_resident_row_join_group_and_complete_cardinality_observation()
    print("PASS resident row/join/group/count errors")
    test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery()
    print("PASS empty/budget/observer/recovery")
    test_mojo_gpu_explode_ieee_words_and_child_metadata_survive_projection()
    print("PASS IEEE transport/alias metadata")
