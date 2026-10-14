"""Coordinated isolated CUT normalization/public execution driver."""
from tests.public_contracts.test_cut_lowering import (
    test_cut_lowering_preserves_string_struct_schema_and_observed_producers,
    test_cut_lowering_defers_edge_order_but_keeps_binder_failure,
    test_cut_lowering_checked_scalar_metadata_uses_empty_literal_anchor,
    test_cut_lowering_literal_window_input_rejects_before_any_column_access,
)
from tests.execution_modes.test_mojo_gpu_cut import (
    test_mojo_gpu_cut_literals_default_labels_nullable_layouts_and_struct_breakpoints,
    test_mojo_gpu_cut_single_nan_edge_empty_edges_filter_and_computed_operands,
    test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery,
    test_mojo_gpu_cut_budget_profiles_and_pending_families_are_explicit,
    test_mojo_gpu_cut_scalar_reduction_siblings_observe_expression_order_and_empty_errors,
    test_mojo_gpu_cut_literal_only_window_rejections_and_literal_sibling_broadcast,
)
from tests.public_contracts.test_ordered_scalar_metadata import (
    test_cut_ordered_scalar_rank_transport_and_resource_metadata,
)
from tests.execution_modes.test_mojo_gpu_rows import (
    test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values,
)


def main() raises:
    test_cut_ordered_scalar_rank_transport_and_resource_metadata()
    print("PASS CUT ordered ranks/resource metadata", flush=True)
    test_cut_lowering_preserves_string_struct_schema_and_observed_producers()
    print("PASS CUT metadata schema/producers", flush=True)
    test_cut_lowering_defers_edge_order_but_keeps_binder_failure()
    print("PASS CUT deferred controls", flush=True)
    test_cut_lowering_checked_scalar_metadata_uses_empty_literal_anchor()
    print("PASS CUT scalar metadata anchors", flush=True)
    test_cut_lowering_literal_window_input_rejects_before_any_column_access()
    print("PASS CUT literal-only binder metadata", flush=True)
    test_mojo_gpu_cut_literals_default_labels_nullable_layouts_and_struct_breakpoints()
    print("PASS CUT literal layouts/structs", flush=True)
    test_mojo_gpu_cut_single_nan_edge_empty_edges_filter_and_computed_operands()
    print("PASS CUT special edges/composition", flush=True)
    test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery()
    print("PASS CUT eager errors/recovery", flush=True)
    test_mojo_gpu_cut_budget_profiles_and_pending_families_are_explicit()
    print("PASS CUT budgets/refusals", flush=True)
    test_mojo_gpu_cut_scalar_reduction_siblings_observe_expression_order_and_empty_errors()
    print("PASS CUT scalar/reduction order/empty/budgets", flush=True)
    test_mojo_gpu_cut_literal_only_window_rejections_and_literal_sibling_broadcast()
    print(
        "PASS CUT literal-only binder/no device and scalar broadcast",
        flush=True,
    )
    test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values()
    print("PASS ordinary row error flags unchanged", flush=True)
