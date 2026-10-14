"""Focused public opaque-byte transport proof."""
from tests.execution_modes.test_mojo_gpu_opaque_bytes import (
    test_mojo_gpu_binary_opaque_raw_bytes_literals_stable_orders_layouts_and_metadata,
    test_mojo_gpu_binary_join_grouped_windows_and_raw_hash_provenance,
    test_mojo_gpu_geometry_transport_preserves_wkb_identity_crs_and_field_metadata,
    test_mojo_gpu_opaque_budgets_observers_empty_inputs_and_explicit_refusals,
)


def main() raises:
    test_mojo_gpu_binary_opaque_raw_bytes_literals_stable_orders_layouts_and_metadata()
    print("PASS Binary raw bytes/literals/orders/layouts/metadata")
    test_mojo_gpu_binary_join_grouped_windows_and_raw_hash_provenance()
    print("PASS Binary joins/groups/windows/raw hashes")
    test_mojo_gpu_geometry_transport_preserves_wkb_identity_crs_and_field_metadata()
    print("PASS Geometry byte identity/CRS/metadata")
    test_mojo_gpu_opaque_budgets_observers_empty_inputs_and_explicit_refusals()
    print("PASS opaque budgets/observers/empty/refusals")
