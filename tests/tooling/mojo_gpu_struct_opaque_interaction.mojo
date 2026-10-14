"""Focused cross-adapter proof; package test module has no main()."""
from tests.execution_modes.test_mojo_gpu_struct_opaque_interaction import (
    test_mojo_gpu_struct_identity_and_opaque_transport_share_owners_masks_and_budget,
)


def main() raises:
    test_mojo_gpu_struct_identity_and_opaque_transport_share_owners_masks_and_budget()
    print(
        "PASS Struct identity/opaque owners, masks, layouts, budgets and recovery"
    )
