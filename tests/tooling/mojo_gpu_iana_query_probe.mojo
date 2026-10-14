"""Focused public IANA field extraction proof; run with the Metal target."""
from tests.execution_modes.test_mojo_gpu_iana_fields import (
    test_mojo_gpu_iana_fields_literal_transitions_footer_extrema_and_nullable_layouts,
    test_mojo_gpu_iana_fields_descriptor_dedup_budgets_conversion_and_refusals,
)


def main() raises:
    test_mojo_gpu_iana_fields_literal_transitions_footer_extrema_and_nullable_layouts()
    print("PASS IANA literal fields/layouts/nulls/extrema/footer")
    test_mojo_gpu_iana_fields_descriptor_dedup_budgets_conversion_and_refusals()
    print("PASS IANA multiple tables/decimal descriptors/budgets/refusals")
