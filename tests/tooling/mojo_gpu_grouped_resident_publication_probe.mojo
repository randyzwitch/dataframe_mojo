from tests.support.resident_grouped_publication_reference import (
    layouts_slices_mixed_siblings_and_resident_consumers,
    empty_foreign_owner_checked_phase_order_and_exact_budget,
)


def main() raises:
    layouts_slices_mixed_siblings_and_resident_consumers()
    print("PASS grouped resident List owners/layouts/slices/consumers")
    empty_foreign_owner_checked_phase_order_and_exact_budget()
    print("PASS grouped empty/owner guards/phase order/budgets/recovery")
