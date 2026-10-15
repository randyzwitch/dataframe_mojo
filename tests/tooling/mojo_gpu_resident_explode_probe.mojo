from tests.support.resident_explode_reference import (
    layouts_and_resident_consumers,
    errors_mismatch_empty_and_recovery,
    exact_budget_and_mapped_views,
)


def main() raises:
    layouts_and_resident_consumers()
    print(
        "PASS resident EXPLODE literals/layouts/high planes/row/group consumers"
    )
    errors_mismatch_empty_and_recovery()
    print("PASS hidden/selected/parent errors/cardinality/empty/recovery")
    exact_budget_and_mapped_views()
    print("PASS exact budget and remapped physical column/row windows")
