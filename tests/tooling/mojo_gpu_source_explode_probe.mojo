from tests.support.source_explode_reference import (
    layouts_clean_metadata_and_private_cache,
    reservations_exact_budget_and_early_admission,
)


def main() raises:
    layouts_clean_metadata_and_private_cache()
    print(
        "PASS source List transport/layouts/private cache/clean metadata/empty"
    )
    reservations_exact_budget_and_early_admission()
    print("PASS CPU staging reservation/exact budget/early refusal/recovery")
