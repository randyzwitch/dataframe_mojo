from tests.support.source_explode_reference import (
    layouts_clean_metadata_and_private_cache,
    reservations_exact_budget_and_early_admission,
    simultaneous_scratch_and_output_literal_budget,
)


def main() raises:
    layouts_clean_metadata_and_private_cache()
    print(
        "PASS source List transport/layouts/private cache/clean metadata/empty"
    )
    reservations_exact_budget_and_early_admission()
    simultaneous_scratch_and_output_literal_budget()
    print("PASS CPU staging reservation/exact budget/early refusal/recovery")
