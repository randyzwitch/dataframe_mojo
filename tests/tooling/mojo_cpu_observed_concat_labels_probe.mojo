"""CPU-only metadata contracts for observed-pair concat label building."""
from tests.public_contracts.test_observed_concat_labels import (
    test_observed_concat_labels_sort_deduplicate_unicode_and_empty_text,
    test_observed_concat_labels_reject_bad_codes_and_pair_shapes,
    test_observed_concat_label_budget_rejection_and_recovery,
    test_observed_concat_labels_accept_empty_pair_domain,
)


def main() raises:
    test_observed_concat_labels_sort_deduplicate_unicode_and_empty_text()
    test_observed_concat_labels_reject_bad_codes_and_pair_shapes()
    test_observed_concat_label_budget_rejection_and_recovery()
    test_observed_concat_labels_accept_empty_pair_domain()
    print("PASS CPU observed text-pair concat label metadata")
