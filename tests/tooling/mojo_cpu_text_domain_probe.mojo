"""Context-free private text-domain metadata contracts."""
from tests.public_contracts.test_dynamic_text_domains import (
    test_dynamic_text_domain_union_maps_are_metadata_and_private,
    test_dynamic_text_domains_keep_original_categorical_and_hash_provenance,
    test_dynamic_text_domain_guards_precede_device_creation,
    test_dynamic_qcut_label_controls_match_cpu_literal_spelling,
)


def main() raises:
    test_dynamic_text_domain_union_maps_are_metadata_and_private()
    test_dynamic_text_domains_keep_original_categorical_and_hash_provenance()
    test_dynamic_text_domain_guards_precede_device_creation()
    test_dynamic_qcut_label_controls_match_cpu_literal_spelling()
    print("PASS CPU private text-domain owners/union/maps/provenance")
