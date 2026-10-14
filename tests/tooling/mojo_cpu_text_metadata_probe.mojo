"""Pure CPU metadata proof: no Scope, DeviceContext or device allocation."""
from tests.public_contracts.test_text_domain_metadata import (
    test_text_domain_metadata_owners_and_logical_literal_tags,
    test_text_domain_metadata_keeps_raw_hash_and_original_code_provenance,
)


def main() raises:
    test_text_domain_metadata_owners_and_logical_literal_tags()
    test_text_domain_metadata_keeps_raw_hash_and_original_code_provenance()
    print("PASS2 pure CPU text-domain metadata contracts")
