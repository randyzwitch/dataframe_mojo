from tests.public_contracts.test_grouped_key_domains import (
    test_grouped_key_domain_original_categorical_and_coded_string_identity,
    test_grouped_key_domain_independent_raw_hashes_restore_and_null_poison,
    test_grouped_key_domain_cpu_worker_bucket_heavy_fault_order,
)


def main() raises:
    test_grouped_key_domain_original_categorical_and_coded_string_identity()
    print(
        "PASS CPU original category/codedString/duplicate identity", flush=True
    )
    test_grouped_key_domain_independent_raw_hashes_restore_and_null_poison()
    print(
        "PASS CPU independent typed raw hashes/bytes/nullable poison",
        flush=True,
    )
    test_grouped_key_domain_cpu_worker_bucket_heavy_fault_order()
    print(
        "PASS CPU workers1/2/4/8 raw-key bucket/heavy fault priority",
        flush=True,
    )
