from tests.execution_modes.test_mojo_gpu_grouped_key_domains import (
    test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts,
    test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority,
    test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission,
)


def main() raises:
    test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts()
    print("PASS original key domains, duplicate codes, byte keys and layouts")
    test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority()
    print("PASS original key hash worker/bucket/heavy fault priority")
    test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission()
    print(
        "PASS key staging budget, empty, checked feeds and metadata admission"
    )
