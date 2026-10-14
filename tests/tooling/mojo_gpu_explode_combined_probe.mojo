"""Raw owner/source and public query proofs for primitive List EXPLODE."""
from std.testing import assert_equal
from tests.support.resident_explode_reference import (
    layouts_and_resident_consumers,
    errors_mismatch_empty_and_recovery,
    exact_budget_and_mapped_views,
)
from tests.support.source_explode_reference import (
    layouts_clean_metadata_and_private_cache,
    reservations_exact_budget_and_early_admission,
)
from tests.execution_modes.test_mojo_gpu_explode import (
    test_mojo_gpu_explode_source_layouts_hidden_spans_high_words_and_metadata,
    test_mojo_gpu_explode_resident_row_join_group_and_complete_cardinality_observation,
    test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery,
    test_mojo_gpu_explode_ieee_words_and_child_metadata_survive_projection,
)


def raw() raises:
    layouts_and_resident_consumers()
    errors_mismatch_empty_and_recovery()
    exact_budget_and_mapped_views()


def source() raises:
    layouts_clean_metadata_and_private_cache()
    reservations_exact_budget_and_early_admission()


def public_queries() raises:
    test_mojo_gpu_explode_source_layouts_hidden_spans_high_words_and_metadata()
    test_mojo_gpu_explode_resident_row_join_group_and_complete_cardinality_observation()
    test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery()
    test_mojo_gpu_explode_ieee_words_and_child_metadata_survive_projection()


def main() raises:
    var failures = 0
    try:
        raw()
        print("PASS resident EXPLODE storage/controls/consumers", flush=True)
    except error:
        failures += 1
        print(
            "FAIL resident EXPLODE storage/controls/consumers",
            String(error),
            flush=True,
        )
    try:
        source()
        print("PASS primitive source List staging/owners", flush=True)
    except error:
        failures += 1
        print(
            "FAIL primitive source List staging/owners",
            String(error),
            flush=True,
        )
    try:
        public_queries()
        print("PASS public EXPLODE four query contracts", flush=True)
    except error:
        failures += 1
        print(
            "FAIL public EXPLODE four query contracts",
            String(error),
            flush=True,
        )
    assert_equal(failures, 0)
