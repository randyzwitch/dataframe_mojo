"""Independent raw foundations before public partitioned CUT/QCUT admission."""
from std.testing import assert_equal
from tests.public_contracts.test_partition_cut_capture import (
    test_partition_cut_capture_keeps_original_controls_and_eager_state,
)
from tests.support.partition_device_ranked_probe import run_probe as ranked_rows
from tests.support.partition_device_schedule_probe import (
    run_probe as schedule,
)
from tests.support.partition_device_fault_probe import run_probe as faults
from tests.support.partition_device_owner_probe import run_probe as owner
from tests.public_contracts.test_partition_operand_metadata import (
    test_partition_operand_metadata_scalar_faults_masks_and_numeric_anchor as operand,
)
from tests.public_contracts.test_partition_qcut_label_domains import (
    test_partition_qcut_label_domains_literal_mapping_owner_budget_and_guards as labels,
)
from tests.support.partition_device_bins_probe import run_probe as publication


def main() raises:
    var failures = 0
    try:
        ranked_rows()
        print("PASS private ranked row ownership/resources", flush=True)
    except error:
        failures += 1
        print(
            "FAIL private ranked row ownership/resources",
            String(error),
            flush=True,
        )
    try:
        schedule()
        print("PASS exact OVER partition schedules", flush=True)
    except error:
        failures += 1
        print("FAIL exact OVER partition schedules", String(error), flush=True)
    try:
        faults()
        print("PASS partition run/stage fault selector", flush=True)
    except error:
        failures += 1
        print(
            "FAIL partition run/stage fault selector", String(error), flush=True
        )
    try:
        owner()
        print("PASS partition source mappings/offsets/owners", flush=True)
    except error:
        failures += 1
        print(
            "FAIL partition source mappings/offsets/owners",
            String(error),
            flush=True,
        )
    try:
        operand()
        print(
            "PASS original partition operand metadata certificate", flush=True
        )
    except error:
        failures += 1
        print(
            "FAIL original partition operand metadata certificate",
            String(error),
            flush=True,
        )
    try:
        labels()
        print("PASS partition labels/immutable domains/resources", flush=True)
    except error:
        failures += 1
        print(
            "FAIL partition labels/immutable domains/resources",
            String(error),
            flush=True,
        )
    try:
        publication()
        print("PASS resident partition CUT/QCUT publication", flush=True)
    except error:
        failures += 1
        print(
            "FAIL resident partition CUT/QCUT publication",
            String(error),
            flush=True,
        )
    try:
        test_partition_cut_capture_keeps_original_controls_and_eager_state()
        print("PASS original public partition capture", flush=True)
    except error:
        failures += 1
        print(
            "FAIL original public partition capture", String(error), flush=True
        )
    assert_equal(failures, 0)
