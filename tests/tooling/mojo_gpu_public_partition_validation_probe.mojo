"""Actual public partition validation, isolated from passing private probes."""
from std.testing import assert_equal
from tests.execution_modes.test_mojo_gpu_partition_cuts import (
    test_public_partition_cut_qcut_resident_outputs as outputs,
    test_public_partition_cut_qcut_eager_fault_order as faults,
    test_public_partition_cuts_original_categorical_keys as categories,
    test_public_partition_cuts_nested_numeric_scopes as nested,
    test_public_partition_when_head_and_sibling_error_contracts as contracts,
    test_public_partition_qcut_signed_zero_breakpoint_literals as zero_words,
    test_public_partition_native_execution_profile as _native_report,
)
from tests.public_contracts.test_partition_cut_capture import (
    test_partition_cut_capture_keeps_original_controls_and_eager_state as capture,
    test_partition_nested_scope_cache_entry_snapshots as cache,
)


def main() raises:
    var failures = 0
    try:
        capture()
        cache()
        print(
            "PASS public partition original certificates and cache snapshots",
            flush=True,
        )
    except error:
        failures += 1
        print("FAIL public partition metadata", String(error), flush=True)
    try:
        outputs()
        print("PASS public partition CUT/QCUT literal outputs", flush=True)
    except error:
        failures += 1
        print("FAIL public partition outputs", String(error), flush=True)
    try:
        faults()
        print(
            "PASS public partition eager/run operand-control priorities",
            flush=True,
        )
    except error:
        failures += 1
        print("FAIL public partition faults", String(error), flush=True)
    try:
        categories()
        print(
            "PASS public partition original category owners and decoded scope",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL public partition category owners", String(error), flush=True
        )
    try:
        nested()
        print("PASS public partition nested numeric scopes", flush=True)
    except error:
        failures += 1
        print("FAIL public partition nested scopes", String(error), flush=True)
    try:
        contracts()
        print(
            "PASS public partition WHEN/head/sibling literal error contracts",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL public partition WHEN/head/sibling contracts",
            String(error),
            flush=True,
        )
    try:
        zero_words()
        print(
            "PASS public partition literal signed-zero edge words", flush=True
        )
    except error:
        failures += 1
        print(
            "FAIL public partition signed-zero words", String(error), flush=True
        )
    try:
        _native_report()
        print(
            "PASS public partition actual Mojo GPU submissions and report",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL public partition native execution report",
            String(error),
            flush=True,
        )
    assert_equal(failures, 0)
