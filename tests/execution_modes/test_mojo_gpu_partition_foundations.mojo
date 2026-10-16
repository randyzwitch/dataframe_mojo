"""Maintained wrappers for the unchanged private partition device probes."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from tests.support.partition_device_ranked_probe import run_probe as ranked
from tests.support.partition_device_schedule_probe import run_probe as schedule
from tests.support.partition_device_fault_probe import run_probe as faults
from tests.support.partition_device_owner_probe import run_probe as owners
from tests.support.partition_device_bins_probe import run_probe as publication


# requires: metal
def test_mojo_gpu_partition_ranked_rows_owners_and_resources() raises:
    ranked()


# requires: metal
def test_mojo_gpu_partition_exact_cpu_schedules() raises:
    schedule()


# requires: metal
def test_mojo_gpu_partition_run_state_operand_fault_order() raises:
    faults()


# requires: metal
def test_mojo_gpu_partition_layout_certificates_and_owners() raises:
    owners()


# requires: metal
def test_mojo_gpu_partition_private_cut_qcut_publication() raises:
    publication()
