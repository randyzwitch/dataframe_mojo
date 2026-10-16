"""Maintained wrappers for the unchanged private partition device probes."""
from tests.support.partition_device_ranked_probe import run_probe as ranked
from tests.support.partition_device_schedule_probe import run_probe as schedule
from tests.support.partition_device_fault_probe import run_probe as faults
from tests.support.partition_device_owner_probe import run_probe as owners
from tests.support.partition_device_bins_probe import run_probe as publication


def test_mojo_gpu_partition_ranked_rows_owners_and_resources() raises:
    ranked()


def test_mojo_gpu_partition_exact_cpu_schedules() raises:
    schedule()


def test_mojo_gpu_partition_run_state_operand_fault_order() raises:
    faults()


def test_mojo_gpu_partition_layout_certificates_and_owners() raises:
    owners()


def test_mojo_gpu_partition_private_cut_qcut_publication() raises:
    publication()
