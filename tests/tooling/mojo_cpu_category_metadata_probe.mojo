"""Unknown-accelerator metadata teardown and context-free category leases."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.accelerator.gpu.residency import Scope, CategoryLease
from tests.execution_modes.test_backend_metal_utc_rolling import (
    test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight,
)


def main() raises:
    test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight()
    print("PASS category metadata Program teardown without GPU target")
    var scope = Scope(2147483647, 32)
    var lease = ArcPointer(CategoryLease(scope, 32))
    var shared = lease.copy()
    assert_equal(scope.info()[0], 32)
    assert_equal(scope.info()[1], 32)
    assert_true(not scope.initialized())
    _ = lease^
    assert_equal(scope.info()[0], 32)
    _ = shared^
    assert_equal(scope.info()[0], 0)
    with assert_raises(contains="memory budget"):
        _ = CategoryLease(scope, 33)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 32)
    assert_true(not scope.initialized())
    print(
        "PASS category lease shared lifetime and rejected budget without context"
    )
