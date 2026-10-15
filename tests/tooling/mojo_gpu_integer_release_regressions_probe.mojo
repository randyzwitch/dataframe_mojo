"""Five original full-Metal release regressions; no capability exemptions."""
from std.testing import assert_equal
from tests.execution_modes.test_backend_metal_distinct_siblings import (
    test_native_metal_distinct_siblings_sum_and_checked_child_restrictions,
)
from tests.execution_modes.test_backend_metal_domains import (
    test_native_metal_aggregate_compositions_and_consumers_stay_resident,
)
from tests.execution_modes.test_backend_metal_partition_aggregates import (
    test_native_metal_partition_window_checked_children_and_bool_reductions,
)
from tests.execution_modes.test_mojo_gpu_cut import (
    test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery,
)
from tests.execution_modes.test_mojo_gpu_qcut import (
    test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery,
)


def main() raises:
    var failures = 0
    print(
        "RUN 1/5 test_native_metal_distinct_siblings_sum_and_checked_child_restrictions",
        flush=True,
    )
    try:
        test_native_metal_distinct_siblings_sum_and_checked_child_restrictions()
        print(
            "PASS 1/5 test_native_metal_distinct_siblings_sum_and_checked_child_restrictions",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 1/5 test_native_metal_distinct_siblings_sum_and_checked_child_restrictions",
            String(error),
            flush=True,
        )
    print(
        "RUN 2/5 test_native_metal_aggregate_compositions_and_consumers_stay_resident",
        flush=True,
    )
    try:
        test_native_metal_aggregate_compositions_and_consumers_stay_resident()
        print(
            "PASS 2/5 test_native_metal_aggregate_compositions_and_consumers_stay_resident",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 2/5 test_native_metal_aggregate_compositions_and_consumers_stay_resident",
            String(error),
            flush=True,
        )
    print(
        "RUN 3/5 test_native_metal_partition_window_checked_children_and_bool_reductions",
        flush=True,
    )
    try:
        test_native_metal_partition_window_checked_children_and_bool_reductions()
        print(
            "PASS 3/5 test_native_metal_partition_window_checked_children_and_bool_reductions",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 3/5 test_native_metal_partition_window_checked_children_and_bool_reductions",
            String(error),
            flush=True,
        )
    print(
        "RUN 4/5 test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery",
        flush=True,
    )
    try:
        test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery()
        print(
            "PASS 4/5 test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 4/5 test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery",
            String(error),
            flush=True,
        )
    print(
        "RUN 5/5 test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery",
        flush=True,
    )
    try:
        test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery()
        print(
            "PASS 5/5 test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 5/5 test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery",
            String(error),
            flush=True,
        )
    assert_equal(failures, 0)
