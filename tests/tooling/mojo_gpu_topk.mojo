"""Focused real-device gate for bounded top-K ordering."""
from tests.execution_modes.test_mojo_gpu_ordering import (
    test_mojo_gpu_ordering_topk_composite_stability_layout_and_budget,
)


def main() raises:
    test_mojo_gpu_ordering_topk_composite_stability_layout_and_budget()
    print("PASS bounded top-K comparator, boundaries, stability and budgets")
