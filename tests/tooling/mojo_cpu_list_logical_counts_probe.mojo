from tests.public_contracts.test_list_logical_counts import (
    test_list_logical_count_null_hidden_span_empty_and_layouts,
    test_list_physical_backing_take_concat_slice_preserves_logical_counts,
    test_explode_multiple_lists_compare_logical_counts_before_placeholders,
)


def main() raises:
    test_list_logical_count_null_hidden_span_empty_and_layouts()
    print("PASS logical null/empty counts and layouts")
    test_list_physical_backing_take_concat_slice_preserves_logical_counts()
    print("PASS physical take/concat/slice backing")
    test_explode_multiple_lists_compare_logical_counts_before_placeholders()
    print("PASS multiple List logical counts and mismatch observation")
