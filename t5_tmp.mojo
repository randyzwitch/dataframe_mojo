from tests.kernel_equivalence.test_vector_expressions import test_filters_select_what_the_eager_filter_keeps, test_projections_equal_the_eager_columns, test_a_dropped_rows_error_is_not_raised
from tests.storage_ownership.test_vectors import *
from tests.kernel_equivalence.test_join_view_selection import *

def main() raises:
    test_flat_window_reads_storage_in_place()
    test_dictionary_slices_merge_and_gather_on_the_way_out()
    test_strings_decode_to_string_t_at_the_rows_read()
    test_string_t_compares_by_bytes()
    test_constant_and_sequence_vectors()
    test_booleans_decode_and_reset_reuses_buffers()
    print("core ok")
    test_filters_select_what_the_eager_filter_keeps()
    test_projections_equal_the_eager_columns()
    test_a_dropped_rows_error_is_not_raised()
    print("expr ok")
    test_selected_rows_read_as_the_rows_they_select()
    test_joined_build_strings_match_the_scalar_model()
    print("join strings ok")
