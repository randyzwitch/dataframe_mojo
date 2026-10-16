from tests.execution_modes.test_mojo_gpu_grouped_typed_explode import (
    test_grouped_typed_expand_literal_raw_domains_and_consumers,
    test_grouped_typed_expand_duplicate_categorical_and_coded_string_identity,
)


def main() raises:
    test_grouped_typed_expand_literal_raw_domains_and_consumers()
    print("PASS typed grouped EXPLODE raw domains ROW/JOIN/GROUP/OVER")
    test_grouped_typed_expand_duplicate_categorical_and_coded_string_identity()
    print(
        "PASS typed grouped EXPLODE original categorical/coded String identity"
    )
