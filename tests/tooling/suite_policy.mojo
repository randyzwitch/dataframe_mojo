"""Execution exceptions and convenient module groups for the maintained suite.

Tests are discovered from source; this policy neither registers ordinary tests
nor describes coverage. Remove an expected failure when its exact defect is fixed.
"""

from std.sys import CompilationTarget


@fieldwise_init
struct ExpectedFailure(Copyable, Movable):
    var entry: String
    var reason: String
    var diagnostic: String
    var exit_code: Int


@fieldwise_init
struct CapabilityRequirement(Copyable, Movable):
    var entry: String
    var capability: String


@fieldwise_init
struct ContractGroup(Copyable, Movable):
    var name: String
    var modules: List[String]


@fieldwise_init
struct SuitePolicy(Copyable, Movable):
    var failures: List[ExpectedFailure]
    var requirements: List[CapabilityRequirement]
    var groups: List[ContractGroup]
    var capabilities: List[String]

    def failure_index(self, entry: String) -> Int:
        for i in range(len(self.failures)):
            if self.failures[i].entry == entry:
                return i
        return -1


def empty_policy() -> SuitePolicy:
    return SuitePolicy(
        List[ExpectedFailure](),
        List[CapabilityRequirement](),
        List[ContractGroup](),
        List[String](),
    )


def mojo_variant_abort_exit_code() -> Int:
    # The same checked variant abort traps as SIGILL on Linux and SIGTRAP on
    # Apple Silicon macOS. Each known fatal defect still needs its exact
    # diagnostic and the one status for the platform executing the suite.
    return 133 if CompilationTarget.is_macos() else 132


def production_policy() -> SuitePolicy:
    var policy = empty_policy()
    policy.capabilities = ["metal"]
    policy.failures.append(
        ExpectedFailure(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_extreme_date_epoch_shifts_keep_independent_gregorian_fields",
            "Native Date calendar extraction wraps the Int64 epoch shift, matching the separately recorded CPU defect. An independent Int128 Gregorian model requires positive year25252734927766554; actual Metal CI returns -25252734927766555. Later calendar fields remain unvalidated until this defect is fixed.",
            "native Date Gregorian year must remain positive across epoch shift; got -25252734927766555",
            0,
        )
    )
    # exit_code zero means an ordinary exception; nonzero means an isolated process.
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reshape.mojo::test_explode_null_list_never_exposes_hidden_child_span",
            "A null list with a nonempty stored child span currently explodes the hidden children instead of producing one null row.",
            "null-list explode must produce exactly one row",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_window_types.mojo::test_uint64_cumulative_sums_preserve_values_above_signed_range",
            "UInt64 cumulative sums reject valid values above Int64.MAX despite a UInt64 result type.",
            "uint64 window sum overflow",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_window_types.mojo::test_uint64_rolling_sums_preserve_values_above_signed_range",
            "UInt64 rolling sums reject valid values above Int64.MAX despite a UInt64 result type.",
            "uint64 window sum overflow",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_nested.mojo::test_struct_null_parent_key_equality_ignores_hidden_field_payloads",
            "Grouping null struct keys distinguishes hidden child payloads.",
            "null struct keys must ignore hidden field payloads",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_nested.mojo::test_struct_join_matches_valid_records_with_null_fields",
            "Equal valid struct keys containing null fields do not match in equality joins.",
            "valid struct join keys must match equal null fields",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_order_windows.mojo::test_coordinate_interpolation_accepts_chunked_equal_length_columns",
            "Coordinate interpolation does not handle chunked columns with equal logical lengths.",
            "Column values and validity must have equal lengths",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal.mojo::test_temporal_subtraction_equal_minimum_ticks_cancel_exactly",
            "Subtracting equal minimum Int64 temporal ticks raises overflow even though the exact difference is zero.",
            "temporal value overflows Int64",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal.mojo::test_temporal_subtraction_rejects_unrepresentable_minimum_rhs",
            "Zero minus minimum Int64 temporal ticks succeeds with a wrapped result instead of rejecting the unrepresentable exact difference.",
            "minimum temporal RHS must reject an out-of-range difference",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal.mojo::test_temporal_strftime_escaped_percent_formats_one_literal_percent",
            "Temporal strftime emits two percent characters for the documented %% escape.",
            "strftime escaped percent must emit one literal percent",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal.mojo::test_duration_minimum_tick_totals_keep_the_negative_sign",
            "Whole seconds of the minimum Int64 nanosecond duration do not retain the correct negative, truncated count.",
            "minimum duration whole seconds must truncate toward zero without sign overflow",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_inspection.mojo::test_empty_selector_expansion_preserves_input_height",
            "An empty selector expansion collapses input height to one.",
            "empty selector expansion must preserve input height",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_inspection.mojo::test_all_selector_preserves_zero_column_frame_height",
            "Selecting all columns of a zero-column frame collapses its height to one.",
            "all selector must preserve zero-column input height",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_inspection.mojo::test_describe_lists_reports_only_parent_counts",
            "describe() rejects lists instead of returning documented parent counts and null summary statistics.",
            "count does not support list or struct expressions yet, found list[int64]",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_inspection.mojo::test_describe_structs_reports_only_parent_counts",
            "describe() rejects structs instead of returning documented parent counts and null summary statistics.",
            "count does not support list or struct expressions yet, found struct[x: int64]",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/planner_correctness/test_errors_schema.mojo::test_filter_pushdown_preserves_preceding_arithmetic_errors",
            "Predicate pushdown suppresses a checked arithmetic overflow that occurs before the later filter in the unoptimized plan.",
            "optimizer must preserve an overflow before a later filter",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_arrow.mojo::test_arrow_dictionary_null_values_decode_as_null_cells",
            "Valid Arrow dictionary indices that refer to null dictionary values raise instead of yielding null cells.",
            "Cannot read a null value",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_arrow.mojo::test_arrow_null_seconds_payload_does_not_overflow_unit_conversion",
            "Null second-unit temporal payloads are rescaled and overflow instead of remaining null.",
            "Arrow seconds value overflows milliseconds",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_arrow.mojo::test_arrow_empty_numeric_array_rejects_missing_required_buffer_slot",
            "Malformed empty Int64 C layout with one buffer slot is accepted instead of rejected.",
            "Arrow numeric layout must reject missing required buffer slot",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_arithmetic.mojo::test_float32_modulo_infinite_divisor_matches_selected_python_policy",
            "Finite floating remainder with an infinite divisor returns NaN instead of the user-selected Python divisor-sign policy.",
            "finite modulo infinite divisor must follow Python remainder, not NaN",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_arithmetic.mojo::test_float64_modulo_infinite_divisor_matches_selected_python_policy",
            "Finite floating remainder with an infinite divisor returns NaN instead of the user-selected Python divisor-sign policy.",
            "finite modulo infinite divisor must follow Python remainder, not NaN",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_selection_types.mojo::test_decimal_bare_filter_literals_adopt_numeric_column_type",
            "Bare integer filter literals do not adopt decimal operand dtype during evaluation for any decimal storage width.",
            "decimal bare filter literals must adopt operand dtype",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_uint64_quantiles_do_not_expose_internal_ordering_bias",
            "Advertised public reduction does not satisfy its documented numeric contract.",
            "UInt64 percentile must not expose signed ordering bias",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_skew_accepts_advertised_numeric_input",
            "Advertised public reduction does not satisfy its documented numeric contract.",
            "expected a numeric column",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_kurtosis_accepts_advertised_numeric_input",
            "Advertised public reduction does not satisfy its documented numeric contract.",
            "expected a numeric column",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_pair_reductions.mojo::test_decimal_pearson_accepts_advertised_numeric_pairs",
            "Verified advertised decimal pair input is rejected by numeric binding.",
            "expected a numeric column",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_pair_reductions.mojo::test_decimal_spearman_accepts_advertised_numeric_pairs",
            "Verified advertised decimal pair input is rejected by numeric binding.",
            "expected a numeric column",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_pair_reductions.mojo::test_decimal_covariance_accepts_advertised_numeric_pairs",
            "Verified advertised decimal pair input is rejected by numeric binding.",
            "expected a numeric column",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/storage_ownership/test_typed_storage.mojo::test_date_export_does_not_narrow_hidden_null_payloads",
            "Arrow date32 export narrows hidden null Int64.MAX payloads and raises instead of preserving valid visible dates plus nulls.",
            "date export must ignore hidden null payloads",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_cast_boundaries.mojo::test_categorical_recode_does_not_require_unused_source_members_in_target",
            "Explicit source dictionary has an unused absent entry; all observed values fit target dictionary. Ordinary DictKeyError[String]() verified.",
            "DictKeyError[String]()",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_cast_boundaries.mojo::test_categorical_recode_relaxed_conversion_nulls_missing_observed_values",
            "strict=False must null an observed value missing from target dictionary and keep representable rows. Ordinary DictKeyError[String]() verified.",
            "DictKeyError[String]()",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_cast_boundaries.mojo::test_binary_to_categorical_rejects_unsupported_target_even_for_valid_utf8",
            "Explicit docs binary-only-to-String boundary, strict false/true and observed valid/invalid byte/null matrix. Unsupported target must reject; first reached successful valid ASCII conversion isolates failure.",
            "binary to categorical must reject unsupported target",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_cast_boundaries.mojo::test_expression_categorical_cast_retains_explicit_target_dictionary",
            "Explicit target dictionary includes unused member and reordered codes. Own assertion: expression categorical cast must retain supplied target dictionary. Verified first dtype assertion; later code assertions await repair.",
            "expression categorical cast must retain supplied target dictionary",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_parquet.mojo::test_parquet_reader_preserves_independent_arbitrary_field_metadata",
            "Public Parquet read drops arbitrary field metadata independently stored by PyArrow; verified via public Arrow export after read. Writer tested separately with explicit foreign C metadata, and independent consumer verifies its preservation.",
            "Parquet arbitrary field metadata must survive import",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_csv_contracts.mojo::test_csv_unterminated_trailing_null_field_preserves_record_width_and_height",
            "Verified independently authored final record ending in an unquoted empty field without a newline raises Explicit height does not match columns; valid CSV must preserve the final field and row. Focused case stops at first affected fixture; later dimensions await repair.",
            "unterminated trailing null CSV field must preserve record width",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/interchange/test_csv_types.mojo::test_csv_writer_emits_categorical_values_instead_of_dictionary_codes",
            "Verified to_csv_string emits UInt32 dictionary codes while write_csv decodes categorical values; public same-text contract requires decoded values. Later layouts await first failure repair.",
            "CSV writer must emit categorical values rather than dictionary codes",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/kernel_equivalence/test_reduction_routes.mojo::test_simd_column_statistics_do_not_wrap_before_widening_to_int128",
            "Verified two Int64.MAX inputs in separate SIMD lanes return -2 before Int128 widening. Wide statistics contract requires exact total. Later lane/layout dimensions await repair.",
            "SIMD column statistics must widen before summing overflowing lanes",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/kernel_equivalence/test_reduction_routes.mojo::test_shifted_sum_shortcut_preserves_final_overflow_and_wide_mean",
            "Verified eight-row shifted SUM of two Int64.MAX values plus zeros returns -2 instead of rejecting final overflow. Subsequent wide-mean check awaits this repair.",
            "shifted SUM must reject a wide final total outside Int64",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/kernel_equivalence/test_reduction_routes.mojo::test_uint64_indexed_moments_preserve_small_differences_before_float_conversion",
            "Verified indexed VAR on UInt64 [0,3,6] returns0 instead of6 (ddof0) because ordering-key signed recoding collapses small differences during Float64 conversion. Focused first assertion verified; later STD/ddof paths await repair.",
            "UInt64 indexed moments must preserve small unsigned differences",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_extended_windows.mojo::test_decimal_pct_change_obeys_documented_float64_result_contract",
            "Verified all three decimal widths return Decimal(38,2) for pct_change, contrary to documented Float64. Focused strict mismatch; later combinations await repair.",
            "decimal pct_change must return Float64",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_set_reshape_types.mojo::test_struct_unique_null_parents_ignore_every_hidden_field_payload",
            "Struct key equality reads hidden fields of null parents, retaining extra equivalent null keys.",
            "struct unique must ignore hidden null-parent fields",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_set_reshape_types.mojo::test_diagonal_concat_missing_categorical_columns_preserve_dictionary_and_null_padding",
            "Missing categorical columns become String null columns; exact underlying diagnostic concat vertical: frame 1 column 0 is x:string, expected x:categorical.",
            "diagonal missing categorical columns must keep logical dtype",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/execution_modes/test_statistics_workers.mojo::test_streamed_misaligned_numeric_chunks_preserve_group_validity_counts",
            "Verified streamed misaligned Float32 chunks produce incorrect grouped nonnull counts at batch65537. Independent count of rows with row%7!=3 gives44939 for first group; actual45173. Other modes complete normal matrix; focused strict first failure does not claim other affected reducers pass.",
            "streamed misaligned chunks must preserve group validity counts",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal_distinct_counts_compare_full_int128_values",
            "Independent quadratic Int128 equality counts five distinct values including null. Decimal128 scaled values beyond Int64, duplicate and signs; eager-global ordinary error is first reached witness.",
            "n_unique of a decimal value beyond 64 bits is not supported",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_eager_global_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_eager_grouped_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_lazy_materialized_global_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_lazy_materialized_grouped_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_lazy_streaming_global_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_precision.mojo::test_decimal128_lazy_streaming_grouped_mean_uses_wide_total",
            "Decimal128(3,2) validscaled[900,900,900,900],hidden999null; exactwide Int128 total3600 gives Float64mean9.0; only first layout0 optimizeFalse reached before ordinary error; later variants are unexecuted.",
            "value exceeds decimal precision 3",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/kernel_equivalence/test_boolean_selection_words.mojo::test_scalar_string_membership_broadcast_preserves_nonnull_literal_validity",
            "Public literal String is_in(column candidate) returns6 unexpected nulls of7; KEEP_NULLS nonnumeric validity fallback packs only one bit rather than broadcasting source validity.",
            "scalar string membership must broadcast valid literal state",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_equality_api.mojo::test_structural_equals_ignores_children_hidden_by_a_null_struct_parent",
            "Two one-row null Struct parents, physically valid Int64 children123/-456. Same schema; self-equalityTrue and null_equalFalseFalse controls pass. Public equals must ignore hidden payload per semantics.md79ff but returnsFalse. Own precise error isolates assertion; unrelated failures propagate.",
            "Structural equality compared child payload hidden by a null struct parent",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal_extremes.mojo::test_maximum_date_weekday_does_not_wrap_the_epoch_alignment_shift",
            "Date Int64.MAX is an exact multiple of7 days; epochThursday requires weekday4. Int64 epoch shift wraps and returns2. Independent Int128 Gregorian model and literal arithmetic verify the expected value.",
            "maximum Date weekday must be Thursday(4); got 2",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_temporal_extremes.mojo::test_extreme_date_calendar_fields_do_not_wrap_the_epoch_conversion",
            "Int64.MAX−719467 Date crosses the civil epoch shift and produces a negative year instead of independently modeled positive25252734927766554. First failing year isolated; later fields await repair.",
            "extreme Date calendar year must remain positive; got -25252734927766555",
            0,
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_variance_retains_numeric_scale",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1221:39: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_standard_deviation_retains_numeric_scale",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1221:39: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_median_retains_numeric_scale",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1221:39: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_quantile_retains_numeric_scale",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1221:39: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_arg_min_accepts_advertised_ordered_input",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1449:37: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_arg_max_accepts_advertised_ordered_input",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1449:37: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_mode_retains_typed_values",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1501:37: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_reduction_types.mojo::test_decimal_value_counts_retains_typed_values",
            "Advertised decimal reduction aborts while accessing physical storage as the wrong variant.",
            "aggregate.mojo:1501:37: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_grouped_types.mojo::test_grouped_decimal_count_skips_null_rows_without_storage_confusion",
            "Verified grouped decimal COUNT reads the wrong storage variant and terminates the process.",
            "execution.mojo:968:28: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )
    policy.failures.append(
        ExpectedFailure(
            "tests/public_contracts/test_set_reshape_types.mojo::test_pivot_count_nullable_decimal128_uses_parent_validity",
            "Nullable Decimal128 pivot COUNT reaches execution COUNT with wrong physical variant; independent expected nonnull counts are 4 and 3.",
            "execution.mojo:968:28: get: wrong variant type",
            mojo_variant_abort_exit_code(),
        )
    )

    # Native cases are omitted from imports unless their capability is enabled.
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal.mojo::test_native_metal_common_numeric_and_boolean_domain_matches_literal_rows_and_cpu_modes",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal.mojo::test_native_metal_payload_budget_accepts_exact_estimate_and_rejects_one_byte_less",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal.mojo::test_native_metal_resident_join_regions_export_complete_owned_typed_results",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal.mojo::test_native_metal_opaque_float64_preserves_ieee_words_through_filter_sort_and_export",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_grouped_and_global_reductions_match_scalar_memberships_all_widths",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_aggregate_compositions_and_consumers_stay_resident",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_boolean_kleene_and_float_extrema_keep_null_nan_zero_and_subnormal_rules",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_wide_sum_proofs_exact_cancellation_and_error_recovery",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_integer_operators_membership_and_cast_matrix_keep_exact_native_words",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_branch_observation_masks_and_precision_errors_leave_runtime_reusable",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_float32_encodings_and_separate_operator_rounding_match_literal_results",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_domains.mojo::test_native_metal_checked_operation_boundaries_reject_before_device_initialization",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_temporal_all_units_and_extreme_ticks_match_independent_calendar",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_extreme_date_epoch_shifts_keep_independent_gregorian_fields",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_fixed_truncation_literal_folding_and_temporal_resident_groups",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_fixed_logical_metadata_survives_nullable_rows_and_extrema",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_temporal.mojo::test_native_metal_temporal_and_decimal_unsupported_requests_reject_before_device_setup",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dictionaries.mojo::test_native_metal_dictionary_unary_lookup_tables_preserve_utf8_nulls_and_report_host_staging",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dictionaries.mojo::test_native_metal_categorical_join_styles_reconcile_independent_codes_and_restore_dictionary_identity",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dictionaries.mojo::test_native_metal_dictionary_distinct_counts_include_nulls_and_feed_aggregate_compositions",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_resident.mojo::test_native_metal_nested_union_and_extreme_slices_preserve_all_typed_payloads_and_order",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_resident.mojo::test_native_metal_distinct_keep_modes_use_logical_null_nan_zero_classes_and_current_slice_positions",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dispatch.mojo::test_native_metal_filtered_scalar_reductions_observe_predicates_without_row_compaction",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dispatch.mojo::test_native_metal_group_dispatch_boundary_preserves_nullable_composite_first_seen_groups",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dispatch.mojo::test_native_metal_integer_sort_dispatch_boundary_keeps_exact_words_null_placement_and_stable_ties",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_dispatch.mojo::test_native_metal_float_join_keys_canonicalize_nan_and_signed_zero_without_matching_nulls",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_cost.mojo::test_native_metal_analytical_estimates_are_value_blind_and_observe_cold_to_warm_shader_metadata",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_cost.mojo::test_native_metal_analytical_unsupported_requests_reject_before_source_or_device_access",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_opaque.mojo::test_native_metal_opaque_float64_row_graphs_copy_literal_null_padding_and_positional_ieee_words",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_opaque.mojo::test_native_metal_opaque_float64_computation_rejects_before_device_setup",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_join_domains.mojo::test_native_metal_typed_composite_join_styles_coalescing_and_empty_sides_match_scalar_pairs",
            "metal",
        )
    )
    policy.requirements.append(
        CapabilityRequirement(
            "tests/execution_modes/test_backend_metal_join_domains.mojo::test_native_metal_bounded_cross_product_heads_preserve_pair_order_without_materializing_fanout",
            "metal",
        )
    )

    # Named selectors select whole modules; newly added tests need no registration.
    policy.groups.append(
        ContractGroup(
            "table.schema",
            [
                "tests/public_contracts/test_schema.mojo",
                "tests/public_contracts/test_tables.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.row_selection",
            [
                "tests/public_contracts/test_row_selection.mojo",
                "tests/public_contracts/test_tables.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.filter",
            [
                "tests/public_contracts/test_filtering.mojo",
                "tests/public_contracts/test_selection_types.mojo",
                "tests/public_contracts/test_tables.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.arithmetic",
            [
                "tests/public_contracts/test_arithmetic.mojo",
                "tests/public_contracts/test_null_logic.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.null_logic",
            [
                "tests/public_contracts/test_expression_boundaries.mojo",
                "tests/public_contracts/test_fill_policies.mojo",
                "tests/public_contracts/test_null_logic.mojo",
                "tests/public_contracts/test_typed_selection.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.order_windows",
            [
                "tests/execution_modes/test_order_batches.mojo",
                "tests/kernel_equivalence/test_parallel_order_selection.mojo",
                "tests/public_contracts/test_extended_windows.mojo",
                "tests/public_contracts/test_order_windows.mojo",
                "tests/public_contracts/test_window_precision.mojo",
                "tests/public_contracts/test_window_types.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.text_binary",
            [
                "tests/public_contracts/test_text_binary.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.temporal",
            [
                "tests/public_contracts/test_temporal.mojo",
                "tests/public_contracts/test_temporal_edges.mojo",
                "tests/public_contracts/test_temporal_extremes.mojo",
                "tests/public_contracts/test_temporal_transitions.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "expr.nested",
            [
                "tests/public_contracts/test_nested.mojo",
                "tests/public_contracts/test_reshape.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "reduce.global",
            [
                "tests/kernel_equivalence/test_reductions.mojo",
                "tests/public_contracts/test_distribution_shapes.mojo",
                "tests/public_contracts/test_expression_boundaries.mojo",
                "tests/public_contracts/test_order_reductions.mojo",
                "tests/public_contracts/test_pair_reductions.mojo",
                "tests/public_contracts/test_reduction_arithmetic.mojo",
                "tests/public_contracts/test_reduction_precision.mojo",
                "tests/public_contracts/test_reduction_types.mojo",
                "tests/public_contracts/test_reductions.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "reduce.grouped",
            [
                "tests/kernel_equivalence/test_aggregate_table.mojo",
                "tests/public_contracts/test_distribution_shapes.mojo",
                "tests/public_contracts/test_expression_boundaries.mojo",
                "tests/public_contracts/test_grouped_types.mojo",
                "tests/public_contracts/test_grouping.mojo",
                "tests/public_contracts/test_order_reductions.mojo",
                "tests/public_contracts/test_pair_reductions.mojo",
                "tests/public_contracts/test_reduction_precision.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.sort",
            [
                "tests/public_contracts/test_selection_types.mojo",
                "tests/public_contracts/test_sorting.mojo",
                "tests/public_contracts/test_tables.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.join",
            [
                "tests/kernel_equivalence/test_join_routes.mojo",
                "tests/planner_correctness/test_asof_sources.mojo",
                "tests/public_contracts/test_asof.mojo",
                "tests/public_contracts/test_asof_errors.mojo",
                "tests/public_contracts/test_join_categorical_coercion.mojo",
                "tests/public_contracts/test_join_types.mojo",
                "tests/public_contracts/test_joins.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.sets_reshape",
            [
                "tests/kernel_equivalence/test_set_routes.mojo",
                "tests/public_contracts/test_concat.mojo",
                "tests/public_contracts/test_duplicates.mojo",
                "tests/public_contracts/test_reshape.mojo",
                "tests/public_contracts/test_set_reshape_types.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "types.cast",
            [
                "tests/public_contracts/test_cast_boundaries.mojo",
                "tests/public_contracts/test_cast_sources.mojo",
                "tests/public_contracts/test_casts.mojo",
                "tests/public_contracts/test_temporal.mojo",
                "tests/public_contracts/test_text_binary.mojo",
                "tests/public_contracts/test_typed_selection.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "api.inspect",
            [
                "tests/public_contracts/test_inspection.mojo",
                "tests/public_contracts/test_sampling_distribution.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "storage.views",
            [
                "tests/storage_ownership/test_isolation.mojo",
                "tests/storage_ownership/test_typed_storage.mojo",
                "tests/storage_ownership/test_views.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "storage.isolation",
            [
                "tests/interchange/test_arrow_edges.mojo",
                "tests/storage_ownership/test_allocation.mojo",
                "tests/storage_ownership/test_csv_builders.mojo",
                "tests/storage_ownership/test_gather_aliases.mojo",
                "tests/storage_ownership/test_isolation.mojo",
                "tests/storage_ownership/test_typed_storage.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "storage.chunks",
            [
                "tests/storage_ownership/test_chunks.mojo",
                "tests/storage_ownership/test_gather_aliases.mojo",
                "tests/storage_ownership/test_typed_storage.mojo",
                "tests/storage_ownership/test_views.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "storage.release",
            [
                "tests/storage_ownership/test_aggregate_key_lifetime.mojo",
                "tests/storage_ownership/test_aggregate_key_views.mojo",
                "tests/storage_ownership/test_release.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "kernel.reduce",
            [
                "tests/execution_modes/test_lifecycle.mojo",
                "tests/kernel_equivalence/test_aggregate_table.mojo",
                "tests/kernel_equivalence/test_grouped_routes.mojo",
                "tests/kernel_equivalence/test_quantile_routes.mojo",
                "tests/kernel_equivalence/test_reduction_routes.mojo",
                "tests/kernel_equivalence/test_reductions.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "kernel.row_ops",
            [
                "tests/interchange/test_arrow_edges.mojo",
                "tests/interchange/test_parquet_edges.mojo",
                "tests/kernel_equivalence/test_bitmap_words.mojo",
                "tests/kernel_equivalence/test_boolean_selection_words.mojo",
                "tests/kernel_equivalence/test_filter_routes.mojo",
                "tests/kernel_equivalence/test_fusion_contracts.mojo",
                "tests/kernel_equivalence/test_join_routes.mojo",
                "tests/kernel_equivalence/test_numeric_cast_routes.mojo",
                "tests/kernel_equivalence/test_row_ops.mojo",
                "tests/storage_ownership/test_gather_aliases.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "kernel.order_keys",
            [
                "tests/kernel_equivalence/test_bucket_keys.mojo",
                "tests/kernel_equivalence/test_group_cardinality.mojo",
                "tests/kernel_equivalence/test_hash_join_routes.mojo",
                "tests/kernel_equivalence/test_join_parallel_boundaries.mojo",
                "tests/kernel_equivalence/test_native_string_rank.mojo",
                "tests/kernel_equivalence/test_order_keys.mojo",
                "tests/kernel_equivalence/test_parallel_order_selection.mojo",
                "tests/kernel_equivalence/test_set_routes.mojo",
                "tests/kernel_equivalence/test_top_rows.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "kernel.state_merge",
            [
                "tests/kernel_equivalence/test_state_merge.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "planner.row_rewrites",
            [
                "tests/planner_correctness/test_errors_schema.mojo",
                "tests/planner_correctness/test_join_aggregate_projection.mojo",
                "tests/planner_correctness/test_row_boundaries.mojo",
                "tests/planner_correctness/test_row_rewrites.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "planner.relational",
            [
                "tests/planner_correctness/test_asof_boundaries.mojo",
                "tests/planner_correctness/test_asof_sources.mojo",
                "tests/planner_correctness/test_branch_rewrites.mojo",
                "tests/planner_correctness/test_join_aggregate_projection.mojo",
                "tests/planner_correctness/test_join_key_pushdown.mojo",
                "tests/planner_correctness/test_join_side_swap.mojo",
                "tests/planner_correctness/test_join_strategies.mojo",
                "tests/planner_correctness/test_name_identity.mojo",
                "tests/planner_correctness/test_nested_boundaries.mojo",
                "tests/planner_correctness/test_relational.mojo",
                "tests/planner_correctness/test_source_reuse.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "planner.errors_schema",
            [
                "tests/planner_correctness/test_errors_schema.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "planner.routing",
            [
                "tests/execution_modes/test_backends.mojo",
                "tests/planner_correctness/test_accelerator_edges.mojo",
                "tests/planner_correctness/test_routing.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "execution.modes",
            [
                "tests/execution_modes/test_aggregate_table_modes.mojo",
                "tests/execution_modes/test_batches.mojo",
                "tests/execution_modes/test_group_strategy.mojo",
                "tests/execution_modes/test_lifecycle.mojo",
                "tests/execution_modes/test_order_batches.mojo",
                "tests/execution_modes/test_stateful_modes.mojo",
                "tests/execution_modes/test_statistics_workers.mojo",
                "tests/execution_modes/test_text_workers.mojo",
                "tests/execution_modes/test_top_bound.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "execution.lifecycle",
            [
                "tests/execution_modes/test_lifecycle.mojo",
                "tests/execution_modes/test_worker_scheduling.mojo",
                "tests/storage_ownership/test_release.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "execution.backends",
            [
                "tests/execution_modes/test_backend_metal.mojo",
                "tests/execution_modes/test_backend_metal_cost.mojo",
                "tests/execution_modes/test_backend_metal_dictionaries.mojo",
                "tests/execution_modes/test_backend_metal_dispatch.mojo",
                "tests/execution_modes/test_backend_metal_domains.mojo",
                "tests/execution_modes/test_backend_metal_join_domains.mojo",
                "tests/execution_modes/test_backend_metal_opaque.mojo",
                "tests/execution_modes/test_backend_metal_resident.mojo",
                "tests/execution_modes/test_backend_metal_temporal.mojo",
                "tests/execution_modes/test_backends.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "io.csv",
            [
                "tests/execution_modes/test_lifecycle.mojo",
                "tests/interchange/test_csv.mojo",
                "tests/interchange/test_csv_contracts.mojo",
                "tests/interchange/test_csv_types.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "io.arrow",
            [
                "tests/interchange/test_arrow.mojo",
                "tests/interchange/test_arrow_edges.mojo",
                "tests/interchange/test_parquet_edges.mojo",
                "tests/interchange/test_spatial_edges.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "io.parquet",
            [
                "tests/interchange/test_parquet.mojo",
                "tests/interchange/test_parquet_edges.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "io.spatial",
            [
                "tests/interchange/test_spatial.mojo",
                "tests/interchange/test_spatial_edges.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "harness.sensitivity",
            [
                "tests/public_contracts/test_comparison_policy.mojo",
                "tests/public_contracts/test_order_windows.mojo",
                "tests/public_contracts/test_tables.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "reference.anchors",
            [
                "tests/public_contracts/test_arithmetic.mojo",
                "tests/public_contracts/test_order_windows.mojo",
                "tests/public_contracts/test_reference_models.mojo",
                "tests/public_contracts/test_tables.mojo",
                "tests/public_contracts/test_temporal.mojo",
                "tests/public_contracts/test_text_binary.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "series.api",
            [
                "tests/public_contracts/test_series_contracts.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "types.dtype",
            [
                "tests/public_contracts/test_dtype_contracts.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.equals",
            [
                "tests/public_contracts/test_equality_api.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "table.group_indices",
            [
                "tests/public_contracts/test_group_indices_api.mojo",
            ],
        )
    )
    policy.groups.append(
        ContractGroup(
            "kernel.parse",
            [
                "tests/kernel_equivalence/test_borrowed_parsers.mojo",
            ],
        )
    )
    return policy^
