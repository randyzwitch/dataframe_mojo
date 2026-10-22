# Module coverage map

This file lists, for each module under `dataframe/`, the contract groups whose tests import it by name. Direct imports are a proxy: tests that go through the public DataFrame and LazyFrame API exercise far more code than this shows, and a module that is imported is not necessarily well tested. Execution-path coverage from `dataframe/trace.mojo` is a separate check. A test file counts when it is `tests/<area>/test_*.mojo` and its groups are its `# contract-group:` markers (`unmarked:<area>` when it has none). Support modules count one level deep: a module imported by a `tests/support` file that a test imports is credited to that test's groups. `__init__.mojo` files are skipped.

Generate with `python3 scripts/check_module_coverage.py --write`. The check (no flag) fails when this file or `tests/coverage_gaps.txt` is out of date.

261 modules, 140 imported directly, 121 not.

| module | contract groups |
| --- | --- |
| `dataframe/_accel_provider.mojo` | execution.backends, planner.routing |
| `dataframe/_build_identity.mojo` | execution.backends |
| `dataframe/accelerator/aggregates.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/capabilities.mojo` | execution.apple_parity, execution.backends, lazy.row_order, planner.routing, public.accelerator_staging, public.apple_parity |
| `dataframe/accelerator/code_domains.mojo` | execution.apple_parity, execution.backends, expr.decimal_extrema, public.apple_parity |
| `dataframe/accelerator/constants.mojo` | execution.apple_parity, execution.backends, expr.cut, planner.routing, public.apple_parity |
| `dataframe/accelerator/cost.mojo` | execution.backends, planner.routing, unmarked:execution_modes |
| `dataframe/accelerator/cpu_work.mojo` | execution.backends, planner.routing |
| `dataframe/accelerator/cuts.mojo` | execution.apple_parity, execution.backends, expr.cut, public.apple_parity |
| `dataframe/accelerator/decimal_prefix.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/decimal_scalar_feed.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/dictionaries.mojo` | execution.apple_parity, execution.backends, expr.decimal_extrema, expr.text_binary, planner.routing, public.accelerator_staging, public.apple_parity, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/accelerator/distinct_counts.mojo` | execution.backends, planner.routing |
| `dataframe/accelerator/distributions.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/calibration.mojo` | execution.apple_parity, execution.backends, public.apple_parity, unmarked:execution_modes |
| `dataframe/accelerator/gpu/coded_group_sum.mojo` | execution.backends |
| `dataframe/accelerator/gpu/dag_runtime.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/device_lists.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/explode.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/grouped_key_owner.mojo` | execution.backends |
| `dataframe/accelerator/gpu/grouped_lists_producer.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/grouped_lists_runtime.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/grouped_schedule.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/grouping.mojo` | unmarked:kernel_equivalence |
| `dataframe/accelerator/gpu/hash_grouping.mojo` | unmarked:kernel_equivalence |
| `dataframe/accelerator/gpu/host_list_reference.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/identity.mojo` | unmarked:execution_modes |
| `dataframe/accelerator/gpu/join_runtime.mojo` | execution.apple_parity, execution.backends, unmarked:execution_modes |
| `dataframe/accelerator/gpu/joins.mojo` | execution.backends, unmarked:execution_modes |
| `dataframe/accelerator/gpu/observed_concat_labels.mojo` | public.apple_parity |
| `dataframe/accelerator/gpu/ordered_faults.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/ordering.mojo` | unmarked:execution_modes |
| `dataframe/accelerator/gpu/paired_coded_sum.mojo` | execution.backends |
| `dataframe/accelerator/gpu/partition_bins.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_faults.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_labels.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_layout.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_runtime.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_schedule.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/partition_state.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/program.mojo` | execution.apple_parity, execution.backends, public.apple_parity, unmarked:execution_modes |
| `dataframe/accelerator/gpu/qcut_labels.mojo` | public.apple_parity |
| `dataframe/accelerator/gpu/query_scope.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/ranked_rows.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/residency.mojo` | execution.apple_parity, execution.backends, public.apple_parity, unmarked:execution_modes |
| `dataframe/accelerator/gpu/resident_fields.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/runtime.mojo` | execution.apple_parity, execution.backends, planner.routing, public.apple_parity, unmarked:execution_modes |
| `dataframe/accelerator/gpu/target.mojo` | execution.backends, planner.routing |
| `dataframe/accelerator/gpu/temporal_descriptors.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/text_domains.mojo` | public.apple_parity |
| `dataframe/accelerator/gpu/timezone_fields.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/gpu/typed_storage.mojo` | execution.backends |
| `dataframe/accelerator/grouped_feeds.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/grouped_key_domains.mojo` | execution.apple_parity, execution.backends, public.apple_parity |
| `dataframe/accelerator/grouped_key_queries.mojo` | execution.backends |
| `dataframe/accelerator/grouped_lists.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/grouped_stream_routes.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/integer_ranges.mojo` | execution.backends |
| `dataframe/accelerator/joins.mojo` | execution.apple_parity, execution.backends, lazy.row_order, planner.routing, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/accelerator/late_payload.mojo` | execution.backends |
| `dataframe/accelerator/lists.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/memory.mojo` | planner.routing |
| `dataframe/accelerator/metadata.mojo` | execution.apple_parity, execution.backends, expr.cut, lazy.frame_verbs, lazy.pivot, lazy.rename_cast, lazy.row_order, lazy.time_windows, public.accelerator_staging, public.apple_parity, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/accelerator/observed_concat.mojo` | execution.backends |
| `dataframe/accelerator/opaque.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/opaque_dictionary.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/ordered_rows.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/paired_moments.mojo` | execution.backends |
| `dataframe/accelerator/partition_cuts.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/partition_operands.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/producer_phases.mojo` | execution.apple_parity, execution.backends, execution.iana_producers, execution.strict_cast_producers |
| `dataframe/accelerator/rows.mojo` | execution.apple_parity, execution.backends, expr.text_binary, lazy.row_order, planner.routing, public.accelerator_staging, public.apple_parity, unmarked:execution_modes |
| `dataframe/accelerator/scalars.mojo` | execution.backends |
| `dataframe/accelerator/selectors.mojo` | execution.backends |
| `dataframe/accelerator/sets.mojo` | execution.backends, planner.routing |
| `dataframe/accelerator/source_compaction.mojo` | public.accelerator_staging, public.apple_parity |
| `dataframe/accelerator/source_lists.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/source_projection.mojo` | execution.backends |
| `dataframe/accelerator/source_structs.mojo` | execution.apple_parity, execution.backends |
| `dataframe/accelerator/sources.mojo` | execution.apple_parity, execution.backends, public.accelerator_staging, public.apple_parity |
| `dataframe/accelerator/structs.mojo` | execution.apple_parity, execution.backends, public.accelerator_staging, public.apple_parity |
| `dataframe/accelerator/text_key_domains.mojo` | public.apple_parity |
| `dataframe/accelerator/window_keys.mojo` | execution.apple_parity, execution.backends, expr.decimal_extrema, public.apple_parity |
| `dataframe/accelerator/windows.mojo` | execution.apple_parity, execution.backends, expr.decimal_extrema, public.apple_parity |
| `dataframe/aggregate.mojo` | kernel.order_keys, kernel.reduce, kernel.state_merge, public.apple_parity |
| `dataframe/aggregate_table.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/binding.mojo` | execution.apple_parity, execution.backends, expr.cut, expr.order_windows, kernel.order_keys, kernel.reduce, kernel.row_ops, public.accelerator_staging, public.apple_parity |
| `dataframe/bool_column.mojo` | expr.order_windows, kernel.order_keys, kernel.row_ops |
| `dataframe/calendar.mojo` | execution.apple_parity, execution.backends |
| `dataframe/cast.mojo` | kernel.row_ops |
| `dataframe/categorical.mojo` | execution.apple_parity, execution.backends, execution.modes, kernel.order_keys, kernel.reduce, planner.relational, planner.routing, public.apple_parity, reduce.grouped, storage.release, table.join |
| `dataframe/column.mojo` | kernel.row_ops |
| `dataframe/comparison_sort.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/decimal.mojo` | execution.apple_parity, execution.backends, expr.order_windows, public.accelerator_staging, public.apple_parity, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/distinct.mojo` | kernel.order_keys |
| `dataframe/distinct_statistics.mojo` | planner.relational |
| `dataframe/dtype.mojo` | execution.apple_parity, execution.backends, expr.decimal_extrema, expr.order_windows, kernel.order_keys, kernel.reduce, planner.routing, public.accelerator_staging, public.apple_parity, storage.chunks, storage.isolation, storage.views, table.filter, table.sets_reshape, table.sort, types.cast, unmarked:execution_modes |
| `dataframe/execution.mojo` | execution.apple_parity, execution.backends, expr.null_logic, kernel.reduce, kernel.row_ops, public.accelerator_staging, public.apple_parity, reduce.global, reduce.grouped |
| `dataframe/execution_settings.mojo` | public.apple_parity |
| `dataframe/expr.mojo` | execution.apple_parity, execution.backends, expr.cut, kernel.reduce, kernel.row_ops, kernel.state_merge, planner.relational, public.apple_parity, unmarked:public_contracts |
| `dataframe/expr_kernels.mojo` | kernel.row_ops |
| `dataframe/frame.mojo` | execution.apple_parity, execution.backends, execution.lifecycle, execution.modes, io.csv, kernel.order_keys, kernel.reduce, kernel.row_ops, planner.relational, table.join |
| `dataframe/fused_compare.mojo` | kernel.row_ops |
| `dataframe/fusion.mojo` | kernel.row_ops |
| `dataframe/gather.mojo` | io.arrow, kernel.row_ops, storage.chunks, storage.isolation |
| `dataframe/hash_agg.mojo` | kernel.order_keys |
| `dataframe/hashing.mojo` | execution.apple_parity, execution.backends, kernel.order_keys, unmarked:public_contracts |
| `dataframe/huge_pages.mojo` | storage.isolation |
| `dataframe/identity_metadata.mojo` | public.apple_parity |
| `dataframe/indexed_reduce.mojo` | kernel.reduce |
| `dataframe/io/arrow.mojo` | execution.backends, execution.lifecycle, io.arrow, kernel.row_ops, storage.chunks, storage.isolation, storage.release, storage.views |
| `dataframe/io/csv_buffers.mojo` | storage.isolation |
| `dataframe/io/csv_reader.mojo` | execution.lifecycle, storage.release |
| `dataframe/io/csv_types.mojo` | execution.lifecycle, storage.release |
| `dataframe/io/parquet.mojo` | execution.backends, execution.lifecycle, io.arrow, io.parquet, io.spatial, kernel.row_ops, storage.chunks, storage.isolation, storage.release, storage.views |
| `dataframe/join_hash.mojo` | kernel.order_keys, kernel.row_ops, table.join |
| `dataframe/join_order.mojo` | planner.relational |
| `dataframe/join_type.mojo` | public.accelerator_staging, public.apple_parity |
| `dataframe/json_value.mojo` | planner.routing, unmarked:execution_modes, unmarked:tooling |
| `dataframe/lazy.mojo` | execution.apple_parity, execution.backends, execution.iana_producers, execution.modes, expr.decimal_extrema, planner.relational, planner.routing, planner.row_rewrites, public.accelerator_staging, public.apple_parity, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/metal.mojo` | execution.apple_parity, execution.backends, execution.iana_producers, execution.strict_cast_producers, expr.cut, expr.horizontal, expr.null_logic, lazy.table_ops, lazy.time_windows, planner.routing, public.apple_parity, unmarked:execution_modes |
| `dataframe/packed_sort.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/parallel.mojo` | execution.apple_parity, execution.backends, execution.lifecycle, execution.modes, expr.decimal_extrema, expr.order_windows, io.arrow, io.csv, io.parquet, kernel.order_keys, kernel.reduce, kernel.regex, kernel.row_ops, public.apple_parity, storage.chunks, storage.isolation, table.join, table.sets_reshape, unmarked:execution_modes, unmarked:public_contracts |
| `dataframe/parse.mojo` | expr.cut, kernel.parse, public.apple_parity |
| `dataframe/partition.mojo` | execution.apple_parity, execution.backends, kernel.order_keys, public.apple_parity, table.sets_reshape, unmarked:execution_modes |
| `dataframe/perfect_aggregate.mojo` | kernel.reduce |
| `dataframe/pipeline.mojo` | execution.apple_parity, execution.backends, execution.modes, table.join |
| `dataframe/rank.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/reductions.mojo` | execution.backends, kernel.reduce, kernel.state_merge |
| `dataframe/row_sort.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/series.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/sorted_range.mojo` | kernel.row_ops |
| `dataframe/string_column.mojo` | execution.apple_parity, execution.backends, execution.modes, kernel.order_keys, kernel.reduce, planner.relational, public.accelerator_staging, public.apple_parity, reduce.grouped, storage.release, table.join |
| `dataframe/string_view.mojo` | execution.modes, expr.regex, expr.text_binary, kernel.order_keys, reference.anchors, storage.chunks, storage.isolation, storage.release, storage.views, types.cast |
| `dataframe/temporal.mojo` | expr.temporal |
| `dataframe/timezone.mojo` | execution.backends |
| `dataframe/top_k.mojo` | expr.order_windows, kernel.order_keys |
| `dataframe/value.mojo` | execution.modes, kernel.reduce, planner.relational, reduce.grouped, storage.release, table.join |
| `dataframe/window.mojo` | kernel.order_keys |
| `dataframe/window_layout.mojo` | execution.apple_parity, execution.backends, expr.order_windows |

## Modules no test imports directly

### `dataframe/` (29)

- `dataframe/asof.mojo`
- `dataframe/display.mojo`
- `dataframe/execution_backend.mojo`
- `dataframe/execution_report.mojo`
- `dataframe/field_metadata.mojo`
- `dataframe/float_ops.mojo`
- `dataframe/geometry.mojo`
- `dataframe/geospatial_metadata.mojo`
- `dataframe/groups.mojo`
- `dataframe/horizontal.mojo`
- `dataframe/list_kernels.mojo`
- `dataframe/mask_filter.mojo`
- `dataframe/nested_column.mojo`
- `dataframe/producer_origin.mojo`
- `dataframe/regex_kernels.mojo`
- `dataframe/replace.mojo`
- `dataframe/row_encode.mojo`
- `dataframe/sampling.mojo`
- `dataframe/selectors.mojo`
- `dataframe/str_kernels.mojo`
- `dataframe/streaming.mojo`
- `dataframe/string_bytes.mojo`
- `dataframe/string_predicates.mojo`
- `dataframe/temporal_kernels.mojo`
- `dataframe/text_functions.mojo`
- `dataframe/time_windows.mojo`
- `dataframe/trace.mojo`
- `dataframe/unicode_alphabetic.mojo`
- `dataframe/wkb.mojo`

### `dataframe/io/` (13)

- `dataframe/io/csv.mojo`
- `dataframe/io/csv_bits.mojo`
- `dataframe/io/csv_decimal.mojo`
- `dataframe/io/csv_decode.mojo`
- `dataframe/io/csv_infer.mojo`
- `dataframe/io/csv_integer.mojo`
- `dataframe/io/csv_numeric.mojo`
- `dataframe/io/csv_scan.mojo`
- `dataframe/io/csv_splitfields.mojo`
- `dataframe/io/geojson.mojo`
- `dataframe/io/shapefile.mojo`
- `dataframe/io/shapefile_binary.mojo`
- `dataframe/io/shapefile_dbf.mojo`

### `dataframe/regex/` (8)

- `dataframe/regex/backtrack.mojo`
- `dataframe/regex/charclass.mojo`
- `dataframe/regex/dfa.mojo`
- `dataframe/regex/pikevm.mojo`
- `dataframe/regex/program.mojo`
- `dataframe/regex/regex.mojo`
- `dataframe/regex/syntax.mojo`
- `dataframe/regex/unicode_tables.mojo`

### `dataframe/accelerator/` (16)

- `dataframe/accelerator/checked_rank.mojo`
- `dataframe/accelerator/decimal_feed.mojo`
- `dataframe/accelerator/decimal_group_feed.mojo`
- `dataframe/accelerator/decimal_row_feed.mojo`
- `dataframe/accelerator/decimal_sum_bounds.mojo`
- `dataframe/accelerator/decimal_sum_receipts.mojo`
- `dataframe/accelerator/explodes.mojo`
- `dataframe/accelerator/faults.mojo`
- `dataframe/accelerator/grouped_float_stream.mojo`
- `dataframe/accelerator/iana_certificate.mojo`
- `dataframe/accelerator/iana_mutations.mojo`
- `dataframe/accelerator/implode_ordered.mojo`
- `dataframe/accelerator/qcuts.mojo`
- `dataframe/accelerator/temporal_expressions.mojo`
- `dataframe/accelerator/timezone_resolution_tables.mojo`
- `dataframe/accelerator/timezone_tables.mojo`

### `dataframe/accelerator/gpu/` (55)

- `dataframe/accelerator/gpu/asof.mojo`
- `dataframe/accelerator/gpu/asof_runtime.mojo`
- `dataframe/accelerator/gpu/category_keys.mojo`
- `dataframe/accelerator/gpu/decimal.mojo`
- `dataframe/accelerator/gpu/decimal_descriptors.mojo`
- `dataframe/accelerator/gpu/floating.mojo`
- `dataframe/accelerator/gpu/fused_int32.mojo`
- `dataframe/accelerator/gpu/fused_int32_plan.mojo`
- `dataframe/accelerator/gpu/fused_int32_runtime.mojo`
- `dataframe/accelerator/gpu/fused_pipeline_cache.mojo`
- `dataframe/accelerator/gpu/grouped_feed.mojo`
- `dataframe/accelerator/gpu/grouped_feed_runtime.mojo`
- `dataframe/accelerator/gpu/grouped_float_geometry.mojo`
- `dataframe/accelerator/gpu/grouped_float_stream.mojo`
- `dataframe/accelerator/gpu/grouped_float_stream_query.mojo`
- `dataframe/accelerator/gpu/grouped_float_stream_runtime.mojo`
- `dataframe/accelerator/gpu/grouped_list_targets.mojo`
- `dataframe/accelerator/gpu/grouped_phase_groups.mojo`
- `dataframe/accelerator/gpu/grouped_stream_dispatch.mojo`
- `dataframe/accelerator/gpu/iana_fields.mojo`
- `dataframe/accelerator/gpu/iana_mutations.mojo`
- `dataframe/accelerator/gpu/iana_resolution.mojo`
- `dataframe/accelerator/gpu/iana_scalar_scopes.mojo`
- `dataframe/accelerator/gpu/iana_ticks.mojo`
- `dataframe/accelerator/gpu/integers.mojo`
- `dataframe/accelerator/gpu/list_groups.mojo`
- `dataframe/accelerator/gpu/list_storage.mojo`
- `dataframe/accelerator/gpu/moments.mojo`
- `dataframe/accelerator/gpu/movement.mojo`
- `dataframe/accelerator/gpu/native_hash_join.mojo`
- `dataframe/accelerator/gpu/observed_concat_runtime.mojo`
- `dataframe/accelerator/gpu/observed_text_pairs.mojo`
- `dataframe/accelerator/gpu/partition_numeric.mojo`
- `dataframe/accelerator/gpu/partition_qcuts.mojo`
- `dataframe/accelerator/gpu/producer_phases_runtime.mojo`
- `dataframe/accelerator/gpu/qcut_runtime.mojo`
- `dataframe/accelerator/gpu/qcuts.mojo`
- `dataframe/accelerator/gpu/quantile_selection.mojo`
- `dataframe/accelerator/gpu/quantiles.mojo`
- `dataframe/accelerator/gpu/ranking.mojo`
- `dataframe/accelerator/gpu/reductions.mojo`
- `dataframe/accelerator/gpu/rows.mojo`
- `dataframe/accelerator/gpu/scratch_pool.mojo`
- `dataframe/accelerator/gpu/source_fields.mojo`
- `dataframe/accelerator/gpu/source_lists_runtime.mojo`
- `dataframe/accelerator/gpu/statistics.mojo`
- `dataframe/accelerator/gpu/temporal.mojo`
- `dataframe/accelerator/gpu/temporal_arithmetic.mojo`
- `dataframe/accelerator/gpu/text_programs.mojo`
- `dataframe/accelerator/gpu/typed_rows.mojo`
- `dataframe/accelerator/gpu/typed_runtime.mojo`
- `dataframe/accelerator/gpu/wide_integer_plan.mojo`
- `dataframe/accelerator/gpu/wide_integer_runtime.mojo`
- `dataframe/accelerator/gpu/wide_integer_sum.mojo`
- `dataframe/accelerator/gpu/windows.mojo`
