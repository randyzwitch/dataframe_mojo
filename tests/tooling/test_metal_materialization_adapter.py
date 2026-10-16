"""A materialization bridge needs lexical type and scalar-selection proof."""
import importlib.util
from pathlib import Path
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "metal_materialization_adapter",
    ROOT / "benchmarks/tools/metal_materialization_adapter.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def sample(
    receiver="materialized",
    selection='col("amount").cast(DataType.FLOAT64).mean()',
    extra="",
):
    return (
        textwrap.dedent(
            f'''\
    from dataframe import DataFrame, LazyFrame, Expr, col
    def build(inputs: Dict[String, DataFrame]) raises -> DataFrame:
        var {receiver} = (
            inputs["sales"].lazy().filter(col("amount") > 0).collect()
        )
    '''
        )
        + extra
        + f'    var answer = {receiver}.select({selection}).item()\n    return {receiver}^\n'
    )


class MetalMaterializationAdapter(unittest.TestCase):
    def test_generic_receiver_rename_and_unchanged_expression_bytes(self):
        expression = 'col("amount").cast(DataType.FLOAT64).mean()'
        for name in ["materialized", "arbitrary_result", "renamed_frame"]:
            source = sample(name)
            derived, proof = adapter.lift_scalar_selects(source)
            self.assertEqual(len(proof["adapted"]), 1)
            self.assertEqual(proof["adapted"][0]["receiver"], name)
            self.assertEqual(
                proof["adapted"][0]["selection_expression_sha256"],
                adapter.digest(expression),
            )
            self.assertIn(
                name + ".lazy().select(" + expression + ").collect().item()",
                derived,
            )
            reversed_materialization = derived.replace(
                name + ".lazy().select(", name + ".select(", 1
            ).replace(".collect().item()", ".item()", 1)
            self.assertEqual(reversed_materialization, source)
            self.assertFalse(proof["whole_query_gpu_certificate"])

    def test_all_reviewed_scalar_aggregates_and_owned_copy_alias(self):
        for method in sorted(adapter.REDUCTIONS):
            _, proof = adapter.lift_scalar_selects(
                sample(selection='col("amount").' + method + "()")
            )
            self.assertEqual(len(proof["adapted"]), 1, method)
        source = sample(extra="    var alias = materialized.copy()\n").replace(
            "var answer = materialized.select", "var answer = alias.select"
        )
        _, proof = adapter.lift_scalar_selects(source)
        self.assertEqual([p["receiver"] for p in proof["adapted"]], ["alias"])

    def test_typed_lazy_parameter_and_annotated_helper_return(self):
        source = """def caller(source: LazyFrame) raises -> DataFrame:
    var frame = source.collect()
    return frame.select(col("x").sum()).item()
"""
        self.assertEqual(
            len(adapter.lift_scalar_selects(source)[1]["adapted"]), 1
        )
        source = """def rows(input: DataFrame) raises -> LazyFrame:
    return input.lazy()
def caller(source: DataFrame) raises -> DataFrame:
    var frame = rows(source).collect()
    return frame.select(col("x").mean()).item()
"""
        self.assertEqual(
            len(adapter.lift_scalar_selects(source)[1]["adapted"]), 1
        )

    def test_raw_unknown_and_reassigned_receivers_are_not_upgraded(self):
        variants = [
            sample().replace(
                'inputs["sales"].lazy().filter(col("amount") > 0).collect()',
                'inputs["sales"]',
            ),
            sample().replace(
                'inputs["sales"].lazy().filter(col("amount") > 0).collect()',
                "unknown()",
            ),
            sample(extra='    materialized = inputs["sales"]\n'),
            sample(
                extra='    if condition:\n        materialized = inputs["sales"]\n'
            ),
            sample(extra="    materialized += unknown\n"),
            sample(extra="    materialized, other = unknown\n"),
        ]
        for source in variants:
            with self.subTest(source=source):
                derived, proof = adapter.lift_scalar_selects(source)
                self.assertEqual(derived, source)
                self.assertEqual(proof["adapted"], [])
                self.assertEqual(len(proof["unadapted"]), 1)

    def test_scope_shadowing_loop_binding_and_closure_are_not_proofs(self):
        source = sample().replace(
            "    var answer = materialized.select",
            "    for materialized in raw_frames:\n        var answer = materialized.select",
        )
        self.assertEqual(adapter.lift_scalar_selects(source)[1]["adapted"], [])
        source = sample().replace(
            "    var answer = materialized.select",
            "    def nested() raises -> DataFrame:\n        var answer = materialized.select",
        )
        self.assertEqual(adapter.lift_scalar_selects(source)[1]["adapted"], [])
        source = sample(extra="    ref borrowed = materialized\n").replace(
            "var answer = materialized.select", "var answer = borrowed.select"
        )
        self.assertEqual(adapter.lift_scalar_selects(source)[1]["adapted"], [])
        source = sample(
            extra='    if condition:\n        var materialized = inputs["sales"]\n'
        )
        self.assertEqual(
            len(adapter.lift_scalar_selects(source)[1]["adapted"]), 1
        )

    def test_non_scalar_options_and_item_arguments_remain_unchanged(self):
        variants = [
            sample(selection='col("amount")'),
            sample(selection="lit(17)"),
            sample(selection='[col("amount").mean()]'),
            sample(selection='col("amount").mean(), batch_size=17'),
            sample().replace(".item()", ".item(0)"),
            sample(selection='col("amount").mean().fill_null(col("other"))'),
        ]
        for source in variants:
            with self.subTest(source=source):
                derived, proof = adapter.lift_scalar_selects(source)
                self.assertEqual(derived, source)
                self.assertEqual(proof["adapted"], [])

    def test_comments_strings_and_idempotence(self):
        source = sample(
            extra='    # materialized.select(col("x").mean()).item()\n    var note = \'materialized.select(col("x").mean()).item()\'\n'
        )
        derived, proof = adapter.lift_scalar_selects(source)
        self.assertEqual(len(proof["adapted"]), 1)
        second, again = adapter.lift_scalar_selects(derived)
        self.assertEqual(second, derived)
        self.assertEqual(again["adapted"], [])

    def test_local_api_shadow_is_not_a_scalar_expression_proof(self):
        source = sample(extra="    var col = custom_expression_factory\n")
        self.assertEqual(adapter.lift_scalar_selects(source)[1]["adapted"], [])
        source = sample().replace(
            "inputs: Dict[String, DataFrame]",
            "inputs: Dict[String, DataFrame], col: CustomFactory",
        )
        self.assertEqual(adapter.lift_scalar_selects(source)[1]["adapted"], [])


if __name__ == "__main__":
    unittest.main()
