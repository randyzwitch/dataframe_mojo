"""Plan nodes as morsel pipeline steps."""
from dataframe.pipeline import (
    STEP_DROP,
    STEP_EXPLODE,
    STEP_FILTER,
    STEP_JOIN,
    STEP_SELECT,
    STEP_UNNEST,
    STEP_WITH_COLUMNS,
    Step,
)
from dataframe.planning.model import (
    EXPLODE,
    FILTER,
    JOIN,
    PlanNode,
    SELECT,
    UNNEST,
    WITH_COLUMNS,
)


def _steps_of(operations: List[PlanNode]) -> List[Step]:
    var steps = List[Step](capacity=len(operations))
    for node in operations:
        var kind = STEP_DROP
        if node.kind == FILTER:
            kind = STEP_FILTER
        elif node.kind == SELECT:
            kind = STEP_SELECT
        elif node.kind == WITH_COLUMNS:
            kind = STEP_WITH_COLUMNS
        elif node.kind == JOIN:
            kind = STEP_JOIN
        elif node.kind == EXPLODE:
            kind = STEP_EXPLODE
        elif node.kind == UNNEST:
            kind = STEP_UNNEST
        steps.append(
            Step(
                kind,
                node.exprs.copy(),
                [node.text] if node.kind == UNNEST else node.names.copy(),
                node.join_build_slot(),
                node.right_keys.copy(),
                node.how,
                node.join_suffix() if node.kind == JOIN else String(""),
                node.coalesce,
            )
        )
    return steps^
