"""CPU-only pieces shared by the Metal provider.

Importing this package never loads a GPU SDK.
"""

from .cost import PlanEstimate, CalibrationKey, CostDecision, select_auto
