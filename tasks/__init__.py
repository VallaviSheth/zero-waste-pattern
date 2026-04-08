"""Tasks package for ZeroWaste-Pattern environment."""

from tasks.base_task import BaseTask
from tasks.basic_packing import BasicPackingTask
from tasks.irregular_shapes import IrregularShapesTask
from tasks.industrial_mode import IndustrialModeTask

__all__ = [
    "BaseTask",
    "BasicPackingTask",
    "IrregularShapesTask",
    "IndustrialModeTask",
]
