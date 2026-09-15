"""MasterPlan2 (MP2) augmentation bridge."""

from utils.plugin import PluginAugmentation
from utils.plugin import PluginAugmentationError as MP2AugmentationError
from utils.plugin import PluginParameterRange as MP2ParameterRange

__all__ = ["MP2Augmentation", "MP2AugmentationError", "MP2ParameterRange"]


class MP2Augmentation(PluginAugmentation):
    allowed_options = frozenset(
        {
            "analysis",
            "automatic",
            "block",
            "disable_oversampling",
            "legacy",
            "legacy_os",
            "limitonly",
            "manual",
            "normal_mode",
            "offset",
            "output_level",
            "peak_level",
            "preview_length",
            "telemetry",
        }
    )
    default_executable_names = ("MasterPlan2CL.exe", "MasterPlan2CL")
