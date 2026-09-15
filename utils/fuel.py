"""FUEL augmentation bridge built on the shared plugin wrapper."""

from utils.plugin import PluginAugmentation
from utils.plugin import PluginAugmentationError as FuelAugmentationError
from utils.plugin import PluginParameterRange as FuelParameterRange

__all__ = ["FuelAugmentation", "FuelAugmentationError", "FuelParameterRange"]


class FuelAugmentation(PluginAugmentation):
    allowed_options = frozenset(
        {
            "analysis",
            "automatic",
            "block",
            "disable_oversampling",
            "limitonly",
            "manual",
            "offset",
            "output_level",
            "peak_level",
            "preview_length",
        }
    )
    default_executable_names = ("FUELCL.exe", "FUELCL")
