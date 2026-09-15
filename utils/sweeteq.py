"""SweetEQ augmentation bridge."""

from utils.plugin import PluginAugmentation
from utils.plugin import PluginAugmentationError as SweetEQAugmentationError
from utils.plugin import PluginParameterRange as SweetEQParameterRange

__all__ = [
    "SweetEQAugmentation",
    "SweetEQAugmentationError",
    "SweetEQParameterRange",
]


class SweetEQAugmentation(PluginAugmentation):
    allowed_options = frozenset(
        {
            "block",
            "disable_oversampling",
            "output_level",
        }
    )
    default_executable_names = ("SweetEQCL.exe", "SweetEQCL")
