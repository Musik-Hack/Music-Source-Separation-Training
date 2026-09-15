"""Shared command-line plugin augmentation bridge for training."""

import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

import numpy as np
import soundfile as sf

PCM_BIT_DEPTHS = {"FLOAT": 32, "PCM_24": 24, "PCM_16": 16}
COMMON_SETTINGS_KEYS = {
    "executable",
    "extra_args",
    "fixed_parameters",
    "parameters",
    "probability",
    "pcm_type",
    "sample_rate",
    "timeout_seconds",
}
FLAG_OPTIONS = {
    "analysis": "--analysis",
    "automatic": "--automatic",
    "disable_oversampling": "--disableos",
    "legacy": "--legacy",
    "limitonly": "--limitonly",
    "manual": "--manual",
    "telemetry": "--telemetry",
}
VALUE_OPTIONS = {
    "block": "--block",
    "legacy_os": "--legacyos",
    "normal_mode": "--normalmode",
    "offset": "--offset",
    "output_level": "--outputlevel",
    "peak_level": "--peakLevel",
    "preview_length": "--previewlength",
}


class PluginAugmentationError(ValueError):
    pass


@dataclass(frozen=True)
class PluginParameterRange:
    minimum: float
    maximum: float
    probability: float = 1.0


def _number(value, label):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise PluginAugmentationError("%s must be a number" % label) from error


def _positive_number(value, label):
    number = _number(value, label)
    if number <= 0:
        raise PluginAugmentationError("%s must be positive" % label)
    return number


def _optional_number(value, label):
    return None if value is None else _number(value, label)


def _probability(value, label):
    number = _number(value, label)
    if not 0 <= number <= 1:
        raise PluginAugmentationError("%s must be in [0, 1]" % label)
    return number


def _boolean(value, label):
    if not isinstance(value, bool):
        raise PluginAugmentationError("%s must be a boolean" % label)
    return value


def _integer_choice(value, choices, label):
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise PluginAugmentationError(
            "%s must be one of: %s" % (label, ", ".join(map(str, choices)))
        )
    return value


def _parameters(value):
    if not isinstance(value, Mapping):
        raise PluginAugmentationError("plugin parameters must be a mapping")
    parameters = {}
    for name, spec in value.items():
        if not isinstance(spec, Mapping):
            raise PluginAugmentationError(
                "plugin parameter %r must be a mapping" % name
            )
        unknown_keys = set(spec) - {"minimum", "maximum", "probability"}
        if unknown_keys:
            raise PluginAugmentationError(
                "plugin parameter %r has unsupported keys: %s"
                % (name, ", ".join(sorted(unknown_keys)))
            )
        minimum = _probability(
            spec.get("minimum", 0), "plugin parameter %r minimum" % name
        )
        maximum = _probability(
            spec.get("maximum", 1), "plugin parameter %r maximum" % name
        )
        if minimum > maximum:
            raise PluginAugmentationError(
                "plugin parameter %r minimum exceeds maximum" % name
            )
        parameters[name] = PluginParameterRange(
            minimum=minimum,
            maximum=maximum,
            probability=_probability(
                spec.get("probability", 1), "plugin parameter %r probability" % name
            ),
        )
    return parameters


def _fixed_parameters(value):
    if not isinstance(value, Mapping):
        raise PluginAugmentationError("plugin fixed_parameters must be a mapping")
    parameters = {}
    for name, parameter_value in value.items():
        parameters[name] = _probability(
            parameter_value, "plugin fixed parameter %r" % name
        )
    return parameters


def _extra_args(value):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PluginAugmentationError("plugin extra_args must be a list of strings")
    return tuple(value)


def _resolve_executable(value, config_path, default_names):
    if value is not None:
        if not isinstance(value, str) or not value:
            raise PluginAugmentationError("plugin executable must be a path")
        executable = Path(value).expanduser()
        if not executable.is_absolute():
            base = Path(config_path).resolve().parent if config_path else Path.cwd()
            executable = base / executable
    else:
        root = Path(__file__).resolve().parents[3]
        executable = next(
            (
                root / "binaries" / name
                for name in default_names
                if (root / "binaries" / name).is_file()
            ),
            None,
        )
        if executable is None:
            path_executable = next(
                (
                    shutil.which(name)
                    for name in default_names
                    if shutil.which(name) is not None
                ),
                None,
            )
            if path_executable is None:
                raise PluginAugmentationError(
                    "plugin executable was not found: %s" % ", ".join(default_names)
                )
            executable = Path(path_executable)

    executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PluginAugmentationError(
            "plugin executable is not executable: %s" % executable
        )
    return executable


@dataclass(frozen=True)
class PluginAugmentation:
    executable: Path
    parameters: Dict[str, PluginParameterRange]
    fixed_parameters: Dict[str, float]
    sample_rate: int
    probability: float
    pcm_type: str
    timeout_seconds: float
    options: Dict[str, object]
    extra_args: Tuple[str, ...]

    allowed_options: ClassVar[frozenset] = frozenset()
    default_executable_names: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def from_config(
        cls,
        settings: Mapping[str, Any],
        config_path: Optional[str] = None,
        sample_rate: int = 44100,
    ) -> "PluginAugmentation":
        if not isinstance(settings, Mapping):
            raise PluginAugmentationError("plugin augmentation must be a mapping")
        unknown_keys = set(settings) - COMMON_SETTINGS_KEYS - cls.allowed_options
        if unknown_keys:
            raise PluginAugmentationError(
                "plugin augmentation has unsupported keys: %s"
                % ", ".join(sorted(unknown_keys))
            )

        pcm_type = settings.get("pcm_type", "FLOAT")
        if pcm_type not in PCM_BIT_DEPTHS:
            raise PluginAugmentationError(
                "plugin pcm_type must be one of: %s" % ", ".join(PCM_BIT_DEPTHS)
            )
        automatic = _boolean(settings.get("automatic", False), "plugin automatic")
        peak_level = _optional_number(settings.get("peak_level"), "plugin peak_level")
        limitonly = _boolean(settings.get("limitonly", False), "plugin limitonly")
        manual = _boolean(settings.get("manual", False), "plugin manual")
        if automatic and (peak_level is not None or limitonly or manual):
            raise PluginAugmentationError(
                "plugin automatic cannot be used with peak_level, limitonly, or manual"
            )

        block = settings.get("block")
        if block is not None:
            block = int(_positive_number(block, "plugin block"))
        legacy_os = settings.get("legacy_os")
        if legacy_os is not None:
            legacy_os = _integer_choice(legacy_os, (0, 1, 2), "plugin legacy_os")
        normal_mode = settings.get("normal_mode")
        if normal_mode is not None:
            normal_mode = _integer_choice(normal_mode, (0, 1, 2), "plugin normal_mode")

        options = {
            "analysis": _boolean(settings.get("analysis", False), "plugin analysis"),
            "automatic": automatic,
            "block": block,
            "disable_oversampling": _boolean(
                settings.get("disable_oversampling", False),
                "plugin disable_oversampling",
            ),
            "legacy": _boolean(settings.get("legacy", False), "plugin legacy"),
            "legacy_os": legacy_os,
            "limitonly": limitonly,
            "manual": manual,
            "normal_mode": normal_mode,
            "offset": _optional_number(settings.get("offset"), "plugin offset"),
            "output_level": _optional_number(
                settings.get("output_level"), "plugin output_level"
            ),
            "peak_level": peak_level,
            "preview_length": _optional_number(
                settings.get("preview_length"), "plugin preview_length"
            ),
            "telemetry": _boolean(settings.get("telemetry", False), "plugin telemetry"),
        }

        parameters = _parameters(settings.get("parameters", {}))
        fixed_parameters = _fixed_parameters(settings.get("fixed_parameters", {}))
        duplicate_parameters = set(parameters) & set(fixed_parameters)
        if duplicate_parameters:
            raise PluginAugmentationError(
                "plugin parameters cannot be both randomized and fixed: %s"
                % ", ".join(sorted(duplicate_parameters))
            )

        return cls(
            executable=_resolve_executable(
                settings.get("executable"), config_path, cls.default_executable_names
            ),
            parameters=parameters,
            fixed_parameters=fixed_parameters,
            sample_rate=int(
                _positive_number(
                    settings.get("sample_rate", sample_rate), "plugin sample_rate"
                )
            ),
            probability=_probability(
                settings.get("probability", 1), "plugin probability"
            ),
            pcm_type=pcm_type,
            timeout_seconds=_positive_number(
                settings.get("timeout_seconds", 30), "plugin timeout_seconds"
            ),
            options=options,
            extra_args=_extra_args(settings.get("extra_args", [])),
        )

    def sample_parameters(self):
        sampled = {}
        for name, spec in self.parameters.items():
            if random.random() < spec.probability:
                sampled[name] = random.uniform(spec.minimum, spec.maximum)
        return sampled

    def build_command(
        self,
        input_path: Path,
        output_path: Path,
        sampled_parameters: Mapping[str, float],
    ) -> Tuple[str, ...]:
        command = [
            str(self.executable),
            str(input_path),
            "--outputfile",
            str(output_path),
        ]
        for name, flag in FLAG_OPTIONS.items():
            if self.options.get(name):
                command.append(flag)
        for name, flag in VALUE_OPTIONS.items():
            if self.options.get(name) is not None:
                command.extend([flag, str(self.options[name])])
        command.extend(["--outputbitdepth", str(PCM_BIT_DEPTHS[self.pcm_type])])
        command.extend(self.extra_args)
        for name, value in self.fixed_parameters.items():
            command.append("--%s=%s" % (name, value))
        for name, value in sampled_parameters.items():
            command.append("--%s=%s" % (name, value))
        return tuple(command)

    def __call__(self, source: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2:
            raise PluginAugmentationError(
                "plugin augmentation requires a (channels, samples) chunk"
            )
        if random.random() >= self.probability:
            return source

        sampled_parameters = self.sample_parameters()
        with tempfile.TemporaryDirectory(prefix="plugin-augmentation-") as directory:
            input_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "output.wav"
            command = self.build_command(input_path, output_path, sampled_parameters)
            sf.write(
                input_path,
                np.ascontiguousarray(source.T),
                self.sample_rate,
                subtype=self.pcm_type,
                format="WAV",
            )
            try:
                subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise PluginAugmentationError(
                    "plugin timed out after %.3f seconds" % self.timeout_seconds
                ) from error
            except subprocess.CalledProcessError as error:
                message = "plugin failed with exit status %s" % error.returncode
                if error.stderr:
                    message += "\n" + error.stderr
                raise PluginAugmentationError(message) from error
            except OSError as error:
                raise PluginAugmentationError(
                    "Could not start plugin: %s" % error
                ) from error

            try:
                output_info = sf.info(output_path)
                output, output_sample_rate = sf.read(
                    output_path, dtype="float32", always_2d=True
                )
            except Exception as error:
                raise PluginAugmentationError(
                    "plugin did not produce readable audio"
                ) from error

        if output_sample_rate != self.sample_rate:
            raise PluginAugmentationError(
                "plugin changed the sample rate from %s to %s"
                % (self.sample_rate, output_sample_rate)
            )
        if output_info.subtype != self.pcm_type:
            raise PluginAugmentationError(
                "plugin produced %s instead of %s"
                % (output_info.subtype, self.pcm_type)
            )
        output = output.T
        if output.shape != source.shape:
            raise PluginAugmentationError(
                "plugin changed the chunk shape from %s to %s"
                % (source.shape, output.shape)
            )
        return output.astype(np.float32, copy=False)
