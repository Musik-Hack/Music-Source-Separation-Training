import os
import random
import shutil
import subprocess
import tempfile
import math
from collections.abc import Sequence

import numpy as np
import soundfile as sf

ALLOWED_SPEC_KEYS = {
    "probability",
    "executable",
    "args",
    "timeout_seconds",
    "sample_rate",
    "pcm_type",
    "suffix",
    "allow_length_change",
}
REQUIRED_SPEC_KEYS = {"executable"}
SUPPORTED_PCM_TYPES = {"FLOAT", "PCM_24", "PCM_16"}
MIXTURE_TARGETS = frozenset(("mix", "mixture"))


class SubprocessAugmentationError(ValueError):
    pass


def _is_mapping(value):
    return hasattr(value, "items") and callable(value.items)


def _is_list(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def validate_subprocess_augmentations(augmentations, config_path=None):
    if not _is_mapping(augmentations):
        return

    if "subprocess" in augmentations:
        settings = augmentations["subprocess"]
        if not _is_mapping(settings):
            raise SubprocessAugmentationError(
                "augmentations.subprocess must be a mapping with all/instrument rules"
            )

        for key, value in settings.items():
            if key == "search_paths":
                _validate_search_paths(value, "augmentations.subprocess.search_paths")
            elif key == "timeout_seconds":
                _require_positive_number(
                    value, "augmentations.subprocess.timeout_seconds"
                )
            elif key == "sample_rate":
                _require_positive_number(value, "augmentations.subprocess.sample_rate")
            elif key == "pcm_type":
                _validate_pcm_type(value, "augmentations.subprocess.pcm_type")
            else:
                if not _is_list(value):
                    raise SubprocessAugmentationError(
                        "augmentations.subprocess.%s must be a list of rules" % key
                    )
                for rule in value:
                    _validate_rule(rule, "augmentations.subprocess.%s" % key)

    if "subprocess_on_mixture" in augmentations:
        _validate_mixture_subprocess_settings(augmentations["subprocess_on_mixture"])


def _validate_search_paths(value, label):
    if not _is_list(value) or not value:
        raise SubprocessAugmentationError("%s must be a non-empty list" % label)
    for path in value:
        _require_string(path, "%s entry" % label)


def _validate_mixture_subprocess_settings(settings):
    allowed_keys = {
        "search_paths",
        "timeout_seconds",
        "sample_rate",
        "pcm_type",
        "rules",
    }
    if _is_mapping(settings):
        unknown_keys = set(settings) - allowed_keys
        if unknown_keys:
            raise SubprocessAugmentationError(
                "augmentations.subprocess_on_mixture has unsupported keys: %s"
                % ", ".join(sorted(unknown_keys))
            )
        for key, value in settings.items():
            if key == "search_paths":
                _validate_search_paths(
                    value, "augmentations.subprocess_on_mixture.search_paths"
                )
            elif key == "timeout_seconds":
                _require_positive_number(
                    value, "augmentations.subprocess_on_mixture.timeout_seconds"
                )
            elif key == "sample_rate":
                _require_positive_number(
                    value, "augmentations.subprocess_on_mixture.sample_rate"
                )
            elif key == "pcm_type":
                _validate_pcm_type(
                    value, "augmentations.subprocess_on_mixture.pcm_type"
                )
        rules = settings.get("rules", [])
    elif _is_list(settings):
        rules = settings
    else:
        raise SubprocessAugmentationError(
            "augmentations.subprocess_on_mixture must be a mapping or list"
        )

    if not rules:
        raise SubprocessAugmentationError(
            "augmentations.subprocess_on_mixture.rules must contain at least one rule"
        )
    for index, rule in enumerate(rules):
        _validate_rule(rule, "augmentations.subprocess_on_mixture.rules[%d]" % index)


def _mixture_rules(settings):
    if _is_mapping(settings):
        return settings.get("rules", [])
    return settings


def apply_subprocess_augmentations(source, config, instrument):
    augmentations = config.get("augmentations", {})
    settings = augmentations.get("subprocess")
    if not settings:
        return source

    rules = []
    for group in ("all", instrument):
        rules.extend(settings.get(group, []))
    if not rules:
        return source

    return _apply_subprocess_rules(source, rules, settings, config)


def apply_subprocess_mixture_augmentations(mix, config):
    augmentations = config.get("augmentations", {})
    subprocess_settings = augmentations.get("subprocess")
    if _is_mapping(subprocess_settings):
        rules = []
        for target in MIXTURE_TARGETS:
            rules.extend(subprocess_settings.get(target, []))
        if rules:
            mix = _apply_subprocess_rules(mix, rules, subprocess_settings, config)

    settings = augmentations.get("subprocess_on_mixture")
    if not settings:
        return mix

    global_settings = settings if _is_mapping(settings) else {}
    return _apply_subprocess_rules(
        mix, _mixture_rules(settings), global_settings, config
    )


def _apply_subprocess_rules(source, rules, settings, config):
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2:
        raise SubprocessAugmentationError(
            "Subprocess augmentations require a (channels, samples) chunk"
        )

    for rule in rules:
        probability = rule.get("probability", 1.0)
        if random.uniform(0, 1) < probability:
            source = _run_subprocess_augmentation(
                source,
                rule,
                settings,
                config.get("_config_path"),
                default_sample_rate=int(
                    config.get("audio", {}).get("sample_rate", 44100)
                ),
            )

    return source


def _validate_rule(rule, label):
    if not _is_mapping(rule):
        raise SubprocessAugmentationError("%s must be a mapping" % label)
    unknown_keys = set(rule) - ALLOWED_SPEC_KEYS
    if unknown_keys:
        raise SubprocessAugmentationError(
            "%s has unsupported keys: %s" % (label, ", ".join(sorted(unknown_keys)))
        )
    missing_keys = REQUIRED_SPEC_KEYS - set(rule)
    if missing_keys:
        raise SubprocessAugmentationError(
            "%s is missing keys: %s" % (label, ", ".join(sorted(missing_keys)))
        )

    _require_string(rule["executable"], "%s.executable" % label)
    probability = rule.get("probability", 1.0)
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise SubprocessAugmentationError("%s.probability must be in [0, 1]" % label)
    if "timeout_seconds" in rule:
        _require_positive_number(rule["timeout_seconds"], "%s.timeout_seconds" % label)
    if "sample_rate" in rule:
        _require_positive_number(rule["sample_rate"], "%s.sample_rate" % label)
    if "pcm_type" in rule:
        _validate_pcm_type(rule["pcm_type"], "%s.pcm_type" % label)
    if "allow_length_change" in rule and not isinstance(
        rule["allow_length_change"], bool
    ):
        raise SubprocessAugmentationError(
            "%s.allow_length_change must be a boolean" % label
        )

    arguments = rule.get("args", [])
    if arguments is None:
        arguments = []
    if not _is_list(arguments):
        raise SubprocessAugmentationError("%s.args must be a list" % label)
    for argument in arguments:
        _require_string(argument, "%s.args entry" % label)
    if not any("{input}" in argument for argument in arguments) or not any(
        "{output}" in argument for argument in arguments
    ):
        raise SubprocessAugmentationError(
            "%s.args must contain {input} and {output}" % label
        )

    suffix = rule.get("suffix", ".wav")
    _require_string(suffix, "%s.suffix" % label)
    if os.path.sep in suffix or (os.path.altsep and os.path.altsep in suffix):
        raise SubprocessAugmentationError("%s.suffix cannot contain a path" % label)
    if os.path.splitext("chunk" + suffix)[1] != suffix:
        raise SubprocessAugmentationError("%s.suffix must include an extension" % label)


def _require_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise SubprocessAugmentationError("%s must be a non-empty string" % label)
    if "\0" in value:
        raise SubprocessAugmentationError("%s cannot contain a NUL byte" % label)


def _require_positive_number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SubprocessAugmentationError("%s must be a positive number" % label)


def _validate_pcm_type(value, label):
    if value not in SUPPORTED_PCM_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PCM_TYPES))
        raise SubprocessAugmentationError("%s must be one of: %s" % (label, supported))


def _search_paths(settings, config_path=None):
    configured_paths = settings.get("search_paths", [])
    config_dir = (
        os.path.dirname(os.path.abspath(str(config_path)))
        if config_path
        else os.getcwd()
    )
    if configured_paths:
        return [
            os.path.abspath(os.path.join(config_dir, str(path)))
            for path in configured_paths
        ]
    return [config_dir]


def _resolve_executable(rule, settings, config_path):
    executable = rule["executable"]
    config_dir = (
        os.path.dirname(os.path.abspath(str(config_path)))
        if config_path
        else os.getcwd()
    )
    if os.path.isabs(executable):
        candidates = [executable]
    elif os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
        candidates = [os.path.abspath(os.path.join(config_dir, executable))]
    else:
        candidates = [
            os.path.join(path, executable)
            for path in _search_paths(settings, config_path)
        ]
        system_path_match = shutil.which(executable)
        if system_path_match:
            candidates.append(system_path_match)

    for candidate in candidates:
        if os.path.exists(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
            raise SubprocessAugmentationError(
                "Subprocess augmentation executable is not executable: %s" % candidate
            )

    raise SubprocessAugmentationError(
        "Subprocess augmentation executable was not found or is not executable: %s"
        % executable
    )


def _run_subprocess_augmentation(
    source, rule, global_settings, config_path, default_sample_rate=44100
):
    executable = _resolve_executable(rule, global_settings, config_path)
    timeout_seconds = float(
        rule.get("timeout_seconds", global_settings.get("timeout_seconds", 30))
    )
    sample_rate = int(
        rule.get("sample_rate", global_settings.get("sample_rate", default_sample_rate))
    )
    suffix = rule.get("suffix", ".wav")
    pcm_type = rule.get("pcm_type", global_settings.get("pcm_type", "FLOAT"))
    _validate_pcm_type(pcm_type, "pcm_type")
    arguments = rule.get("args", []) or []
    allow_length_change = rule.get("allow_length_change", False)

    with tempfile.TemporaryDirectory(prefix="msst-subprocess-") as directory:
        input_path = os.path.join(directory, "input" + suffix)
        output_path = os.path.join(directory, "output" + suffix)
        command = [
            executable,
            *[
                argument.replace("{input}", input_path).replace("{output}", output_path)
                for argument in arguments
            ],
        ]
        sf.write(
            input_path,
            np.ascontiguousarray(source.T),
            sample_rate,
            subtype=pcm_type,
            format="WAV",
        )

        try:
            subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise SubprocessAugmentationError(
                "Subprocess augmentation timed out after %.3f seconds: %s"
                % (timeout_seconds, executable)
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or b"").decode(errors="replace").strip()
            if stderr:
                stderr = "\n" + stderr[-2000:]
            raise SubprocessAugmentationError(
                "Subprocess augmentation failed with exit status %s: %s%s"
                % (error.returncode, executable, stderr)
            ) from error
        except OSError as error:
            raise SubprocessAugmentationError(
                "Could not start subprocess augmentation %s: %s" % (executable, error)
            ) from error

        try:
            output_info = sf.info(output_path)
            output, output_sample_rate = sf.read(
                output_path, dtype="float32", always_2d=True
            )
        except Exception as error:
            raise SubprocessAugmentationError(
                "Subprocess augmentation did not produce readable audio: %s"
                % executable
            ) from error

    if output_sample_rate != sample_rate:
        raise SubprocessAugmentationError(
            "Subprocess augmentation changed the sample rate from %s to %s: %s"
            % (sample_rate, output_sample_rate, executable)
        )

    if output_info.subtype != pcm_type:
        raise SubprocessAugmentationError(
            "Subprocess augmentation produced %s instead of %s: %s"
            % (output_info.subtype, pcm_type, executable)
        )

    output = output.T
    if output.shape[0] != source.shape[0]:
        raise SubprocessAugmentationError(
            "Subprocess augmentation changed the channel count from %s to %s: %s"
            % (source.shape[0], output.shape[0], executable)
        )
    if output.shape[1] != source.shape[1]:
        if not allow_length_change:
            raise SubprocessAugmentationError(
                "Subprocess augmentation changed the length from %s to %s: %s"
                % (source.shape[1], output.shape[1], executable)
            )
        if output.shape[1] > source.shape[1]:
            output = output[:, : source.shape[1]]
        else:
            output = np.pad(output, ((0, 0), (0, source.shape[1] - output.shape[1])))

    return output.astype(np.float32, copy=False)
