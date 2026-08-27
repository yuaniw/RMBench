import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):

    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):

    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class HistoryCheckpointWeightLoader(WeightLoader):
    """Loads a Pi0 checkpoint while initializing only explicitly allowed history parameters."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r".*(history_conditioner|history_modulation|history_token_proj).*",
        )


@dataclasses.dataclass(frozen=True)
class PretrainedHistoryCheckpointWeightLoader(WeightLoader):
    """Loads dense pretrained Pi0 weights and initializes the added LoRA and history parameters."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        params_path = pathlib.Path(self.params_path).expanduser()
        loaded_params = _model.restore_params(download.maybe_download(str(params_path)), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r".*(lora|history_conditioner|history_modulation|history_token_proj).*",
        )


@dataclasses.dataclass(frozen=True)
class PretrainedDualSiglipHistoryCheckpointWeightLoader(WeightLoader):
    """Loads official Pi0 weights and duplicates its SigLIP for frozen history inference."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        params_path = pathlib.Path(self.params_path).expanduser()
        loaded_params = _model.restore_params(
            download.maybe_download(str(params_path)), restore_type=np.ndarray
        )
        merged = _merge_params(
            loaded_params,
            params,
            missing_regex=(
                r".*(lora|history_conditioner|history_modulation|history_token_proj|"
                r"history_image_encoder).*"
            ),
        )
        return _copy_subtree(
            merged,
            loaded_params,
            params,
            source_prefix="PaliGemma/img/",
            target_prefix="history_image_encoder/",
        )


@dataclasses.dataclass(frozen=True)
class OfficialPi0WithTrainedSiglipWeightLoader(WeightLoader):
    """Loads official Pi0 weights and overlays SigLIP from a trained checkpoint."""

    official_params_path: str
    trained_siglip_params_path: str

    def load(self, params: at.Params) -> at.Params:
        official_path = pathlib.Path(self.official_params_path).expanduser()
        trained_siglip_path = pathlib.Path(self.trained_siglip_params_path).expanduser()
        official_params = _model.restore_params(
            download.maybe_download(str(official_path)), restore_type=np.ndarray
        )
        trained_params = _model.restore_params(
            download.maybe_download(str(trained_siglip_path)), restore_type=np.ndarray
        )
        merged = _merge_params(
            official_params,
            params,
            missing_regex=r".*(lora|history_conditioner|history_modulation|history_token_proj).*",
        )
        return _overlay_subtree(
            merged,
            trained_params,
            params,
            subtree_prefix="PaliGemma/img/",
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _overlay_subtree(
    base_params: at.Params,
    overlay_params: at.Params,
    reference_params: at.Params,
    *,
    subtree_prefix: str,
) -> at.Params:
    """Replaces one complete parameter subtree while preserving the base elsewhere."""
    flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_overlay = flax.traverse_util.flatten_dict(overlay_params, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference_params, sep="/")

    result = dict(flat_base)
    for key, reference_value in flat_reference.items():
        if key.startswith(subtree_prefix):
            result[key] = flat_overlay[key].astype(reference_value.dtype)
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _copy_subtree(
    base_params: at.Params,
    source_params: at.Params,
    reference_params: at.Params,
    *,
    source_prefix: str,
    target_prefix: str,
) -> at.Params:
    """Copies a complete source subtree to a differently named target subtree."""
    flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_source = flax.traverse_util.flatten_dict(source_params, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference_params, sep="/")

    result = dict(flat_base)
    for key, reference_value in flat_reference.items():
        if key.startswith(target_prefix):
            source_key = source_prefix + key.removeprefix(target_prefix)
            result[key] = flat_source[source_key].astype(reference_value.dtype)
    return flax.traverse_util.unflatten_dict(result, sep="/")
