from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Literal

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def validate_checkpoint_params(params: dict, checkpoint_dir: pathlib.Path | str) -> None:
    """Reject non-finite weights before model compilation or simulator actions."""
    invalid = []
    for path, value in flax.traverse_util.flatten_dict(params).items():
        if value is None:
            continue  # NNX stores optional parameters as None leaves.
        array = np.asarray(value)
        if not np.isfinite(array).all():
            invalid.append("/".join(map(str, path)))
    if invalid:
        examples = ", ".join(invalid[:5])
        raise ValueError(
            f"Invalid checkpoint {checkpoint_dir}: {len(invalid)} parameter tensors contain NaN/Inf "
            f"(examples: {examples}). Evaluation cannot use these weights. Check the training loss "
            "and select a finite checkpoint or retrain; changing Python/GPU/XLA flags cannot repair them."
        )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    asset_id: str | None = None,
    history_overflow: Literal["error", "hold", "slide", "grow"] = "error",
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        asset_id: Asset directory name inside the checkpoint. Defaults to the training data config's asset id.
        history_overflow: Behavior after a Transformer history cache reaches its trained maximum length. ``slide``
            keeps the most recent window, while ``grow`` doubles the cache capacity and preserves all prior
            history for effectively unbounded episodes.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    # Validate on the host, before allocating GPU memory or compiling kernels.
    params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
    validate_checkpoint_params(params, checkpoint_dir)
    model = train_config.model.load(jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.bfloat16), params))
    del params

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        norm_asset_id = asset_id or data_config.asset_id
        if norm_asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", norm_asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        history_overflow=history_overflow,
    )
