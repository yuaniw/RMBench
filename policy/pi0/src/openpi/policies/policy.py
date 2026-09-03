from collections.abc import Sequence
import logging
import os
import pathlib
from typing import Any, Literal, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import history as _history
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        history_overflow: Literal["error", "hold", "slide", "grow"] = "error",
    ):
        self._history_enabled = model.history_conditioner is not None
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        if self._history_enabled:
            self._update_history = nnx_utils.module_jit(model.update_history)
            self._history_conditioning_mode = model.history_conditioner.config.conditioning_mode
            if self._history_conditioning_mode == "film":
                self._history_diagnostics = nnx_utils.module_jit(model.history_film_diagnostics)
            elif self._history_conditioning_mode == "adaln":
                self._history_diagnostics = nnx_utils.module_jit(model.history_adaln_diagnostics)
            else:
                self._history_diagnostics = nnx_utils.module_jit(model.history_prefix_diagnostics)
            self._init_history_cache = model.init_history_cache
            self._history_encoder_type = model.history_conditioner.config.encoder_type
            self._history_max_length = model.history_conditioner.config.max_length
            self._history_cache = self._init_history_cache(1)
            self._history_attention_debug = os.getenv("OPENPI_HISTORY_ATTENTION_DEBUG") == "1"
            self._last_history_attention = None
            if self._history_encoder_type == "transformer":
                self._history_attention_diagnostics = nnx_utils.module_jit(model.history_attention_diagnostics)
            self._history_condition = None
            self._history_overflow = history_overflow
            self._history_steps = 0
            self._last_history_diagnostics = {}
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def _prepare_inputs(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    def _history_has_capacity(self) -> bool:
        if self._history_encoder_type != "transformer":
            return True
        current_capacity = self._history_cache.input_tokens.shape[1]
        if int(self._history_cache.length) < current_capacity:
            return True
        if self._history_overflow == "hold":
            return False
        if self._history_overflow == "slide":
            return True
        if self._history_overflow == "grow":
            self._history_cache = _history.grow_transformer_history_cache(
                self._history_cache, current_capacity * 2
            )
            return True
        raise ValueError(f"History exceeds the configured maximum of {self._history_max_length} steps.")

    def update_history(self, obs: dict) -> bool:
        if not self._history_enabled:
            raise ValueError("Cannot update history for a policy without a history encoder.")
        if not self._history_has_capacity():
            return False
        inputs = self._prepare_inputs(obs)
        self._history_condition, self._history_cache = self._update_history(
            _model.Observation.from_dict(inputs), self._history_cache
        )
        self._history_steps += 1
        return True

    def reset_history(self) -> None:
        if not self._history_enabled:
            raise ValueError("Cannot reset history for a policy without a history encoder.")
        self._history_cache = self._init_history_cache(1)
        self._history_condition = None
        self._history_steps = 0
        self._last_history_diagnostics = {}
        self._last_history_attention = None

    def _maybe_print_history_attention(self) -> None:
        if not self._history_attention_debug or self._history_encoder_type != "transformer":
            return
        attention = np.asarray(self._history_attention_diagnostics(self._history_cache))
        self._last_history_attention = attention
        length = int(self._history_cache.length)
        mean_by_frame = attention[:, :, :, :length].mean(axis=(0, 1, 2))
        print(
            "[history-attention] "
            f"length={length} first_frame={mean_by_frame[0]:.6f} "
            f"weights={np.array2string(mean_by_frame, precision=4, separator=',')}"
        )

    @override
    def infer(self, obs: dict, *, update_history: bool = True) -> dict:  # type: ignore[misc]
        inputs = self._prepare_inputs(obs)
        if self._history_enabled:
            if update_history and self._history_has_capacity():
                self._history_condition, self._history_cache = self._update_history(
                    _model.Observation.from_dict(inputs), self._history_cache
                )
                self._history_steps += 1
            if self._history_condition is None:
                raise ValueError("History policy must observe at least one frame before action inference.")
            diagnostics = self._history_diagnostics(self._history_condition)
            self._last_history_diagnostics = {
                key: np.asarray(value[0]) for key, value in diagnostics.items()
            }
            self._last_history_diagnostics["history_length"] = self._history_steps
            self._maybe_print_history_attention()

        self._rng, sample_rng = jax.random.split(self._rng)
        if self._history_enabled:
            actions = self._sample_actions(
                sample_rng,
                _model.Observation.from_dict(inputs),
                history_condition=self._history_condition,
                **self._sample_kwargs,
            )
        else:
            actions = self._sample_actions(
                sample_rng,
                _model.Observation.from_dict(inputs),
                **self._sample_kwargs,
            )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    def get_history_film_diagnostics(self) -> dict[str, np.ndarray | int]:
        if not self._history_enabled:
            raise ValueError("History diagnostics require a history-enabled policy.")
        return self._last_history_diagnostics

    def get_history_diagnostics(self) -> dict[str, np.ndarray | int]:
        if not self._history_enabled:
            raise ValueError("History diagnostics require a history-enabled policy.")
        return self._last_history_diagnostics

    def get_history_attention_diagnostics(self) -> np.ndarray:
        if not self._history_enabled:
            raise ValueError("History diagnostics require a history-enabled policy.")
        return self._last_history_attention

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
