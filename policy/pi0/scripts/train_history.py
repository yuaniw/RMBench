import dataclasses
import functools
import logging
import platform
import time

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import train as _train
import wandb

from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import history_data_loader as _history_data_loader
from openpi.training import sharding
from openpi.training import utils as training_utils


def sample_history_input_modes(rng, anchors, probabilities):
    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    modes = jax.random.categorical(rng, jnp.log(probabilities), shape=anchors.shape)
    return jnp.where((modes == 2) & (anchors == 0), 1, modes)


def history_input_mode_masks(modes):
    return modes == 2, modes == 0, modes != 2


def encode_online_history_features(model, visual):
    """Encode one padded episode with the shared SigLIP and cut history gradients."""
    episode_images = visual.reshape((-1, *visual.shape[2:]))
    episode_images = episode_images.astype(jnp.float32) / 255.0 * 2.0 - 1.0
    visual_features = model.encode_history_images(episode_images)
    visual_features = visual_features.reshape(
        (*visual.shape[:2], visual_features.shape[-2], visual_features.shape[-1])
    )
    return jax.lax.stop_gradient(visual_features)


def history_grad_step(config: _config.TrainConfig, rng, state: training_utils.TrainState, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    (
        observation,
        actions,
        action_valid_mask,
        visual,
        history_states,
        history_mask,
        anchors,
        history_actions,
    ) = batch

    def loss_fn(model, loss_rng):
        mode_rng, flow_rng = jax.random.split(loss_rng)
        single_token_mode = model.history_conditioner.config.conditioning_mode == "single_token"
        if config.history_data.online_image_history:
            visual_features = encode_online_history_features(model, visual)
        else:
            visual_features = visual

        if single_token_mode:
            modes = sample_history_input_modes(
                mode_rng,
                anchors.reshape((-1,)),
                (
                    config.history_data.full_input_probability,
                    config.history_data.image_dropout_probability,
                    config.history_data.strict_past_probability,
                ),
            )
            use_previous, current_image_visible, current_state_visible = history_input_mode_masks(modes)
            history_condition, encoded_sequence = model.encode_single_token_history(
                visual_features,
                history_states,
                history_mask,
                anchors,
                use_previous=use_previous[None],
                train=True,
            )
        else:
            modes = jnp.zeros(anchors.shape[1:], dtype=jnp.int32)
            encoded_sequence = None
            current_image_visible = None
            current_state_visible = None
            history_condition = model.encode_history_conditions(
                visual_features,
                history_states,
                history_mask,
                anchors,
                train=True,
            )
        # The episode history stays replicated; only the gathered anchor batch is split across the FSDP mesh.
        history_condition = sharding.activation_sharding_constraint(history_condition)
        per_action_loss = model.compute_loss(
            flow_rng,
            observation,
            actions,
            train=config.history_data.augment_current_observation,
            history_condition=history_condition,
            current_image_visible=current_image_visible,
            current_state_visible=current_state_visible,
        )
        valid = action_valid_mask.astype(per_action_loss.dtype)
        per_anchor_loss = jnp.sum(per_action_loss * valid, axis=-1) / jnp.sum(valid, axis=-1)
        main_loss = jnp.mean(per_anchor_loss)
        history_action_loss = jnp.zeros((), dtype=main_loss.dtype)
        state_action_loss = jnp.zeros((), dtype=main_loss.dtype)
        if single_token_mode:
            history_action_predictions = model.predict_history_actions(encoded_sequence)
            state_action_predictions = model.predict_state_actions(history_states)
            target = history_actions.astype(history_action_predictions.dtype)
            aux_valid = history_mask[..., None].astype(history_action_predictions.dtype)
            history_action_loss = jnp.sum(
                jnp.square(history_action_predictions - target) * aux_valid
            ) / (jnp.sum(aux_valid) * target.shape[-1])
            state_action_loss = jnp.sum(
                jnp.square(state_action_predictions - target) * aux_valid
            ) / (jnp.sum(aux_valid) * target.shape[-1])
        loss = (
            main_loss
            + config.history_data.history_action_loss_weight * history_action_loss
            + config.history_data.state_action_loss_weight * state_action_loss
        )
        if model.history_conditioner.config.conditioning_mode == "film":
            scale, shift = jnp.split(model.history_modulation(history_condition), 2, axis=-1)
            info = {
                "history_condition_norm": jnp.mean(jnp.linalg.norm(history_condition, axis=-1)),
                "history_scale_abs": jnp.mean(jnp.abs(scale)),
                "history_shift_abs": jnp.mean(jnp.abs(shift)),
            }
        elif model.history_conditioner.config.conditioning_mode == "prefix_tokens":
            diagnostics = model.history_prefix_diagnostics(history_condition)
            info = {
                "history_token_norm": jnp.mean(diagnostics["token_norms"]),
                "history_token_variance": jnp.mean(diagnostics["token_variance"]),
                "history_token_pairwise_cosine": jnp.mean(diagnostics["token_pairwise_cosine"]),
            }
        else:
            valid_sequence = history_mask[..., None].astype(encoded_sequence.dtype)
            count = jnp.sum(valid_sequence, axis=1, keepdims=True)
            mean = jnp.sum(encoded_sequence * valid_sequence, axis=1, keepdims=True) / count
            temporal_variance = jnp.sum(
                jnp.square(encoded_sequence - mean) * valid_sequence
            ) / (jnp.sum(valid_sequence) * encoded_sequence.shape[-1])
            normalized = encoded_sequence / jnp.maximum(
                jnp.linalg.norm(encoded_sequence, axis=-1, keepdims=True), 1e-6
            )
            adjacent_valid = (history_mask[:, 1:] & history_mask[:, :-1]).astype(encoded_sequence.dtype)
            adjacent_cosine = jnp.sum(normalized[:, 1:] * normalized[:, :-1], axis=-1)
            info = {
                "history_token_norm": jnp.mean(jnp.linalg.norm(history_condition, axis=-1)),
                "history_temporal_variance": temporal_variance,
                "history_adjacent_cosine": jnp.sum(adjacent_cosine * adjacent_valid)
                / jnp.sum(adjacent_valid),
                "history_action_aux_loss": history_action_loss,
                "state_action_aux_loss": state_action_loss,
                "full_main_loss": jnp.sum(per_anchor_loss * (modes == 0))
                / jnp.maximum(jnp.sum(modes == 0), 1),
                "image_dropout_main_loss": jnp.sum(per_anchor_loss * (modes == 1))
                / jnp.maximum(jnp.sum(modes == 1), 1),
                "strict_past_main_loss": jnp.sum(per_anchor_loss * (modes == 2))
                / jnp.maximum(jnp.sum(modes == 2), 1),
                "full_fraction": jnp.mean(modes == 0),
                "image_dropout_fraction": jnp.mean(modes == 1),
                "strict_past_fraction": jnp.mean(modes == 2),
            }
        info["main_flow_loss"] = main_loss
        return loss, info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, rng)
    history_grads = grads.filter(nnx_utils.PathRegex(".*history_.*"))
    history_encoder_grads = grads.filter(
        nnx_utils.PathRegex(".*history_conditioner/(input_adapter|encoder)/.*")
    )
    history_action_head_grads = grads.filter(
        nnx_utils.PathRegex(".*history_conditioner/(action_head|action_head_norm)/.*")
    )
    llm_grads = grads.filter(nnx_utils.PathRegex(".*llm.*"))
    action_expert_grads = grads.filter(nnx_utils.PathRegex(".*llm.*_1.*"))
    action_lora_grads = grads.filter(nnx_utils.PathRegex(".*llm.*_1.*lora.*"))
    return grads, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "history_grad_norm": optax.global_norm(history_grads),
        "history_encoder_grad_norm": optax.global_norm(history_encoder_grads),
        "history_action_head_grad_norm": optax.global_norm(history_action_head_grads),
        "llm_grad_norm": optax.global_norm(llm_grads),
        "action_expert_grad_norm": optax.global_norm(action_expert_grads),
        "action_lora_grad_norm": optax.global_norm(action_lora_grads),
        **info,
    }


def apply_history_grads(config: _config.TrainConfig, state: training_utils.TrainState, grads):
    model = nnx.merge(state.model_def, state.params)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    return dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )


def make_history_batch_sharding(mesh: jax.sharding.Mesh, batch):
    anchor_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    observation, actions, action_valid_mask, visual, history_states, history_mask, anchors, history_actions = batch
    return (
        jax.tree.map(lambda _: anchor_sharding, observation),
        anchor_sharding,
        anchor_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
    )


def main(config: _config.TrainConfig):
    _train.init_logging()
    logging.info(f"Running history training on: {platform.node()}")
    if config.history_data is None:
        raise ValueError("train_history.py requires a history_data config.")
    if config.model.history is None:
        raise ValueError("train_history.py requires a history-enabled model.")
    if config.batch_size != 1:
        raise ValueError("History training uses one episode per dataloader batch.")
    if jax.device_count() != config.fsdp_devices:
        raise ValueError(
            "History training requires all visible devices to form one FSDP group: "
            f"got {jax.device_count()} visible devices and fsdp_devices={config.fsdp_devices}."
        )
    if config.history_data.anchors_per_episode % jax.device_count() != 0:
        raise ValueError("anchors_per_episode must be divisible by the number of visible devices.")
    probabilities = (
        config.history_data.full_input_probability,
        config.history_data.image_dropout_probability,
        config.history_data.strict_past_probability,
    )
    if not jnp.isclose(sum(probabilities), 1.0):
        raise ValueError("History input-mode probabilities must sum to one.")
    if any(probability < 0 for probability in probabilities):
        raise ValueError("History input-mode probabilities must be nonnegative.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    _train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    data_loader = _history_data_loader.create_history_data_loader(
        config, num_workers=config.num_workers, shuffle=True
    )
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    logging.info(f"Initialized history loader:\n{training_utils.array_tree_to_info(first_batch)}")

    state, state_sharding = _train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    if resuming:
        state = _checkpoints.restore_state(checkpoint_manager, state, data_loader)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    batch_sharding = make_history_batch_sharding(mesh, first_batch)
    grad_sharding = state_sharding.params.filter(config.trainable_filter)
    grad_fn = jax.jit(
        functools.partial(history_grad_step, config),
        in_shardings=(replicated_sharding, state_sharding, batch_sharding),
        out_shardings=(grad_sharding, replicated_sharding),
    )
    apply_fn = jax.jit(
        functools.partial(apply_history_grads, config),
        in_shardings=(state_sharding, grad_sharding),
        out_shardings=state_sharding,
        donate_argnums=(0, 1),
    )
    accumulate = config.history_data.gradient_accumulate_episodes
    start_step = int(state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps)
    batch = first_batch

    for step in pbar:
        step_started_at = time.perf_counter()
        accumulated_grads = None
        micro_infos = []
        for micro_step in range(accumulate):
            micro_rng = jax.random.fold_in(train_rng, step * accumulate + micro_step)
            with sharding.set_mesh(mesh):
                grads, info = grad_fn(micro_rng, state, batch)
            accumulated_grads = grads if accumulated_grads is None else jax.tree.map(
                jnp.add, accumulated_grads, grads
            )
            micro_infos.append(info)
            batch = next(data_iter)
        averaged_grads = jax.tree.map(lambda x: x / accumulate, accumulated_grads)
        state = apply_fn(state, averaged_grads)
        jax.block_until_ready(state)
        step_time_seconds = time.perf_counter() - step_started_at

        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(micro_infos)
            info = jax.device_get(jax.tree.map(jnp.mean, stacked))
            memory_stats = jax.local_devices()[0].memory_stats()
            info["step_time_seconds"] = step_time_seconds
            info["peak_device_memory_gib"] = memory_stats["peak_bytes_in_use"] / 1024**3
            pbar.write(", ".join(f"{key}={value:.4f}" for key, value in info.items()))
            wandb.log(info, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            save_step = step + 1 if step == config.num_train_steps - 1 else step
            _checkpoints.save_state(checkpoint_manager, state, data_loader, save_step)

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
