import csv
import pathlib

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        asset_id,
        checkpoint_id,
        pi0_step,
        eval_save_dir,
        history_overflow,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.asset_id = asset_id
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        self.history_enabled = config.model.history is not None
        if self.history_enabled:
            self.history_conditioning_mode = config.model.history.conditioning_mode
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            asset_id=self.asset_id,
            history_overflow=history_overflow,
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.history_film_dir = pathlib.Path(eval_save_dir) / "history_film"
        self.history_film_records = []
        self.history_film_episode = 0

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, *, advance_history=True):
        img_front, img_right, img_left = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
        if self.history_enabled and advance_history:
            self.policy.update_history(self.observation_window)

    def get_action(self, rollout_step):
        assert self.observation_window is not None, "update observation_window first!"
        result = self.policy.infer(self.observation_window, update_history=not self.history_enabled)
        if self.history_enabled:
            diagnostics = self.policy.get_history_diagnostics()
            self.history_film_records.append({
                "rollout_step": rollout_step,
                **diagnostics,
            })
        return result["actions"]

    def _save_history_film_diagnostics(self):
        if not self.history_film_records:
            return

        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        episode_dir = self.history_film_dir / f"episode_{self.history_film_episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        rollout_steps = np.asarray([record["rollout_step"] for record in self.history_film_records])
        history_lengths = np.asarray([record["history_length"] for record in self.history_film_records])
        history_condition = np.stack([record["history_condition"] for record in self.history_film_records])
        scale = np.stack([record["scale"] for record in self.history_film_records])
        shift = np.stack([record["shift"] for record in self.history_film_records])
        scale_dynamic = np.stack([record["scale_dynamic"] for record in self.history_film_records])
        shift_dynamic = np.stack([record["shift_dynamic"] for record in self.history_film_records])

        np.savez_compressed(
            episode_dir / "history_film.npz",
            rollout_step=rollout_steps,
            history_length=history_lengths,
            history_condition=history_condition,
            scale=scale,
            shift=shift,
            scale_dynamic=scale_dynamic,
            shift_dynamic=shift_dynamic,
        )

        summary_fields = (
            "policy_call",
            "rollout_step",
            "history_length",
            "scale_mean_abs",
            "shift_mean_abs",
            "scale_dynamic_mean_abs",
            "shift_dynamic_mean_abs",
            "scale_dynamic_max_abs",
            "shift_dynamic_max_abs",
        )
        with (episode_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields)
            writer.writeheader()
            for index in range(len(self.history_film_records)):
                writer.writerow({
                    "policy_call": index,
                    "rollout_step": rollout_steps[index],
                    "history_length": history_lengths[index],
                    "scale_mean_abs": np.mean(np.abs(scale[index])),
                    "shift_mean_abs": np.mean(np.abs(shift[index])),
                    "scale_dynamic_mean_abs": np.mean(np.abs(scale_dynamic[index])),
                    "shift_dynamic_mean_abs": np.mean(np.abs(shift_dynamic[index])),
                    "scale_dynamic_max_abs": np.max(np.abs(scale_dynamic[index])),
                    "shift_dynamic_max_abs": np.max(np.abs(shift_dynamic[index])),
                })

        figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        axes[0, 0].plot(rollout_steps, np.mean(np.abs(scale), axis=1), marker="o", label="|scale| mean")
        axes[0, 0].plot(rollout_steps, np.mean(np.abs(shift), axis=1), marker="o", label="|shift| mean")
        axes[0, 0].set_title("Total FiLM magnitude (includes bias)")
        axes[0, 0].set_ylabel("Mean absolute value")
        axes[0, 0].legend()

        axes[0, 1].plot(
            rollout_steps,
            np.mean(np.abs(scale_dynamic), axis=1),
            marker="o",
            label="|dynamic scale| mean",
        )
        axes[0, 1].plot(
            rollout_steps,
            np.mean(np.abs(shift_dynamic), axis=1),
            marker="o",
            label="|dynamic shift| mean",
        )
        axes[0, 1].set_title("History-dependent FiLM magnitude")
        axes[0, 1].legend()

        scale_image = axes[1, 0].imshow(scale_dynamic.T, aspect="auto", origin="lower", cmap="coolwarm")
        axes[1, 0].set_title("History-dependent scale by channel")
        axes[1, 0].set_ylabel("Action-expert channel")
        figure.colorbar(scale_image, ax=axes[1, 0])
        shift_image = axes[1, 1].imshow(shift_dynamic.T, aspect="auto", origin="lower", cmap="coolwarm")
        axes[1, 1].set_title("History-dependent shift by channel")
        figure.colorbar(shift_image, ax=axes[1, 1])

        tick_positions = np.arange(len(rollout_steps))
        tick_labels = [f"{step}\n(h={length})" for step, length in zip(rollout_steps, history_lengths, strict=True)]
        for axis in axes[1]:
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, rotation=45, ha="right")
            axis.set_xlabel("Rollout step (history length)")
        for axis in axes[0]:
            axis.set_xlabel("Rollout step")
            axis.grid(alpha=0.25)
        figure.suptitle(f"History FiLM diagnostics - episode {self.history_film_episode}")
        figure.savefig(episode_dir / "history_film.png", dpi=160)
        plt.close(figure)

        print(f"Saved history FiLM diagnostics to {episode_dir}")
        self.history_film_records = []
        self.history_film_episode += 1

    def _save_history_prefix_diagnostics(self):
        if not self.history_film_records:
            return

        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        episode_dir = self.history_film_dir / f"episode_{self.history_film_episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        rollout_steps = np.asarray([record["rollout_step"] for record in self.history_film_records])
        history_lengths = np.asarray([record["history_length"] for record in self.history_film_records])
        history_tokens = np.stack([record["history_tokens"] for record in self.history_film_records])
        token_norms = np.stack([record["token_norms"] for record in self.history_film_records])
        token_variance = np.asarray([record["token_variance"] for record in self.history_film_records])
        token_cosine = np.asarray([
            record["token_pairwise_cosine"] for record in self.history_film_records
        ])

        np.savez_compressed(
            episode_dir / "history_prefix.npz",
            rollout_step=rollout_steps,
            history_length=history_lengths,
            history_tokens=history_tokens,
            token_norms=token_norms,
            token_variance=token_variance,
            token_pairwise_cosine=token_cosine,
        )

        figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for token_index in range(token_norms.shape[1]):
            axes[0].plot(rollout_steps, token_norms[:, token_index], label=f"token {token_index}")
        axes[0].set_title("History-token norms")
        axes[0].set_xlabel("Rollout step")
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].plot(rollout_steps, token_variance, marker="o")
        axes[1].set_title("Variance across history tokens")
        axes[1].set_xlabel("Rollout step")
        axes[2].plot(rollout_steps, token_cosine, marker="o")
        axes[2].set_title("Mean pairwise token cosine")
        axes[2].set_xlabel("Rollout step")
        figure.suptitle(f"History prefix diagnostics - episode {self.history_film_episode}")
        figure.savefig(episode_dir / "history_prefix.png", dpi=160)
        plt.close(figure)

        print(f"Saved history prefix diagnostics to {episode_dir}")
        self.history_film_records = []
        self.history_film_episode += 1

    def reset_obsrvationwindows(self):
        if self.history_enabled:
            if self.history_conditioning_mode == "film":
                self._save_history_film_diagnostics()
            else:
                self._save_history_prefix_diagnostics()
        self.instruction = None
        self.observation_window = None
        if self.history_enabled:
            self.policy.reset_history()
        print("successfully unset obs and language intruction")
