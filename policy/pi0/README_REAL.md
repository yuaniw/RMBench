# Franka 左臂双视角真机推理

本文说明如何加载 `memory_260915` 双视角真机模型，并把真实相机和关节状态接入 inference。模型输入是主视角 `cam_high`、左腕相机 `cam_left_wrist` 和左臂 8 维状态；右腕相机不使用。

当前训练配置名是：

```text
pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1
```

模型输出 50 步 action chunk，每步 8 维：左臂 7 个关节目标和 1 个夹爪目标。

## 1. 准备代码和环境

在 RMBench 根目录执行：

```bash
cd policy/pi0
uv sync
```

需要把这套真机训练对应的代码改动和仓库原有的 `policy/pi0/src` 一起保留。最小真机推理改动清单见：

```text
docs/franka_infer_git_add.txt
```

这 6 个文件必须来自同一个代码版本：

```text
src/openpi/models/history.py
src/openpi/models/pi0.py
src/openpi/policies/aloha_policy.py
src/openpi/policies/policy_config.py
src/openpi/training/config.py
src/openpi/shared/franka_memory.py
```

不要只把这 6 个文件复制到一个旧版 openpi。它们依赖仓库已有的 `openpi.models.model`、`openpi.policies.policy`、`openpi.transforms`、`openpi.training.checkpoints` 以及 `packages/openpi-client`。最稳妥的方式是从本仓库 checkout 同一个代码提交，再应用这 6 个文件。

第一次启动需要下载或准备 PaliGemma tokenizer 和 Python 依赖。若部署机器没有外网，需要提前把 tokenizer 放入 openpi 的下载缓存；只有模型权重已经在本地时，才设置 `HF_HUB_OFFLINE=1`。

## 2. 准备 checkpoint

推理 checkpoint 必须是一个完整的 Orbax step 目录。目录至少包含：

```text
<checkpoint>/
├── _CHECKPOINT_METADATA
├── params/                         # 完整模型参数，不能只传部分分片
└── assets/
    └── memory_260915-franka-left-2view-v1/
        └── norm_stats.json         # 必须和该 checkpoint 配套
```

当前本地训练目录是：

```text
real_data/migration_runs/memory_260915-franka-left-2view-v1/checkpoints/
pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1/
dual-30k-v1/<step>/
```

`<step>` 应替换成已经完整写完的 checkpoint，例如 `16500`。正式 30k 完成后，使用完整的 `30000` 目录；不要在异步保存尚未结束时复制目录，也不要混用不同 step 的 `params` 和 `assets`。

一个 step 通常约 5 GB。普通 GitHub Git 不适合上传这些权重，建议把整个 checkpoint 放到 Hugging Face Hub 或对象存储，再把下载地址写到实验记录里。推理只需要 `params/`、`assets/.../norm_stats.json` 和顶层 metadata，不需要 `train_state/`。

## 3. 最小加载代码

下面的代码在 `policy/pi0` 目录执行。`CHECKPOINT` 可以改为本地完整 step 目录；如果用 Hugging Face 或对象存储，则先下载成同样的目录结构。

```python
from pathlib import Path

import numpy as np

from openpi.policies import policy_config
from openpi.shared import franka_memory
from openpi.training import config


CHECKPOINT = Path(
    "real_data/migration_runs/memory_260915-franka-left-2view-v1/"
    "checkpoints/pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1/"
    "dual-30k-v1/16500"
)

cfg = config.get_config(franka_memory.TRAIN_CONFIG)
policy = policy_config.create_trained_policy(
    cfg,
    CHECKPOINT,
    default_prompt=franka_memory.PROMPT,
    history_overflow="grow",
)
```

`history_overflow="grow"` 会在在线 history 超过初始 384 帧时扩容并保留已有 history。每个新任务或新 episode 必须调用一次：

```python
policy.reset_history()
```

## 4. 构造一帧 observation

策略接口要求图像是 **CHW**，即 `[3, H, W]`；状态是 `[8]`。图像可以是 RGB `uint8` `[0,255]`，也可以是 RGB `float32` `[0,1]`。如果相机驱动给出 HWC `[H,W,3]`，先转成 CHW。

```python
def as_chw_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {image.shape}")
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] == 3:
        chw = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Expected RGB image in CHW or HWC format, got {image.shape}")
    if chw.dtype.kind == "f":
        if not np.isfinite(chw).all() or chw.min() < 0 or chw.max() > 1:
            raise ValueError("Float images must be finite RGB values in [0, 1]")
        return chw.astype(np.float32)
    if chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    return chw


def make_observation(front_hwc_rgb, left_wrist_hwc_rgb, measured_state):
    state = np.asarray(measured_state, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("measured_state must be finite with shape (8,)")
    return {
        "state": state,
        "images": {
            "cam_high": as_chw_rgb(front_hwc_rgb),
            "cam_left_wrist": as_chw_rgb(left_wrist_hwc_rgb),
        },
        "prompt": franka_memory.PROMPT,
    }
```

相机原始分辨率是：

```text
主视角：640×480 RGB
左腕：  424×240 RGB
```

不要在机器人端手动拉伸到正方形，也不要把两路图像拼成一张图。模型 transform 会自动执行等比例 resize 和黑边 padding 到 `224×224`：主视角有效区域是 `224×168`，左腕有效区域是 `224×126`。不要传 BGR；使用 OpenCV 时先执行 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`。

缺少 `cam_right_wrist` 是正确的，配置会自动生成黑色占位图并把 `right_wrist_0_rgb` mask 设为 False。不要传 `cam_low` 或 `cam_right_wrist` 的伪造图像。

## 5. 推理循环

`policy.infer()` 默认会把当前 observation 加入 history，并返回一整个 action chunk：

```python
policy.reset_history()

while episode_is_running:
    front = read_front_camera_rgb()       # HWC RGB, normally 480×640×3
    left_wrist = read_left_wrist_rgb()    # HWC RGB, normally 240×424×3
    measured_state = read_left_franka_state()  # [left_joint_1..7, left_gripper]

    observation = make_observation(front, left_wrist, measured_state)
    result = policy.infer(observation)
    action_chunk = np.asarray(result["actions"])

    if action_chunk.shape != (50, 8) or not np.isfinite(action_chunk).all():
        raise RuntimeError(f"Invalid policy output: {action_chunk.shape}")

    # Execute a short receding-horizon prefix, then read a new observation.
    execute_absolute_targets(action_chunk[:8])
```

`result["actions"]` 是 **绝对目标**，不是 delta：

```text
actions[:, :7] = absolute left joint targets
actions[:, 7]  = absolute gripper target
```

训练时 7 个关节使用相对当前 observation 的 chunk-origin delta，夹爪保持绝对值；推理输出 transform 已经把关节 delta 加回当前 state。执行器不能再次把当前 state 加到 `result["actions"]` 上，否则会重复加 delta。夹爪单位、开闭方向和安全限位仍需按真机驱动接口确认。

如果业务代码先调用 `policy.update_history(observation)`，后续必须用 `policy.infer(observation, update_history=False)`，避免同一帧被加入两次。通常直接使用上面的 `policy.infer(observation)` 即可。

## 6. 快速离线检查

在安装好依赖、准备好 checkpoint 后，可以先执行下面的检查，确认配置、输入和输出维度：

```bash
cd policy/pi0
PYTHONPATH=src:packages/openpi-client/src \
JAX_PLATFORMS=cpu \
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

from openpi.policies import policy_config
from openpi.shared import franka_memory
from openpi.training import config

checkpoint = Path("/path/to/dual-30k-v1/16500")
cfg = config.get_config(franka_memory.TRAIN_CONFIG)
policy = policy_config.create_trained_policy(cfg, checkpoint, history_overflow="grow")
policy.reset_history()
observation = {
    "state": np.zeros(8, np.float32),
    "images": {
        "cam_high": np.zeros((3, 480, 640), np.uint8),
        "cam_left_wrist": np.zeros((3, 240, 424), np.uint8),
    },
    "prompt": franka_memory.PROMPT,
}
actions = np.asarray(policy.infer(observation)["actions"])
assert actions.shape == (50, 8)
assert np.isfinite(actions).all()
print("Franka dual-view inference check passed:", actions.shape)
PY
```

首次 inference 会触发 JAX 编译，可能需要几十秒；后续相同 shape 的调用会快很多。CPU 只适合检查接口，真机部署应使用与 checkpoint 匹配的 CUDA/JAX 环境。

## 7. 常见错误

| 现象 | 原因和处理 |
|---|---|
| `History policy must observe at least one frame` | 先调用 `policy.reset_history()`，然后用至少一帧 observation 调 `infer()`。 |
| history 超过 384 帧后报错 | 创建 policy 时传 `history_overflow="grow"`。每个新任务仍要 reset。 |
| `images` 维度错误 | 输入需要 CHW `[3,H,W]`；HWC 相机帧先转置。 |
| 左腕 mask 为 False | `images` 中缺少 `cam_left_wrist`；双视角推理必须提供该键。 |
| 输出是 32 维 | 检查是否使用了 `franka_memory.TRAIN_CONFIG`，而不是预训练 Pi0 或 simulation config。真机配置输出应为 `[50,8]`。 |
| `params` 加载失败 | checkpoint 必须是完整 Orbax step 目录，不能只复制一个大分片或只复制 `train_state`。 |
| norm stats 找不到 | `assets/memory_260915-franka-left-2view-v1/norm_stats.json` 必须位于同一个 checkpoint 目录下。 |
| 夹爪动作异常 | 不要再次做 delta；检查真机驱动的夹爪单位、范围和开闭极性。 |

这套模型的 history cache 是训练阶段的离线主视角 SigLIP 特征。部署时不需要复制 `history_cache/`；在线 policy 会从当前主视角图像重新编码 history。双视角只在当前 observation 中增加左腕输入。
